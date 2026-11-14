import os
import glob
import pandas as pd
import numpy as np
import allantools as at
import matplotlib.pyplot as plt
from scipy.stats import chi2

# ============================================================
# CONFIGURAZIONE
# ============================================================
CARTELLA = r"D:\Gaetano\SID\dati\Sdruno"
STORICO_NSY = os.path.join(CARTELLA, "Storico_NSY")

# Catalogo completo delle varianti storiche.
# Lo script userà SOLO quelle per cui esiste almeno un file
# nei giorni richiesti (auto-detect). Così BW01/BW02/PESATO
# non generano warning se il tracker non li produce più, ma
# restano pronti se in futuro (es. PESATO 2027) tornano attivi.
PREFISSI = {
    "V1": "GAESID_NSY_V1",
    "V2": "GAESID_NSY",
    "BW02": "GAESID_NSY_BW02",
    "BW01": "GAESID_NSY_BW01",
    "PESATO": "GAESID_NSY_PESATO"
}

MASCHERA_RESET_S = 600
MAX_GAP_S = 5

# QUI INSERISCI I GIORNI CHE VUOI ANALIZZARE
GIORNI_DA_ANALIZZARE = ["2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06"]

# ============================================================
# RICERCA FILE (Sdruno + Storico_NSY)
# ============================================================
def trova_file(pre, data_str, verbose=False):
    """
    Catena di fallback per il file di una variante in un dato giorno:
      1) Sdruno non-ricorsiva:      <CARTELLA>/<pre>_<data>.csv
      2) Storico_NSY ricorsiva:     <STORICO_NSY>/**/<pre>_<data>.csv
    Se il fallback (2) restituisce più candidati, sceglie il più
    recente per mtime e (se verbose) logga gli altri in console.
    Restituisce (path, provenienza) oppure (None, None).
    """
    fname = f"{pre}_{data_str}.csv"

    # 1) Sdruno (non ricorsiva)
    p1 = os.path.join(CARTELLA, fname)
    if os.path.exists(p1):
        return p1, "Sdruno"

    # 2) Storico_NSY (ricorsiva)
    if os.path.isdir(STORICO_NSY):
        candidati = glob.glob(os.path.join(STORICO_NSY, "**", fname), recursive=True)
        if candidati:
            candidati.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            if verbose and len(candidati) > 1:
                print(f"[INFO] {fname}: {len(candidati)} candidati in Storico_NSY, "
                      f"uso il più recente ({os.path.basename(os.path.dirname(candidati[0]))}).")
                for c in candidati[1:]:
                    print(f"       ignorato: {c}")
            return candidati[0], "Storico_NSY"

    return None, None

def trova_ultimo_giorno():
    """Ultimo giorno per cui esiste un file V2, cercando in entrambe le cartelle."""
    pat = "GAESID_NSY_2026-*.csv"
    files = glob.glob(os.path.join(CARTELLA, pat))
    if os.path.isdir(STORICO_NSY):
        files += glob.glob(os.path.join(STORICO_NSY, "**", pat), recursive=True)
    if not files:
        return None
    # Estrae la data dal nome file (robusto al path)
    date_strs = []
    for f in files:
        base = os.path.basename(f)
        try:
            date_strs.append(base.split("_")[-1].replace(".csv", ""))
        except Exception:
            pass
    if not date_strs:
        return None
    return sorted(date_strs)[-1]

def prefissi_presenti(date_list, catalogo=PREFISSI):
    """Restituisce solo i prefissi con almeno un file nei giorni richiesti."""
    presenti = {}
    for nome, pre in catalogo.items():
        for data_str in date_list:
            p, _ = trova_file(pre, data_str)
            if p is not None:
                presenti[nome] = pre
                break
    return presenti

# ============================================================
# CARICAMENTO E ALLINEAMENTO
# ============================================================
def carica_e_allinea(date_list, prefissi_attivi, mask_s=MASCHERA_RESET_S, max_gap_s=MAX_GAP_S):
    dfs_grezzi = {nome: [] for nome in prefissi_attivi.keys()}

    for data_str in date_list:
        for nome, pre in prefissi_attivi.items():
            p, prov = trova_file(pre, data_str, verbose=True)
            if p is not None:
                if prov == "Storico_NSY":
                    print(f"[INFO] {nome} {data_str}: caricato da Storico_NSY ({p})")
                d = pd.read_csv(p, comment="#")
                d["Timestamp_UTC"] = pd.to_datetime(d["Timestamp_UTC"], utc=True)
                d = d.drop_duplicates('Timestamp_UTC')
                
                # Carica Corr_Mag solo se esiste (V2 ha 9 colonne, V1 ne ha 7)
                # FIX: Includere "Timestamp_UTC" nella lista delle colonne da mantenere!
                cols = ["Timestamp_UTC", "PLL_Freq_Hz", "Reset_Flag"]
                if "Corr_Mag" in d.columns:
                    cols.append("Corr_Mag")
                d = d[cols]
                dfs_grezzi[nome].append(d)
            else:
                print(f"[WARN] File mancante: {nome} per il giorno {data_str}")

    dfs_completi = {}
    for nome, lista_df in dfs_grezzi.items():
        if lista_df:
            df_n = pd.concat(lista_df).set_index("Timestamp_UTC").sort_index()
            rename_dict = {
                "PLL_Freq_Hz": f"Freq_{nome}", 
                "Reset_Flag": f"Reset_{nome}"
            }
            if "Corr_Mag" in df_n.columns:
                rename_dict["Corr_Mag"] = f"Corr_{nome}"
            dfs_completi[nome] = df_n.rename(columns=rename_dict)

    if not dfs_completi: return None

    # OUTER join per non perdere dati se una variante manca
    df_merged = pd.concat(dfs_completi.values(), axis=1, join='outer')

    # 1) Individua i gap sull'indice ORIGINALE (prima del reindex)
    gap_mask_orig = df_merged.index.to_series().diff().dt.total_seconds().fillna(1) > max_gap_s
    gap_starts = df_merged.index[gap_mask_orig]

    # 2) Reindex a 1 Hz
    full_idx = pd.date_range(df_merged.index[0], df_merged.index[-1], freq="1s")
    df_full = df_merged.reindex(full_idx)

    # 3) Costruisci maschera unificata a 1 Hz
    mask_1s = pd.Series(False, index=df_full.index)
    
    for nome in dfs_completi.keys():
        # Reset flag
        if f'Reset_{nome}' in df_full.columns:
            mask_1s |= (df_full[f'Reset_{nome}'].fillna(0) != 0)
        # Maschera Corr_Mag (Lock Marginale) per V2 (soglia 0.4)
        if f'Corr_{nome}' in df_full.columns:
            mask_1s |= (df_full[f'Corr_{nome}'].fillna(0) < 0.4)

    # Aggiungi i gap alla maschera (arrotondando al secondo)
    for t in gap_starts:
        t_floor = t.floor('s')
        if t_floor in mask_1s.index:
            mask_1s.loc[t_floor] = True

    # 4) Espandi la maschera in avanti per mask_s secondi (600s)
    bad_mask = mask_1s.rolling(mask_s, min_periods=1).max().astype(bool)

    freq_cols = [c for c in df_full.columns if c.startswith("Freq_")]
    df_full.loc[bad_mask, freq_cols] = np.nan

    return df_full

# ============================================================
# ADEV ROBUSTO
# ============================================================
def calcola_adev_robusto(serie_freq, tau):
    is_valid = ~np.isnan(serie_freq.values)
    groups = np.cumsum(np.concatenate([[True], is_valid[1:] != is_valid[:-1]]))

    adevs, weights, edfs = [], [], []
    min_len = max(5 * tau, 1200)

    for g in np.unique(groups):
        mask_g = groups == g
        if not is_valid[mask_g][0]: continue

        segment = serie_freq.values[mask_g]
        segment = segment[~np.isnan(segment)]

        if len(segment) >= min_len:
            t, a, e, n = at.oadev(segment, rate=1.0, data_type="freq", taus=[tau])
            if not np.isnan(a[0]):
                adevs.append(a[0])
                weights.append(max(n[0], 0))
                edf = 0.5 * (len(segment) / tau - 1)
                edfs.append(edf)

    if not adevs or sum(weights) == 0:
        return np.nan, np.nan, np.nan

    combined_adev = np.sqrt(np.sum(np.array(weights) * np.array(adevs)**2) / np.sum(weights))

    E = sum(edfs)
    if E < 10:
        return combined_adev, np.nan, np.nan

    V = combined_adev**2
    lo = np.sqrt(V * E / chi2.ppf(0.975, E))
    hi = np.sqrt(V * E / chi2.ppf(0.025, E))

    return combined_adev, lo, hi

# ============================================================
# ANALISI
# ============================================================
def analizza(date_list):
    print(f"\n{'='*60}\n ANALISI MULTI-PLL: {', '.join(date_list)}\n{'='*60}")

    prefissi_attivi = prefissi_presenti(date_list)
    if not prefissi_attivi:
        print("Nessun file trovato per i giorni richiesti.")
        return
    print(f"Varianti rilevate: {', '.join(prefissi_attivi.keys())}")

    df = carica_e_allinea(date_list, prefissi_attivi)
    if df is None:
        print("Impossibile procedere: file mancanti o non allineati.")
        return

    freq_cols = [c for c in df.columns if c.startswith("Freq_")]
    if not freq_cols:
        print("Nessuna colonna di frequenza trovata.")
        return

    ore = len(df.dropna(subset=[freq_cols[0]])) / 3600.0
    print(f"Ore totali valide analizzate: {ore:.2f}")

    print(f"\n--- ALLAN DEVIATION (con IC 95%) ---")
    print(f"{'Config':<8} | {'ADEV 300s [mHz]':<22} | {'ADEV 600s [mHz]':<22}")
    print("-" * 60)
    for nome in prefissi_attivi.keys():
        if f'Freq_{nome}' in df.columns:
            a300, lo300, hi300 = calcola_adev_robusto(df[f'Freq_{nome}'], 300)
            a600, lo600, hi600 = calcola_adev_robusto(df[f'Freq_{nome}'], 600)

            str300 = f"{a300*1e3:.4f} [{lo300*1e3:.4f}-{hi300*1e3:.4f}]" if not np.isnan(lo300) else f"{a300*1e3:.4f} (n/a)"
            str600 = f"{a600*1e3:.4f} [{lo600*1e3:.4f}-{hi600*1e3:.4f}]" if not np.isnan(lo600) else f"{a600*1e3:.4f} (n/a)"

            print(f"{nome:<8} | {str300:<22} | {str600:<22}")

    print(f"\n--- GLITCH RATE ---")
    print(f"{'Config':<8} | {'Rate 6*MAD (/h)':<15} | {'Rate 20mHz (/h)':<15} | {'RMS vs V2 (mHz)':<15}")
    print("-" * 70)

    ref_freq = df['Freq_V2'] if 'Freq_V2' in df.columns else None

    for nome in prefissi_attivi.keys():
        if f'Freq_{nome}' in df.columns:
            freq = df[f'Freq_{nome}']
            diff = freq.diff().abs()
            valid_diff = diff.dropna()
            if len(valid_diff) < 100: continue

            mad = valid_diff.rolling(window=3600, center=True, min_periods=60).median().fillna(valid_diff.median())

            rate_mad = (valid_diff > 6 * mad).sum() / ore
            rate_abs = (valid_diff > 0.020).sum() / ore

            rms_str = "-"
            if ref_freq is not None and nome != "V2":
                valid = pd.concat([freq, ref_freq], axis=1).dropna()
                if len(valid) > 100:
                    rms_val = np.sqrt(np.mean((valid.iloc[:,0] - valid.iloc[:,1])**2))
                    rms_str = f"{rms_val*1e3:.3f}"

            print(f"{nome:<8} | {rate_mad:>15.2f} | {rate_abs:>15.2f} | {rms_str:>15}")

    # --- GRAFICO ADEV ---
    plt.figure(figsize=(10, 6))
    taus = np.logspace(1, 3.5, 15)
    colors = {"V1": "black", "V2": "blue", "BW02": "green",
              "BW01": "orange", "PESATO": "red"}

    for nome in prefissi_attivi.keys():
        if f'Freq_{nome}' not in df.columns: continue
        adevs = [calcola_adev_robusto(df[f'Freq_{nome}'], int(tau))[0] for tau in taus]
        plt.loglog(taus, np.array(adevs)*1e3, label=nome,
                   color=colors.get(nome, 'gray'), marker='o', markersize=3)

    plt.xlabel("Tempo di media tau (s)")
    plt.ylabel("Allan Deviation (mHz)")
    plt.title(f"Confronto Allan Deviation - {', '.join(date_list)}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png = f"ADEV_multi_pll_{'_'.join(date_list)}.png"
    plt.savefig(out_png, dpi=150)
    print(f"\nGrafico ADEV salvato in: {out_png}")
    plt.show()

if __name__ == "__main__":
    date_list = GIORNI_DA_ANALIZZARE
    if not date_list:
        ultimo = trova_ultimo_giorno()
        if ultimo:
            date_list = [ultimo]
            print(f"Nessun giorno specificato. Uso l'ultimo disponibile: {ultimo}")
        else:
            print("Nessun file trovato nella cartella.")
            exit()

    analizza(date_list)