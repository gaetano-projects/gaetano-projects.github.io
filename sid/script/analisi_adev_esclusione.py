import pandas as pd
import numpy as np
import allantools as at
import os

CARTELLA = r"D:\Gaetano\SID\dati\Sdruno"
GIORNI = ["2026-10-26", "2026-10-27"]
SOGLIA_EVENTO_MHZ = 20.0
MARGINE_S = 120  # ±120s attorno al picco

def carica_e_allinea(giorni):
    dfs = {}
    prefissi = {"V1": "GAESID_NSY_V1", "V2": "GAESID_NSY", "BW02": "GAESID_NSY_BW02",
                "BW01": "GAESID_NSY_BW01", "PESATO": "GAESID_NSY_PESATO"}
    
    for nome, pre in prefissi.items():
        lista = []
        for g in giorni:
            p = os.path.join(CARTELLA, f"{pre}_{g}.csv")
            if os.path.exists(p):
                d = pd.read_csv(p, comment="#")
                d['Timestamp_UTC'] = pd.to_datetime(d['Timestamp_UTC'], utc=True)
                d = d.drop_duplicates('Timestamp_UTC').set_index('Timestamp_UTC')
                d = d[['PLL_Freq_Hz', 'Reset_Flag']].rename(columns={
                    'PLL_Freq_Hz': f'Freq_{nome}',
                    'Reset_Flag': f'Reset_{nome}'
                })
                lista.append(d)
        if lista:
            dfs[nome] = pd.concat(lista).sort_index()
    
    if not dfs: return None
    df = pd.concat(dfs.values(), axis=1, join='inner')
    
    # Rimuovi reset
    reset = pd.Series(False, index=df.index)
    for n in dfs.keys():
        reset |= (df[f'Reset_{n}'].fillna(1) != 0)
    df = df[~reset]
    return df

def identifica_eventi(df):
    """Trova i timestamp in cui la V2 supera la soglia."""
    diff_v2 = df['Freq_V2'].diff().abs() * 1000  # in mHz
    eventi = df[diff_v2 > SOGLIA_EVENTO_MHZ].index
    return eventi

def maschera_eventi(df, eventi, margine_s=MARGINE_S):
    """Crea una maschera booleana: True = dato valido, False = da escludere."""
    mask = pd.Series(True, index=df.index)
    for t in eventi:
        t_start = t - pd.Timedelta(seconds=margine_s)
        t_end = t + pd.Timedelta(seconds=margine_s)
        mask &= ~((df.index >= t_start) & (df.index <= t_end))
    return mask

def calcola_adev_robusto(serie_freq, tau):
    """ADEV calcolata su segmenti contigui, come validato."""
    is_valid = ~np.isnan(serie_freq.values)
    groups = np.cumsum(np.concatenate([[True], is_valid[1:] != is_valid[:-1]]))
    
    adevs, weights = [], []
    min_len = max(5 * tau, 1200)
    
    for g in np.unique(groups):
        mask_g = groups == g
        if not is_valid[mask_g][0]: continue
        segment = serie_freq.values[mask_g]
        segment = segment[~np.isnan(segment)]
        if len(segment) >= min_len:
            t, a, _, n = at.oadev(segment, rate=1.0, data_type="freq", taus=[tau])
            if not np.isnan(a[0]):
                adevs.append(a[0])
                weights.append(max(n[0], 0))
    
    if not adevs or sum(weights) == 0: return np.nan
    return np.sqrt(np.sum(np.array(weights) * np.array(adevs)**2) / np.sum(weights))

def main():
    print("Caricamento dati...")
    df = carica_e_allinea(GIORNI)
    if df is None:
        print("Dati non trovati.")
        return
    
    print(f"Ore totali: {len(df)/3600:.1f}\n")
    
    # 1. Identifica eventi sulla V2
    eventi = identifica_eventi(df)
    print(f"Eventi > {SOGLIA_EVENTO_MHZ} mHz sulla V2: {len(eventi)}")
    print(f"Finestra di esclusione: ±{MARGINE_S} s attorno a ciascun evento\n")
    
    # 2. Crea maschera (applicata a tutte le configurazioni)
    mask = maschera_eventi(df, eventi)
    print(f"Secondi validi dopo l'esclusione: {mask.sum()} su {len(df)} ({100*mask.sum()/len(df):.1f}%)\n")
    
    # 3. Calcola ADEV su serie completa e su serie "pulita"
    print("--- ADEV (serie completa vs serie con eventi esclusi) ---")
    print(f"{'Config':<8} | {'ADEV 300s':<20} | {'ADEV 600s':<20}")
    print(f"{'':<8} | {'Completa -> Pulita':<20} | {'Completa -> Pulita':<20}")
    print("-" * 60)
    
    for nome in ["V1", "V2", "BW02", "BW01", "PESATO"]:
        if f'Freq_{nome}' not in df.columns: continue
        
        freq = df[f'Freq_{nome}']
        
        # ADEV completa
        a300_full = calcola_adev_robusto(freq, 300)
        a600_full = calcola_adev_robusto(freq, 600)
        
        # ADEV pulita (applica maschera)
        freq_pulita = freq.where(mask)
        a300_pulita = calcola_adev_robusto(freq_pulita, 300)
        a600_pulita = calcola_adev_robusto(freq_pulita, 600)
        
        # Calcola la riduzione percentuale
        red300 = (1 - a300_pulita/a300_full) * 100 if a300_full > 0 else 0
        red600 = (1 - a600_pulita/a600_full) * 100 if a600_full > 0 else 0
        
        print(f"{nome:<8} | {a300_full*1e3:.3f} -> {a300_pulita*1e3:.3f} ({red300:+.1f}%) | {a600_full*1e3:.3f} -> {a600_pulita*1e3:.3f} ({red600:+.1f}%)")
    
    # 4. Calcola la percentuale di varianza spiegata dagli eventi (solo V2)
    print("\n--- QUOTA DI VARIANZA SPIEGATA DAGLI EVENTI (V2) ---")
    freq_v2 = df['Freq_V2']
    var_totale = np.var(freq_v2.dropna())  # varianza su tutta la serie
    var_pulita = np.var(freq_v2.where(mask).dropna())  # varianza sui dati puliti
    
    # Varianza spiegata = (var_totale - var_pulita) / var_totale
    # Ma attenzione: la varianza dei dati "puliti" è calcolata su meno campioni, 
    # quindi questa formula è approssimativa. Usiamo la varianza dei residui.
    
    # Stima più robusta: la varianza dei dati esclusi
    freq_esclusi = freq_v2[~mask]
    if len(freq_esclusi) > 0:
        # La varianza spiegata può essere stimata come 
        # (var_totale - var_quiete) / var_totale
        quota = (1 - var_pulita / var_totale) * 100 if var_totale > 0 else 0
        print(f"  Varianza totale (V2): {var_totale:.6f} Hz^2")
        print(f"  Varianza su serie pulita: {var_pulita:.6f} Hz^2")
        print(f"  Varianza spiegata dagli eventi: {quota:.1f}%")
        print(f"  (Nota: stima approssimativa, la varianza su meno campioni è meno precisa)")
    
    # 5. Verifica finale: la V2 pulita è ora simile alla V1?
    print("\n--- VERIFICA FINALE ---")
    v2_pulita_600 = calcola_adev_robusto(df['Freq_V2'].where(mask), 600)
    v1_pulita_600 = calcola_adev_robusto(df['Freq_V1'].where(mask), 600)
    
    if not np.isnan(v2_pulita_600) and not np.isnan(v1_pulita_600):
        ratio = v2_pulita_600 / v1_pulita_600
        print(f"  ADEV 600s (pulita): V2={v2_pulita_600*1e3:.3f} mHz, V1={v1_pulita_600*1e3:.3f} mHz")
        print(f"  Rapporto V2/V1: {ratio:.2f}")
        
        if ratio < 1.3:
            print("  -> La V2 pulita è simile alla V1. Gli eventi spiegano la differenza.")
            print("  -> Ipotesi confermata: l'ADEV extra della V2 è dovuta ai transienti fisici.")
        else:
            print("  -> La V2 pulita è ancora più rumorosa della V1. C'è una componente di rumore intrinseco.")
            print("  -> Potrebbe essere necessario indagare ulteriormente.")

if __name__ == "__main__":
    main()