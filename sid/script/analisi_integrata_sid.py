import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, wilcoxon, binomtest

# ============================================================
# CONFIGURAZIONE
# ============================================================
DATA_DIR = r"D:\Gaetano\SID\dati\Sdruno"
FLARE_DIR = r"D:\Gaetano\SID\dati\Sdruno\Storico_NSY\Flares"
XRAY_DIR = r"D:\Gaetano\SID\dati\Sdruno\Storico_NSY\XRAY"

DATE = "2026-10-26"
PLLS = ["V1", "V2"]
PREFISSI = {"V1": "GAESID_NSY_V1", "V2": "GAESID_NSY"}

PRE_MIN = 10          # minuti pre-flare per baseline
POST_MIN = 45         # minuti post-onset per analisi
SOGLIE_MHZ = [0.5, 1.0, 2.0, 5.0, 10.0]  # soglie di rilevazione
TARGET_FA_AL_GIORNO = 1.0
SMOOTH_S = 10         # smoothing del Doppler (secondi)

# ============================================================
# CARICAMENTO DATI
# ============================================================
def carica_flares(date_str):
    path = os.path.join(FLARE_DIR, f"FLARES_{date_str}.json")
    if not os.path.exists(path):
        print(f"[WARN] File flares mancante: {path}")
        return []
    with open(path, 'r') as f:
        raw = json.load(f)
    flares = []
    for f in raw:
        flares.append({
            'onset': pd.to_datetime(f['inizio'], utc=True),
            'peak': pd.to_datetime(f['picco'], utc=True),
            'end': pd.to_datetime(f['fine'], utc=True),
            'classe': f['classe'],
            'tipo': f['tipo'],
            'limb': f.get('limb', False),
        })
    return flares

def carica_goes(date_str):
    path = os.path.join(XRAY_DIR, f"GOES_XRAY_{date_str}.csv")
    if not os.path.exists(path):
        print(f"[WARN] File GOES mancante: {path}")
        return None
    df = pd.read_csv(path, comment='#', names=['timestamp_utc','flux_wm2'])
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
    df = df.set_index('timestamp_utc').sort_index()
    # dX/dt a 1 min (W/m²/s)
    df['dxdt'] = df['flux_wm2'].diff() / 60.0
    return df

def carica_tracker(date_str):
    dfs = {}
    for pll, pref in PREFISSI.items():
        path = os.path.join(DATA_DIR, f"{pref}_{date_str}.csv")
        if not os.path.exists(path):
            print(f"[WARN] File tracker mancante: {path}")
            continue
        d = pd.read_csv(path, comment='#')
        d['Timestamp_UTC'] = pd.to_datetime(d['Timestamp_UTC'], utc=True)
        d = d.drop_duplicates('Timestamp_UTC').set_index('Timestamp_UTC').sort_index()
        dfs[pll] = d
    return dfs

# ============================================================
# ANALISI SINGOLO FLARE
# ============================================================
def analizza_flare(tracker_df, goes_df, flare, pll):
    t0 = flare['onset']
    t_start = t0 - pd.Timedelta(minutes=PRE_MIN)
    t_end = t0 + pd.Timedelta(minutes=POST_MIN)

    # Estrai finestra tracker
    seg = tracker_df.loc[t_start:t_end, 'PLL_Freq_Hz'].dropna()
    if len(seg) < 60:
        return None

    # Baseline (pre-flare)
    pre_mask = seg.index < t0
    baseline = np.median(seg[pre_mask].values) if pre_mask.sum() > 0 else seg.iloc[0]
    doppler = (seg.values - baseline) * 1000  # mHz

    # Smoothing
    doppler_smooth = pd.Series(doppler).rolling(SMOOTH_S, center=True, min_periods=1).mean().values

    # Finestra evento
    post_mask = seg.index >= t0
    doppler_ev = doppler_smooth[post_mask]
    times_ev = seg.index[post_mask]

    if len(doppler_ev) == 0:
        return None

    # Rilevazione a varie soglie
    risultati = []
    for soglia in SOGLIE_MHZ:
        sopra = np.abs(doppler_ev) > soglia
        rilevato = bool(sopra.any())
        latenza = int(np.argmax(sopra)) if rilevato else np.nan
        picco = float(np.max(np.abs(doppler_ev)))
        risultati.append({
            'PLL': pll, 'soglia_mHz': soglia,
            'rilevato': rilevato, 'latenza_s': latenza,
            'picco_mHz': picco,
        })

    # Correlazione con dX/dt (interpolato a 1 Hz)
    corr_val = np.nan
    lag_ottimo = np.nan
    if goes_df is not None:
        goes_win = goes_df.loc[t_start:t_end]
        if len(goes_win) > 10:
            dxdt_interp = np.interp(
                times_ev.astype('int64') / 1e9,
                goes_win.index.astype('int64') / 1e9,
                goes_win['dxdt'].fillna(0).values
            )
            if np.std(dxdt_interp) > 0 and np.std(doppler_ev) > 0:
                # Cross-correlazione per lag
                corr = np.correlate(doppler_ev - np.mean(doppler_ev),
                                    dxdt_interp - np.mean(dxdt_interp), mode='full')
                lag_ottimo = (np.argmax(np.abs(corr)) - (len(dxdt_interp) - 1))
                corr_val = np.corrcoef(doppler_ev, dxdt_interp)[0, 1]

    for r in risultati:
        r['corr_dxdt'] = corr_val
        r['lag_s'] = lag_ottimo
        r['classe'] = flare['classe']
        r['onset'] = t0
    return risultati

# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*70}")
    print(f" ANALISI FLARE {DATE}")
    print(f"{'='*70}")

    flares = carica_flares(DATE)
    goes = carica_goes(DATE)
    tracker = carica_tracker(DATE)

    if not flares:
        print("Nessun flare trovato.")
        return
    if not tracker:
        print("Nessun dato tracker trovato.")
        return

    print(f"Flare trovati: {len(flares)}")
    print(f"Tracker: {list(tracker.keys())}")
    if goes is not None:
        print(f"GOES: {len(goes)} punti")

    tutti = []
    for flare in flares:
        print(f"\n--- {flare['onset'].strftime('%H:%M')} {flare['classe']} (limb={flare['limb']}) ---")
        for pll, df in tracker.items():
            res = analizza_flare(df, goes, flare, pll)
            if res:
                # Prendi il picco
                r0 = res[0]
                print(f"  {pll}: picco={r0['picco_mHz']:+7.2f} mHz  "
                      f"corr={r0['corr_dxdt'] if not np.isnan(r0['corr_dxdt']) else 'n/a'}  "
                      f"lag={r0['lag_s']:.0f}s" if not np.isnan(r0['lag_s']) else "")
                tutti.extend(res)

    if not tutti:
        print("Nessun risultato.")
        return

    df = pd.DataFrame(tutti)
    df.to_csv(f"risultati_flare_{DATE}.csv", index=False)
    print(f"\nSalvati {len(df)} risultati in risultati_flare_{DATE}.csv")

    # Riepilogo a soglia 2 mHz
    soglia = 2.0
    sub = df[df['soglia_mHz'] == soglia]
    print(f"\n--- RILEVAZIONE @ {soglia} mHz ---")
    pivot = sub.pivot_table(index=['onset', 'classe'], columns='PLL',
                            values='rilevato', aggfunc='first')
    print(pivot.to_string())

    # McNemar (se entrambi V1 e V2 presenti)
    if 'V1' in pivot.columns and 'V2' in pivot.columns:
        n10 = ((pivot['V1']==True) & (pivot['V2']==False)).sum()
        n01 = ((pivot['V1']==False) & (pivot['V2']==True)).sum()
        print(f"\nMcNemar: V1+/V2- = {n10}, V1-/V2+ = {n01}")
        if n10 + n01 > 0:
            p = binomtest(n10, n10+n01, 0.5).pvalue
            print(f"p = {p:.4f}")

    # Plot
    fig, axes = plt.subplots(len(flares), 1, figsize=(12, 3*len(flares)))
    if len(flares) == 1: axes = [axes]
    for ax, flare in zip(axes, flares):
        t0 = flare['onset']
        t_start = t0 - pd.Timedelta(minutes=PRE_MIN)
        t_end = t0 + pd.Timedelta(minutes=POST_MIN)
        for pll, df_pll in tracker.items():
            seg = df_pll.loc[t_start:t_end, 'PLL_Freq_Hz'].dropna()
            if len(seg) < 2: continue
            baseline = np.median(seg[seg.index < t0].values)
            doppler = (seg.values - baseline) * 1000
            ax.plot(seg.index, doppler, label=pll, alpha=0.7)
        ax.axvline(t0, color='red', ls='--', label='onset')
        ax.axhline(0, color='gray', ls=':', lw=0.5)
        ax.set_title(f"{flare['onset'].strftime('%H:%M')} {flare['classe']}")
        ax.set_ylabel("Doppler (mHz)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"flare_plot_{DATE}.png", dpi=110)
    print(f"Plot: flare_plot_{DATE}.png")
    plt.show()

if __name__ == "__main__":
    main()