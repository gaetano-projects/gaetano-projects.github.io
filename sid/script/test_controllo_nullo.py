import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

CARTELLA = r"D:\Gaetano\SID\dati\Sdruno"
GIORNI = ["2026-10-26", "2026-10-27"]
SOGLIA_EVENTO_MHZ = 20.0
SOGLIA_QUIETE_MHZ = 5.0
FINESTRA_S = 60
N_QUIETE_TARGET = 31  # Rinomino per evitare conflitti

def carica_dati(giorni):
    dfs = {}
    for nome, pre in {"V1": "GAESID_NSY_V1", "V2": "GAESID_NSY"}.items():
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
    
    reset = pd.Series(False, index=df.index)
    for n in dfs.keys():
        reset |= (df[f'Reset_{n}'].fillna(1) != 0)
    df = df[~reset]
    return df

def calcola_metriche(win):
    v1 = win['Freq_V1'].values - np.median(win['Freq_V1'].values)
    v2 = win['Freq_V2'].values - np.median(win['Freq_V2'].values)
    
    if np.std(v1) == 0 or np.std(v2) == 0:
        return None
    
    picco_v1 = np.max(np.abs(v1)) * 1000
    picco_v2 = np.max(np.abs(v2)) * 1000
    ratio = picco_v1 / picco_v2 if picco_v2 > 0 else np.nan
    corr = np.corrcoef(v1, v2)[0, 1]
    
    return {
        'picco_v1': picco_v1,
        'picco_v2': picco_v2,
        'ratio': ratio,
        'corr': corr
    }

def main():
    print("Caricamento dati...")
    df = carica_dati(GIORNI)
    if df is None: return
    
    # Identifica eventi
    df['diff_v2'] = df['Freq_V2'].diff().abs() * 1000
    eventi = df[df['diff_v2'] > SOGLIA_EVENTO_MHZ].copy()
    eventi['gap'] = eventi.index.to_series().diff().dt.total_seconds().fillna(999)
    eventi['gruppo'] = (eventi['gap'] > 120).cumsum()
    gruppi_eventi = eventi.groupby('gruppo')
    
    # Calcola metriche per gli eventi
    metriche_eventi = []
    for _, group in gruppi_eventi:
        idx_max = group['diff_v2'].idxmax()
        t_start = idx_max - pd.Timedelta(seconds=FINESTRA_S)
        t_end = idx_max + pd.Timedelta(seconds=FINESTRA_S)
        mask = (df.index >= t_start) & (df.index <= t_end)
        win = df[mask]
        if len(win) < 30: continue
        m = calcola_metriche(win)
        if m: metriche_eventi.append(m)
    
    print(f"Episodi evento analizzati: {len(metriche_eventi)}")
    
    # Crea maschera per gli eventi (per escluderli dalle finestre di quiete)
    mask_eventi = pd.Series(False, index=df.index)
    for _, group in gruppi_eventi:
        idx_max = group['diff_v2'].idxmax()
        t_start = idx_max - pd.Timedelta(seconds=FINESTRA_S + 60)
        t_end = idx_max + pd.Timedelta(seconds=FINESTRA_S + 60)
        mask_eventi |= ((df.index >= t_start) & (df.index <= t_end))
    
    # Identifica finestre di quiete
    rolling_max_diff = df['diff_v2'].rolling(window=2*FINESTRA_S, center=True).max()
    mask_quiete_possibili = (rolling_max_diff < SOGLIA_QUIETE_MHZ) & (~mask_eventi)
    
    indici_quiete = df.index[mask_quiete_possibili]
    
    # Usa una variabile locale per il numero effettivo
    n_quiete_effettivo = min(N_QUIETE_TARGET, len(indici_quiete))
    
    if len(indici_quiete) < N_QUIETE_TARGET:
        print(f"Attenzione: trovate solo {len(indici_quiete)} finestre di quiete (richieste {N_QUIETE_TARGET}).")
    
    # Seleziona finestre casuali (con seed fisso per riproducibilità)
    np.random.seed(42)
    if len(indici_quiete) > n_quiete_effettivo:
        indici_scelti = np.random.choice(indici_quiete, size=n_quiete_effettivo, replace=False)
    else:
        indici_scelti = indici_quiete
    
    # Calcola metriche per le finestre di quiete
    metriche_quiete = []
    for t_center in indici_scelti:
        t_start = t_center - pd.Timedelta(seconds=FINESTRA_S)
        t_end = t_center + pd.Timedelta(seconds=FINESTRA_S)
        mask = (df.index >= t_start) & (df.index <= t_end)
        win = df[mask]
        if len(win) < 30: continue
        m = calcola_metriche(win)
        if m: metriche_quiete.append(m)
    
    print(f"Finestre di quiete analizzate: {len(metriche_quiete)}\n")
    
    # Estrai gli array
    corr_eventi = np.array([m['corr'] for m in metriche_eventi])
    corr_quiete = np.array([m['corr'] for m in metriche_quiete])
    ratio_eventi = np.array([m['ratio'] for m in metriche_eventi])
    ratio_quiete = np.array([m['ratio'] for m in metriche_quiete])
    
    def fisher_mean(r):
        r = np.clip(r, -0.999, 0.999)
        return np.tanh(np.mean(np.arctanh(r)))
    
    print("--- CONFRONTO CORRELAZIONI (V1-V2) ---")
    print(f"  Eventi: media (Fisher) = {fisher_mean(corr_eventi):.3f}, mediana = {np.median(corr_eventi):.3f}")
    print(f"  Quiete: media (Fisher) = {fisher_mean(corr_quiete):.3f}, mediana = {np.median(corr_quiete):.3f}")
    
    stat, p = stats.mannwhitneyu(corr_eventi, corr_quiete, alternative='greater')
    print(f"  Test Mann-Whitney (eventi > quiete): p = {p:.4f}")
    
    print("\n--- CONFRONTO RAPPORTI (V1/V2) ---")
    pesi_eventi = np.array([m['picco_v2']**2 for m in metriche_eventi])
    pesi_quiete = np.array([m['picco_v2']**2 for m in metriche_quiete])
    
    ratio_pesato_eventi = np.average(ratio_eventi, weights=pesi_eventi)
    ratio_pesato_quiete = np.average(ratio_quiete, weights=pesi_quiete)
    
    print(f"  Eventi: media = {np.mean(ratio_eventi):.3f}, pesato energia = {ratio_pesato_eventi:.3f}")
    print(f"  Quiete: media = {np.mean(ratio_quiete):.3f}, pesato energia = {ratio_pesato_quiete:.3f}")
    
    stat, p = stats.mannwhitneyu(ratio_eventi, ratio_quiete, alternative='greater')
    print(f"  Test Mann-Whitney (eventi > quiete): p = {p:.4f}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(corr_eventi, bins=15, alpha=0.5, label='Eventi', color='red')
    ax1.hist(corr_quiete, bins=15, alpha=0.5, label='Quiete', color='blue')
    ax1.set_xlabel('Correlazione V1-V2')
    ax1.set_ylabel('Conteggio')
    ax1.set_title('Distribuzione Correlazioni')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(ratio_eventi, bins=15, alpha=0.5, label='Eventi', color='red')
    ax2.hist(ratio_quiete, bins=15, alpha=0.5, label='Quiete', color='blue')
    ax2.set_xlabel('Rapporto V1/V2')
    ax2.set_ylabel('Conteggio')
    ax2.set_title('Distribuzione Rapporti')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()