import pandas as pd
import numpy as np
import scipy.signal as signal
from scipy.ndimage import median_filter
import os

DATA_DIR = r"D:\Gaetano\SID\dati\Sdruno"
DATA = "2026-10-26"
FILE_V1 = os.path.join(DATA_DIR, f"GAESID_NSY_V1_{DATA}.csv")
FILE_V2 = os.path.join(DATA_DIR, f"GAESID_NSY_{DATA}.csv")

def hampel_filter(data, window_size=7, n_sigmas=3.0):
    med = median_filter(data, size=window_size)
    mad = median_filter(np.abs(data - med), size=window_size)
    threshold = n_sigmas * 1.4826 * mad
    return np.where(np.abs(data - med) > threshold, med, data)

def filtra_frozen_clean(freq_series, offset_comune):
    s = pd.Series(freq_series.values).interpolate(method='linear', limit_direction='both')
    freq = s.values.astype(float)
    
    # 1. Sottraiamo lo STESSO offset a entrambi (non la mediana individuale!)
    freq = freq - offset_comune
    
    # 2. Hampel (nota: se V2 è più rumoroso, Hampel agirà di più. Questo è voluto)
    freq = hampel_filter(freq, window_size=7, n_sigmas=3.0)
    
    # 3. Filtro
    fs = 1.0; nyq = 0.5 * fs
    lowcut = 1.0 / (60 * 60); highcut = 1.0 / (5 * 60)
    sos = signal.butter(2, [lowcut/nyq, highcut/nyq], btype='band', output='sos')
    filtered = signal.sosfiltfilt(sos, freq)
    return filtered * 1000.0

print("Caricamento grezzi 07:00-16:00...")
df_v1 = pd.read_csv(FILE_V1, comment='#')
df_v1['Timestamp_UTC'] = pd.to_datetime(df_v1['Timestamp_UTC'], utc=True)
df_v1 = df_v1.drop_duplicates('Timestamp_UTC').set_index('Timestamp_UTC').sort_index().between_time('07:00', '16:00')

df_v2 = pd.read_csv(FILE_V2, comment='#')
df_v2['Timestamp_UTC'] = pd.to_datetime(df_v2['Timestamp_UTC'], utc=True)
df_v2 = df_v2.drop_duplicates('Timestamp_UTC').set_index('Timestamp_UTC').sort_index().between_time('07:00', '16:00')

df = pd.concat([df_v1['PLL_Freq_Hz'], df_v2['PLL_Freq_Hz']], axis=1, keys=['V1', 'V2']).dropna()

# Usiamo un offset comune: la mediana di V1 (che è la baseline più pulita)
offset_comune = np.median(df['V1'])

print(f"Offset comune applicato: {offset_comune:.3f} Hz")
print("Applicando la pipeline pulita (stesso offset per entrambi)...")

d1_clean = filtra_frozen_clean(df['V1'], offset_comune)
d2_clean = filtra_frozen_clean(df['V2'], offset_comune)

# Estraiamo la finestra 10:00-11:00
mask = (df.index >= '2026-10-26 10:00:00') & (df.index <= '2026-10-26 11:00:00')
d1_win = d1_clean[mask]
d2_win = d2_clean[mask]

print(f"\n--- RISULTATI FINESTRA 10:00-11:00 (Pipeline Pulita) ---")
print(f"Std V1: {np.std(d1_win):.3f} mHz")
print(f"Std V2: {np.std(d2_win):.3f} mHz")
print(f"Correlazione V1-V2: {np.corrcoef(d1_win, d2_win)[0, 1]:.3f}")

print(f"\n--- PRIMI 5 VALORI FILTRATI (10:00 - 10:00:04) ---")
print("V1_filtrato_mHz | V2_filtrato_mHz")
for i in range(5):
    print(f"{d1_win[i]:15.3f} | {d2_win[i]:15.3f}")