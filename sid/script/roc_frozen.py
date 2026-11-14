import pandas as pd
import numpy as np
import scipy.signal as signal
from scipy.ndimage import median_filter
import os

DATA_DIR = r"D:\Gaetano\SID\dati\Sdruno"
DATA = "2026-10-26"
FILE_V1 = os.path.join(DATA_DIR, f"GAESID_NSY_V1_{DATA}.csv")
FILE_V2 = os.path.join(DATA_DIR, f"GAESID_NSY_{DATA}.csv")

ORA_INIZIO, ORA_FINE = 9, 16
SOGLIE_MHZ = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
MIN_DURATA_S = 60  # Un SID dura minuti, non secondi
PAD_S = 1800

def hampel_filter(data, window_size=7, n_sigmas=3.0):
    """Filtro Hampel: median filter + MAD."""
    med = median_filter(data, size=window_size)
    mad = median_filter(np.abs(data - med), size=window_size)
    threshold = n_sigmas * 1.4826 * mad
    # Sostituisce gli outliers con la mediana locale
    return np.where(np.abs(data - med) > threshold, med, data)

def filtra_frozen(freq_series):
    """Applica la pipeline: Hampel -> BP 5-60 min -> padding reflect."""
    # 1. Gestione NaN: interpolazione lineare (evita propagazione di inf)
    s = pd.Series(freq_series.values)
    s = s.interpolate(method='linear', limit_direction='both')
    freq = s.values.astype(float)
    
    # 2. Hampel
    freq = hampel_filter(freq, window_size=7, n_sigmas=3.0)
    
    # 3. Rimuovi offset DC (centra il segnale a zero)
    freq = freq - np.median(freq)
    
    # 4. Passa-banda 5-60 min (usando SOS per stabilità numerica)
    fs = 1.0  # 1 Hz
    nyq = 0.5 * fs
    lowcut = 1.0 / (60 * 60)   # 60 min
    highcut = 1.0 / (5 * 60)   # 5 min
    
    # Ordine 2 e output='sos' (Second-Order Sections)
    sos = signal.butter(2, [lowcut/nyq, highcut/nyq], btype='band', output='sos')
    
    # 5. Filtro zero-phase (sosfiltfilt è molto più stabile di filtfilt)
    filtered = signal.sosfiltfilt(sos, freq)
    
    return filtered * 1000.0  # in mHz

def stima_fa(freq_series, pll_name):
    doppler = filtra_frozen(freq_series)
    ore = len(doppler) / 3600.0
    
    risultati = []
    for soglia in SOGLIE_MHZ:
        sopra = np.abs(doppler) > soglia
        diff = np.diff(sopra.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        if sopra[0]: starts = np.r_[0, starts]
        if sopra[-1]: ends = np.r_[ends, len(sopra)]
        
        # Conta solo eventi che durano più di MIN_DURATA_S
        n_eventi = 0
        for s0, e0 in zip(starts, ends):
            if (e0 - s0) >= MIN_DURATA_S:
                n_eventi += 1
        
        fa_giorno = n_eventi / ore * 24
        
        risultati.append({'PLL': pll_name, 'Soglia_mHz': soglia, 'FA_per_giorno': fa_giorno, 'N_eventi': n_eventi})
    return pd.DataFrame(risultati)

def main():
    print(f"\n=== ROC CON PIPELINE FROZEN (Diurno {ORA_INIZIO}-{ORA_FINE} UTC) ===")
    for name, path in [("V1", FILE_V1), ("V2", FILE_V2)]:
        df = pd.read_csv(path, comment='#')
        df['Timestamp_UTC'] = pd.to_datetime(df['Timestamp_UTC'], utc=True)
        df = df.drop_duplicates('Timestamp_UTC').set_index('Timestamp_UTC').sort_index()
        # Taglio dei bordi: escludiamo 30 min all'inizio e 30 min alla fine
        # per evitare l'edge effect di sosfiltfilt (scoperto con debug_fa_v2)
        df = df.between_time('09:30', '15:30')
        
        res = stima_fa(df['PLL_Freq_Hz'], name)
        print(f"\n--- {name} ---")
        print(f"{'Soglia (mHz)':<14} | {'FA/giorno':<10} | {'Eventi (N)':<10}")
        print("-" * 44)
        for _, row in res.iterrows():
            print(f"{row['Soglia_mHz']:<14.1f} | {row['FA_per_giorno']:<10.2f} | {row['N_eventi']:<10}")
        
        # Stima soglia per 1 FA/giorno
        target = 1.0
        sotto = res[res['FA_per_giorno'] <= target]
        if len(sotto) > 0:
            print(f"-> Soglia stimata per 1 FA/giorno: {sotto.iloc[0]['Soglia_mHz']:.1f} mHz")
        else:
            print(f"-> Nessuna soglia testata dà FA <= 1/giorno.")

if __name__ == "__main__":
    main()