import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

CARTELLA_SNR = r"D:\Gaetano\SID\dati\Sdruno"
GIORNI_DA_ANALIZZARE = 10  # Più giorni ci sono, meglio è

def stima_rumore():
    files = glob.glob(os.path.join(CARTELLA_SNR, "snr_log_*.csv"))
    files.sort()
    files = files[-GIORNI_DA_ANALIZZARE:]
    
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, comment="#")
            df['Timestamp_UTC'] = pd.to_datetime(df['Timestamp_UTC'], utc=True)
            
            # Fallback per vecchi file
            if 'Noise_dB' not in df.columns:
                df['Noise_dB'] = df['Power_dB'] - df['SNR_band_dB']
                
            frames.append(df[['Timestamp_UTC', 'Noise_dB']])
        except Exception as e:
            print(f"Errore su {os.path.basename(f)}: {e}")

    if not frames:
        print("Nessun dato trovato.")
        return

    df_all = pd.concat(frames, ignore_index=True).dropna(subset=['Noise_dB'])
    
    # --- 1. PULIZIA DEI PLATEAU (Filtro sui salti bruschi) ---
    # Calcoliamo la differenza tra campioni consecutivi per trovare i gradini
    diff = df_all['Noise_dB'].diff().abs()
    # Se il salto è > 3 dB, è un plateau (RFI o glitch). Li marchiamo come NaN
    df_pulito = df_all.copy()
    df_pulito.loc[diff > 3.0, 'Noise_dB'] = np.nan
    df_pulito = df_pulito.dropna()
    
    # --- 2. CALCOLO DEL PAVIMENTO (5° Percentile) ---
    # Usiamo il dataset pulito (senza plateau)
    rumore_locale_db = np.percentile(df_pulito['Noise_dB'], 5)
    
    # --- 3. CALCOLO DELLA MEDIA IN LINEARE (La correzione di Claude) ---
    # Media geometrica (vecchio modo, sbagliato)
    # rumore_medio_db = np.mean(df_pulito['Noise_dB'])
    
    # Media lineare (nuovo modo, corretto)
    p_lin = 10 ** (df_pulito['Noise_dB'] / 10.0)
    rumore_medio_db = 10 * np.log10(np.mean(p_lin))
    
    # --- 4. CALCOLO DELL'ATMOSFERICO ---
    p_tot_lin = 10 ** (rumore_medio_db / 10.0)
    p_loc_lin = 10 ** (rumore_locale_db / 10.0)
    p_atm_lin = max(p_tot_lin - p_loc_lin, 1e-12)
    rumore_atmosferico_db = 10 * np.log10(p_atm_lin)
    perc_atm = (p_atm_lin / p_tot_lin) * 100

    # --- STAMPA ---
    print(f"\n--- STIMA RUMORE (Analizzati {len(files)} giorni) ---")
    print(f"Pavimento Locale (5° perc): {rumore_locale_db:.2f} dB")
    print(f"Rumore Medio Totale (lin): {rumore_medio_db:.2f} dB")
    print(f"Rumore Atmosferico Vero:   {rumore_atmosferico_db:.2f} dB")
    print(f"Percentuale Atmosferico:   {perc_atm:.1f}% (stima indicativa)")

    # --- GRAFICO ---
    plt.figure(figsize=(10, 5))
    plt.hist(df_pulito['Noise_dB'], bins=50, color='purple', alpha=0.7, edgecolor='black')
    plt.axvline(rumore_locale_db, color='red', linestyle='--', label=f'Pavimento ({rumore_locale_db:.1f} dB)')
    plt.axvline(rumore_medio_db, color='blue', linestyle='-', label=f'Media Lineare ({rumore_medio_db:.1f} dB)')
    plt.title("Distribuzione del Rumore (con Plateau rimossi)")
    plt.xlabel("Rumore (dB)")
    plt.ylabel("Frequenza")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    stima_rumore()