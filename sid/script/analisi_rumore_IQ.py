import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats

CARTELLA_IQ = r"D:\Gaetano\SID\pysdrplay"
FS = 5000.0  # Frequenza di campionamento (5 kHz)

def analizza_iq():
    # Trova tutti i file .npy nella cartella
    files = [f for f in os.listdir(CARTELLA_IQ) if f.endswith('.npy')]
    
    if not files:
        print(f"Nessun file .npy trovato in {CARTELLA_IQ}")
        return

    print(f"Trovati {len(files)} file. Inizio analisi...\n")
    
    risultati = []
    psd_medio = None
    freqs_psd = None
    
    for filename in files:
        path = os.path.join(CARTELLA_IQ, filename)
        try:
            # Carica l'array complesso
            data = np.load(path)
            
            # --- 1. CALCOLO DELLA KURTOSIS ---
            # Usiamo il modulo della potenza istantanea
            potenza = np.abs(data)**2
            
            # Rimuoviamo la media (per evitare che il segnale continuo di NSY influenzi il calcolo)
            potenza_centrata = potenza - np.mean(potenza)
            
            # Kurtosis (Fisher=False restituisce il valore classico: 3 = gaussiano)
            kurt = stats.kurtosis(potenza_centrata, fisher=False)
            
            # --- 2. CALCOLO PSD (Welch) ---
            # Usiamo la funzione di Welch per una stima pulita
            f, psd = signal.welch(data, fs=FS, nperseg=8192, return_onesided=False)
            # fftshift per mettere lo 0 al centro
            f = np.fft.fftshift(f)
            psd = np.fft.fftshift(psd)
            
            if psd_medio is None:
                psd_medio = psd.copy()
                freqs_psd = f.copy()
            else:
                psd_medio += psd
            
            # Salva i risultati per la tabella
            risultati.append((filename, kurt, len(data)/FS/60)) # durata in minuti
            
        except Exception as e:
            print(f"Errore su {filename}: {e}")

    if not risultati:
        return

    # Media della PSD
    psd_medio /= len(files)
    psd_medio_db = 10 * np.log10(psd_medio + 1e-12)

    # --- 3. STAMPA DEI RISULTATI ---
    print(f"{'File':<30} | {'Kurtosis':<10} | {'Durata (min)':<12}")
    print("-" * 60)
    for nome, kurt, durata in risultati:
        stato = "Impulsivo (Sferiche)" if kurt > 5 else "Gaussiano (RFI/Termico)"
        print(f"{nome:<30} | {kurt:>8.2f}  | {durata:>8.1f}   -> {stato}")

    # --- 4. GRAFICO ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Analisi Rumore IQ (NSY) - Attenzione: segnale filtrato a 150 Hz", fontsize=14, fontweight='bold')
    
    # Pannello 1: Kurtosis
    nomi = [r[0] for r in risultati]
    kurts = [r[1] for r in risultati]
    ax1.bar(range(len(nomi)), kurts, color='purple', alpha=0.7)
    ax1.axhline(3, color='red', linestyle='--', label='Gaussiano (K=3)')
    ax1.axhline(10, color='orange', linestyle='--', label='Impulsivo (Sferiche)')
    ax1.set_ylabel('Kurtosis')
    ax1.set_title('Kurtosis dei file IQ (più è alto, più il rumore è impulsivo)')
    ax1.set_xticks(range(len(nomi)))
    ax1.set_xticklabels(nomi, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Pannello 2: PSD Media
    ax2.plot(freqs_psd, psd_medio_db, color='blue', linewidth=1.5)
    ax2.set_xlim(-300, 300) # Zoommiamo sui ±300 Hz (dato che il filtro taglia a 150 Hz)
    ax2.set_xlabel('Frequenza (Hz)')
    ax2.set_ylabel('Densità di Potenza (dB/Hz)')
    ax2.set_title('PSD Media (Zoom ±300 Hz)')
    ax2.grid(True, alpha=0.3)
    
    # Evidenzia le frequenze di rete (50, 100, 150 Hz)
    for f_rete in [50, 100, 150, -50, -100, -150]:
        ax2.axvline(f_rete, color='red', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analizza_iq()