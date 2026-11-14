import time
import numpy as np
import scipy.signal as signal
import os
import sys
from pysdrplay import SDRPlayRadio, Bw, Event, PowerOverloadCbEventId

# --- CONFIGURAZIONE ---
TARGET_FREQ = 45_905.0
OFFSET_HZ = 20000.0
TUNE_FREQ = TARGET_FREQ + OFFSET_HZ
HW_SAMPLE_RATE = 2_000_000
IF_BANDWIDTH = Bw.BW_0_200

IF_GR_LIST = [59, 55, 50, 45, 40, 35, 30, 25, 20]
LNA_STATE = 0 
TEMPO_MEDIA_S = 30

# --- VARIABILI GLOBALI PER OVERLOAD ---
overload_count = 0
overload_da_ackare = False

def event_callback(event_id, tuner, params, ctx):
    global overload_count, overload_da_ackare
    try:
        if event_id == Event.POWER_OVERLOAD_CHANGE:
            tipo = params.contents.powerOverloadParams.powerOverloadChangeType
            if tipo == PowerOverloadCbEventId.OVERLOAD_DETECTED:
                overload_count += 1
            overload_da_ackare = True
    except Exception as e:
        pass

def calcola_snr(iq_data):
    n = np.arange(len(iq_data))
    t = n / HW_SAMPLE_RATE
    baseband = iq_data * np.exp(1j * 2 * np.pi * OFFSET_HZ * t)
    
    b, a = signal.butter(4, 4500 / (HW_SAMPLE_RATE/2), btype='low')
    s1 = signal.lfilter(b, a, baseband)
    s2 = s1[::100]
    FS_FFT = 20000.0
    
    b_sig = signal.firwin(1001, 45.0, fs=FS_FFT, window=("kaiser", 8.0))
    filtered = signal.lfilter(b_sig, 1.0, s2)
    
    n_fft = 2048
    bw_signal = 90.0
    window = np.hanning(n_fft)
    s2_fft = s2[-n_fft:] * window
    spettro = np.abs(np.fft.fft(s2_fft)) ** 2
    freqs = np.fft.fftfreq(n_fft, d=1.0/FS_FFT)
    
    mask_noise = (freqs > 500.0) & (freqs < 1500.0)
    if np.any(mask_noise):
        sum_w2 = np.sum(window ** 2)
        mediana_bin = np.median(spettro[mask_noise])
        p_noise = mediana_bin * bw_signal / (FS_FFT * sum_w2)
    else:
        p_noise = 1e-12
        
    p_in = np.mean(np.abs(filtered) ** 2)
    p_signal = max(p_in - p_noise, p_noise * 1e-3)
    
    snr_db = 10 * np.log10(p_signal / p_noise + 1e-12)
    power_db = 10 * np.log10(p_in + 1e-12)
    noise_db = 10 * np.log10(p_noise + 1e-12)
    
    return snr_db, power_db, noise_db

def main():
    global iq_buffer, overload_count, overload_da_ackare
    print("Inizializzazione radio...")
    radio = SDRPlayRadio()
    radio.select_device(radio.devices[0])
    radio.set_frequency(TUNE_FREQ)
    radio.set_sample_rate(HW_SAMPLE_RATE)
    radio.set_bandwidth(IF_BANDWIDTH)
    radio.set_agc(False)
    radio.set_dc_offset(dc_enable=True, iq_enable=True)
    
    iq_buffer = []
    
    def callback(xi, xq, params, num_samples, reset, ctx):
        global iq_buffer
        if len(iq_buffer) > 4_000_000: 
            iq_buffer = iq_buffer[-2_000_000:]
        samples = np.empty(num_samples, dtype=np.complex64)
        samples.real = np.ctypeslib.as_array(xi, shape=(num_samples,))
        samples.imag = np.ctypeslib.as_array(xq, shape=(num_samples,))
        iq_buffer.extend(samples)
    
    risultati = []
    
    print(f"\n{'Gain IF':<8} | {'SNR (dB)':<10} | {'Power (dB)':<12} | {'Noise (dB)':<12} | {'Overload?':<10}")
    print("-" * 65)
    
    for gain in IF_GR_LIST:
        print(f"Misurazione gain IF_GR = {gain} dB...")
        radio.set_gain(lna_state=LNA_STATE, gr_db=gain)
        
        iq_buffer = [] 
        overload_count = 0
        overload_da_ackare = False
        
        radio.start(callback, on_event=event_callback)
        
        time.sleep(3.0) # Assestamento
        
        # --- NUOVO: Pulizia del transitorio di cambio gain ---
        if overload_da_ackare:
            try: radio.acknowledge_power_overload()
            except: pass
            overload_da_ackare = False
        overload_count = 0  # Azzera il contatore per iniziare la misura pulita
        
        snr_list, pwr_list, noise_list = [], [], []
        for _ in range(TEMPO_MEDIA_S):
            time.sleep(1.0)
            
            # --- NUOVO: Gestione overload in tempo reale ---
            if overload_da_ackare:
                try: radio.acknowledge_power_overload()
                except: pass
                overload_da_ackare = False
            # ---------------------------------------------
            
            if len(iq_buffer) >= 2_000_000:
                iq = np.array(iq_buffer[-2_000_000:], dtype=np.complex64)
                s, p, n = calcola_snr(iq)
                snr_list.append(s)
                pwr_list.append(p)
                noise_list.append(n)
        
        radio.stop()
        
        # Acknowledge overload se necessario per sbloccare lo stato
        if overload_da_ackare:
            try: radio.acknowledge_power_overload()
            except: pass
        
        if snr_list:
            snr_medio = np.mean(snr_list)
            pwr_medio = np.mean(pwr_list)
            noise_medio = np.mean(noise_list)
            ovl_str = "SI" if overload_count > 0 else "NO"
            
            risultati.append({
                "gain": gain, "snr": snr_medio, "pwr": pwr_medio, 
                "noise": noise_medio, "overload": overload_count > 0
            })
            print(f"{gain:<8} | {snr_medio:>+6.2f}    | {pwr_medio:>+6.2f}     | {noise_medio:>+6.2f}     | {ovl_str:<10}")
        else:
            print(f"{gain:<8} | ERRORE: Dati insufficienti")
            
    # --- ALGORITMO DI SCELTA FINALE ---
    if risultati:
        print("\n" + "="*65)
        
        # 1. Scarta i gain che hanno causato ADC Overload
        validi = [r for r in risultati if not r["overload"]]
        
        if not validi:
            print("⚠ ATTENZIONE: Tutti i gain testati hanno causato ADC Overload!")
            print("   Uso il gain con il minor numero di overload (o il primo).")
            validi = risultati
        
        # 2. Scegli il gain con l'SNR massimo tra quelli validi
        migliore = max(validi, key=lambda x: x["snr"])
        
        print(f"⭐ MIGLIOR GAIN IF: {migliore['gain']} dB (Scelto per SNR massimo)")
        print(f"   SNR: {migliore['snr']:+.2f} dB")
        print(f"   Potenza Segnale: {migliore['pwr']:+.2f} dB")
        print(f"   Rumore: {migliore['noise']:+.2f} dB")
        print("="*65)

if __name__ == "__main__":
    main()