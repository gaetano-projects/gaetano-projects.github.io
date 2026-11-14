#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
event_snr_calculator.py (v2.2)
Integra il calcolo del Segnale Puro (rumore sottratto) dai log snr_log.
Confronta tre canali: Power_dB, Amp_100Hz_dB (normalizzato) e Signal_Pure_dB.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.stats import linregress
import os
import glob

DIR_SDRUNO = r"D:\Gaetano\SID\dati\Sdruno"
DIR_STORICO = r"D:\Gaetano\SID\dati\Sdruno\Storico_NSY"

def trova_file_csv(nome_file, is_log=False):
    prefisso = "snr_log_" if is_log else ""
    if is_log and not nome_file.startswith(prefisso):
        nome_file = prefisso + nome_file
        
    if os.path.exists(nome_file): return nome_file
    percorso_1 = os.path.join(DIR_SDRUNO, nome_file)
    if os.path.exists(percorso_1): return percorso_1
    percorso_2 = glob.glob(os.path.join(DIR_STORICO, "**", nome_file), recursive=True)
    return percorso_2[0] if percorso_2 else None

def carica_csv_robusto(file_path):
    try:
        df = pd.read_csv(file_path, skipinitialspace=True, on_bad_lines='skip')
        if 'Timestamp_UTC' in df.columns: return df
    except Exception: pass
    
    skip_rows = 0
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if 'Timestamp_UTC' in line:
                skip_rows = i
                break
    df = pd.read_csv(file_path, skiprows=skip_rows, skipinitialspace=True, on_bad_lines='skip')
    df.columns = df.columns.str.strip()
    return df

def carica_e_unisci(file_main, date_str):
    """Carica il file principale e, se esiste, unisce i dati di Signal_Pure dal log."""
    path_main = trova_file_csv(file_main)
    if not path_main:
        print(f"ERRORE: File principale {file_main} non trovato.")
        return None

    print(f"Caricamento principale: {path_main}")
    df = carica_csv_robusto(path_main)
    df['Timestamp_UTC'] = pd.to_datetime(df['Timestamp_UTC'], errors='coerce')
    df = df.dropna(subset=['Timestamp_UTC'])
    
    for col in ['Power_dB', 'Amp_100Hz_dB']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            raise ValueError(f"Colonna mancante: {col}")
    
    # Normalizzazione Claude
    df['Amp_100Hz_dB'] = df['Amp_100Hz_dB'] / 2.0

    # --- CARICAMENTO LOG E CALCOLO SIGNAL_PURE ---
    path_log = trova_file_csv(f"snr_log_{date_str}.csv", is_log=True)
    if path_log:
        print(f"Caricamento log SNR: {path_log}")
        try:
            df_log = carica_csv_robusto(path_log)
            df_log['Timestamp_UTC'] = pd.to_datetime(df_log['Timestamp_UTC'], errors='coerce')
            df_log = df_log.dropna(subset=['Timestamp_UTC', 'Power_dB', 'Noise_dB'])
            
            # Formula da visualizza_snr_log.py
            p_tot_lin = 10 ** (pd.to_numeric(df_log['Power_dB']) / 10.0)
            p_noise_lin = 10 ** (pd.to_numeric(df_log['Noise_dB']) / 10.0)
            p_sig_lin = np.maximum(p_tot_lin - p_noise_lin, 1e-12)
            df_log['Signal_Pure_dB'] = 10 * np.log10(p_sig_lin)
            
            # Arrotondiamo i timestamp al secondo per un merge sicuro
            df_log['Timestamp_UTC'] = df_log['Timestamp_UTC'].dt.round('s')
            df['Timestamp_UTC'] = df['Timestamp_UTC'].dt.round('s')
            
            # Uniamo i dati
            df = pd.merge_asof(df, df_log[['Timestamp_UTC', 'Signal_Pure_dB']], 
                               on='Timestamp_UTC', direction='nearest', tolerance=pd.Timedelta('1s'))
            print("[INFO] Canale Signal_Pure_dB (rumore sottratto) integrato con successo.")
        except Exception as e:
            print(f"[WARNING] Impossibile integrare il log SNR: {e}")
            df['Signal_Pure_dB'] = np.nan
    else:
        print(f"[WARNING] File snr_log_{date_str}.csv non trovato. Il canale Signal_Pure sarà ignorato.")
        df['Signal_Pure_dB'] = np.nan

    return df

def calculate_event_snr(csv_file, flare_time_str, pre_mins=15, post_mins=30):
    date_str = flare_time_str.split(' ')[0] if ' ' in flare_time_str else "2026-11-06" 
    # Cerca di dedurre la data dal nome del file se non specificata
    import re
    match = re.search(r'(\d{4}-\d{2}-\d{2})', csv_file)
    if match:
        date_str = match.group(1)

    df = carica_e_unisci(csv_file, date_str)
    if df is None: return

    t_onset = pd.to_datetime(f"{date_str} {flare_time_str.split()[-1]}")
    t_start_base = t_onset - timedelta(minutes=pre_mins)
    t_end_event = t_onset + timedelta(minutes=post_mins)
    
    print(f"Data: {date_str} | Onset GOES: {t_onset.strftime('%H:%M')} UTC")
    print(f"Baseline: {t_start_base.strftime('%H:%M')} - {t_onset.strftime('%H:%M')}")
    print(f"Evento:   {t_onset.strftime('%H:%M')} - {t_end_event.strftime('%H:%M')}")
    print("-" * 85)

    # Lista dei canali da analizzare
    canali = ['Power_dB', 'Amp_100Hz_dB']
    if not df['Signal_Pure_dB'].isna().all():
        canali.append('Signal_Pure_dB')

    # Smoothing
    for ch in canali:
        df[f'{ch}_smooth'] = df.rolling('60s', on='Timestamp_UTC', min_periods=10)[ch].mean()

    results = {}
    for ch in canali:
        mask_base = (df['Timestamp_UTC'] >= t_start_base) & (df['Timestamp_UTC'] < t_onset)
        mask_event = (df['Timestamp_UTC'] >= t_onset) & (df['Timestamp_UTC'] <= t_end_event)
        
        base_data = df.loc[mask_base, ch].dropna()
        event_data_smooth = df.loc[mask_event, f'{ch}_smooth'].dropna()
        
        if len(base_data) < 10 or len(event_data_smooth) < 10:
            print(f"Dati insufficienti per {ch}.")
            continue

        # 1. DETREND LINEARE DELLA BASELINE
        x_base = np.arange(len(base_data))
        slope, intercept, _, _, _ = linregress(x_base, base_data.values)
        trend = slope * x_base + intercept
        detrended_base = base_data.values - trend
        
        median_base_detrended = np.median(detrended_base)
        mad_base = np.median(np.abs(detrended_base - median_base_detrended)) * 1.4826
        
        # 2. PICCO ROBUSTO (Mediana dei 3 minuti attorno al picco)
        peak_idx = event_data_smooth.idxmax()
        peak_time = df.loc[peak_idx, 'Timestamp_UTC']
        
        t_peak_start = peak_time - timedelta(minutes=1.5)
        t_peak_end = peak_time + timedelta(minutes=1.5)
        mask_peak = (df['Timestamp_UTC'] >= t_peak_start) & (df['Timestamp_UTC'] <= t_peak_end)
        
        max_event_robust = df.loc[mask_peak, f'{ch}_smooth'].median()
        
        median_base_original = base_data.median()
        delta_db = max_event_robust - median_base_original
        
        snr = (delta_db / mad_base) if mad_base > 0 else 0.0
            
        results[ch] = {
            'Med_Base_Orig': median_base_original,
            'MAD_Detrend': mad_base,
            'Max_Robusto': max_event_robust,
            'Delta_dB': delta_db,
            'Event_SNR': snr,
            'Peak_Time': peak_time.strftime('%H:%M:%S')
        }

    # Stampa Tabella
    print(f"{'Canale':<15} | {'Med Base':<9} | {'MAD (detr)':<10} | {'Max Robusto':<11} | {'Delta (dB)':<10} | {'Event SNR':<9} | {'Tempo Picco':<10}")
    print("-" * 85)
    for ch, data in results.items():
        print(f"{ch:<15} | {data['Med_Base_Orig']:<9.2f} | {data['MAD_Detrend']:<10.2f} | {data['Max_Robusto']:<11.2f} | {data['Delta_dB']:<10.2f} | {data['Event_SNR']:<9.1f} | {data['Peak_Time']:<10}")
    print("-" * 85)

    # Plot
    fig, axes = plt.subplots(len(canali), 1, figsize=(12, 4 * len(canali)), sharex=True)
    if len(canali) == 1: axes = [axes]
    
    plot_df = df[(df['Timestamp_UTC'] >= t_start_base - timedelta(minutes=10)) & 
                 (df['Timestamp_UTC'] <= t_end_event + timedelta(minutes=10))]

    colors = {'Power_dB': ('lightgreen', 'green'), 
              'Amp_100Hz_dB': ('lightblue', 'blue'), 
              'Signal_Pure_dB': ('#FFD700', '#FFA500')}

    for ax, ch in zip(axes, canali):
        c_grezzo, c_smooth = colors[ch]
        ax.plot(plot_df['Timestamp_UTC'], plot_df[ch], color=c_grezzo, alpha=0.5, label=f'{ch} grezzo')
        ax.plot(plot_df['Timestamp_UTC'], plot_df[f'{ch}_smooth'], color=c_smooth, label=f'{ch} smooth 60s')
        ax.set_ylabel(f'{ch}')
        ax.axvspan(t_start_base, t_onset, color='gray', alpha=0.2, label='Baseline (Detrend)')
        ax.axvspan(t_onset, t_end_event, color='yellow', alpha=0.2, label='Finestra Evento')
        ax.axvline(t_onset, color='red', linestyle='--', label='Onset')
        ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)

    axes[0].set_title(f'Event SNR - Flare {flare_time_str} UTC')
    plt.xticks(rotation=45); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event SNR Robusto e Normalizzato (con Segnale Puro)")
    parser.add_argument("csv_file", help="Nome o percorso del file CSV (GAESID_NSY_...)")
    parser.add_argument("--time", required=True, help="Orario di onset GOES (es. '11:00')")
    parser.add_argument("--pre", type=int, default=15, help="Minuti di baseline (default: 15)")
    parser.add_argument("--post", type=int, default=30, help="Minuti di evento (default: 30)")
    args = parser.parse_args()
    calculate_event_snr(args.csv_file, args.time, args.pre, args.post)