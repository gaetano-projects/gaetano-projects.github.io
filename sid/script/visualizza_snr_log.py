#!/usr/bin/env python3
"""
visualizza_snr_log.py - Visualizzatore grafico dei log SNR (snr_log_*.csv)
Versione a 2 pannelli: Curve (Power, Segnale Puro, SNR, Noise) unificate + Stato/ADC.

Tasti attivi nella finestra:
  [I] = Cambia data
  [Q] = Esci

Note:
- Post-processing puro: legge i CSV esistenti, non scrive nulla nel CSV a 9 colonne.
- Noise_90Hz_dB = Power_dB - SNR_band_dB (rumore nella banda del segnale, ±45 Hz).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone, timedelta
import tkinter as tk
from tkinter import messagebox


# Configurazione percorsi
CARTELLE_SNR_LOG = [
    r"D:\Gaetano\SID\dati\Sdruno",
    r"D:\Gaetano\SID\dati\Sdruno\Storico_NSY",
]
SOGLIA_SNR_DB = 4.8          # reference visiva (allineata al tracker)
SITE_LAT = 45.07
SITE_LON = 7.68
GIORNI_MEDIANA = 7
MOSTRA_MEDIANE_7GG = False 
SMOOTH_WINDOW_S = 30 

# Stato globale per la GUI
_STATO = {
    'fig': None, 
    'data_corrente': None
}


# ----------------------------------------------------------------------
# Calendario
# ----------------------------------------------------------------------
def scegli_data_da_calendario():
    try:
        from tkcalendar import Calendar
    except ImportError:
        messagebox.showerror("Errore", "Libreria 'tkcalendar' non installata.\nEsegui: pip install tkcalendar")
        return None

    root = tk.Tk()
    root.title("Seleziona Data")
    root.geometry("450x400")
    root.configure(bg='#0d0d1a')
    root.attributes('-topmost', True)

    oggi = datetime.now(timezone.utc)

    cal = Calendar(
        root, selectmode='day',
        year=oggi.year, month=oggi.month, day=oggi.day,
        date_pattern='yyyy-mm-dd',
        locale='it_IT',
        background='#1a1a2e', foreground='white',
        headersbackground='#2a2a4e', headersforeground='#00FF99',
        selectbackground='#00FF99', selectforeground='black',
        normalbackground='#0d0d1a', normalforeground='white',
        weekendbackground='#1a1a2e', weekendforeground='#aaaacc',
        othermonthforeground='#444466', othermonthbackground='#0d0d1a',
        font=('Consolas', 10),
    )
    cal.pack(padx=20, pady=20, fill='both', expand=True)

    risultato = {'data': None}

    def conferma():
        data_str = cal.get_date()
        path = trova_file_snr_log(data_str)
        if path is None:
            messagebox.showerror(
                "File non trovato",
                f"Nessun file snr_log per il {data_str} nelle cartelle:\n"
                + "\n".join(CARTELLE_SNR_LOG)
            )
            return
        risultato['data'] = data_str
        root.destroy()

    def annulla():
        root.destroy()

    btn_frame = tk.Frame(root, bg='#0d0d1a')
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="Visualizza", command=conferma,
              bg='#00FF99', fg='black', font=('Consolas', 10, 'bold')).pack(side='left', padx=10)
    tk.Button(btn_frame, text="Annulla", command=annulla,
              bg='#333355', fg='white', font=('Consolas', 10)).pack(side='left', padx=10)

    root.mainloop()
    return risultato['data']


# ----------------------------------------------------------------------
# Ricerca file e caricamento
# ----------------------------------------------------------------------
def trova_file_snr_log(data_str):
    nome_file = f"snr_log_{data_str}.csv"
    for cartella in CARTELLE_SNR_LOG:
        path = os.path.join(cartella, nome_file)
        if os.path.exists(path):
            return path
    return None

def carica_snr_log(path):
    df = pd.read_csv(path, comment="#")
    df['Timestamp_UTC'] = pd.to_datetime(df['Timestamp_UTC'], utc=True)
    
    # Usa la colonna Noise_dB fornita direttamente dal tracker
    # (Fallback di sicurezza per i vecchi file a 60s che non l'avevano)
    if 'Noise_dB' not in df.columns:
        df['Noise_dB'] = df['Power_dB'] - df['SNR_band_dB']
    
    # Calcolo Segnale Puro di NSY (Sottrazione in scala lineare)
    p_tot_lin = 10 ** (df['Power_dB'] / 10.0)
    p_noise_lin = 10 ** (df['Noise_dB'] / 10.0)
    p_sig_lin = p_tot_lin - p_noise_lin
    
    # Evita logaritmi di numeri negativi o zero
    p_sig_lin = np.maximum(p_sig_lin, 1e-12) 
    df['Signal_Pure_dB'] = 10 * np.log10(p_sig_lin)
    
    return df

def calcola_mediana_7gg(data_str, giorni=GIORNI_MEDIANA):
    data_dt = datetime.strptime(data_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    frames = []
    for i in range(1, giorni + 1):
        d = (data_dt - timedelta(days=i)).strftime("%Y-%m-%d")
        p = trova_file_snr_log(d)
        if p is None:
            continue
        try:
            df = carica_snr_log(p)
            df['_h'] = df['Timestamp_UTC'].dt.hour
            df['_m'] = df['Timestamp_UTC'].dt.minute
            frames.append(df[['_h', '_m', 'Power_dB', 'SNR_band_dB', 'Noise_dB']])
        except Exception:
            continue

    if not frames:
        return None

    concat = pd.concat(frames, ignore_index=True)
    med = concat.groupby(['_h', '_m'], as_index=False).median(numeric_only=True)

    def _to_ts(row):
        return data_dt.replace(hour=int(row['_h']), minute=int(row['_m']), second=0, microsecond=0)

    med['Timestamp_UTC'] = med.apply(_to_ts, axis=1)
    med = med.sort_values('Timestamp_UTC').reset_index(drop=True)
    return med


# ----------------------------------------------------------------------
# Sunrise / sunset
# ----------------------------------------------------------------------
def calcola_sunrise_sunset(data_str):
    try:
        from astral import Observer
        from astral.sun import sun
    except ImportError:
        return None, None

    try:
        data_dt = datetime.strptime(data_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        obs = Observer(latitude=SITE_LAT, longitude=SITE_LON)
        s = sun(obs, date=data_dt.date(), tzinfo=timezone.utc)
        return s['sunrise'], s['sunset']
    except Exception:
        return None, None

def apri_finestra_stima_rumore(data_str):
    """Calcola e mostra la stima del rumore locale vs atmosferico per la data corrente."""
    path = trova_file_snr_log(data_str)
    if not path:
        print(f"[K] File non trovato per {data_str}")
        return

    try:
        df = carica_snr_log(path)
        df = df.dropna(subset=['Noise_dB'])
        
        if df.empty:
            print("[K] Nessun dato valido per il rumore.")
            return
            
        # --- 1. PULIZIA DEI PLATEAU (Filtro sui salti bruschi > 3 dB) ---
        # Calcoliamo la differenza tra campioni consecutivi per trovare i gradini
        diff = df['Noise_dB'].diff().abs()
        
        # Creiamo una copia pulita e marchiamo i salti come NaN
        df_pulito = df.copy()
        df_pulito.loc[diff > 3.0, 'Noise_dB'] = np.nan
        df_pulito = df_pulito.dropna(subset=['Noise_dB'])
        
        if df_pulito.empty:
            print("[K] Dati insufficienti dopo la pulizia dei plateau.")
            return

        # --- 2. CALCOLO DEL PAVIMENTO (5° Percentile) ---
        rumore_locale_db = np.percentile(df_pulito['Noise_dB'], 5)
        
        # --- 3. CALCOLO DELLA MEDIA IN LINEARE (Correzione Fisica) ---
        p_lin = 10 ** (df_pulito['Noise_dB'] / 10.0)
        rumore_medio_db = 10 * np.log10(np.mean(p_lin))
        
        # --- 4. CALCOLO DELL'ATMOSFERICO ---
        p_tot_lin = 10 ** (rumore_medio_db / 10.0)
        p_loc_lin = 10 ** (rumore_locale_db / 10.0)
        p_atm_lin = max(p_tot_lin - p_loc_lin, 1e-12)
        
        rumore_atmosferico_db = 10 * np.log10(p_atm_lin)
        perc_atm = (p_atm_lin / p_tot_lin) * 100

        # --- STAMPA IN CONSOLE ---
        print(f"\n--- STIMA RUMORE ({data_str}) ---")
        print(f"Pavimento Locale (5° perc): {rumore_locale_db:.2f} dB")
        print(f"Rumore Medio Totale (lin):  {rumore_medio_db:.2f} dB")
        print(f"Rumore Atmosferico Vero:    {rumore_atmosferico_db:.2f} dB")
        print(f"Percentuale Atmosferico:    {perc_atm:.1f}% del totale")

        # --- GRAFICO ---
        fig_k = plt.figure(figsize=(10, 5))
        try:
            fig_k.canvas.manager.set_window_title(f"Stima Rumore - {data_str}")
        except Exception:
            pass
            
        plt.hist(df_pulito['Noise_dB'], bins=50, color='purple', alpha=0.7, edgecolor='black')
        plt.axvline(rumore_locale_db, color='red', linestyle='--', label=f'Pavimento Locale ({rumore_locale_db:.1f} dB)')
        plt.axvline(rumore_medio_db, color='blue', linestyle='-', label=f'Media Lineare ({rumore_medio_db:.1f} dB)')
        
        plt.title(f"Distribuzione del Rumore - {data_str}\nAtmosferico stimato: {perc_atm:.1f}% del totale (Plateau rimossi)", fontsize=12)
        plt.xlabel("Rumore (dB)")
        plt.ylabel("Frequenza (numero di campioni)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"[K] Errore calcolo rumore: {e}")

# ----------------------------------------------------------------------
# Gestione tasti
# ----------------------------------------------------------------------
def on_key_press(event):
    if event.key is None:
        return
        
    if event.key.lower() == 'q':
        plt.close(_STATO['fig'])
        return
        
    if event.key.lower() == 'i':
        print("[GUI] Selezione nuova data...")
        nuova_data = scegli_data_da_calendario()
        if nuova_data:
            _STATO['data_corrente'] = nuova_data
            disegna_grafico(_STATO['fig'], nuova_data)
        return

    if event.key.lower() == 'r':
        print("[GUI] Refresh dati in corso...")
        if _STATO['data_corrente']:
            disegna_grafico(_STATO['fig'], _STATO['data_corrente'])
        return

    if event.key.lower() == 'r':
        print("[GUI] Refresh dati in corso...")
        if _STATO['data_corrente']:
            disegna_grafico(_STATO['fig'], _STATO['data_corrente'])
        return

    if event.key.lower() == 'k':
        print("[GUI] Calcolo stima rumore...")
        if _STATO['data_corrente']:
            apri_finestra_stima_rumore(_STATO['data_corrente'])
        return

# ----------------------------------------------------------------------
# Disegno
# ----------------------------------------------------------------------
def _applica_marker(axes, sunrise, sunset, df):
    if sunrise is not None and sunset is not None:
        for ax in axes:
            ax.axvline(sunrise, color='#FFD700', linestyle=':', linewidth=1.0, alpha=0.7)
            ax.axvline(sunset, color='#FF8C00', linestyle=':', linewidth=1.0, alpha=0.7)

    # Marker :20 sul primo pannello (indice 0)
    if len(axes) > 0:
        t0 = df['Timestamp_UTC'].iloc[0].replace(minute=20, second=0, microsecond=0)
        t1 = df['Timestamp_UTC'].iloc[-1]
        t = t0
        while t <= t1:
            axes[0].axvline(t, color='#FF6347', linestyle=':', linewidth=0.6, alpha=0.35)
            t += timedelta(hours=1)


def disegna_grafico(fig, data_str):
    fig.clf()

    path = trova_file_snr_log(data_str)
    if not path:
        fig.text(0.5, 0.5, f"File non trovato per il {data_str}",
                 ha='center', va='center', fontsize=14, color='red')
        fig.canvas.draw_idle()
        return

    try:
        df = carica_snr_log(path)
        
        # Calcolo delle curve smussate (Rolling Mean)
        # Assumendo che i dati siano a 1 Hz, la finestra è direttamente in secondi
        df['Power_dB_smooth'] = df['Power_dB'].rolling(window=SMOOTH_WINDOW_S, center=True, min_periods=1).mean()
        df['Signal_Pure_dB_smooth'] = df['Signal_Pure_dB'].rolling(window=SMOOTH_WINDOW_S, center=True, min_periods=1).mean()
        df['SNR_band_dB_smooth'] = df['SNR_band_dB'].rolling(window=SMOOTH_WINDOW_S, center=True, min_periods=1).mean()
        df['Noise_dB_smooth'] = df['Noise_dB'].rolling(window=SMOOTH_WINDOW_S, center=True, min_periods=1).mean()
        
    except Exception as e:
        fig.text(0.5, 0.5, f"Errore lettura file:\n{e}",
                 ha='center', va='center', fontsize=12, color='red')
        fig.canvas.draw_idle()
        return

    if df.empty:
        fig.text(0.5, 0.5, f"Nessun dato nel file per il {data_str}",
                 ha='center', va='center', fontsize=12, color='orange')
        fig.canvas.draw_idle()
        return

    if MOSTRA_MEDIANE_7GG:
        med = calcola_mediana_7gg(data_str)
    else:
        med = None
        
    sunrise, sunset = calcola_sunrise_sunset(data_str)

    # Decodifica robusta: gestisce sia il vecchio formato (stringhe) che il nuovo (numerico: 1, 0, -1)
    def decode_stato(val):
        val_str = str(val).strip()
        if val_str in ['PRESENTE', '1', '1.0']: 
            return 1      # PRESENTE
        elif val_str in ['ASSENTE', '0', '0.0']: 
            return -1     # ASSENTE (nel CSV è 0, ma per l'asse Y lo mappiamo a -1)
        elif val_str in ['ATTESA', '-1', '-1.0', '2']: 
            return 0      # ATTESA (nel CSV è -1, ma per l'asse Y lo mappiamo a 0)
        return 0          # Default (ATTESA) se il valore è corrotto o mancante

    # Ricerca dinamica della colonna dello stato (compatibilità con vecchi e nuovi CSV)
    if 'Stato_NSY' in df.columns:
        df['Stato_Num'] = df['Stato_NSY'].apply(decode_stato)
    elif 'Stato_NSY_Num' in df.columns:
        df['Stato_Num'] = df['Stato_NSY_Num'].apply(decode_stato)
    elif 'Stato' in df.columns:
        df['Stato_Num'] = df['Stato'].apply(decode_stato)
    else:
        # Se non trova nessuna colonna di stato, mette tutto in ATTESA (0) per non far crashare lo script
        print("[WARNING] Nessuna colonna di stato trovata nel CSV. Uso ATTESA come default.")
        df['Stato_Num'] = 0

    # Gestione di eventuali valori NaN (celle vuote)
    df['Stato_Num'] = df['Stato_Num'].fillna(0)

    try:
        fig.canvas.manager.set_window_title(
            f"Stato Segnale NSY - {data_str}  [I] Cambia Data  [R] Refresh  [K] Stima Rumore  [Q] Esci")
    except Exception:
        pass

    # Creiamo 2 pannelli con proporzione 2/3 (curva) e 1/3 (stato)
    axes = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f"Stato Segnale NSY - {data_str}", fontsize=14, fontweight='bold')

    # ---- 1. PANNELLO UNIFICATO (Power, Signal, SNR, Noise) ----
    ax1 = axes[0]
    
    # Curve smussate (linee principali)
    ax1.plot(df['Timestamp_UTC'], df['Power_dB_smooth'], color='green', linewidth=1.5, label='Power Totale (dB)')
    ax1.plot(df['Timestamp_UTC'], df['Signal_Pure_dB_smooth'], color='darkgreen', linewidth=1.5, linestyle=':', label='Segnale NSY Puro (dB)')
    ax1.plot(df['Timestamp_UTC'], df['SNR_band_dB_smooth'], color='#00BFFF', linewidth=1.5, label='SNR (dB)')
    ax1.plot(df['Timestamp_UTC'], df['Noise_dB_smooth'], color='#B388FF', linewidth=1.5, label='Noise ±45 Hz (dB)')
    
    # Mediane 7gg (se attive)
    if med is not None:
        ax1.plot(med['Timestamp_UTC'], med['Power_dB'], color='green', linewidth=1.4, linestyle='--', alpha=0.5, label=f'Mediana {GIORNI_MEDIANA}gg (Power)')
        ax1.plot(med['Timestamp_UTC'], med['SNR_band_dB'], color='#00BFFF', linewidth=1.4, linestyle='--', alpha=0.5, label=f'Mediana {GIORNI_MEDIANA}gg (SNR)')
        ax1.plot(med['Timestamp_UTC'], med['Noise_dB'], color='#B388FF', linewidth=1.4, linestyle='--', alpha=0.5, label=f'Mediana {GIORNI_MEDIANA}gg (Noise)')

    # Linea di riferimento SNR
    ax1.axhline(SOGLIA_SNR_DB, color='red', linestyle='--', linewidth=1.2, label=f'Reference {SOGLIA_SNR_DB} dB (SNR)')
    
    ax1.set_ylabel('Ampiezza / Rapporto (dB)', color='black')
    ax1.grid(True, alpha=0.3)
    # Legenda in orizzontale per non coprire le curve
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_title("Potenza, Segnale Puro, SNR e Rumore (Scala Condivisa)", fontsize=10)

    # ---- 2. Stato NSY + ADC ----
    ax2 = axes[1]
    
    map_y_stato = {1: 1.2, 0: 0.8, -1: 0.4}
    map_y_adc = {0: -0.4, 1: -0.8}
    
    df['Stato_Y'] = df['Stato_Num'].map(map_y_stato)
    df['ADC_Y'] = df['ADC_Overload'].map(map_y_adc)
    
    ax2.plot(df['Timestamp_UTC'], df['Stato_Y'], color='purple', drawstyle='steps-post', linewidth=1.5, label='Stato NSY')
    ax2.plot(df['Timestamp_UTC'], df['ADC_Y'], color='orange', drawstyle='steps-post', linewidth=1.2, label='ADC Overload')
    
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.5, linewidth=1.0)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_yticks([1.2, 0.8, 0.4, -0.4, -0.8])
    ax2.set_yticklabels(['PRESENTE', 'ATTESA', 'ASSENTE', 'Normale', 'Overload'])
    
    for label in ax2.get_yticklabels()[:3]:
        label.set_color('purple')
        label.set_fontsize(8)
    for label in ax2.get_yticklabels()[3:]:
        label.set_color('orange')
        label.set_fontsize(8)
        
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Stato NSY (metà superiore) e ADC Overload (metà inferiore)", fontsize=10)
    ax2.legend(loc='upper left', fontsize=8, ncol=2)

    _applica_marker(axes, sunrise, sunset, df)

    # Auto-zoom X-axis
    t_min = df['Timestamp_UTC'].min()
    t_max = df['Timestamp_UTC'].max()
    padding = pd.Timedelta(minutes=15)
    axes[-1].set_xlim(t_min - padding, t_max + padding)

    # Usa un locatore automatico che si adatta alla larghezza dei dati
    locator = mdates.AutoDateLocator(minticks=4, maxticks=12)
    formatter = mdates.DateFormatter('%H:%M')
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)
    
    # Ruota e allinea automaticamente le etichette delle date
    fig.autofmt_xdate()
    axes[-1].set_xlabel('Ora UTC')

    fig.text(0.99, 0.01, "[I] Cambia data   [R] Refresh   [K] Stima Rumore   [Q] Esci",
             ha='right', va='bottom', fontsize=9, family='monospace',
             color='#333333',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F8F8',
                       edgecolor='#AAAAAA', alpha=0.9))

    # Layout ottimizzato per 2 pannelli
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.canvas.draw_idle()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Visualizzatore Log SNR NSY (Versione Unificata a 2 Pannelli)")
    print("=" * 60)

    data_str = scegli_data_da_calendario()
    if not data_str:
        print("[INFO] Selezione annullata.")
        return

    _STATO['data_corrente'] = data_str
    # Ridotta l'altezza della figura per adattarsi ai 2 pannelli
    _STATO['fig'] = plt.figure(figsize=(14, 9)) 
    _STATO['fig'].canvas.mpl_connect('key_press_event', on_key_press)

    disegna_grafico(_STATO['fig'], data_str)
    plt.show()


if __name__ == "__main__":
    main()