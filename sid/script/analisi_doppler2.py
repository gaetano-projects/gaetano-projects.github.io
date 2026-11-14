# ============================================================
# analisi_doppler.py
#
# Analisi Doppler dei flare solari sui dati VLF NSY.
# - Legge il CSV del tracker (Timestamp, Power, Phase, PLL_Freq, Reset)
# - Scarica/legge X-ray GOES
# - Scarica/legge flares DONKI (con fallback manuale per date future)
# - Plotta 4 pannelli
# - Calcola correlazione Doppler-flare con lag
#
# NON tocca i dati di Visualizza_37.
# ============================================================

import os
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone, timedelta
from scipy.ndimage import median_filter
from scipy.signal import correlate
from astral import LocationInfo
from astral.sun import elevation as solar_elevation
from scipy.ndimage import median_filter, uniform_filter1d
import time
from scipy.stats import ttest_ind

from scipy.signal import butter, sosfiltfilt, medfilt
from scipy.signal import savgol_filter

# --- Utility DSP condivise ---
from dsp_utils import filtro_hampel, sanity_doppler_con_durata

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

# --- Metodo frozen v1.0 (doppio null) — opzionale ---
try:
    from frozen_method import valuta_frozen_da_df
    _FROZEN_OK = True
except ImportError:
    _FROZEN_OK = False
    valuta_frozen_da_df = None

if _FROZEN_OK:
    print("### FROZEN v1.0 attivo: verdetto doppio-null abilitato ###")

print("### VERSIONE SCRIPT: 13-NOV-2026 v2.1 (Blind Test Ready) ###")
# ============================================================
# CONFIGURAZIONE
# ============================================================
DATA = "2026-09-29"   # formato YYYY-MM-DD

# Disabilita le scorciatoie predefinite di matplotlib che confliggono
# con i tasti dell'applicazione (f=fullscreen, s=save, g=grid, l=log, k=xscale).
plt.rcParams['keymap.fullscreen'] = []   # libera 'f' per il filtro
plt.rcParams['keymap.save']       = []   # libera 's'
plt.rcParams['keymap.grid']       = []   # libera 'g'
plt.rcParams['keymap.xscale']     = []   # libera 'k'
plt.rcParams['keymap.yscale']     = []   # libera 'l'
plt.rcParams['keymap.pan']        = []   # libera 'p'
plt.rcParams['keymap.zoom']       = []   # libera 'o'

plt.rcParams['keymap.home']       = []   # libera 'h'
plt.rcParams['keymap.back']       = []   # libera 'c' o 'left'
plt.rcParams['keymap.forward']    = []   # libera 'v' o 'right'

CSV_TRACKER = rf"D:\Gaetano\SID\dati\Sdruno\GAESID_NSY_{DATA}.csv"

# Directory dove cercare i CSV del tracker.
# Ordine di priorità: il primo che contiene il file vince.
# La prima è la cartella "attiva" (giorni recenti); la seconda è
# l'archivio storico (giorni vecchi).
CARTELLE_CSV = [
    r"D:\Gaetano\SID\dati\Sdruno",
    r"D:\Gaetano\SID\dati\Sdruno\Storico_NSY",
]

CARTELLA_XRAY = r"D:\Gaetano\SID\pysdrplay\Dati\Storico_NSY\XRAY"
CARTELLA_FLARES = r"D:\Gaetano\SID\pysdrplay\Dati\Storico_NSY\Flares"
os.makedirs(CARTELLA_XRAY, exist_ok=True)
os.makedirs(CARTELLA_FLARES, exist_ok=True)

PATH_XRAY = os.path.join(CARTELLA_XRAY, f"GOES_XRAY_{DATA}.csv")
PATH_FLARES = os.path.join(CARTELLA_FLARES, f"FLARES_{DATA}.json")

# Per le date entro N giorni da oggi, la cache viene ignorata e i dati
# vengono sempre riscaricati da internet (SWPC / DONKI). Questo perché
# i dati recenti possono essere incompleti al primo scaricamento e
# vengono aggiornati dal provider nelle ore/giorni successivi.
# Per le date più vecchie, la cache è considerata definitiva.
GIORNI_REFRESH_CACHE = 2

# Parametri analisi
FINESTRA_DETREND_SEC = 7200     # 30 min per detrend PLL_Freq
FINESTRA_FLARE_SEC   = 1800     # ±30 min intorno al flare (era 900)
LAG_MAX_SEC          = 150      # ±10 min per cross-correlazione (era 120)
ESCLUDI_ALBA_TRAMONTO = True    # escludi flare vicini ad alba/tramonto
ORA_ALBA_UTC = 5
ORA_TRAMONTO_UTC = 17

# --- Soglia elevazione Sole per SID atteso ---
SOGLIA_ELEVAZIONE_SOLE_DEG = 20.0   # era 25.0

# --- Diagnostica qualità del lock ---
AMP100_DROP_DB       = 12.0    # calo di Amp_100Hz_dB sotto (mediana giornaliera - 12 dB) = lock perso
FRAZIONE_INVALIDA_MAX = 0.20   # oltre 20% di campioni invalidi → finestra scartata
CORR_MIN_SID         = 0.50    # |corr| minima per parlare di SID reale
# Soglie PLL aggiornate 03/10/2026 dopo il fix di Claude sul tracker:
# prima la std di PLL_Freq era ~50 mHz (soglia 0.05 Hz = 50 mHz, inutilizzabile).
# Ora la std è ~0.5 mHz, quindi la soglia minima scende di 100×.
DELTA_PLL_MIN_HZ     = 0.0005  # 0.5 mHz — Doppler minimo per SID (a livello del rumore diurno)
DPLL_RAPPORTO_MAX    = 5.0     # era 20 — ΔPLL/Δfase oltre questo = PLL rumoroso
LAG_COERENTE_SEC     = 180     # lag massimo tra X-ray e Doppler per SID reale
DELTA_FASE_ARTEFATTO = 0.005   # 5 mHz — salto Doppler senza controparte PLL è sospetto

# --- Finestre temporali per analisi evento (tutte in secondi) ---
# Baseline VECCHIA (centrata sul picco): mantenuta per retrocompatibilità
# ma NON più usata per il calcolo del Δ.
BASE_PRE_SEC         = 600    # (deprecato)
BASE_POST_SEC        = 120    # (deprecato)

# Baseline NUOVA (bilaterale, basata su inizio/fine del flare)
# Pre-evento: da inizio-3600s a inizio-300s
# Post-evento: da fine+300s a fine+3600s
BASE_PRE_EVENTO_SEC  = 3600
BASE_POST_EVENTO_SEC = 3600
BASE_SKIP_SEC        = 300    # margine di esclusione attorno all'evento

EVENTO_PRE_SEC       = 30     # evento: da inizio-30s
EVENTO_POST_SEC      = 180    # evento: fino a fine+180s (per fase/PLL)
EVENTO_POST_PWR_SEC  = 600    # (deprecato, ora usa EVENTO_POST_SEC)

# --- Soglie per SID di potenza ---
PWR_SID_MIN_DB       = 0.30   # differenza media minima (dB)
PWR_SID_MIN_SNR      = 2.0    # SNR minimo sulla potenza
PWR_SID_WEAK_SNR     = 1.2    # soglia per candidato debole
PWR_SID_PICCO_MIN_DB = 1.5   # Soglia minima per il picco (evita falsi positivi da rumore)

# Soglia dinamica per il picco di potenza (si adatta al rumore locale)
PWR_SID_PICCO_MIN_DB  = 1.5   # Floor minimo assoluto (inverno)
PWR_SID_PICCO_K_SIGMA = 4.0   # Moltiplicatore per la MAD (estate/rumore alto)
PWR_SID_SFA_MIN_DB = 2.0  # Soglia più alta per i cali (evita falsi dropout)

# Soglia per il canale fase (SNR intra-giorno del bump Doppler).
# Ricalibrata il 2026-10-07 dopo la misura della deviazione di Allan:
#   sigma_Allan(900 s) = 0.030 mHz (media di 03/10 e 05/10)
# La formula precedente (se_evento = std/sqrt(N)) era ottimistica di ~2.8x.
# Ora usiamo sigma_Allan(900s) come denominatore.
# Valori attesi con la nuova formula:
#   C7.9 (giorno pulito): SNR ~ 2.7
#   M1.8 (giorno pulito): SNR ~ 0.5 (sotto rumore!)
#   M5+ (giorno pulito):  SNR ~ 17-33
# Soglia promuove C7+ e M+, ma NON i M1-M2 (era 2.5, ora 1.5).
SNR_FASE_MIN         = 2.5
SIGMA_ALLAN_MHZ_900S = 0.030   # mHz, dalla misura notturna (03/10-05/10)

# Soglia di sanity check per il canale fase.
# Valori oltre 5 mHz sono fisicamente impossibili per il Doppler VLF
# (un X10 produce ~1-2 mHz). Se il delta medio o la std quieta
# superano questa soglia, c'e' una discontinuita' di fase non
# mascherata nel CSV (tipico dei giorni con manutenzione TX).
SOGLIA_ARTEFATTO_FASE_MHZ = 5.0

# Soglia per il canale FASE INTEGRATA (Δφ/σ).
# NOTA: Questa metrica è diagnostica. La soglia 5.0 è provvisoria
# e necessita di calibrazione sul 95° percentile dei giorni quieti.
SNR_FASE_INT_MIN     = 5.0

# Soglie Doppler filtrato (Δfase) — il canale fisico
DFASE_MIN_HZ      = 0.005   # Doppler debole (C)
DFASE_FORTE_HZ    = 0.020   # Doppler forte (M/X)
#DPLL_RAPPORTO_MAX = 20.0    # ΔPLL/Δfase oltre questo → PLL rumoroso

CARTELLA_PLOT_EVENTI = os.path.join(CARTELLA_FLARES, "eventi")
os.makedirs(CARTELLA_PLOT_EVENTI, exist_ok=True)

# ============================================================
# MATCHED DOPPLER (Regola secondaria, 12 Nov 2026)
# Riferimento validato: check_matched_doppler.py
# ============================================================
DOPPLER_POOL_PATH   = r"D:\Gaetano\SID\pysdrplay\doppler_pool_v1.0.parquet"
DOPPLER_POOL_SHA256 = None
MATCHED_DOPPLER_ON  = True
MIN_GEMELLE         = 10
PERCENTILE_SOGLIA   = 95.0
AMP100_TOL_DB       = 3.0
ORA_TOL_H           = 1.0

# ============================================================
# GEOMETRIA DEL PERCORSO (NSY Niscemi → Torino)
# ============================================================
TX_LAT, TX_LON = 37.13, 14.39    # NSY — Niscemi (CL)
RX_LAT, RX_LON = 45.07, 7.68     # Torino

# Punto medio del percorso (approssimazione piana, ok per ~1000 km)
MID_LAT = (TX_LAT + RX_LAT) / 2.0
MID_LON = (TX_LON + RX_LON) / 2.0

# ============================================================
# FILTRI PER CANALE (calibrati come Visualizza_37)
# ============================================================
# Ampiezza: filtro uniforme 60 s (media mobile) → cattura SAA anche deboli (C2+)
# Fase:     filtro mediano 30 s → cattura SPA, robusto ai picchi
# PLL_Freq: filtro mediano 121 s → cattura solo Doppler di M/X
SMOOTH_AMPIEZZA_SEC = 60
SMOOTH_FASE_SEC     = 30
SMOOTH_PLL_SEC      = 30

# ============================================================
# POLARITÀ DEL SITO (osservazione empirica stazione NSY→Torino)
# ============================================================
POLARITA_SID_AMPIEZZA = +1   # +1 = aumenta, -1 = cala

# Polarità del Doppler (PLL_Freq / derivata fase).
# +1 = il Doppler aumenta durante un SID (fase va verso positivo).
# -1 = il Doppler diminuisce durante un SID.
# Da verificare empiricamente sul prossimo M1+.
# Valore di default: +1 (ipotesi: aumento di ionizzazione → aumento
# dell'altezza di riflessione → Doppler positivo).
POLARITA_SID_DOPPLER = 1

# Stato per la finestra di messaggio "nessun flare" — così non se ne
# accumulano più di una se l'utente preme R più volte.
_FIG_MESSAGGIO = {'fig': None}

_FINESTRA_MSG_TK = {'win': None}

import numpy as np
import pandas as pd

def esegui_sanity_check_pll(df):
    eventi_grezzi = []
    
    if 'Timestamp_UTC' in df.columns:
        ts_series = pd.to_datetime(df['Timestamp_UTC'])
    elif df.index.name == 'Timestamp_UTC' or isinstance(df.index, pd.DatetimeIndex):
        ts_series = df.index
    else:
        ts_series = pd.to_datetime(df.iloc[:, 0])

    baseline_potenza = df['Power_dB'].rolling(window=1800, center=True, min_periods=300).median()
    baseline_amp100 = df['Amp_100Hz_dB'].rolling(window=1800, center=True, min_periods=300).median()
    baseline_freq = df['PLL_Freq_Hz'].rolling(window=1800, min_periods=300).median()
    
    residuo_freq = (df['PLL_Freq_Hz'] - baseline_freq).fillna(0)
    mad_freq = residuo_freq.abs().rolling(window=1800, center=True, min_periods=300).median() * 1.4826
    mad_freq = mad_freq.fillna(0.005)
    residuo_np = residuo_freq.values
    
    derivata_freq = df['PLL_Freq_Hz'].diff().abs().fillna(0)
    derivata_np = derivata_freq.values

    def is_terminatore(ts):
        ora_decimale = ts.hour + ts.minute / 60.0
        return 4.5 <= ora_decimale <= 6.5 or 16.5 <= ora_decimale <= 18.5

    def fmt(val, decimals=1):
        if abs(val) < (10 ** -decimals): return f"{0.0:+.{decimals}f}"
        return f"{val:+.{decimals}f}"

    # FASE 1: Raccolta spike grezzi
    for i in range(len(df)):
        row = df.iloc[i]
        ts = ts_series.iloc[i]
        
        freq_val = row['PLL_Freq_Hz']
        pwr_val = row['Power_dB']
        amp100_val = row['Amp_100Hz_dB']
        
        pwr_base = baseline_potenza.iloc[i]
        amp100_base = baseline_amp100.iloc[i]
        freq_base = baseline_freq.iloc[i]
        freq_mad = mad_freq.iloc[i]

        df_val = abs(freq_val - freq_base) if pd.notna(freq_base) else 0
        dp_val = (pwr_base - pwr_val) if pd.notna(pwr_base) else 0
        da_val = (amp100_val - amp100_base) if pd.notna(amp100_base) else 0

        soglia_spike = max(4.0 * freq_mad, 0.005)
        
        is_reset_flag = ('Reset_Flag' in df.columns and row['Reset_Flag'] == 1)
        is_impossible_jump = df_val > 500.0 
        is_flatline = (freq_val == 0.0 and pd.notna(freq_base) and abs(freq_base) > 0.1)
        
        if df_val > soglia_spike or dp_val > 10.0 or da_val > 15.0 or is_reset_flag or is_impossible_jump or is_flatline:
            eventi_grezzi.append({
                'ts': ts, 'idx': i, 
                'df': df_val, 'signed_df': freq_val - freq_base if pd.notna(freq_base) else 0,
                'dp': dp_val, 'signed_da': amp100_val - amp100_base if pd.notna(amp100_base) else 0,
                'freq_val': freq_val, 'pwr_val': pwr_val, 'amp100_val': amp100_val,
                'is_reset': is_reset_flag, 'is_jump': is_impossible_jump, 'is_flat': is_flatline
            })

    # FASE 2: Clustering (fino a 180s, esteso a 900s se c'è un dropout)
    clusters = []
    if eventi_grezzi:
        corrente = [eventi_grezzi[0]]
        for successivo in eventi_grezzi[1:]:
            has_dropout = any(e['is_reset'] or e['is_jump'] or e['is_flat'] for e in corrente)
            limite = 900 if has_dropout else 180
            
            if (successivo['ts'] - corrente[-1]['ts']).total_seconds() < limite:
                corrente.append(successivo)
            else:
                clusters.append(corrente)
                corrente = [successivo]
        clusters.append(corrente)

    # FASE 3: Valutazione
    risultato = []
    for cl in clusters:
        idx_inizio = cl[0]['idx']
        idx_fine = cl[-1]['idx']
        
        ts_inizio = cl[0]['ts'].strftime('%H:%M:%S')
        ts_fine = cl[-1]['ts'].strftime('%H:%M:%S')
        intervallo = f"{ts_inizio} - {ts_fine}" if ts_inizio != ts_fine else ts_inizio
        durata = (cl[-1]['ts'] - cl[0]['ts']).total_seconds()
        
        max_dp = max(e['dp'] for e in cl)
        max_da = max(e['signed_da'] for e in cl)
        max_signed_df = max(cl, key=lambda x: abs(x['signed_df']))['signed_df']
        max_slope_mhz = (np.max(derivata_np[idx_inizio:idx_fine+1]) if idx_fine >= idx_inizio else 0) * 1000

        rientro_fase = 0.0
        if abs(max_signed_df) > 0.005 and abs(max_signed_df) < 500.0: 
            start_w = max(0, idx_inizio - 30)
            end_w = min(len(df), idx_fine + 300)
            window_resid = residuo_np[start_w:end_w]
            if len(window_resid) > 10:
                pre_evento = window_resid[:min(60, len(window_resid))]
                offset_locale = np.mean(pre_evento) if len(pre_evento) > 0 else 0
                resid_centrato = window_resid - offset_locale
                area_netta = abs(np.sum(resid_centrato))
                area_totale = np.sum(np.abs(resid_centrato))
                if area_totale > 0:
                    rientro_fase = area_netta / area_totale

        # --- LOGICA DI PRIORITÀ v5.4 ---
        has_dropout = any(e['is_reset'] or e['is_jump'] or e['is_flat'] for e in cl)
        
        if has_dropout:
            motivo = "Dropout / Reset Hardware"
            str_df, str_rientro, str_slope = "n/d", "n/d", "n/d"
        else:
            str_df = fmt(max_signed_df*1000)
            str_rientro = f"{rientro_fase:.2f}"
            str_slope = f"{max_slope_mhz:.1f}"
            
            if max_da > 15.0:
                motivo = "Disturbo 100Hz"
            elif is_terminatore(cl[0]['ts']) and abs(max_signed_df) > 0.005:
                motivo = "Spike PLL (Terminatore)"
            elif abs(max_signed_df) > 0.005:
                if durata < 20:
                    motivo = "Glitch / Spike breve"
                elif rientro_fase < 0.3:
                    motivo = "Candidato SFD (Fase Rientrata)"
                elif rientro_fase <= 0.6:
                    motivo = "Ambiguo (Fase 0.3-0.6)"
                else:
                    motivo = "Fase Persistente"
            elif max_dp > 10.0:
                motivo = "Crollo Potenza"
            else:
                motivo = "Spike PLL"

            if abs(max_signed_df) > 0.015:
                motivo += " (Critico)"
            if max_slope_mhz > 10.0:
                motivo += " [Slope Elevata]"

        worst_pwr = min(e['pwr_val'] for e in cl)
        worst_amp = max(e['amp100_val'] for e in cl)
        
        risultato.append((
            intervallo, f"[{len(cl)} spike] {motivo}", 
            str_df, fmt(-max_dp), fmt(max_da), str_rientro,
            str_slope, f"{worst_pwr:.1f}", f"{worst_amp:.1f}"
        ))

    return risultato

def mostra_finestra_sanity_check(df, fig_principale=None):
    from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QLabel
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QColor

    dialog = QDialog()
    dialog.setWindowTitle("Sanity Check PLL - Diagnostica v5.4 (Blind Test Ready)")
    dialog.resize(1200, 650)
    
    dialog.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
    dialog.setAttribute(Qt.WA_DeleteOnClose)
    layout = QVBoxLayout()

    try:
        eventi = esegui_sanity_check_pll(df)
    except Exception as e:
        import traceback
        print(f"[DEBUG] Errore: {e}", flush=True)
        traceback.print_exc()
        return

    if not eventi:
        msg = QMessageBox()
        msg.setWindowTitle("Sanity Check PLL")
        msg.setText("Nessun problema nel PLL durante la giornata.")
        msg.setIcon(QMessageBox.Information)
        msg.exec_()
        return

    table = QTableWidget()
    table.setRowCount(len(eventi))
    table.setColumnCount(9)
    table.setHorizontalHeaderLabels([
        "Intervallo", "Motivo", "Δ Freq (mHz)", "Δ Power (dB)", "Δ Amp100 (dB)", 
        "Rientro Fase", "Max Slope (mHz/s)", "Power Ass.", "Amp100 Ass."
    ])

    for row_idx, ev in enumerate(eventi):
        for col_idx, val in enumerate(ev):
            item = QTableWidgetItem(str(val))
            if col_idx != 1: item.setTextAlignment(Qt.AlignCenter)
            
            if col_idx == 1:
                text = str(val)
                if "Dropout" in text:
                    item.setForeground(QColor("darkred"))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                elif "Slope Elevata" in text:
                    item.setForeground(QColor("magenta"))
                elif "Candidato SFD" in text:
                    item.setForeground(Qt.darkGreen)
                elif "Ambiguo" in text or "Terminatore" in text or "Glitch" in text:
                    item.setForeground(Qt.darkYellow)
                elif "100Hz" in text:
                    item.setForeground(Qt.blue)
                    
            table.setItem(row_idx, col_idx, item)

    header = table.horizontalHeader()
    header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
    header.setSectionResizeMode(1, QHeaderView.Stretch)
    for i in range(2, 9): header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
    
    table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    table.setAlternatingRowColors(True)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    layout.addWidget(table)

    legenda_testo = """
    <html>
    <body>
    <b>Significato dei Motivi (v5.4):</b>
    <ul style="margin-top: 2px; margin-bottom: 2px; margin-left: -25px;">
        <li><b style='color:darkred;'>Dropout / Reset Hardware:</b> Salto impossibile (&gt;500 mHz) o flatline. I valori Δ sono "n/d".</li>
        <li><b style='color:#0000cc;'>Disturbo 100Hz:</b> Δ Amp100 &gt; +15 dB. Interferenza locale.</li>
        <li><b style='color:#b8860b;'>Finestra Terminatore:</b> Anomalia durante alba/tramonto.</li>
        <li><b style='color:#006600;'>Candidato SFD:</b> Durata &gt; 20s, Fase Rientrata (&lt; 0.3).</li>
        <li><b style='color:#b8860b;'>Glitch / Spike breve:</b> Durata &lt; 20s.</li>
        <li><b style='color:#b8860b;'>Ambiguo:</b> Durata &gt; 20s, Fase tra 0.3 e 0.6.</li>
        <li><b style='color:#b8860b;'>Fase Persistente:</b> Fase &gt; 0.6 (senza crollo di potenza).</li>
        <li><b>(Critico):</b> Δ Freq &gt; 15 mHz. &nbsp; <b>[Slope Elevata]:</b> Pendenza &gt; 10 mHz/s.</li>
    </ul>
    </body>
    </html>
    """
    legenda_label = QLabel(legenda_testo)
    legenda_label.setWordWrap(True)
    legenda_label.setStyleSheet("QLabel { background-color: #f9f9f9; color: #333333; border: 1px solid #cccccc; border-radius: 6px; padding: 12px; font-size: 12px; font-family: Arial, sans-serif; }")
    layout.addWidget(legenda_label)

    dialog.setLayout(layout)

    if fig_principale is not None:
        if not hasattr(fig_principale, '_qt_dialogs'): fig_principale._qt_dialogs = []
        fig_principale._qt_dialogs.append(dialog)

    dialog.show()
    

def maschera_flare_vicini(timestamps, picco, flares,
                          margine_pre_s=300, margine_post_s=900,
                          tol_stesso_s=90):
    """
    Maschera booleana (True = campione da escludere dalle finestre di
    quiete) per i campioni che cadono dentro [inizio - margine_pre,
    fine + margine_post] di un ALTRO flare. Il flare corrente (picco
    entro tol_stesso_s da `picco`) non viene mascherato.
    Inizio/fine mancanti: picco -5 min / +15 min (come detrend_mascherato).
    Il margine_post copre la coda ionosferica del flare e il transitorio
    del filtro passa-banda (t_lento=2400 s).
    """
    ts = np.asarray(timestamps).astype('datetime64[s]')
    out = np.zeros(len(ts), dtype=bool)
    if not flares:
        return out
    t_cur = np.datetime64(picco.replace(tzinfo=None), 's')
    for f in flares:
        pk = parse_flare_datetime(f.get('picco', ''))
        if pk is None:
            continue
        pk64 = np.datetime64(pk.replace(tzinfo=None), 's')
        if abs((pk64 - t_cur) / np.timedelta64(1, 's')) <= tol_stesso_s:
            continue
        a = parse_flare_datetime(f.get('inizio', '')) or (pk - timedelta(minutes=5))
        b = parse_flare_datetime(f.get('fine', ''))   or (pk + timedelta(minutes=15))
        a64 = np.datetime64(a.replace(tzinfo=None), 's') - np.timedelta64(int(margine_pre_s), 's')
        b64 = np.datetime64(b.replace(tzinfo=None), 's') + np.timedelta64(int(margine_post_s), 's')
        out |= (ts >= a64) & (ts <= b64)
    return out

def maschera_lock_ok(df):
    """Restituisce una maschera booleana True dove il lock è accettabile."""
    ok = np.ones(len(df), dtype=bool)
    if 'Corr_Mag' in df.columns:
        cm = pd.Series(df['Corr_Mag'].values).rolling(60, center=True, min_periods=10).median().values
        ok &= ~(np.isfinite(cm) & (cm < 0.40))
    elif 'Amp_100Hz_dB' in df.columns:
        # Fallback per giorni vecchi senza Corr_Mag
        soglia_lock = df.attrs.get('soglia_lock_amp100', None)
        if soglia_lock is not None:
            amp = df['Amp_100Hz_dB'].values
            ok &= ~(np.isfinite(amp) & (amp < soglia_lock))
    return ok

def snr_locale_bump(df, picco,
                    win_evento_sec=(-600, 300),
                    win_quiete_pre=(-2400, -900),
                    win_quiete_post=(900, 2400),
                    altri_flare=None):
    """
    Calcola l'SNR intra-giorno del bump Doppler intorno a un flare.

    VERSIONE CORRETTA (2026-10-07 v3):
    Usa la deviazione di Allan alla scala della finestra evento come
    denominatore, invece di se_evento = std/sqrt(N).

    Motivo: i campioni del Doppler filtrato NON sono indipendenti
    (tempo di correlazione ~120 s, misurato dalla deviazione di Allan).
    La formula std/sqrt(N) gonfia l'SNR di ~2.8x.

    Il denominatore usato è la std locale della finestra di quiete
    (std_quieta), che si adatta alle condizioni di rumore del giorno.

    Finestre:
      evento: -10 min ... +5 min dal picco
      quiete_pre: -40 min ... -15 min
      quiete_post: +15 min ... +40 min
    """
    if picco is None:
        return None

    # --- Calcola la serie Doppler filtrata sull'intero df ---
    fase_raw = df['Phase_Deg'].values.astype(float).copy()
    disc_mask = (df['Discontinuita'].values == 1) \
                if 'Discontinuita' in df.columns else None
    if disc_mask is not None:
        fase_raw[disc_mask] = np.nan

    fase_interp = pd.Series(fase_raw).interpolate(
        method='linear', limit_direction='both').bfill().ffill().values
    d_fase_hz = np.gradient(fase_interp) / 360.0
    d_fase_hz = d_fase_hz - np.nanmedian(d_fase_hz)
    fase_smooth = median_filter(d_fase_hz, size=SMOOTH_FASE_SEC)
    if disc_mask is not None:
        fase_smooth = fase_smooth.copy()
        fase_smooth[disc_mask] = np.nan

    # Sanity check: nessun Doppler VLF reale supera 20 mHz.
    # Un reset azzera il contatore di fase e l'interpolazione
    # lineare del buco crea rampe con pendenza ~590 Hz che il
    # median filter non rimuove ai bordi. Marchiamo come NaN
    # tutto ciò che è fisicamente impossibile.
    SOGLIA_DOPPLER_FISICA_HZ = 0.020
    mask_impossibile = np.abs(fase_smooth) > SOGLIA_DOPPLER_FISICA_HZ
    fase_smooth = fase_smooth.copy()
    fase_smooth[mask_impossibile] = np.nan

    fd = FiltroDoppler(t_lento=2400, t_veloce=120, fs=1.0)
    fd.calcola(fase_smooth)
    fase_f = (fd._filtrato.copy() if fd._filtrato is not None
              else fase_smooth.copy())
    if disc_mask is not None:
        fase_f[disc_mask] = np.nan

    # --- Calcola i tempi relativi al picco ---
    ts = df['Timestamp'].values
    t_pk = np.datetime64(picco.replace(tzinfo=None))
    t_sec = (ts - t_pk) / np.timedelta64(1, 's')

    # --- Maschere ---
    m_evento = (t_sec >= win_evento_sec[0]) & (t_sec <= win_evento_sec[1])
    m_pre    = (t_sec >= win_quiete_pre[0]) & (t_sec <= win_quiete_pre[1])
    m_post   = (t_sec >= win_quiete_post[0]) & (t_sec <= win_quiete_post[1])
    m_quiete = m_pre | m_post
    # Escludi dalla quiete i campioni su altri flare (inizio-5 min .. fine+15 min)
    m_altri = maschera_flare_vicini(df['Timestamp'].values, picco, altri_flare)
    m_quiete = m_quiete & ~m_altri
    contaminato = bool(np.any(m_evento & m_altri))

    lock_ok = maschera_lock_ok(df)
    
    f_evento = fase_f[m_evento]
    f_evento = f_evento[np.isfinite(f_evento)]
    
    q = fase_f[m_quiete]
    f_quiete = q[np.isfinite(q) & lock_ok[m_quiete]]

    # FALLBACK: se la quiete è troppo contaminata, allarga a ±90 min
    if len(f_quiete) < 120:
        win_quiete_pre = (-5400, -900)
        win_quiete_post = (900, 5400)
        m_pre = (t_sec >= win_quiete_pre[0]) & (t_sec <= win_quiete_pre[1])
        m_post = (t_sec >= win_quiete_post[0]) & (t_sec <= win_quiete_post[1])
        m_quiete = m_pre | m_post
        m_quiete = m_quiete & ~m_altri
        q = fase_f[m_quiete]
        f_quiete = q[np.isfinite(q) & lock_ok[m_quiete]]

    if len(f_evento) < 60 or len(f_quiete) < 120:
        return None

    std_quieta = float(np.std(f_quiete))
    if std_quieta <= 0:
        return None

    # --- Delta RMS (cattura i segnali bipolari che si annullano in media) ---
    media_quiete = float(np.mean(f_quiete))
    # RMS della differenza rispetto alla media della quiete
    delta_medio = float(np.sqrt(np.mean((f_evento - media_quiete)**2)))

    # --- SNR con deviazione locale (std_quieta) ---
    # Usiamo la std della finestra di quiete locale invece della sigma 
    # di Allan notturna fissa, per non gonfiare l'SNR nei giorni rumorosi.
    snr_corretto = delta_medio / std_quieta if std_quieta > 0 else None

    # --- Forma (segno della massima deviazione) ---
    dev = f_evento - media_quiete
    forma_su = bool(dev[np.argmax(np.abs(dev))] > 0)

    # --- Info diagnostiche ---
    n_evento = len(f_evento)
    range_evento = float(np.max(f_evento) - np.min(f_evento))
    idx_max_rel = int(np.argmax(f_evento))
    idx_min_rel = int(np.argmin(f_evento))
    picco_dopo_minimo = idx_max_rel > idx_min_rel

    return {
        # Metriche principali
        'snr_locale':        snr_corretto,
        'ampiezza_bump_mhz': delta_medio * 1000.0,
        'std_quieta_mhz':    std_quieta * 1000.0,
        'sigma_allan_mhz':   SIGMA_ALLAN_MHZ_900S,
        'bump_su':           forma_su,
        'n_evento':          n_evento,
        'n_quiete':          int(np.isfinite(f_quiete).sum()),
        'evento_contaminato': contaminato,
        # Diagnostiche
        'range_evento_mhz':  range_evento * 1000.0,
        'picco_dopo_minimo': picco_dopo_minimo,
    }

def snr_fase_integrata(df, picco,
                       win_evento_sec=(-600, 300),
                       win_quiete_pre=(-2400, -900),
                       win_quiete_post=(900, 2400),
                       altri_flare=None):
    """
    Calcola l'SNR della FASE INTEGRATA (integrale del Doppler filtrato).

    Perché è più sensibile del Doppler:
    - Il Doppler istantaneo ha rumore sigma_D su ogni campione
    - La fase integrata su T secondi accumula il segnale come D*T
      e il rumore come sigma_D * sqrt(T)
    - Quindi SNR_fase_int = SNR_Doppler * sqrt(N_eff)
    - Con N_eff ~ 7.5 (durata 900 s / tempo_correlazione 120 s),
      il guadagno e' ~2.7x

    Applica un detrend mediano 1200 s per rimuovere la deriva lenta
    accumulata dall'integrazione.

    Ritorna dict o None se dati insufficienti.
    """
    if picco is None:
        return None

    # --- Calcola la serie Doppler filtrata sull'intero df ---
    fase_raw = df['Phase_Deg'].values.astype(float).copy()
    disc_mask = (df['Discontinuita'].values == 1) \
                if 'Discontinuita' in df.columns else None
    if disc_mask is not None:
        fase_raw[disc_mask] = np.nan

    fase_interp = pd.Series(fase_raw).interpolate(
        method='linear', limit_direction='both').bfill().ffill().values
    d_fase_hz = np.gradient(fase_interp) / 360.0
    d_fase_hz = d_fase_hz - np.nanmedian(d_fase_hz)
    fase_smooth = median_filter(d_fase_hz, size=SMOOTH_FASE_SEC)
    if disc_mask is not None:
        fase_smooth = fase_smooth.copy()
        fase_smooth[disc_mask] = np.nan

    # Sanity check: nessun Doppler VLF reale supera 20 mHz.
    # Un reset azzera il contatore di fase e l'interpolazione
    # lineare del buco crea rampe con pendenza ~590 Hz che il
    # median filter non rimuove ai bordi. Marchiamo come NaN
    # tutto ciò che è fisicamente impossibile.
    SOGLIA_DOPPLER_FISICA_HZ = 0.020
    mask_impossibile = np.abs(fase_smooth) > SOGLIA_DOPPLER_FISICA_HZ
    fase_smooth = fase_smooth.copy()
    fase_smooth[mask_impossibile] = np.nan

    fd = FiltroDoppler(t_lento=2400, t_veloce=120, fs=1.0)
    fd.calcola(fase_smooth)
    fase_f = (fd._filtrato.copy() if fd._filtrato is not None
              else fase_smooth.copy())
    if disc_mask is not None:
        fase_f[disc_mask] = np.nan

    # --- Integra il Doppler filtrato per ottenere la fase cumulata ---
    # Sostituisci NaN con 0 (l'integrale non può accumulare buchi)
    fase_f_clean = np.where(np.isfinite(fase_f), fase_f, 0.0)
    fase_int = np.cumsum(fase_f_clean) * 1.0     # cicli

    # Detrend mediano 1200 s: rimuove la deriva lenta accumulata
    fase_int = detrend_mediano(fase_int, 1200)

    # Riapplica NaN dove c'erano discontinuita'
    if disc_mask is not None:
        fase_int = fase_int.copy()
        fase_int[disc_mask] = np.nan

    # --- Calcola i tempi relativi al picco ---
    ts = df['Timestamp'].values
    t_pk = np.datetime64(picco.replace(tzinfo=None))
    t_sec = (ts - t_pk) / np.timedelta64(1, 's')

    # --- Maschere ---
    m_evento = (t_sec >= win_evento_sec[0]) & (t_sec <= win_evento_sec[1])
    m_pre    = (t_sec >= win_quiete_pre[0]) & (t_sec <= win_quiete_pre[1])
    m_post   = (t_sec >= win_quiete_post[0]) & (t_sec <= win_quiete_post[1])
    m_quiete = m_pre | m_post
    # Escludi dalla quiete i campioni su altri flare (inizio-5 min .. fine+15 min)
    m_altri = maschera_flare_vicini(df['Timestamp'].values, picco, altri_flare)
    m_quiete = m_quiete & ~m_altri
    contaminato = bool(np.any(m_evento & m_altri))

    lock_ok = maschera_lock_ok(df)
    
    f_evento = fase_int[m_evento]
    f_evento = f_evento[np.isfinite(f_evento)]
    
    q = fase_int[m_quiete]
    f_quiete = q[np.isfinite(q) & lock_ok[m_quiete]]

    # FALLBACK: se la quiete è troppo contaminata, allarga a ±90 min
    if len(f_quiete) < 120:
        win_quiete_pre = (-5400, -900)
        win_quiete_post = (900, 5400)
        m_pre = (t_sec >= win_quiete_pre[0]) & (t_sec <= win_quiete_pre[1])
        m_post = (t_sec >= win_quiete_post[0]) & (t_sec <= win_quiete_post[1])
        m_quiete = m_pre | m_post
        m_quiete = m_quiete & ~m_altri
        q = fase_int[m_quiete]  # <-- QUI USA fase_int
        f_quiete = q[np.isfinite(q) & lock_ok[m_quiete]]


    if len(f_evento) < 60 or len(f_quiete) < 120:
        return None

    std_quiete = float(np.std(f_quiete))
    if std_quiete <= 0:
        return None


    # --- Delta picco-picco della fase integrata (in cicli) ---
    # L'integrale del Doppler è una curva liscia: qui max-min è robusto.
    delta_fase_int = float(np.max(f_evento) - np.min(f_evento))
    
    # Assegniamo 'su' di default per non far fallire il controllo del classificatore
    forma_su = True

    # Fix: Uso il picco-picco su sigma (evita l'errore matematico della vecchia formula)
    se = std_quiete
    snr = delta_fase_int / se if se > 0 else None

    # Forma (segno della massima deviazione)
    media_q_int = np.mean(f_quiete)
    dev_int = f_evento - media_q_int
    forma_su = bool(dev_int[np.argmax(np.abs(dev_int))] > 0)

    return {
        'snr_fase_int':       snr,
        'delta_fase_cicli':   delta_fase_int,
        'delta_fase_gradi':   delta_fase_int * 360.0,
        'std_quiete_cicli':   std_quiete,
        'std_quiete_gradi':   std_quiete * 360.0,
        'forma_su_int':       forma_su,
        'n_evento':           len(f_evento),
        'n_quiete':           int(len(f_quiete)),
        'evento_contaminato': contaminato,
    }

def mostra_messaggio_overlay(fig, titolo, messaggio, sottotesto=None):
    """
    Piccolo messaggio in basso a destra dell'ultimo pannello (X-ray).
    Usa coordinate relative agli axes: è sempre visibile.
    """
    # Rimuovi overlay precedenti
    for ax in fig.axes:
        for txt in list(ax.texts):
            if getattr(txt, '_sid_overlay', False):
                txt.remove()
    for txt in list(fig.texts):
        if getattr(txt, '_sid_overlay', False):
            txt.remove()

    if not fig.axes:
        print("[MSG-OVERLAY] nessun axes disponibile")
        return
    ax = fig.axes[-1]   # ultimo pannello (X-ray)

    testo = f"{titolo} — {messaggio}"
    if sottotesto:
        testo += f"  |  {sottotesto}"

    txt = ax.text(0.99, 0.10, testo,
                  transform=ax.transAxes,
                  ha='right', va='bottom',
                  fontsize=8, family='monospace', color='#333333',
                  bbox=dict(boxstyle='round,pad=0.4',
                            facecolor='#FFF8F0',
                            edgecolor='#FF6347',
                            linewidth=1.2, alpha=0.95),
                  zorder=1000)
    txt._sid_overlay = True
    print(f"[MSG-OVERLAY] testo aggiunto a {ax}  y=0.10")
    fig.canvas.draw_idle()

def rimuovi_overlay(fig):
    """Rimuove l'eventuale overlay 'nessun evento' dalla figura."""
    for ax in fig.axes:
        for txt in list(ax.texts):
            if getattr(txt, '_sid_overlay', False):
                txt.remove()
    for txt in list(fig.texts):
        if getattr(txt, '_sid_overlay', False):
            txt.remove()

def _ottieni_root_tk():
    """
    Restituisce una root Tk valida:
      1) quella di default di Tk, se esiste;
      2) quella della figura matplotlib principale (TkAgg);
      3) una nuova root nascosta, come ultimo fallback.
    """
    import tkinter as tk

    # 1) root di default
    if getattr(tk, '_default_root', None) is not None:
        try:
            if tk._default_root.winfo_exists():
                return tk._default_root
        except Exception:
            pass

    # 2) root ricavata dalla figura matplotlib principale
    try:
        import matplotlib.pyplot as plt
        fignums = plt.get_fignums()
        if fignums:
            fig = plt.figure(fignums[0])
            win = fig.canvas.manager.window
            root = win.winfo_toplevel()
            if root is not None and root.winfo_exists():
                return root
    except Exception as e:
        print(f"[MSG] root da matplotlib non disponibile: {e}")

    # 3) fallback: nuova root nascosta
    try:
        root = tk.Tk()
        root.withdraw()
        return root
    except Exception as e:
        print(f"[MSG] impossibile creare root Tk: {e}")
        return None
    
def mostra_finestra_messaggio(titolo, messaggio, sottotesto=None):
    """
    Finestra di messaggio nativa Tkinter. Non blocca lo script,
    resta sempre sopra la finestra principale e si può riaprire
    dopo essere stata chiusa.
    """
    import tkinter as tk

    root = _ottieni_root_tk()
    if root is None:
        # Fallback estremo: nessuna root Tk disponibile
        print(f"[MSG] {titolo} — {messaggio}")
        return None

    win = _FINESTRA_MSG_TK['win']
    if win is not None:
        try:
            if not win.winfo_exists():
                win = None
        except Exception:
            win = None

    if win is None:
        win = tk.Toplevel(root)
        _FINESTRA_MSG_TK['win'] = win
    else:
        for w in win.winfo_children():
            w.destroy()

    win.title(titolo)
    win.configure(bg='#0d0d1a')
    win.geometry("540x220")
    win.resizable(False, False)

    tk.Label(win, text=titolo, bg='#0d0d1a', fg='#FF6347',
             font=('Consolas', 12, 'bold')).pack(pady=(16, 8))
    tk.Label(win, text=messaggio, bg='#0d0d1a', fg='white',
             font=('Consolas', 10), wraplength=490,
             justify='center').pack(pady=4, padx=20)
    if sottotesto:
        tk.Label(win, text=sottotesto, bg='#0d0d1a', fg='#888888',
                 font=('Consolas', 9, 'italic'), wraplength=490,
                 justify='center').pack(pady=(4, 0), padx=20)

    tk.Button(win, text="Chiudi", command=win.destroy,
              bg='#333355', fg='white', font=('Consolas', 10),
              relief='flat', padx=20, pady=5).pack(pady=14)

    # Porta in primo piano in modo affidabile
    try:
        win.deiconify()
        win.lift()
        win.attributes('-topmost', True)
        win.after(700, lambda: (
            win.attributes('-topmost', False)
            if win.winfo_exists() else None
        ))
        win.focus_force()
    except Exception as e:
        print(f"[MSG] Errore portare in primo piano: {e}")

    # Forza il disegno immediato della finestra, altrimenti su Windows
    # a volte resta "bianca" finché l'utente non la tocca.
    try:
        win.update_idletasks()
        win.update()
    except Exception:
        pass

    return win
  
class FiltroDoppler:
    def __init__(self, t_lento=2400, t_veloce=120, fs=1.0):
        self.sos = butter(2, [1/t_lento, 1/t_veloce],
                          btype='band', fs=fs, output='sos')
        self.fs = fs
        self.attivo = False
        self._grezzo = None
        self._filtrato = None

    def _preprocess(self, x):
        x = np.asarray(x, dtype=float).copy()
        mask = np.isnan(x)
        if mask.any() and (~mask).any():
            idx = np.arange(len(x))
            x[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
        x = medfilt(x - np.median(x), 31)
        return x

    def calcola(self, doppler):
        """Ricalcola il filtro sull'intero buffer disponibile."""
        x = self._preprocess(doppler)
        if len(x) < 4 * int(self.fs * 60):   # troppo pochi dati (<~4 min)
            self._grezzo = doppler
            self._filtrato = x
            return
        pad = min(len(x) // 4, int(self.fs * 3600))
        xp = np.pad(x, pad, mode='edge')
        y = sosfiltfilt(self.sos, xp)[pad:-pad]
        self._grezzo = doppler
        self._filtrato = y

    def toggle(self):
        self.attivo = not self.attivo
        return self._filtrato if self.attivo else self._grezzo

# ============================================================
# SELEZIONE DATA — finestra calendario
# ============================================================
def scegli_data_da_calendario(data_default=None):
    """
    Apre una finestra tkinter con calendario per scegliere il giorno da
    analizzare. Ritorna un datetime UTC (a mezzanotte) o None se annullato.

    Se data_default è una stringa 'YYYY-MM-DD', il calendario si apre su
    quel giorno. Se è None, si apre su oggi UTC.
    """
    import tkinter as tk
    try:
        from tkcalendar import Calendar
    except ImportError:
        print("[GUI] tkcalendar non installato. Lancia: pip install tkcalendar")
        return None

    oggi = datetime.now(timezone.utc)
    if data_default:
        try:
            data_iniziale = datetime.strptime(data_default, "%Y-%m-%d")
        except Exception:
            data_iniziale = oggi
    else:
        data_iniziale = oggi

    root = tk.Tk()
    root.title("Analisi Doppler — Scegli data")
    root.attributes('-topmost', True)
    root.resizable(False, False)
    root.configure(bg='#0d0d1a')
    root.geometry("460x440")

    tk.Label(root, text="Scegli il giorno da analizzare",
             bg='#0d0d1a', fg='#00FF99',
             font=('Consolas', 12, 'bold')).pack(pady=(14, 4))
    tk.Label(root, text="(il file CSV deve esistere in Sdruno)",
             bg='#0d0d1a', fg='#888888',
             font=('Consolas', 9)).pack(pady=(0, 8))

    cal = Calendar(
        root, selectmode='day',
        year=data_iniziale.year,
        month=data_iniziale.month,
        day=data_iniziale.day,
        date_pattern='yyyy-mm-dd',
        locale='it_IT',
        background='#1a1a2e', foreground='white',
        headersbackground='#2a2a4e', headersforeground='#00FF99',
        selectbackground='#00FF99', selectforeground='black',
        normalbackground='#0d0d1a', normalforeground='white',
        weekendbackground='#1a1a2e', weekendforeground='#aaaacc',
        othermonthforeground='#444466', othermonthbackground='#0d0d1a',
        font=('Consolas', 9),
    )
    try:
        cal.calevent_create(oggi.date(), 'Oggi', 'oggi')
        cal.tag_config('oggi', background='#005500', foreground='white')
    except Exception:
        pass
    cal.pack(padx=10, pady=8, fill='both', expand=True)

    lbl_stato = tk.Label(root, text="", bg='#0d0d1a', fg='#FF6347',
                         font=('Consolas', 9))
    lbl_stato.pack(pady=(0, 4))

    risultato = {'data': None}

    def conferma():
        data_str = cal.get_date()
        try:
            dt = datetime.strptime(data_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            lbl_stato.config(text="Data non valida")
            return
        
        # Verifica che il CSV esista in almeno una delle directory
        path_csv = trova_csv(data_str)
        if path_csv is None:
            lbl_stato.config(
                text=f"CSV non trovato: GAESID_NSY_{data_str}.csv")
            return
        
        risultato['data'] = dt
        root.destroy()

    def annulla():
        risultato['data'] = None
        root.destroy()

    frame_btn = tk.Frame(root, bg='#0d0d1a')
    frame_btn.pack(pady=(0, 12))
    tk.Button(frame_btn, text="Analizza", command=conferma,
              bg='#00FF99', fg='black',
              font=('Consolas', 10, 'bold'), relief='flat',
              padx=16, pady=6).pack(side='left', padx=8)
    tk.Button(frame_btn, text="Annulla", command=annulla,
              bg='#333355', fg='white',
              font=('Consolas', 10), relief='flat',
              padx=16, pady=6).pack(side='left', padx=8)

    root.protocol("WM_DELETE_WINDOW", annulla)
    root.bind('<Return>', lambda e: conferma())
    root.wait_window(root)

    return risultato['data']

def e_oggi(data_str):
    """True se data_str corrisponde alla data odierna UTC."""
    oggi_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return data_str == oggi_utc

def e_recente(data_str, giorni=GIORNI_REFRESH_CACHE):
    """
    True se data_str è oggi o entro `giorni` giorni fa.
    In questo caso la cache viene ignorata e i dati vengono riscaricati.
    """
    oggi_utc = datetime.now(timezone.utc).date()
    try:
        d_req = datetime.strptime(data_str, "%Y-%m-%d").date()
    except Exception:
        return False
    delta = (oggi_utc - d_req).days
    return 0 <= delta <= giorni
    
def elevazione_sole_punto_medio(dt_utc):
    """
    Restituisce l'elevazione solare (gradi) al punto medio del percorso
    al tempo dt_utc (datetime con tz=UTC).
    """
    loc = LocationInfo(latitude=MID_LAT, longitude=MID_LON)
    return solar_elevation(loc.observer, dt_utc)

def calcola_elevazione_sole_serie(timestamps, step_sec=60):
    """
    Calcola l'elevazione solare al punto medio del percorso per una
    serie di timestamp. Calcola solo ogni step_sec (default 60 s) per
    efficienza, poi interpola linearmente.

    Parametri:
      timestamps : array/lista di timestamp (pd.Timestamp o datetime)
      step_sec   : passo di campionamento per il calcolo (default 60 s)

    Ritorna un array numpy della stessa lunghezza di timestamps.
    """
    n = len(timestamps)
    if n == 0:
        return np.array([])

    elev = np.full(n, np.nan)
    idx_campione = list(range(0, n, step_sec))
    if idx_campione[-1] != n - 1:
        idx_campione.append(n - 1)

    for i in idx_campione:
        t = pd.Timestamp(timestamps[i]).to_pydatetime()
        try:
            elev[i] = elevazione_sole_punto_medio(t)
        except Exception:
            elev[i] = 0.0

    elev = pd.Series(elev).interpolate(
        method='linear', limit_direction='both').bfill().ffill().values
    return elev

def soglia_elevazione_per_classe(classe):
    """
    Soglia di elevazione solare dipendente dalla classe del flare.
    Flare più forti hanno SFE più grandi e tollerano meglio la
    transizione ionosferica.

    Classi e soglie:
      X1+     :  5°  (SFE dominante su tutto)
      M5-M9   : 12°  (SFE forte, tollera transizione)
      M1-M4   : 20°  (soglia standard)
      C6-C9   : 20°
      C1-C5   : 25°  (SFE debole, serve D-layer stabile)
      Sconosciuta: 20° (default)
    """
    if not classe:
        return 20.0
    c = classe[0].upper()
    if c == 'X':
        return 5.0
    if c == 'M':
        try:
            num = float(classe[1:])
            return 12.0 if num >= 5.0 else 20.0
        except (ValueError, IndexError):
            return 20.0
    if c == 'C':
        try:
            num = float(classe[1:])
            return 20.0 if num >= 6.0 else 25.0
        except (ValueError, IndexError):
            return 25.0
    return 20.0

def flare_ha_sid_atteso(flare):
    """
    Un flare produce un SID solo se la D-layer è formata lungo il percorso.
    Controlliamo l'elevazione solare al picco, all'inizio e alla fine del flare
    e prendiamo il massimo (un flare può iniziare prima dell'alba e finire dopo).
    """
    tempi = []
    for k in ('inizio', 'picco', 'fine'):
        t = parse_flare_datetime(flare.get(k, ''))
        if t is not None:
            tempi.append(t)
    if not tempi:
        return False, 0.0
    elev_max = max(elevazione_sole_punto_medio(t) for t in tempi)
    
    # Soglia dipendente dalla classe del flare (2026-10-08)
    soglia = soglia_elevazione_per_classe(flare.get('classe', ''))
    
    # ARROTONDAMENTO: evitiamo che 24.98 venga scartato se la soglia è 25.0
    elev_round = round(elev_max, 1)
    soglia_round = round(soglia, 1)
    
    return elev_round >= soglia_round, elev_max

# ============================================================
# X-RAY GOES
# ============================================================
def scarica_xray(data_str):
    """
    Scarica X-ray GOES (0.1-0.8 nm) da SWPC.
    Regole:
      - Oggi            → scarica SEMPRE (no cache)
      - Giorni passati  → usa cache se esiste ed è valida
      - File corrotto   → cancella e riscarica
      - Data > 7 gg fa  → SWPC non ha più i dati, usa cache o vuoto
    """
    global PATH_XRAY
    PATH_XRAY = os.path.join(CARTELLA_XRAY, f"GOES_XRAY_{data_str}.csv")

    # 1. Cache utilizzabile (solo per date passate)
    if cache_valida(PATH_XRAY, data_str):
        print(f"[XRAY] Uso cache: {PATH_XRAY}")
        ts, flux = leggi_xray_locale(PATH_XRAY)
        if ts:
            return ts, flux
        # Se la cache è vuota, cancellala e prosegui
        print(f"[XRAY] Cache vuota → elimino e riscarico")
        try:
            os.remove(PATH_XRAY)
        except Exception:
            pass

    # 2. Se data > 7 giorni fa, SWPC non ha più i dati
    oggi_utc = datetime.now(timezone.utc).date()
    try:
        d_req = datetime.strptime(data_str, "%Y-%m-%d").date()
        giorni_fa = (oggi_utc - d_req).days
    except Exception:
        giorni_fa = 0

    if giorni_fa > 7:
        print(f"[XRAY] {data_str} è {giorni_fa} giorni fa; "
              f"SWPC conserva solo 7 giorni → nessun download possibile")
        return [], []

    # 3. Download da SWPC
    print(f"[XRAY] Download da SWPC...")
    for sat in ("primary", "secondary"):
        url = f"https://services.swpc.noaa.gov/json/goes/{sat}/xrays-7-day.json"
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            dati = r.json()
            ts, flux = [], []
            for c in dati:
                if '0.1-0.8' not in c.get('energy', ''):
                    continue
                time_tag = c.get('time_tag', '').replace('T', ' ').replace('Z', '').strip()
                try:
                    dt = datetime.strptime(time_tag, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                if dt.strftime("%Y-%m-%d") != data_str:
                    continue
                try:
                    val = float(c.get('flux', 0))
                except Exception:
                    continue
                if val > 0:
                    ts.append(dt)
                    flux.append(val)
            if ts:
                salva_xray_locale(ts, flux)
                print(f"[XRAY] Scaricati {len(ts)} campioni da {sat}")
                return ts, flux
            else:
                print(f"[XRAY] {sat}: nessun campione per {data_str}")
        except Exception as e:
            print(f"[XRAY] Errore {sat}: {e}")

    # 4. Se la cache esisteva ma era stantia, prova a rileggerla
    if os.path.exists(PATH_XRAY):
        print(f"[XRAY] Download fallito → provo cache esistente")
        ts, flux = leggi_xray_locale(PATH_XRAY)
        if ts:
            return ts, flux

    print(f"[XRAY] Nessun dato disponibile per {data_str}")
    return [], []

def leggi_xray_locale(path):
    ts, flux = [], []
    try:
        with open(path, 'r') as f:
            for riga in f:
                riga = riga.strip()
                if riga.startswith('#') or not riga:
                    continue
                parti = riga.split(',')
                if len(parti) != 2:
                    continue
                try:
                    dt = datetime.strptime(parti[0].strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    val = float(parti[1].strip())
                    if val > 0:
                        ts.append(dt)
                        flux.append(val)
                except Exception:
                    continue
    except Exception as e:
        print(f"[XRAY] Errore lettura: {e}")
    return ts, flux

def salva_xray_locale(ts, flux):
    try:
        with open(PATH_XRAY, 'w') as f:
            f.write("# GOES X-ray (0.1-0.8 nm)\n")
            f.write("# timestamp_utc,flux_wm2\n")
            for dt, v in zip(ts, flux):
                f.write(f"{dt.strftime('%Y-%m-%d %H:%M:%S')},{v:.6e}\n")
    except Exception as e:
        print(f"[XRAY] Errore salvataggio: {e}")

# ============================================================
# FLARES DONKI (CON CACHE E FALLBACK MANUALE)
# ============================================================
def scarica_flares(data_str, ts_xray=None, flux_xray=None):
    """
    Scarica i flare da fonti ufficiali con fallback multipli:
      - Date recenti (<=30 gg): SWPC JSON (no rate limit)
      - Date vecchie: DONKI CCMC (endpoint storico)
      - Fallback: file TXT SWPC archive
    Poi unisce con l'auto-detect X-ray (se attivo).
    """
    global PATH_FLARES
    PATH_FLARES = os.path.join(CARTELLA_FLARES, f"FLARES_{data_str}.json")

    # --- Cache valida per date vecchie ---
    if cache_valida(PATH_FLARES, data_str):
        print(f"[FLARE] Uso cache: {PATH_FLARES}")
        try:
            with open(PATH_FLARES, 'r') as f:
                flares = json.load(f)
            if flares:
                return flares
            os.remove(PATH_FLARES)
        except Exception:
            try:
                os.remove(PATH_FLARES)
            except Exception:
                pass

    # --- Costruisci le URL in base all'età della data ---
    data_utc = datetime.strptime(data_str, "%Y-%m-%d").replace(
        tzinfo=timezone.utc)
    oggi_utc = datetime.now(timezone.utc).date()
    delta_giorni = (oggi_utc - data_utc.date()).days
    anno = data_utc.strftime("%Y")
    data_compatta = data_utc.strftime("%Y%m%d")

    url_txt_swpc = f"https://www.swpc.noaa.gov/archive/{anno}/{data_compatta}events.txt"

    if delta_giorni <= 30:
        urls = [
            ("json_noaa", "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json"),
            ("json_noaa", "https://services.swpc.noaa.gov/json/goes/secondary/xray-flares-7-day.json"),
            ("txt_swpc",  url_txt_swpc),
        ]
    else:
        urls = [
            ("json_donki", f"https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/FLR"
                           f"?startDate={data_str}&endDate={data_str}"),
            ("txt_swpc",   url_txt_swpc),
        ]

    # --- Scarica (prima fonte che risponde) ---
    flares_ufficiali = []
    for tipo, url in urls:
        try:
            print(f"[FLARE] Provo {tipo}: {url[:80]}...")
            r = requests.get(url, timeout=(5, 10))
            r.raise_for_status()

            if tipo == "json_noaa":
                flares_ufficiali = _parse_json_swpc(r.text, data_str)
                if flares_ufficiali:
                    print(f"[FLARE] SWPC JSON: {len(flares_ufficiali)} flare")
                    break

            elif tipo == "json_donki":
                flares_ufficiali = _parse_json_donki(r.text, data_str)
                if flares_ufficiali:
                    print(f"[FLARE] DONKI: {len(flares_ufficiali)} flare")
                    break

            elif tipo == "txt_swpc":
                flares_ufficiali = _parse_txt_swpc(r.text, data_str)
                if flares_ufficiali:
                    print(f"[FLARE] SWPC TXT: {len(flares_ufficiali)} flare")
                    break
                else:
                    print(f"[FLARE] SWPC TXT: 0 flare utili per {data_str}")

        except Exception as e:
            print(f"[FLARE] {tipo} errore: {e}")

    if not flares_ufficiali:
        print(f"[FLARE] Nessun flare ufficiale per {data_str}")

    # --- Auto-detect X-ray DISATTIVATO ---
    # Usiamo solo i flare ufficiali (SWPC/DONKI), come Visualizza_37.
    # Per riattivare, decommentare le 3 righe sotto.
    flares_xray = []
    # if ts_xray and flux_xray:
    #     flares_xray = rileva_flare_da_xray(ts_xray, flux_xray)
    #     print(f"[FLARE] X-ray auto-detect: {len(flares_xray)} flare")


    # --- Merge: priorità al timing X-ray, classe da ufficiali ---
    flares = []
    usati_ufficiali = set()
    for fx in flares_xray:
        t_picco_x = parse_flare_datetime(fx['picco'])
        match = None
        for k, fu in enumerate(flares_ufficiali):
            if k in usati_ufficiali:
                continue
            t_picco_u = parse_flare_datetime(fu['picco'])
            if t_picco_u is None:
                continue
            if abs((t_picco_u - t_picco_x).total_seconds()) < 600:
                match = (k, fu)
                break
        if match:
            k, fu = match
            usati_ufficiali.add(k)
            flares.append({
                'classe': fu['classe'],
                'inizio': fx['inizio'],
                'picco': fx['picco'],
                'fine': fx['fine'],
                '_fonte': 'xray+ufficiale',
            })
        else:
            flares.append(fx)

    # Aggiungi gli ufficiali non matchati
    for k, fu in enumerate(flares_ufficiali):
        if k not in usati_ufficiali:
            flares.append(fu)

    flares.sort(key=lambda f: f['picco'])

    # --- Salva cache per date passate ---
    if not e_oggi(data_str) and flares:
        try:
            with open(PATH_FLARES, 'w') as f:
                json.dump(flares, f, indent=2)
            print(f"[FLARE] Cache aggiornata: {len(flares)} flare totali")
        except Exception as e:
            print(f"[FLARE] Cache non salvata: {e}")

    return flares


def _parse_json_swpc(text, data_str):
    """Parsa il JSON di SWPC xray-flares-7-day.json."""
    flares = []
    try:
        dati = json.loads(text)
    except Exception:
        return flares
    for f in dati:
        try:
            # SWPC usa 'begin_time', 'peak_time', 'end_time',
            # 'max_class', 'max_flux'
            picco = f.get('peak_time', '') or f.get('max_time', '')
            if not picco:
                continue
            # Filtra per la data richiesta
            t_picco = datetime.strptime(
                picco[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            if t_picco.strftime("%Y-%m-%d") != data_str:
                continue
            classe = f.get('max_class', '') or f.get('class', '')
            if not classe or classe[0] not in 'CMX':
                continue
            flares.append({
                'classe': classe,
                'inizio': f.get('begin_time', ''),
                'picco': picco,
                'fine': f.get('end_time', ''),
                '_fonte': 'swpc',
            })
        except Exception:
            continue
    return flares


def _parse_json_donki(text, data_str):
    """Parsa il JSON di DONKI CCMC FLR."""
    flares = []
    try:
        dati = json.loads(text)
    except Exception:
        return flares
    for f in dati:
        try:
            classe = f.get('classType', '')
            if not classe or classe[0] not in 'CMX':
                continue
            picco = f.get('peakTime', '')
            if not picco:
                continue
            flares.append({
                'classe': classe,
                'inizio': f.get('beginTime', ''),
                'picco': picco,
                'fine': f.get('endTime', ''),
                '_fonte': 'donki',
            })
        except Exception:
            continue
    return flares


def _parse_txt_swpc(text, data_str):
    """
    Parsa il file events.txt di SWPC (formato storico).
    Cerca le righe con 'XRA' che contengono i flare X-ray.
    Formato tipico:
      :Product: 20261005events.txt
      ...
      2026 10 05 1147 C1.5 ...
    """
    flares = []
    for riga in text.splitlines():
        riga = riga.strip()
        if not riga or riga.startswith(':'):
            continue
        # Cerca righe con pattern: YYYY MM DD HHMM classe ...
        # Formato tipico SWPC events:
        # "2026 10 05 1147 C1.5  ..."
        parti = riga.split()
        if len(parti) < 5:
            continue
        try:
            # Verifica che i primi 3 siano anno mese giorno
            anno = int(parti[0])
            mese = int(parti[1])
            giorno = int(parti[2])
            ora_str = parti[3]  # HHMM
            classe = parti[4]
            if classe[0] not in 'CMX':
                continue
            if f"{anno:04d}-{mese:02d}-{giorno:02d}" != data_str:
                continue
            if len(ora_str) != 4:
                continue
            hh = int(ora_str[:2])
            mm = int(ora_str[2:])
            picco = (f"{anno:04d}-{mese:02d}-{giorno:02d}"
                     f"T{hh:02d}:{mm:02d}:00Z")
            flares.append({
                'classe': classe,
                'inizio': picco,
                'picco': picco,
                'fine': picco,
                '_fonte': 'swpc_txt',
            })
        except Exception:
            continue
    return flares

def parse_flare_datetime(s):
    """Converte stringa DONKI in datetime UTC."""
    if not s:
        return None
    s = s.strip().rstrip('Z')
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None

# ============================================================
# AUTO-DETECTION FLARE DA X-RAY
# ============================================================
def rileva_flare_da_xray(ts_xray, flux_xray):
    """
    Rileva flare dalla curva X-ray GOES cercando i picchi sul GRADIENTE
    del log-flusso, non sul flusso stesso.

    Vantaggio: due flare ravvicinati che si sovrappongono nel flusso
    (valle poco profonda) producono DUE picchi distinti nel gradiente,
    perché tra loro il flusso è quasi piatto.
    """
    from scipy.signal import find_peaks

    if not ts_xray or not flux_xray:
        return []

    SOGLIA_C1 = 1e-6
    # Prominenza minima del GRADIENTE (in decadi/min).
    # 0.005 = 1.2% al minuto. Bassa per catturare anche C1 deboli.
    PROMINENZA_GRAD = 0.005
    # Distanza minima tra picchi di gradiente (minuti)
    DISTANZA_MIN = 4

    t_arr = np.array([t.timestamp() for t in ts_xray])
    f_arr = np.array(flux_xray, dtype=float)
    log_f = np.log10(np.clip(f_arr, 1e-12, None))

    # Smoothing leggero del flusso (60 s = 1 campione a 1/min)
    # per stabilizzare il gradiente senza distruggere i picchi.
    lf_smooth = pd.Series(log_f).rolling(3, center=True,
                                          min_periods=1).mean().values

    # Gradiente in decadi/minuto
    if len(t_arr) > 1:
        dt_sec = float(np.median(np.diff(t_arr)))
        if dt_sec <= 0:
            dt_sec = 60.0
    else:
        dt_sec = 60.0
    dt_min = dt_sec / 60.0
    gradiente = np.gradient(lf_smooth) / dt_min

    print(f"[DEBUG-XRAY] dt={dt_sec:.1f}s, "
          f"gradiente max={gradiente.max():.4f} decadi/min")

    # Trova i picchi del gradiente
    picchi_grad, props = find_peaks(
        gradiente,
        prominence=PROMINENZA_GRAD,
        distance=DISTANZA_MIN,
    )

    print(f"[DEBUG-XRAY] Picchi di gradiente: {len(picchi_grad)}")
    for idx in picchi_grad:
        print(f"  idx={idx} t={ts_xray[idx].strftime('%H:%M')} "
              f"grad={gradiente[idx]:.4f} "
              f"prom={props['prominences'][list(picchi_grad).index(idx)]:.4f}")

    flare = []
    for k, idx_grad in enumerate(picchi_grad):
        # Il picco del FLUSSO è il massimo locale del flusso
        # in una finestra di ±5 minuti intorno al picco del gradiente
        finestra = 5
        i_a = max(0, idx_grad - finestra)
        i_b = min(len(lf_smooth), idx_grad + finestra + 1)
        idx_picco = i_a + int(np.argmax(lf_smooth[i_a:i_b]))
        picco_val = f_arr[idx_picco]

        if picco_val < SOGLIA_C1:
            continue

        if picco_val >= 1e-4:
            classe = f"X{picco_val/1e-4:.1f}"
        elif picco_val >= 1e-5:
            classe = f"M{picco_val/1e-5:.1f}"
        else:
            classe = f"C{picco_val/1e-6:.1f}"

        flare.append({
            'classe': classe,
            'inizio': ts_xray[max(0, idx_grad - 2)].strftime('%Y-%m-%dT%H:%M:%SZ'),
            'picco': ts_xray[idx_picco].strftime('%Y-%m-%dT%H:%M:%SZ'),
            'fine': ts_xray[min(len(ts_xray) - 1, idx_grad + 10)].strftime('%Y-%m-%dT%H:%M:%SZ'),
            '_fonte': 'xray_auto',
        })

    print(f"[DEBUG-XRAY] Flare finali (C1+): {len(flare)}")
    for f in flare:
        print(f"  picco={f['picco']}  classe={f['classe']}")

    return flare

def trova_csv(data_str):
    """
    Cerca il CSV del tracker per la data specificata in tutte le
    directory di CARTELLE_CSV. Ritorna il percorso completo se
    trovato, altrimenti None.

    Parametri:
      data_str : stringa 'YYYY-MM-DD'

    Ritorna:
      path (str) oppure None
    """
    nome_file = f"GAESID_NSY_{data_str}.csv"
    for cartella in CARTELLE_CSV:
        path = os.path.join(cartella, nome_file)
        if os.path.exists(path):
            return path
    return None

# ============================================================
# LETTURA CSV TRACKER
# ============================================================
def leggi_csv_tracker(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV non trovato: {path}\n"
            f"Directory cercate:\n" +
            "\n".join(f"  - {c}" for c in CARTELLE_CSV))
    print(f"[CSV] Lettura {path}")
    df = pd.read_csv(path, comment='#')

    if 'Timestamp_UTC' not in df.columns:
        raise ValueError("[CSV] Colonna Timestamp_UTC mancante")
    df['Timestamp'] = pd.to_datetime(df['Timestamp_UTC'], utc=True)

    # Colonne opzionali (retrocompatibilità con tracker vecchi)
    if 'PLL_Freq_Hz' not in df.columns:
        print("[CSV] Colonna PLL_Freq_Hz assente → NaN")
        df['PLL_Freq_Hz'] = np.nan
    if 'Reset_Flag' not in df.columns:
        print("[CSV] Colonna Reset_Flag assente → 0")
        df['Reset_Flag'] = 0
    if 'Amp_100Hz_dB' not in df.columns:
        print("[CSV] Colonna Amp_100Hz_dB assente → NaN")
        df['Amp_100Hz_dB'] = np.nan

    df = df.sort_values('Timestamp').reset_index(drop=True)

    # Statistica di riferimento per la soglia di lock (su tutto il giorno)
    amp100_validi = df['Amp_100Hz_dB'].dropna()
    if len(amp100_validi) > 0:
        df.attrs['amp100_mediana_giorno'] = float(amp100_validi.median())
        df.attrs['soglia_lock_amp100']    = float(amp100_validi.median() - AMP100_DROP_DB)
        # Soglia assoluta: sotto questo livello non c'è portante NSY, solo rumore
        AMP100_SOGLIA_ASSOLUTA_DB = 50.0
        df.attrs['soglia_lock_amp100'] = max(
            df.attrs['soglia_lock_amp100'],
            AMP100_SOGLIA_ASSOLUTA_DB
        )
        print(f"[CSV] Amp_100Hz_dB: mediana={df.attrs['amp100_mediana_giorno']:+.2f} dB, "
              f"soglia lock={df.attrs['soglia_lock_amp100']:+.2f} dB")
    else:
        df.attrs['amp100_mediana_giorno'] = None
        df.attrs['soglia_lock_amp100']    = None

       # ============================================================
    # Rilevamento discontinuità del tracker (riavvii, gap, glitch)
    # ============================================================
    # Un "riavvio" del tracker produce tre firme possibili:
    #   1) Reset_Flag=1 esplicito (dal tracker stesso)
    #   2) Gap temporale > 5 s (tracker spento e riacceso)
    #   3) Salto di fase > 1000° tra due campioni consecutivi (contatore azzerato)
    # Le tre vengono unite in un'unica colonna 'Discontinuita'.

    mask_reset = (df['Reset_Flag'].values == 1)

    dt = df['Timestamp'].diff().dt.total_seconds().to_numpy(copy=True)
    dt[0] = 0.0
    mask_gap = (dt > 5.0)

    fase_vals = df['Phase_Deg'].values.astype(float)
    d_fase = np.abs(np.diff(fase_vals, prepend=fase_vals[0]))
    # Soglia adattiva: 10× la mediana delle differenze, con un minimo di 1000°.
    # Si adatta automaticamente a offset RF diversi (es. -5 Hz vs -1.5 Hz).
    d_fase_mediana = np.median(d_fase[np.isfinite(d_fase)])
    soglia_salto = max(10.0 * d_fase_mediana, 1000.0)
    mask_salto = (d_fase > soglia_salto)

    mask_disc = mask_reset | mask_gap | mask_salto

    # Estendi ciascuna discontinuità di 600 campioni (10 min) in avanti.
    # Dopo un riavvio il PLL impiega 5-10 minuti a riagganciarsi:
    # in quella finestra la fase è dominata dalla transizione di aggancio
    # e va esclusa dall'analisi.
    WARMUP_S = 600
    idx_disc = np.where(mask_disc)[0]
    for idx in idx_disc:
        end = min(len(mask_disc), idx + WARMUP_S + 1)
        mask_disc[idx:end] = True

    # Warmup iniziale: i primi 600 campioni (10 min) sono sempre in
    # acquisizione PLL. Vanno esclusi dall'analisi come se fossero
    # discontinuità.
    #
    # IMPORTANTE: se il file è molto corto (es. tracker appena riavviato),
    # un warmup fisso di 600 righe coprirebbe l'intero file, azzerando
    # tutti i dati. In quel caso riduciamo il warmup proporzionalmente
    # (30% del file) in modo da lasciare ~70% di dati analizzabili.
    WARMUP_INIZIALE_S = 600
    if len(mask_disc) < WARMUP_INIZIALE_S * 1.5:
        WARMUP_INIZIALE_S = max(int(len(mask_disc) * 0.3), 30)
        print(f"[CSV] File corto ({len(mask_disc)} righe): "
              f"warmup ridotto a {WARMUP_INIZIALE_S} righe")
    fine_warmup = min(WARMUP_INIZIALE_S, len(mask_disc))
    mask_disc[0:fine_warmup] = True
    
    df['Discontinuita'] = mask_disc.astype(int)

    n_tot = len(df)
    n_disc = int(mask_disc.sum())
    df.attrs['n_discontinuita'] = n_disc

    if n_disc > 0:
        print(f"[CSV] Discontinuita' rilevate: {n_disc}/{n_tot} "
              f"(reset={int(mask_reset.sum())}, "
              f"gap={int(mask_gap.sum())}, "
              f"salto_fase={int(mask_salto.sum())})")
        # Mostra gli indici delle prime 5 discontinuità
        idx_disc = np.where(mask_disc)[0][:5]
        for idx in idx_disc:
            t_str = df['Timestamp'].iloc[idx].strftime('%H:%M:%S')
            motivo = []
            if mask_reset[idx]:   motivo.append('reset')
            if mask_gap[idx]:     motivo.append('gap')
            if mask_salto[idx]:   motivo.append('salto')
            print(f"[CSV]   idx={idx:5d}  t={t_str}  motivo={','.join(motivo)}")
    else:
        print("[CSV] Nessuna discontinuita' rilevata.")

    return df

# ============================================================
# DETREND FASE (ROBUSTO, SENZA SPIKE AI BORDI)
# ============================================================
def detrend_fase(fase, finestra_sec=1800, discontinuita=None):
    """
    Detrend della fase con fit LINEARE locale, ma RISPETTANDO i segmenti
    separati da discontinuità. Ogni segmento viene detrenduto in modo
    indipendente, così nessuna finestra mobile attraversa mai un salto
    del contatore di fase.

    Parametri:
        fase         : array della fase (gradi)
        finestra_sec : ampiezza della finestra di fit (campioni)
        discontinuita: array booleano/int con 1 sui campioni da separare.
                       Se None, la serie è trattata come un unico segmento.

    Restituisce un array della stessa lunghezza, con i valori detrenduti.
    I punti di discontinuità restituiscono 0.0.
    """
    fase = np.asarray(fase, dtype=float)
    n = len(fase)
    if n == 0:
        return fase

    # Nessuna discontinuità → un solo segmento
    if discontinuita is None or not np.any(discontinuita):
        return _detrend_segmento(fase, finestra_sec)

    # Segmentazione: ogni True in discontinuita è l'inizio di un nuovo segmento
    disc = np.asarray(discontinuita).astype(bool)
    # Forziamo l'inizio e la fine della serie come bordi di segmento
    disc[0] = True

    out = np.zeros(n)
    idx_start = None
    for i in range(n):
        if disc[i]:
            # Chiudi il segmento precedente
            if idx_start is not None:
                seg = fase[idx_start:i]
                out[idx_start:i] = _detrend_segmento(seg, finestra_sec)
            idx_start = i
    # Chiudi l'ultimo segmento
    if idx_start is not None and idx_start < n:
        seg = fase[idx_start:n]
        out[idx_start:n] = _detrend_segmento(seg, finestra_sec)

    return out

def _detrend_segmento(fase_seg, finestra_sec=1800):
    """
    Detrend lineare locale su un singolo segmento continuo.
    Ignora i NaN: se un punto è NaN, il suo output resta NaN.
    Se il segmento ha meno di 60 punti validi, restituisce tutti NaN.
    """
    n = len(fase_seg)
    if n == 0:
        return fase_seg

    mask_validi = np.isfinite(fase_seg)
    if mask_validi.sum() < 60:
        return np.full(n, np.nan)

    # Interpola solo per il calcolo del fit (i valori originali restano)
    fase_fit = pd.Series(fase_seg).interpolate(
        method='linear', limit_direction='both'
    ).bfill().ffill().values

    meta = finestra_sec // 2
    out = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(fase_seg[i]):
            continue
        start = max(0, i - meta)
        end   = min(n, i + meta)
        if end - start < 30:
            continue
        t_local = np.arange(end - start, dtype=float)
        try:
            coeffs = np.polyfit(t_local, fase_fit[start:end], 1)
            out[i] = fase_seg[i] - np.polyval(coeffs, i - start)
        except Exception:
            pass
    return out
# ============================================================
# DETREND GENERICO (per PLL_Freq)
# ============================================================
def detrend_mediano(serie, finestra):
    """
    Detrend standard (usato per X-ray e altri canali).
    Usa Savitzky-Golay per stimare il trend.
    """
    if len(serie) == 0:
        return serie
    s = pd.Series(serie)
    s_interp = s.interpolate(method='linear', limit_direction='both').bfill().ffill()
    y = s_interp.values
    
    window_length = int(finestra)
    if window_length % 2 == 0:
        window_length += 1
    if window_length > len(y):
        window_length = len(y) if len(y) % 2 != 0 else len(y) - 1
        
    if window_length < 5:
        return serie
        
    try:
        trend = savgol_filter(y, window_length=window_length, polyorder=2)
    except Exception:
        trend = pd.Series(y).rolling(window=window_length, center=True).median().interpolate().values

    y_detrend = y - trend
    y_detrend[np.isnan(serie)] = np.nan
    return y_detrend

def detrend_mascherato(serie, timestamps, flares, finestra=1800):
    """
    Detrend intelligente: maschera le finestre dei flare prima di calcolare il trend.
    Evita l'effetto 'anticipazione' (fossa pre-evento).
    """
    if len(serie) == 0:
        return serie
        
    s = pd.Series(serie).interpolate().bfill().ffill()
    s_masked = s.copy()
    
    if flares:
        for f in flares:
            t_ini = parse_flare_datetime(f.get('inizio', ''))
            t_fin = parse_flare_datetime(f.get('fine', ''))
            t_picco = parse_flare_datetime(f.get('picco', ''))
            
            if t_ini is None and t_picco is not None: 
                t_ini = t_picco - timedelta(minutes=5)
            if t_fin is None and t_picco is not None: 
                t_fin = t_picco + timedelta(minutes=15)
            
            if t_ini is not None and t_fin is not None:
                # Maschera allargata a 30 min per includere l'intera SPA
                t_a = t_ini - timedelta(minutes=30)
                t_b = t_fin + timedelta(minutes=30)
                idx_mask = (timestamps >= t_a) & (timestamps <= t_b)
                s_masked.loc[idx_mask.values] = np.nan
                
    s_interp = s_masked.interpolate(method='linear', limit_direction='both').bfill().ffill().values
    
    window_length = int(finestra)
    if window_length % 2 == 0:
        window_length += 1
    if window_length > len(s_interp): 
        window_length = len(s_interp) - 1 if len(s_interp) % 2 == 0 else len(s_interp)
        
    if window_length < 5:
        return serie
        
    try:
        trend = savgol_filter(s_interp, window_length=window_length, polyorder=2)
    except Exception:
        trend = pd.Series(s_interp).rolling(window=window_length, center=True).median().interpolate().values

    y_detrend = s.values - trend
    y_detrend[np.isnan(serie)] = np.nan
    return y_detrend

# ============================================================
# CROSS-CORRELAZIONE con X-ray
# ============================================================
def calcola_cross_correlazione(serie, xray, lag_max_sec):
    """
    Calcola la cross-correlazione Pearson pura per ogni lag.
    Corregge il bug del denominatore (len(s)) calcolando la correlazione
    solo sui campioni effettivamente sovrapposti.
    """
    serie = np.asarray(serie, dtype=float)
    xray  = np.asarray(xray,  dtype=float)

    valid = np.isfinite(serie) & np.isfinite(xray)
    if valid.sum() < 60:
        return None, None

    # Prendi il segmento continuo più lungo
    segmenti = []
    in_segmento = False
    inizio = 0
    for i, v in enumerate(valid):
        if v and not in_segmento:
            inizio = i
            in_segmento = True
        elif not v and in_segmento:
            segmenti.append((inizio, i))
            in_segmento = False
    if in_segmento:
        segmenti.append((inizio, len(valid)))

    if not segmenti:
        return None, None

    segmenti.sort(key=lambda s: s[1] - s[0], reverse=True)
    a, b = segmenti[0]
    if (b - a) < 60:
        return None, None

    s = serie[a:b]
    x = xray[a:b]
    
    if np.std(s) == 0 or np.std(x) == 0:
        return None, None

    # Standardizza le intere serie (per il calcolo veloce della covarianza)
    s_norm = (s - np.mean(s)) / (np.std(s) + 1e-12)
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-12)

    # Calcolo cross-correlazione
    cross = correlate(s_norm, x_norm, mode='full')
    centro = len(s) - 1
    lag_max_campioni = min(lag_max_sec, len(s) // 2)
    
    idx_range = np.arange(centro - lag_max_campioni, centro + lag_max_campioni + 1)
    idx_range = idx_range[(idx_range >= 0) & (idx_range < len(cross))]
    if len(idx_range) == 0:
        return None, None

    # Correzione di Pearson: dividiamo per il numero di campioni sovrapposti
    # (non per len(s) fisso).
    lags = idx_range - centro
    n_overlap = len(s) - np.abs(lags)
    cross_corretta = cross[idx_range] / n_overlap

    # Trova il massimo in valore assoluto (o orientato se usi la polarità)
    abs_cross = np.abs(cross_corretta)
    if not np.any(np.isfinite(abs_cross)):
        return None, None
        
    idx_best_rel = np.argmax(abs_cross)
    idx_best = idx_range[idx_best_rel]
    
    return idx_best - centro, cross_corretta[idx_best_rel]

def ttest_welch_neff(x, y, tau_sec, fs=1.0):
    """
    Welch t-test corretto per serie temporali autocorrelate.
    Usa N_eff = N / (tau * fs) per non gonfiare i gradi di libertà.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    
    nx, ny = len(x), len(y)
    if nx < 10 or ny < 10:
        return None, None, None
        
    n_eff_x = max(2.0, nx / (tau_sec * fs))
    n_eff_y = max(2.0, ny / (tau_sec * fs))

    m1, m2 = np.mean(x), np.mean(y)
    v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)

    se2 = v1 / n_eff_x + v2 / n_eff_y
    if se2 <= 0:
        return 0.0, 1.0, 1.0 # t, df, p
        
    se = np.sqrt(se2)
    t_stat = (m1 - m2) / se
    
    # Gradi di libertà Welch-Satterthwaite con N_eff
    num = se2 ** 2
    den = (v1 / n_eff_x)**2 / (n_eff_x - 1) + (v2 / n_eff_y)**2 / (n_eff_y - 1)
    df = num / den if den > 0 else 1.0
    
    from scipy.stats import t
    p_value = 2 * t.sf(abs(t_stat), df)
    
    return t_stat, df, p_value

def classifica_diagnostica(r):
    """
    Classifica l'evento in base ai risultati di analizza_flare.
    """

    soglia_picco_din = r.get('soglia_picco_dinamica', PWR_SID_PICCO_MIN_DB)
    
    fraz   = r.get('frazione_invalida', 0.0)
    lag_f  = r.get('lag_fase')
    corr_f = r.get('corr_fase')
    lag_p  = r.get('lag_pll')
    corr_p = r.get('corr_pll')
    d_fase = r.get('delta_fase')
    d_pll  = r.get('delta_pll')
    d_pwr  = r.get('delta_pwr')
    n_res  = r.get('n_reset', 0)
    soglia_dinamica = r.get('d_fase_soglia', 0.001)

    # ---------- 0) ARTEFATTO FASE (flag, non return) ----------
    bump_amp_check = r.get('bump_ampiezza_mhz')
    bump_std_check = r.get('bump_std_quieta')

    fase_artefatto = False
    motivo_fase_artefatto = ""
    if (bump_amp_check is not None and
            abs(bump_amp_check) > SOGLIA_ARTEFATTO_FASE_MHZ):
        fase_artefatto = True
        motivo_fase_artefatto = (f"delta medio={bump_amp_check:.2f} mHz "
                                  f"(> {SOGLIA_ARTEFATTO_FASE_MHZ} mHz)")
    elif (bump_std_check is not None and
            bump_std_check > SOGLIA_ARTEFATTO_FASE_MHZ):
        fase_artefatto = True
        motivo_fase_artefatto = (f"std quieta={bump_std_check:.2f} mHz "
                                  f"(> {SOGLIA_ARTEFATTO_FASE_MHZ} mHz)")

    # ---------- 1) ARTEFATTO ----------
    if fraz > FRAZIONE_INVALIDA_MAX:
        return ('ARTEFATTO',
                f"⚠ ARTEFATTO: {fraz*100:.0f}% della finestra invalida "
                f"(reset={n_res}, lock basso). Misure inaffidabili.",
                '#FF0000')

    grad_max = r.get('gradiente_max_fase')
    SOGLIA_GRADIENTE_SALTO_HZ = 0.050
    salto_isolato = (
        grad_max is not None and
        grad_max > SOGLIA_GRADIENTE_SALTO_HZ and
        (d_pll is None or abs(d_pll) < DELTA_PLL_MIN_HZ) and
        (corr_f is None or abs(corr_f) < 0.3)
    )
    if salto_isolato:
        return ('ARTEFATTO',
                f"⚠ ARTEFATTO: salto istantaneo di fase "
                f"(gradiente max={grad_max:.3f} Hz/s) senza Doppler "
                f"associato → probabile reset/slip del PLL.",
                '#FF0000')

    # ============================================================
    # CALCOLO DI TUTTE LE METRICHE (prima di qualsiasi return)
    # ============================================================
    # Canale FASE (bump)
    bump_snr  = r.get('bump_snr_locale')
    bump_su   = r.get('bump_forma_su')
    bump_amp  = r.get('bump_ampiezza_mhz')
    bump_std  = r.get('bump_std_quieta')

    bump_snr_abs = abs(bump_snr) if bump_snr is not None else 0.0
    fase_ok = (
        not fase_artefatto and
        bump_snr is not None and bump_snr_abs > SNR_FASE_MIN and
        fraz < FRAZIONE_INVALIDA_MAX
    )

    # Canale FASE INTEGRATA
    fase_int_snr = r.get('snr_fase_int')
    fase_int_su  = r.get('forma_su_int')
    fase_int_deg = r.get('delta_fase_int_deg')
    fase_int_snr_abs = abs(fase_int_snr) if fase_int_snr is not None else 0.0
    fase_int_ok = (
        not fase_artefatto and
        fase_int_snr is not None and fase_int_snr_abs > SNR_FASE_INT_MIN and
        fraz < FRAZIONE_INVALIDA_MAX
    )

    # Canale POTENZA
    d_pwr_media = r.get('delta_pwr_media')
    snr_pwr_media = r.get('snr_pwr_media')
    d_pwr_picco = r.get('delta_pwr_picco')
    snr_pwr_picco = r.get('snr_pwr_picco')

    dt_picco = r.get('dt_picco_pwr_sec')
    allineato = True
    if dt_picco is not None:
        allineato = -600 <= dt_picco <= 600

    potenza_ok = (
        d_pwr_picco is not None and abs(d_pwr_picco) > soglia_picco_din and
        snr_pwr_picco is not None and snr_pwr_picco > PWR_SID_MIN_SNR and
        d_pwr_media is not None and abs(d_pwr_media) > 0.10 and
        allineato and
        fraz < FRAZIONE_INVALIDA_MAX
    )

    # Canale PLL (Doppler grezzo)
    rapporto_pll_fase = None
    if d_pll is not None and d_fase is not None and d_fase > 1e-6:
        rapporto_pll_fase = d_pll / d_fase

    pll_troppo_rumoroso = (
        rapporto_pll_fase is not None and
        rapporto_pll_fase > DPLL_RAPPORTO_MAX
    )

    # Usiamo il valore assoluto (cieco alla polarità)
    corr_p_or = abs(corr_p) if corr_p is not None else None
    corr_f_or = abs(corr_f) if corr_f is not None else None

    sid_pll_ok = (
        d_fase is not None and d_fase > soglia_dinamica and
        d_pll is not None and d_pll >= DELTA_PLL_MIN_HZ and
        corr_p_or is not None and corr_p_or > CORR_MIN_SID and
        lag_p is not None and abs(lag_p) < LAG_COERENTE_SEC and
        not pll_troppo_rumoroso and
        fraz < FRAZIONE_INVALIDA_MAX
    )
    
    sid_fase_ok = (
        corr_f_or is not None and corr_f_or > CORR_MIN_SID and
        lag_f is not None and abs(lag_f) < LAG_COERENTE_SEC
    )

    # ============================================================
    # GERARCHIA DI CLASSIFICAZIONE (i più forti prima)
    # ============================================================

    # ---------- 2) SID_COMPLETO (potenza + fase) ----------
    # Richiede che il Doppler locale (fase_ok) sia significativo.
    # La Fase Integrata (fase_int_ok) non basta più da sola.
    # SID_COMPLETO: richiede potenza OK e almeno una tra Fase Doppler e Fase Integrata
    # SID_COMPLETO: richiede potenza OK E Doppler locale significativo
    if potenza_ok and not fase_artefatto and fase_ok:
        if fase_int_ok:
            dettaglio_fase = (f"SNR_int={fase_int_snr:.2f} "
                              f"(Δφ={fase_int_deg:+.2f}°)")
        else:
            dettaglio_fase = f"SNR_fase={bump_snr:.2f} (Doppler)"
        return ('SID_COMPLETO',
                f"✓✓ SID COMPLETO: potenza + fase. "
                f"Picco pwr={d_pwr_picco:+.2f} dB (SNR={snr_pwr_picco:+.1f}), "
                f"{dettaglio_fase}.",
                '#00CC00')

    # ---------- 2b) SID_REALE (da PLL o Doppler filtrato) ----------
    if sid_pll_ok and fraz < FRAZIONE_INVALIDA_MAX:
        dettaglio = f"ΔPLL={d_pll:.3f} Hz, corr={corr_p:+.2f}, lag={lag_p:+d}s"
        if sid_fase_ok and lag_f is not None and lag_p is not None:
            if abs(lag_f - lag_p) < 60:
                dettaglio += "  |  fase e PLL concordi"
            else:
                dettaglio += f"  |  solo PLL (fase lag={lag_f:+d}s)"
        return ('SID_REALE',
                f"✓ SID REALE: Doppler correlato con X-ray. {dettaglio}.",
                '#00AA00')

    cf_f = r.get('corr_fase_filtrata')
    lf_f = r.get('lag_fase_filtrata')
    cf_f_abs = abs(cf_f) if cf_f is not None else None
    if (cf_f_abs is not None and cf_f_abs > 0.55 and
            lf_f is not None and abs(lf_f) < 240 and
            bump_snr_abs > SNR_FASE_MIN and  # <-- VINCOLO DI AMPIEZZA AGGIUNTO
            fraz < FRAZIONE_INVALIDA_MAX):
        return ('SID_REALE',
                f"✓ SID REALE (correlazione filtrata): "
                f"|corr|={cf_f_abs:.2f}, lag={lf_f:+d}s. "
                f"Il Doppler filtrato 2–40 min segue il gradiente X-ray.",
                '#00AA00')

    # ---------- 2c) SID_POTENZA ----------
    if fraz < FRAZIONE_INVALIDA_MAX:
        dettaglio_pwr = (
            f"media={d_pwr_media:+.2f} dB (SNR={snr_pwr_media:+.1f}), "
            f"picco={d_pwr_picco:+.2f} dB (SNR={snr_pwr_picco:+.1f})"
            if d_pwr_media is not None and d_pwr_picco is not None
            else "N/D"
        )

        if potenza_ok:
            nota_fase = ""
            if fase_artefatto:
                nota_fase = (f" Canale fase escluso "
                              f"({motivo_fase_artefatto}).")
            elif fase_int_snr is not None and abs(fase_int_snr) > SNR_FASE_INT_MIN:
                nota_fase = (f" Fase integrata: Δφ/σ={fase_int_snr:.1f} "
                             f"(Δφ={fase_int_deg:+.1f}°).")
            return ('SID_POTENZA',
                    f"✓ SID DI POTENZA: risposta in ampiezza coerente con X-ray. "
                    f"{dettaglio_pwr}.{nota_fase}",
                    '#00AA00')

        # --- INIZIO BLOCCO CORRETTO (Bipolare) ---
        if (d_pwr_picco is not None and
                abs(d_pwr_picco) > soglia_picco_din * 0.5 and
                snr_pwr_picco is not None and
                snr_pwr_picco > PWR_SID_WEAK_SNR and
                d_pwr_media is not None and abs(d_pwr_media) > 0.10 and
                allineato):
            return ('SID_POTENZA_DEBOLE',
                    f"◐ CANDIDATO SID in potenza: picco significativo ma "
                    f"SNR marginale. {dettaglio_pwr}. Da verificare con altri "
                    f"canali o giorni.",
                    '#FFD700')
        # --- FINE BLOCCO CORRETTO ---

    # ---------- 2d) SID da correlazione di potenza (ramo 2d spostato) ----------
    cp_f = r.get('corr_pwr_f')
    lp_f = r.get('lag_pwr_f')
    if (cp_f is not None and abs(cp_f) > 0.60 and
            lp_f is not None and abs(lp_f) < 180 and
            fraz < 0.15):
        return ('SID_POTENZA',
                f"✓ SID in POTENZA: correlazione pulita con X-ray "
                f"(|corr|={abs(cp_f):+.2f}, lag={lp_f:+d}s). Il Doppler è sotto "
                f"il rumore ma la potenza conferma l'evento.",
                '#00AA00')

    # ---------- 2e) SID_FASE ----------
    # Richiediamo che ci sia almeno una debole traccia di coerenza con lo X-ray
    # o nella potenza o nella correlazione del Doppler.
    cp_f_val = r.get('corr_pwr_f')
    cf_f_val = r.get('corr_fase_filtrata')
    coerenza_xray = (
        (cp_f_val is not None and abs(cp_f_val) > 0.3) or
        (cf_f_val is not None and abs(cf_f_val) > 0.3) or
        (d_pwr_media is not None and abs(d_pwr_media) > 0.1)
    )

    if not fase_artefatto and coerenza_xray:
        if fase_ok:
            return ('SID_FASE',
                    f"✓ SID_FASE: bump Doppler significativo. "
                    f"SNR_fase={bump_snr:.2f} (forma {'su' if bump_su else 'giu'}, "
                    f"amp={bump_amp:+.4f} mHz). Potenza sotto soglia.",
                    '#66CC66')

    # ---------- 2f) CANDIDATO DEBOLE ----------
    if (d_pwr_picco is not None and abs(d_pwr_picco) > soglia_picco_din * 0.1 and
            snr_pwr_picco is not None and snr_pwr_picco > 0.3 and
            d_pwr_media is not None and abs(d_pwr_media) > 0.05 and
            allineato and
            fraz < FRAZIONE_INVALIDA_MAX):
        return ('CANDIDATO_DEBOLE',
                f"◐ CANDIDATO DEBOLE: picco in potenza "
                f"({d_pwr_picco:+.3f} dB, SNR={snr_pwr_picco:+.2f}) coerente "
                f"con il flare ma sotto le soglie standard. "
                f"Da verificare su altri giorni o con riferimento esterno.",
                '#FFD700')

    # ---------- 3) NESSUN EVENTO ----------
    pwr_piccolo = (
        (d_pwr_media is None or abs(d_pwr_media) < PWR_SID_MIN_DB)
    )
    fase_piccolo = (d_fase is None or abs(d_fase) < soglia_dinamica)
    pll_piccolo  = (d_pll is None or abs(d_pll) < 3.0 * DELTA_PLL_MIN_HZ)

    if pwr_piccolo and fase_piccolo and pll_piccolo:
        return ('NESSUN_EVENTO',
                "○ NESSUN EVENTO: nessuna variazione misurabile in potenza, "
                "fase o Doppler.",
                '#888888')

    # ---------- 4) DERIVA / TID / JITTER ----------
    if d_fase is not None and abs(d_fase) > soglia_dinamica and not sid_pll_ok:
        if d_pll is None or d_pll < DELTA_PLL_MIN_HZ:
            return ('DERIVA_TID',
                    f"◐ DERIVA/TID/JITTER: Doppler fase {d_fase:.4f} Hz ma "
                    f"ΔPLL trascurabile ({d_pll if d_pll is None else round(d_pll,4)} Hz). "
                    "Non è un SID.",
                    '#FF8C00')
        else:
            return ('INCERTO',
                    f"◑ INCERTO: Doppler presente (ΔPLL={d_pll:.3f} Hz) ma "
                    "non correlato con X-ray. Possibile TID intensa o interferenza.",
                    '#FFD700')

    return ('INCERTO',
            "◑ INCERTO: dati insufficienti per una classificazione netta.",
            '#FFD700')

_DOPPLER_POOL_CACHE = {'df': None, 'loaded': False}

def _sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load_doppler_pool(path=None):
    global _DOPPLER_POOL_CACHE
    if _DOPPLER_POOL_CACHE['loaded']:
        return _DOPPLER_POOL_CACHE['df']
    if not MATCHED_DOPPLER_ON:
        _DOPPLER_POOL_CACHE['loaded'] = True
        return None

    path = path or DOPPLER_POOL_PATH
    if not os.path.exists(path):
        print(f"[MATCHED] Pool assente: {path} — regola secondaria disabilitata")
        _DOPPLER_POOL_CACHE['loaded'] = True
        return None

    # Verifica hash solo se specificato
    if DOPPLER_POOL_SHA256 is not None:
        try:
            actual = _sha256_file(path)
            if actual != DOPPLER_POOL_SHA256:
                print(f"[MATCHED] HASH NON VALIDO (atteso {DOPPLER_POOL_SHA256[:8]}..., trovato {actual[:8]}...)")
                _DOPPLER_POOL_CACHE['loaded'] = True
                return None
        except Exception as e:
            print(f"[MATCHED] Errore lettura hash: {e}")
            _DOPPLER_POOL_CACHE['loaded'] = True
            return None

    try:
        pool = pd.read_parquet(path)
        required = {'ora', 'bump_mhz', 'amp100', 'date'}
        if not required.issubset(set(pool.columns)):
            print(f"[MATCHED] Colonne mancanti nel pool: {required - set(pool.columns)}")
            _DOPPLER_POOL_CACHE['loaded'] = True
            return None
        _DOPPLER_POOL_CACHE['df'] = pool
        print(f"[MATCHED] Pool Doppler caricato: {len(pool)} finestre")
    except Exception as e:
        print(f"[MATCHED] Errore caricamento pool: {e}")
        _DOPPLER_POOL_CACHE['df'] = None
    _DOPPLER_POOL_CACHE['loaded'] = True
    return _DOPPLER_POOL_CACHE['df']

def calcola_bump_evento_da_df(df, picco, finestra_min=(-10, 5)):
    if picco is None:
        return None
    # disc_mask: solo Reset_Flag (replica check_matched_doppler)
    disc_mask = (df['Reset_Flag'].values == 1).copy()
    for idx in np.where(disc_mask)[0]:
        end = min(len(disc_mask), idx + 601)
        disc_mask[idx:end] = True
    disc_mask[0:min(600, len(disc_mask))] = True

    fase_raw = df['Phase_Deg'].values.astype(float).copy()
    fase_raw[disc_mask] = np.nan
    fase_interp = pd.Series(fase_raw).interpolate(
        method='linear', limit_direction='both').bfill().ffill().values
    d_fase_hz = np.gradient(fase_interp) / 360.0
    d_fase_hz = d_fase_hz - np.nanmedian(d_fase_hz)
    fase_smooth = median_filter(d_fase_hz, size=SMOOTH_FASE_SEC)
    fase_smooth[disc_mask] = np.nan
    fase_smooth = fase_smooth.copy()
    fase_smooth[np.abs(fase_smooth) > 0.020] = np.nan

    fd = FiltroDoppler(t_lento=2400, t_veloce=120, fs=1.0)
    fd.calcola(fase_smooth)
    if fd._filtrato is None:
        return None
    doppler_f = fd._filtrato.copy() * 1000.0
    doppler_f[disc_mask] = np.nan

    t_a = picco + timedelta(minutes=finestra_min[0])
    t_b = picco + timedelta(minutes=finestra_min[1])
    mask_ev = (df['Timestamp'] >= t_a) & (df['Timestamp'] <= t_b)
    if not mask_ev.any():
        return None

    f_win = doppler_f[mask_ev.values]
    if np.all(~np.isfinite(f_win)):
        return None

    mediana = np.nanmedian(f_win)
    bump_mhz = float(np.nanmax(np.abs(f_win - mediana)))
    amp100_mean = float(df.loc[mask_ev, 'Amp_100Hz_dB'].mean())
    ora_evento = picco.hour + picco.minute / 60.0

    return {'bump_mhz': bump_mhz, 'amp100': amp100_mean, 'ora': ora_evento}

def matched_doppler_percentile(bump_mhz, amp100_evento, ora_evento, data_evento, pool):
    if pool is None or bump_mhz is None or amp100_evento is None:
        return None, 0
    same_day = (pool['date'] == data_evento)
    mask = (
        (np.abs(pool['amp100'] - amp100_evento) <= AMP100_TOL_DB)
        & (np.abs(pool['ora'] - ora_evento) <= ORA_TOL_H)
        & (~same_day)
        & pool['bump_mhz'].notna()
    )
    gem = pool.loc[mask, 'bump_mhz'].values
    n_gem = int(len(gem))
    if n_gem < MIN_GEMELLE:
        return None, n_gem
    pct = 100.0 * float(np.sum(gem <= bump_mhz) / n_gem)
    return pct, n_gem

# ============================================================
# ANALISI FLARE
# ============================================================
def analizza_flare(df, flare, ts_xray, flux_xray, flares=None):
    
    if flares is None:
        flares = [flare]   # fallback: solo il flare corrente
    picco = parse_flare_datetime(flare['picco'])
    if picco is None:
        return None

    # --- BUG FIX 2026-10-08 (Fix 1) ---
    t_ini_loc = parse_flare_datetime(flare.get('inizio', ''))
    t_fin_loc = parse_flare_datetime(flare.get('fine', ''))
    if t_ini_loc is None:
        t_ini_loc = picco - timedelta(minutes=5)
    if t_fin_loc is None:
        t_fin_loc = picco + timedelta(minutes=15)

    t_start = min(
        picco - timedelta(seconds=FINESTRA_FLARE_SEC),
        t_ini_loc - timedelta(seconds=BASE_PRE_EVENTO_SEC + 600)
    )
    t_end = max(
        picco + timedelta(seconds=FINESTRA_FLARE_SEC),
        t_fin_loc + timedelta(seconds=BASE_POST_EVENTO_SEC + 600)
    )

    mask = (df['Timestamp'] >= t_start) & (df['Timestamp'] <= t_end)
    df_win = df[mask].copy()
    if len(df_win) < 60:
        return None

    # ============================================================
    # REALITY CHECK X-RAY (evita i falsi positivi da TID)
    # ============================================================
    # Controlliamo solo la finestra stretta intorno al picco (±10 min).
    # Se in questo intervallo lo X-ray non ha mai superato la soglia C1 
    # (1e-6), non può esserci stato un SID. Scartiamo l'evento.
    if ts_xray and flux_xray:
        t_xray = np.array([t.timestamp() for t in ts_xray])
        f_xray = np.array(flux_xray)
        t_peak_ts = picco.timestamp()
        
        mask_peak = (t_xray >= t_peak_ts - 600) & (t_xray <= t_peak_ts + 600)
        
        if np.any(mask_peak):
            if np.max(f_xray[mask_peak]) < 1e-6:
                print(f"  [SKIP] Nessun flare X-ray > C1 vicino al picco. Scarto.")
                return None
            
    # X-ray interpolato — usa il GRADIENTE di log10(flux), non log10(flux).
    if ts_xray:
        t_xray = np.array([t.timestamp() for t in ts_xray])
        flux_xray_arr = np.array(flux_xray)
        t_tracker = np.array([t.timestamp() for t in df_win['Timestamp']])

        # Interpola log10 del flusso sulla griglia del tracker
        logflux = np.interp(t_tracker, t_xray, np.log10(flux_xray_arr + 1e-12))

        # Smoothing (60 s) per stabilizzare il gradiente
        logflux_smooth = (pd.Series(logflux)
                            .rolling(60, center=True, min_periods=1)
                            .mean()
                            .values)

        # Gradiente temporale: d(log10 flux)/dt in unità di 1/s
        xray_interp = np.gradient(logflux_smooth)

        # Log-flusso detrenduto: serve per correlare con la POTENZA,
        # che risponde al flusso (gradino), non al suo gradiente.
        logflux_det = detrend_mediano(logflux_smooth, 3600)
    else:
        xray_interp = np.zeros(len(df_win))
        logflux_det = np.zeros(len(df_win))

    # Segnali grezzi
    pwr  = df_win['Power_dB'].values
    fase = df_win['Phase_Deg'].values
    pll  = df_win['PLL_Freq_Hz'].values

    # 1) AMPIEZZA
    pwr_smooth = uniform_filter1d(pwr, size=SMOOTH_AMPIEZZA_SEC, mode='nearest')
    pwr_det    = pwr_smooth

    # 2) FASE — usa la DERIVATA (Doppler istantaneo in Hz)
    fase_raw = df_win['Phase_Deg'].values.astype(float)
    disc_mask = (df_win['Discontinuita'].values == 1) if 'Discontinuita' in df_win.columns else None

    fase_for_grad = fase_raw.copy()
    if disc_mask is not None:
        fase_for_grad[disc_mask] = np.nan

    fase_for_grad = pd.Series(fase_for_grad).interpolate(
        method='linear', limit_direction='both'
    ).bfill().ffill().values

    d_fase_hz = np.gradient(fase_for_grad) / 360.0

    d_fase_hz = filtro_hampel(d_fase_hz, window=7, n_sigma=3.0)

    offset_d_fase = np.nanmedian(d_fase_hz)
    d_fase_hz = d_fase_hz - offset_d_fase
    fase_smooth = median_filter(d_fase_hz, size=SMOOTH_FASE_SEC)

    # Calcolo del gradiente massimo (usato per il sanity check degli artefatti)
    gradiente_max_fase = None
    if len(d_fase_hz) > 0:
        d_fase_validi = d_fase_hz[np.isfinite(d_fase_hz)]
        if len(d_fase_validi) > 0:
            gradiente_max_fase = float(np.max(np.abs(d_fase_validi)))

    if disc_mask is not None:
        fase_smooth = fase_smooth.copy()
        fase_smooth[disc_mask] = np.nan

        idx_disc = np.where(disc_mask)[0]
        if len(idx_disc) > 0:
            fine_blocchi = []
            prev = idx_disc[0]
            for idx in idx_disc[1:]:
                if idx != prev + 1:
                    fine_blocchi.append(prev)
                prev = idx
            fine_blocchi.append(prev)
            for f_idx in fine_blocchi:
                end = min(len(fase_smooth), f_idx + 31)
                fase_smooth[f_idx+1:end] = np.nan

        fase_smooth = fase_smooth.copy()
        fase_smooth = sanity_doppler_con_durata(
            fase_smooth, soglia_hz=0.020, min_durata_s=60
        )

    # 3) PLL_Freq — pulito dalle discontinuità
    pll_vals = df_win['PLL_Freq_Hz'].values.astype(float).copy()
    if disc_mask is not None:
        pll_vals[disc_mask] = np.nan
    pll_vals = pd.Series(pll_vals).interpolate(
        method='linear', limit=10, limit_direction='both'
    ).values
    pll_det        = detrend_mediano(pll_vals, 600)
    pll_det_smooth = median_filter(pll_det, size=SMOOTH_PLL_SEC)

    # DELTA orientati
    if POLARITA_SID_AMPIEZZA > 0:
        delta_pwr = np.nanmax(pwr_det)
    else:
        delta_pwr = -np.nanmin(pwr_det)

    delta_fase = np.nanmax(fase_smooth)    - np.nanmin(fase_smooth)
    delta_pll  = np.nanmax(pll_det_smooth) - np.nanmin(pll_det_smooth)

    # ============================================================
    # FINESTRE TEMPORALI E BASELINE
    # ============================================================
    t_peak = picco
    t_ini = t_ini_loc
    t_fin = t_fin_loc
    if t_ini is None:
        t_ini = t_peak - timedelta(minutes=5)
    if t_fin is None:
        t_fin = t_peak + timedelta(minutes=15)

    t_ev_a = t_ini - timedelta(seconds=EVENTO_PRE_SEC)
    t_ev_b = t_fin + timedelta(seconds=EVENTO_POST_SEC)

    t_base_pre_a  = t_ini - timedelta(seconds=BASE_PRE_EVENTO_SEC)
    t_base_pre_b  = t_ini - timedelta(seconds=BASE_SKIP_SEC)
    t_base_post_a = t_fin + timedelta(seconds=BASE_SKIP_SEC)
    t_base_post_b = t_fin + timedelta(seconds=BASE_POST_EVENTO_SEC)

    mask_base = (((df_win['Timestamp'] >= t_base_pre_a) &
                  (df_win['Timestamp'] <= t_base_pre_b)) |
                 ((df_win['Timestamp'] >= t_base_post_a) &
                  (df_win['Timestamp'] <= t_base_post_b)))

    mask_ev = ((df_win['Timestamp'] >= t_ev_a) &
               (df_win['Timestamp'] <= t_ev_b))

    # FIX A: escludi transizione ionosferica
    try:
        elev_serie = calcola_elevazione_sole_serie(
            df_win['Timestamp'].values, step_sec=60)
        mask_transizione = (elev_serie > 0.0) & (elev_serie < 20.0)
        mask_transizione = pd.Series(
            mask_transizione, index=mask_base.index)
        mask_base = mask_base & ~mask_transizione
    except Exception as e:
        print(f"  [WARN] Esclusione transizione ionosferica fallita: {e}")

    # FIX B: escludi altri flare vicini
    for other in flares:
        if other is flare:
            continue
        other_pk = parse_flare_datetime(other.get('picco', ''))
        if other_pk is None:
            continue
        other_ini = parse_flare_datetime(other.get('inizio', ''))
        other_fin = parse_flare_datetime(other.get('fine', ''))
        if other_ini is None:
            other_ini = other_pk - timedelta(minutes=5)
        if other_fin is None:
            other_fin = other_pk + timedelta(minutes=15)

        other_ini_ext = other_ini - timedelta(seconds=300)
        other_fin_ext = other_fin + timedelta(seconds=300)

        interseca = (
            other_ini_ext <= t_base_post_b and
            other_fin_ext >= t_base_pre_a
        )
        if not interseca:
            continue

        mask_base &= ~(
            (df_win['Timestamp'] >= other_ini_ext) &
            (df_win['Timestamp'] <= other_fin_ext)
        )

    # Soglia Doppler dinamica
    fase_base = fase_smooth[mask_base.values]
    fase_base_validi = fase_base[np.isfinite(fase_base)]

    if len(fase_base_validi) >= 30:
        std_baseline = float(np.nanstd(fase_base_validi))
        range_baseline = float(np.nanmax(fase_base_validi) -
                                np.nanmin(fase_base_validi))
        d_fase_soglia = max(0.001, 4.0 * std_baseline)
    else:
        std_baseline = None
        range_baseline = None
        d_fase_soglia = 0.001

    # ============================================================
    # TEST STATISTICO (Welch corretto con N_eff)
    # ============================================================
    # Usiamo N_eff basato sul tempo di correlazione.
    TAU_FASE_S = 120.0
    TAU_PWR_S  = 60.0

    p_fase = None
    p_pll  = None
    p_pwr  = None

    # Fase
    fase_base = fase_smooth[mask_base.values]
    fase_ev   = fase_smooth[mask_ev.values]
    res_f = ttest_welch_neff(fase_ev, fase_base, tau_sec=TAU_FASE_S)
    if res_f[2] is not None:
        _, _, p_fase = res_f

    # PLL
    pll_base = pll_det_smooth[mask_base.values]
    pll_ev   = pll_det_smooth[mask_ev.values]
    res_p = ttest_welch_neff(pll_ev, pll_base, tau_sec=TAU_PWR_S)
    if res_p[2] is not None:
        _, _, p_pll = res_p

    # Potenza
    pwr_base = pwr_det[mask_base.values]
    pwr_ev   = pwr_det[mask_ev.values]
    res_w = ttest_welch_neff(pwr_ev, pwr_base, tau_sec=TAU_PWR_S)
    if res_w[2] is not None:
        _, _, p_pwr = res_w

    # ============================================================
    # SID DI POTENZA — baseline e metriche
    # ============================================================
    mask_pwr_base = mask_base
    mask_pwr_ev   = mask_ev

    delta_pwr_media    = None
    snr_pwr_media      = None
    delta_pwr_picco    = None
    snr_pwr_picco      = None
    dt_picco_flare_sec = None
    soglia_picco_dinamica = PWR_SID_PICCO_MIN_DB
    std_ref_pre = 0.0
    pwr_detrended = df_win['Power_dB'].values.copy()

    try:
        mask_base_arr = mask_pwr_base.values
        t_base_sec = np.array([t.timestamp()
                                for t in df_win.loc[mask_base_arr, 'Timestamp']])
        pwr_base_raw = df_win.loc[mask_base_arr, 'Power_dB'].values
        base_ok = np.isfinite(pwr_base_raw)

        if base_ok.sum() >= 30:
            t_base_ok = t_base_sec[base_ok]
            pwr_base_ok = pwr_base_raw[base_ok]
            coeff = np.polyfit(t_base_ok - t_base_ok[0], pwr_base_ok, 1)
            t_all_sec = np.array([t.timestamp() for t in df_win['Timestamp']])
            trend = np.polyval(coeff, t_all_sec - t_base_ok[0])
            pwr_detrended = df_win['Power_dB'].values - trend
        else:
            pwr_detrended = df_win['Power_dB'].values

        pwr_base_vals = pwr_detrended[mask_pwr_base.values]
        pwr_ev_vals   = pwr_detrended[mask_pwr_ev.values]
        pwr_base_vals = pwr_base_vals[np.isfinite(pwr_base_vals)]
        pwr_ev_vals   = pwr_ev_vals[np.isfinite(pwr_ev_vals)]

        if len(pwr_base_vals) >= 20 and len(pwr_ev_vals) >= 20:

            delta_pwr_media = float(
                np.mean(pwr_ev_vals) - np.mean(pwr_base_vals)
            )
            std_pwr_base = float(np.std(pwr_base_vals))
            snr_pwr_media = (delta_pwr_media / std_pwr_base
                             if std_pwr_base > 0 else 0.0)

            t_ev_sec = np.array([t.timestamp()
                                 for t in df_win.loc[mask_pwr_ev, 'Timestamp']])
            mask_ev_validi = np.isfinite(pwr_detrended[mask_pwr_ev.values])
            t_ev_validi = t_ev_sec[mask_ev_validi]

            # --- FIX CLAUDE (10 Novembre): Metrica 2 (Picco Potenza) ---
            t_win_sec = np.array([t.timestamp() for t in df_win['Timestamp']])
            pwr_win_raw = df_win['Power_dB'].values

            t_ini_ts = t_ini_loc.timestamp()
            t_pk_ts = t_peak.timestamp()

            # Baseline pre-onset: 15 min a 2 min prima dell'inizio
            mask_ref_pre = (t_win_sec >= t_ini_ts - 900) & (t_win_sec <= t_ini_ts - 120)
            # --- CALCOLO RUMORE PRE-EVENTO ROBUSTO (Detrend + MAD) ---
            if np.any(mask_ref_pre):
                pwr_pre_vals = pwr_win_raw[mask_ref_pre]
                t_pre_vals = t_win_sec[mask_ref_pre]

                # 1. Escludi campioni contaminati da altri flare
                m_altri_pre = maschera_flare_vicini(
                    df_win['Timestamp'].values[mask_ref_pre], 
                    picco, 
                    flares
                )
                pwr_pre_clean = pwr_pre_vals[~m_altri_pre]
                t_pre_clean = t_pre_vals[~m_altri_pre]
                
                # FIX: Rimuovi i NaN prima di fare polyfit e median
                mask_valid_pre = np.isfinite(pwr_pre_clean)
                pwr_pre_clean = pwr_pre_clean[mask_valid_pre]
                t_pre_clean = t_pre_clean[mask_valid_pre]

                if len(pwr_pre_clean) > 30:
                    # 2. Detrend lineare per togliere la deriva lenta (Morning Dip)
                    coeff_pre = np.polyfit(t_pre_clean - t_pre_clean[0], pwr_pre_clean, 1)
                    trend_pre = np.polyval(coeff_pre, t_pre_clean - t_pre_clean[0])
                    residui_pre = pwr_pre_clean - trend_pre
                    
                    # 3. Baseline = mediana; Rumore = MAD (robusta agli outlier)
                    ref_pwr_pre = float(np.median(pwr_pre_clean))
                    mad_pre = 1.4826 * np.median(np.abs(residui_pre - np.median(residui_pre)))
                    std_ref_pre = max(mad_pre, 0.1)
                else:
                    # FIX DEFINITIVO: Livello da potenza grezza, rumore da potenza detrenduta
                    # Fallback: mediana dell'intera finestra se la pre-evento è mascherata 
                    # (es. caso C1.5 del 08/11, dove il C2.1 vicino oscura la finestra standard)
                    ref_pwr_pre = np.nanmedian(pwr_win_raw)
                    std_ref_pre = np.nanstd(pwr_base_vals)
            else:
                # FIX DEFINITIVO: Livello da potenza grezza, rumore da potenza detrenduta
                # Fallback: mediana dell'intera finestra se la finestra pre-evento non esiste
                ref_pwr_pre = np.nanmedian(pwr_win_raw)
                std_ref_pre = np.nanstd(pwr_base_vals)

            # Picco: massima deviazione assoluta (su o giù) tra inizio-60s e picco+600s
            # Pulisce i NaN prima del filtro per evitare propagazione
            pwr_win_clean = pd.Series(pwr_win_raw).interpolate().bfill().ffill().values
            pwr_smooth_win = uniform_filter1d(pwr_win_clean, size=60, mode='nearest')
            mask_peak_win = (t_win_sec >= t_ini_ts - 60) & (t_win_sec <= t_pk_ts + 600)

            if np.any(mask_peak_win):
                win_vals = pwr_smooth_win[mask_peak_win]
            else:
                win_vals = pwr_smooth_win
            
            # FIX 1: Se la baseline pre-evento è NaN, usa la mediana dell'intera finestra
            if not np.isfinite(ref_pwr_pre):
                ref_pwr_pre = np.nanmedian(df_win['Power_dB'].values)

            dev = win_vals - ref_pwr_pre
            
            # FIX 2: Ignora i NaN e applica la guardia fisica (> 15 dB)
            mask_fisica = np.isfinite(dev) & (np.abs(dev) <= 15.0)
            if np.any(mask_fisica):
                dev_validi = dev[mask_fisica]
                idx_max_dev = np.argmax(np.abs(dev_validi))
                delta_pwr_picco = dev_validi[idx_max_dev]
            else:
                delta_pwr_picco = 0.0

            # 4. Calcolo della soglia dinamica (Asimmetrica)
            if delta_pwr_picco < 0:
                soglia_picco_dinamica = max(
                    PWR_SID_SFA_MIN_DB,
                    PWR_SID_PICCO_K_SIGMA * std_ref_pre
                )
            else:
                soglia_picco_dinamica = max(
                    PWR_SID_PICCO_MIN_DB,
                    PWR_SID_PICCO_K_SIGMA * std_ref_pre
                )

            snr_pwr_picco = abs(delta_pwr_picco) / std_ref_pre if std_ref_pre > 0 else 0.0

            # Timing: gradiente vs gradiente
            mask_validi_ev = np.isfinite(pwr_detrended[mask_pwr_ev.values])
            pwr_ev_validi = pwr_detrended[mask_pwr_ev.values][mask_validi_ev]
            xray_ev_validi = xray_interp[mask_pwr_ev.values][mask_validi_ev]
            t_ev_validi = t_ev_sec[mask_validi_ev]

            pwr_smooth_ev = uniform_filter1d(pwr_ev_validi, size=60, mode='nearest')
            grad_pwr = np.gradient(pwr_smooth_ev)
            grad_xray_ev = np.gradient(xray_ev_validi)

            t_ev_validi_arr = np.array(t_ev_validi)
            t_peak_sec = t_peak.timestamp()
            mask_ricerca = (t_ev_validi_arr - t_peak_sec >= -300) & (t_ev_validi_arr - t_peak_sec <= 900)

            if np.any(mask_ricerca):
                grad_pwr_ric = np.where(mask_ricerca, grad_pwr, 0)
                grad_xray_ev_ric = np.where(mask_ricerca, grad_xray_ev, 0)

                if POLARITA_SID_AMPIEZZA > 0:
                    idx_grad_max_pwr = int(np.argmax(grad_pwr_ric))
                    idx_grad_max_xray = int(np.argmax(grad_xray_ev_ric))
                else:
                    idx_grad_max_pwr = int(np.argmin(grad_pwr_ric))
                    idx_grad_max_xray = int(np.argmin(grad_xray_ev_ric))

                if len(t_ev_validi) > idx_grad_max_pwr and len(t_ev_validi) > idx_grad_max_xray:
                    t_grad_pwr = t_ev_validi[idx_grad_max_pwr]
                    t_grad_xray = t_ev_validi[idx_grad_max_xray]
                    dt_picco_flare_sec = t_grad_pwr - t_grad_xray
                else:
                    dt_picco_flare_sec = None
            else:
                dt_picco_flare_sec = None

    except Exception as e:
        print(f"  [ERR] Fit baseline potenza: {e}")

    # ==================================================
    # SNR LOCALE DEL BUMP
    # ==================================================
    snr_bump = snr_locale_bump(df, picco, altri_flare=flares)
    if snr_bump is not None:
        print(f"  [BUMP] delta_medio={snr_bump['ampiezza_bump_mhz']:+.4f} mHz  "
              f"std_locale={snr_bump['std_quieta_mhz']:.4f} mHz  "
              f"SNR_fase={snr_bump['snr_locale']:.2f}  "
              f"forma={'su' if snr_bump['bump_su'] else 'giu'}  "
              f"(N={snr_bump['n_evento']})")

    snr_fint = snr_fase_integrata(df, picco, altri_flare=flares)
    if snr_fint is not None:
        print(f"  [FASE_INT] delta={snr_fint['delta_fase_gradi']:+.2f} deg  "
              f"std_quiete={snr_fint['std_quiete_gradi']:.2f} deg  "
              f"SNR_int={snr_fint['snr_fase_int']:.2f}  "
              f"forma={'su' if snr_fint['forma_su_int'] else 'giu'}")

    # ==================================================
    # CROSS-CORRELAZIONE (Passo 4 - Pearson corretta)
    # ==================================================
    lag_fase, corr_fase = calcola_cross_correlazione(fase_smooth, xray_interp, LAG_MAX_SEC)
    lag_pll,  corr_pll  = calcola_cross_correlazione(pll_det_smooth, xray_interp, LAG_MAX_SEC)

    if len(xray_interp) >= 4 * 60:
        pad_x = min(len(xray_interp) // 4, 3600)
        xp_x = np.pad(xray_interp, pad_x, mode='edge')
        xray_interp_f = sosfiltfilt(
            butter(2, [1/2400, 1/120], btype='band', fs=1.0, output='sos'),
            xp_x
        )[pad_x:-pad_x]
    else:
        xray_interp_f = xray_interp

    fd_flare = FiltroDoppler(t_lento=2400, t_veloce=120, fs=1.0)
    fd_flare.calcola(fase_smooth)
    fase_filtrata = (fd_flare._filtrato.copy()
                     if fd_flare._filtrato is not None
                     else fase_smooth.copy())
    if disc_mask is not None and len(disc_mask) == len(fase_filtrata):
        fase_filtrata[disc_mask] = np.nan

    lag_fase_f, corr_fase_f = calcola_cross_correlazione(
        fase_filtrata, xray_interp_f, LAG_MAX_SEC
    )

    pwr_per_corr = POLARITA_SID_AMPIEZZA * pwr_detrended
    lag_pwr_x, corr_pwr_x = calcola_cross_correlazione(
        pwr_per_corr, logflux_det, LAG_MAX_SEC
    )
    lag_pwr = lag_pwr_x
    corr_pwr = corr_pwr_x

    if len(pwr_per_corr) >= 4 * 60:
        pad_p = min(len(pwr_per_corr) // 4, 3600)
        xp_p = np.pad(pwr_per_corr, pad_p, mode='edge')
        pwr_corr_f = sosfiltfilt(
            butter(2, [1/2400, 1/120], btype='band', fs=1.0, output='sos'),
            xp_p
        )[pad_p:-pad_p]
    else:
        pwr_corr_f = pwr_per_corr

    if len(logflux_det) >= 4 * 60:
        pad_l = min(len(logflux_det) // 4, 3600)
        xp_l = np.pad(logflux_det, pad_l, mode='edge')
        logflux_f = sosfiltfilt(
            butter(2, [1/2400, 1/120], btype='band', fs=1.0, output='sos'),
            xp_l
        )[pad_l:-pad_l]
    else:
        logflux_f = logflux_det

    lag_pwr_f, corr_pwr_f = calcola_cross_correlazione(
        pwr_corr_f, logflux_f, LAG_MAX_SEC
    )

    # ============================================================
    # Qualità del lock nella finestra del flare
    # ============================================================
    reset_vals  = df_win['Reset_Flag'].values
    amp100_vals = df_win['Amp_100Hz_dB'].values
    n_reset = int(np.sum(reset_vals == 1))
    soglia_lock = df.attrs.get('soglia_lock_amp100', None)

    lock_ok = maschera_lock_ok(df_win)
    inval = (reset_vals == 1) | (~lock_ok)
        
    if 'Discontinuita' in df_win.columns:
        inval = inval | (df_win['Discontinuita'].values == 1)
    if 'ADC_Overload' in df_win.columns:
        inval = inval | (df_win['ADC_Overload'].values == 1)

    n_amp_bassi = int(np.sum(np.isfinite(amp100_vals) &
                             (amp100_vals < soglia_lock))) if soglia_lock is not None else 0
    amp100_media = float(np.nanmean(amp100_vals)) if np.any(np.isfinite(amp100_vals)) else None
    amp100_std   = float(np.nanstd(amp100_vals)) if np.any(np.isfinite(amp100_vals)) else None

    # La frazione invalida va calcolata SOLO sulla finestra evento
    mask_rilevante = mask_ev
    if np.any(mask_rilevante):
        frazione_invalida = float(inval[mask_rilevante].mean())
    else:
        frazione_invalida = float(inval.mean())

    # --- DEBUG: dettaglio delle cause di invalidità ---
    n_tot_win = int(np.sum(mask_rilevante))
    if n_tot_win > 0:
        n_reset_win = int(np.sum(reset_vals[mask_rilevante] == 1))
        n_discontinuita_win = int(np.sum(df_win['Discontinuita'].values[mask_rilevante] == 1)) if 'Discontinuita' in df_win.columns else 0
        n_amp_bassi_win = int(np.sum(np.isfinite(amp100_vals[mask_rilevante]) & (amp100_vals[mask_rilevante] < soglia_lock))) if soglia_lock is not None else 0
        n_adc_win = int(np.sum(df_win['ADC_Overload'].values[mask_rilevante] == 1)) if 'ADC_Overload' in df_win.columns else 0

        print(f"  [DEBUG_INVALID] Finestra {picco.strftime('%H:%M')} — Totale campioni rilevanti: {n_tot_win}")
        print(f"    Reset_Flag=1:        {n_reset_win:>5} ({100*n_reset_win/n_tot_win:.1f}%)")
        print(f"    Discontinuita=1:     {n_discontinuita_win:>5} ({100*n_discontinuita_win/n_tot_win:.1f}%)")
        print(f"    Amp100 < {soglia_lock:.1f}:      {n_amp_bassi_win:>5} ({100*n_amp_bassi_win/n_tot_win:.1f}%)")
        print(f"    ADC_Overload=1:      {n_adc_win:>5} ({100*n_adc_win/n_tot_win:.1f}%)")
        print(f"    Frazione totale:     {frazione_invalida*100:.1f}%")
    # --- FINE DEBUG ---

        if 'Corr_Mag' in df_win.columns:
            cm_vals = df_win['Corr_Mag'].values[mask_rilevante]
            cm_smooth = pd.Series(cm_vals).rolling(60, center=True, min_periods=10).median().values
            n_cm_bassi = int(np.sum(np.isfinite(cm_smooth) & (cm_smooth < 0.40)))
            print(f"    Corr_Mag < 0.40:     {n_cm_bassi:>5} ({100*n_cm_bassi/n_tot_win:.1f}%)")

    # ============================================================
    # Classificazione diagnostica
    # ============================================================
    result = {
        'ora': picco.strftime('%H:%M'),
        'picco_dt': picco,
        'classe': flare['classe'],
        'delta_pwr':  delta_pwr,
        'delta_fase': delta_fase,
        'd_fase_soglia':  d_fase_soglia,
        'range_baseline': range_baseline,
        'delta_pll':  delta_pll,
        'lag_fase':   lag_fase,
        'corr_fase':  corr_fase,
        'lag_fase_filtrata':  lag_fase_f,
        'corr_fase_filtrata': corr_fase_f,
        'bump_ampiezza_mhz': snr_bump['ampiezza_bump_mhz'] if snr_bump else None,
        'bump_std_quieta':   snr_bump['std_quieta_mhz'] if snr_bump else None,
        'bump_snr_locale':   snr_bump['snr_locale'] if snr_bump else None,
        'bump_forma_su':     snr_bump['bump_su'] if snr_bump else None,
        'snr_fase_int':      snr_fint['snr_fase_int'] if snr_fint else None,
        'delta_fase_int_deg': snr_fint['delta_fase_gradi'] if snr_fint else None,
        'forma_su_int':      snr_fint['forma_su_int'] if snr_fint else None,
        'std_quiete_gradi':  snr_fint['std_quiete_gradi'] if snr_fint else None,
        'lag_pwr_x':  lag_pwr_x,
        'corr_pwr_x': corr_pwr_x,
        'lag_pwr':    lag_pwr,
        'corr_pwr':   corr_pwr,
        'lag_pwr_f':  lag_pwr_f,
        'corr_pwr_f': corr_pwr_f,
        'lag_pll':    lag_pll,
        'corr_pll':   corr_pll,
        'p_fase':     p_fase,
        'p_pll':      p_pll,
        'p_pwr':      p_pwr,
        'delta_pwr_media': delta_pwr_media,
        'snr_pwr_media':   snr_pwr_media,
        'delta_pwr_picco': delta_pwr_picco,
        'snr_pwr_picco':   snr_pwr_picco,
        'frazione_invalida': frazione_invalida,
        'n_reset':     n_reset,
        'n_amp_bassi': n_amp_bassi,
        'amp100_media': amp100_media,
        'amp100_std':   amp100_std,
        'soglia_lock':  soglia_lock,
        'dt_picco_pwr_sec': dt_picco_flare_sec,
        'std_baseline': std_baseline,
        'gradiente_max_fase': gradiente_max_fase,
        'soglia_picco_dinamica': soglia_picco_dinamica,
        'std_ref_pre': std_ref_pre,
    }

    categoria, spiegazione, colore = classifica_diagnostica(result)
    result['categoria']   = categoria
    result['spiegazione'] = spiegazione
    result['colore_cat']  = colore

    # ------------------------------------------------------------
    # REGOLA SECONDARIA — Matched Doppler (con guardia e baseline pre-evento)
    # ------------------------------------------------------------
    if (MATCHED_DOPPLER_ON and result['categoria'] == 'ARTEFATTO'
            and frazione_invalida >= FRAZIONE_INVALIDA_MAX):

        # --- 1. GUARDIA: l'invalidità è dovuta *solo* ad Amp100? ---
        m = mask_rilevante.values
        # Il nuovo nome riflette il fatto che l'invalidità può essere dovuta a Corr_Mag o Amp100
        solo_lock_basso = (
            not (reset_vals[m] == 1).any()
            and not (('Discontinuita' in df_win.columns) and df_win['Discontinuita'].values[m].any())
            and not (('ADC_Overload' in df_win.columns) and (df_win['ADC_Overload'].values[m] == 1).any())
        )
        
        if solo_lock_basso:
            pool = load_doppler_pool()
            if pool is not None:
                ev = calcola_bump_evento_da_df(df, picco)
                if ev is not None:
                    t_ini_ts = t_ini_loc.timestamp()
                    t_win_sec = np.array([t.timestamp() for t in df_win['Timestamp']])
                    mask_pre_ev = (t_win_sec >= t_ini_ts - 2400) & (t_win_sec <= t_ini_ts - 600)

                    if np.any(mask_pre_ev):
                        amp100_pre = np.nanmean(df_win['Amp_100Hz_dB'].values[mask_pre_ev])
                    else:
                        amp100_pre = ev['amp100']

                    if np.isnan(amp100_pre):
                        amp100_pre = ev['amp100']

                    pct, n_gem = matched_doppler_percentile(
                        bump_mhz=ev['bump_mhz'],
                        amp100_evento=amp100_pre,
                        ora_evento=ev['ora'],
                        data_evento=picco.strftime('%Y-%m-%d'),
                        pool=pool,
                    )

                    result['matched_doppler_bump_mhz'] = ev['bump_mhz']
                    result['matched_doppler_amp100']   = amp100_pre
                    result['matched_doppler_pct']      = pct
                    result['matched_doppler_n_gem']    = n_gem

                    if pct is not None and pct >= PERCENTILE_SOGLIA:
                        result['categoria']   = 'SID_DOPPLER_ABBINATO'
                        result['colore_cat']  = '#00AA00'
                        result['spiegazione'] = (
                            f"✓ SID DOPPLER ABBINATO: primaria ARTEFATTO "
                            f"({frazione_invalida*100:.0f}% invalido), "
                            f"ma matched Doppler a {pct:.1f}° percentile "
                            f"(bump={ev['bump_mhz']:.3f} mHz, N_gem={n_gem}, "
                            f"Amp100_pre={amp100_pre:+.1f} dB)."
                        )
                        print(f"  [MATCHED] → SID_DOPPLER_ABBINATO (pct={pct:.1f}, N={n_gem})")
                    else:
                        if pct is not None:
                            print(f"  [MATCHED] non promosso (pct={pct:.1f}, N={n_gem})")
                        else:
                            print(f"  [MATCHED] gemelle insufficienti (N={n_gem} < {MIN_GEMELLE})")
        else:
            print(f"  [MATCHED] Scartato: l'ARTEFATTO non è dovuto solo al lock (reset/overload).")

    if _FROZEN_OK:
        try:
            frozen = valuta_frozen_da_df(df, ts_xray, flux_xray, picco)
            if frozen is not None:
                result.update(frozen)
            else:
                print(f"  [FROZEN] non calcolabile (dati insufficienti "
                      f"o picco fuori finestra)")
        except Exception as e:
            print(f"  [FROZEN] errore: {e}")

    return result

def plot_riassuntivo(df, ts_xray, flux_xray, flares, risultati, fig=None):
    # Se fig è None → nuova figura. Se esiste → la svuota e riusa.
    if fig is None:
        fig = plt.figure(figsize=(16, 15))
    else:
        fig.clf()

    axes = fig.subplots(5, 1, sharex=True,
                        gridspec_kw={'height_ratios': [1.2, 1.2, 1.0, 1.0, 1.0]})

    # ----------------------------------------------------------
    # 1) AMPIEZZA
    # ----------------------------------------------------------
    pwr = df['Power_dB'].values
    pwr_smooth = uniform_filter1d(pwr, size=SMOOTH_AMPIEZZA_SEC, mode='nearest')
    # Potenza lineare: inversa di 10*log10(P).
    pwr_lin = 10.0 ** (pwr / 10.0)
    pwr_smooth_lin = 10.0 ** (pwr_smooth / 10.0)

    scala_lineare_prec = getattr(fig, '_pwr_scala_lineare', False)

    y_raw_plot = pwr_lin if scala_lineare_prec else pwr
    y_smooth_plot = pwr_smooth_lin if scala_lineare_prec else pwr_smooth

    linea_pwr_raw, = axes[0].plot(df['Timestamp'], y_raw_plot,
                                  color='green', alpha=0.25,
                                  linewidth=0.5, label='grezzo')
    linea_pwr_smooth, = axes[0].plot(df['Timestamp'], y_smooth_plot,
                                     color='green', linewidth=1.1,
                                     label=f'smooth {SMOOTH_AMPIEZZA_SEC}s')

    label_pwr = 'Power (lin.)' if scala_lineare_prec else 'Power (dB)'
    axes[0].set_ylabel(label_pwr, color='green')

    pwr_detr_raw = detrend_mascherato(pwr, df['Timestamp'], flares, finestra=1800)
    pwr_detr_smooth = detrend_mascherato(pwr_smooth, df['Timestamp'], flares, finestra=1800)

    fig._pwr_lines = (linea_pwr_raw, linea_pwr_smooth)
    fig._pwr_data = {
        'db_raw': pwr, 'db_smooth': pwr_smooth,
        'lin_raw': pwr_lin, 'lin_smooth': pwr_smooth_lin,
        'det_raw': pwr_detr_raw, 'det_smooth': pwr_detr_smooth
    }
    fig._pwr_scala_lineare = scala_lineare_prec
    fig._ax_pwr = axes[0]
    data_plot = df['Timestamp'].iloc[0].strftime('%Y-%m-%d')
    axes[0].set_title(f'Analisi Doppler flare — {data_plot}  '
                      f'(CSV: {df["Timestamp"].iloc[0].strftime("%H:%M")} → '
                      f'{df["Timestamp"].iloc[-1].strftime("%H:%M")} UTC)',
                      color='black', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='y', colors='green')
    axes[0].legend(loc='upper left', fontsize=7, framealpha=0.7)

    # ----------------------------------------------------------
    # 2) FASE — derivata (Doppler in Hz)
    # ----------------------------------------------------------
    fase_raw_plot = df['Phase_Deg'].values.astype(float).copy()
    disc_mask = (df['Discontinuita'].values == 1) if 'Discontinuita' in df.columns else None
    if disc_mask is not None:
        fase_raw_plot[disc_mask] = np.nan

    fase_for_grad_plot = pd.Series(fase_raw_plot).interpolate(
        method='linear', limit_direction='both'
    ).bfill().ffill().values

    d_fase_hz_plot = np.gradient(fase_for_grad_plot) / 360.0
    d_fase_hz_plot = filtro_hampel(d_fase_hz_plot, window=7, n_sigma=3.0)
    offset_plot = np.nanmedian(d_fase_hz_plot)
    d_fase_hz_plot = d_fase_hz_plot - offset_plot
    fase_smooth = median_filter(d_fase_hz_plot, size=SMOOTH_FASE_SEC)

    fase_smooth = sanity_doppler_con_durata(fase_smooth, soglia_hz=0.020, min_durata_s=60)

    if disc_mask is not None:
        fase_smooth = fase_smooth.copy()
        fase_smooth[disc_mask] = np.nan

        EXTENSIONE_POST_DISC = 30
        idx_disc = np.where(disc_mask)[0]
        if len(idx_disc) > 0:
            fine_blocchi = []
            prev = idx_disc[0]
            for idx in idx_disc[1:]:
                if idx != prev + 1:
                    fine_blocchi.append(prev)
                prev = idx
            fine_blocchi.append(prev)
            for f_idx in fine_blocchi:
                end = min(len(fase_smooth), f_idx + EXTENSIONE_POST_DISC + 1)
                fase_smooth[f_idx + 1:end] = np.nan

    filtro_era_attivo = False
    if hasattr(fig, '_fd'):
        filtro_era_attivo = fig._fd.attivo

    fd = FiltroDoppler(t_lento=2400, t_veloce=120, fs=1.0)
    fd.calcola(fase_smooth)
    fd.attivo = filtro_era_attivo

    if disc_mask is not None and fd._filtrato is not None:
        fase_filtrata = fd._filtrato.copy()
        fase_filtrata[disc_mask] = np.nan
        idx_disc = np.where(disc_mask)[0]
        if len(idx_disc) > 0:
            fine_blocchi = []
            prev = idx_disc[0]
            for idx in idx_disc[1:]:
                if idx != prev + 1:
                    fine_blocchi.append(prev)
                prev = idx
            fine_blocchi.append(prev)
            for f_idx in fine_blocchi:
                end = min(len(fase_filtrata), f_idx + 6)
                fase_filtrata[f_idx + 1:end] = np.nan
        fd._filtrato = fase_filtrata

    y_plot = fd._filtrato if fd.attivo else fd._grezzo
    linea_doppler, = axes[1].plot(df['Timestamp'], y_plot,
                                  color='blue', linewidth=0.9)

    fig._fd = fd
    fig._linea_doppler = linea_doppler
    fig._ax_doppler = axes[1]

    label_doppler = 'Doppler fase (Hz)'
    if fd.attivo:
        label_doppler += '  [filtrato 2–40 min]'
    axes[1].set_ylabel(label_doppler, color='blue')

    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='y', colors='blue')

    if 'Discontinuita' in df.columns:
        mask_disc = df['Discontinuita'].values == 1
        idx_disc = np.where(mask_disc)[0]
        if len(idx_disc) > 0:
            blocchi = []
            inizio = idx_disc[0]
            prev = idx_disc[0]
            for idx in idx_disc[1:]:
                if idx != prev + 1:
                    blocchi.append((inizio, prev))
                    inizio = idx
                prev = idx
            blocchi.append((inizio, prev))
            for i_start, i_end in blocchi:
                t0 = df['Timestamp'].iloc[i_start]
                t1 = df['Timestamp'].iloc[i_end]
                axes[1].axvspan(t0, t1, color='#8899AA', alpha=0.15,
                                linewidth=0)
                axes[2].axvspan(t0, t1, color='#8899AA', alpha=0.10,
                                linewidth=0)

    # ----------------------------------------------------------
    # 3) PLL_Freq — pulito dalle discontinuità
    # ----------------------------------------------------------
    pll_vals = df['PLL_Freq_Hz'].values.astype(float).copy()
    if 'Discontinuita' in df.columns:
        pll_vals[df['Discontinuita'].values == 1] = np.nan
    pll_vals = pd.Series(pll_vals).interpolate(
        method='linear', limit=30, limit_direction='both'
    ).bfill().ffill().values
    pll_det = detrend_mediano(pll_vals, 600)
    pll_det_smooth = median_filter(pll_det, size=SMOOTH_PLL_SEC)

    axes[2].plot(df['Timestamp'], pll_det_smooth, color='purple', linewidth=0.9)
    axes[2].set_ylabel('PLL_Freq (Hz)', color='purple')
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(axis='y', colors='purple')

    # ----------------------------------------------------------
    # 4) NUOVO — Amp_100Hz_dB (qualità lock PLL)
    # ----------------------------------------------------------
    amp100 = df['Amp_100Hz_dB'].values.astype(float).copy()

    amp100_smooth = uniform_filter1d(amp100, size=SMOOTH_AMPIEZZA_SEC,
                                     mode='nearest')

    if disc_mask is not None:
        amp100 = amp100.copy()
        amp100_smooth = amp100_smooth.copy()
        amp100[disc_mask] = np.nan
        amp100_smooth[disc_mask] = np.nan

    soglia_lock = df.attrs.get('soglia_lock_amp100', None)
    amp100_detr_raw = detrend_mascherato(amp100, df['Timestamp'], flares, finestra=1800)
    amp100_detr_smooth = detrend_mascherato(amp100_smooth, df['Timestamp'], flares, finestra=1800)

    linea_amp_raw, = axes[3].plot(df['Timestamp'], amp100, color='#00BFFF', alpha=0.25,
                 linewidth=0.5, label='Amp 100 Hz (grezzo)')
    linea_amp_smooth, = axes[3].plot(df['Timestamp'], amp100_smooth, color='#00BFFF',
                 linewidth=1.1, label=f'Amp 100 Hz (smooth {SMOOTH_AMPIEZZA_SEC}s)')

    linea_soglia = None
    if soglia_lock is not None:
        linea_soglia = axes[3].axhline(soglia_lock, color='red', linestyle='--',
                        linewidth=1.0, alpha=0.7,
                        label=f'Rif. Amp100 ({soglia_lock:+.1f} dB)')

    fig._amp100_lines = (linea_amp_raw, linea_amp_smooth)
    fig._linea_soglia_amp100 = linea_soglia
    fig._amp100_data = {
        'raw': amp100, 'smooth': amp100_smooth,
        'det_raw': amp100_detr_raw, 'det_smooth': amp100_detr_smooth
    }
    fig._ax_amp100 = axes[3]
    fig._detrend_attivo = False

    validi = amp100_smooth[np.isfinite(amp100_smooth)]
    if len(validi) > 10:
        v_min = float(np.min(validi))
        v_max = float(np.max(validi))
        margine = max((v_max - v_min) * 0.20, 1.0)
        axes[3].set_ylim(v_min - margine, v_max + margine)

    reset_mask = df['Reset_Flag'].values == 1
    if np.any(reset_mask):
        t_reset = df['Timestamp'].values[reset_mask]
        for t in t_reset:
            axes[3].axvline(t, color='red', alpha=0.15, linewidth=0.5)
    axes[3].set_ylabel('Amp 100 Hz (dB)', color='#00BFFF')
    axes[3].grid(True, alpha=0.3)
    axes[3].tick_params(axis='y', colors='#00BFFF')
    axes[3].legend(loc='upper left', fontsize=7, framealpha=0.7)

    # ----------------------------------------------------------
    # 5) X-ray + marker flare
    # ----------------------------------------------------------
    if ts_xray:
        t_min = df['Timestamp'].iloc[0]
        t_max = df['Timestamp'].iloc[-1]
        ts_xray_f = [t for t in ts_xray if t_min <= t <= t_max]
        flux_xray_f = [f for t, f in zip(ts_xray, flux_xray) if t_min <= t <= t_max]
        if ts_xray_f:
            axes[4].plot(ts_xray_f, flux_xray_f, color='orange', linewidth=0.7)
            axes[4].set_yscale('log')
    axes[4].set_ylabel('X-ray GOES (W/m²)', color='orange')
    axes[4].set_xlabel('Ora UTC', fontsize=11)
    axes[4].grid(True, alpha=0.3, which='both')
    axes[4].tick_params(axis='y', colors='orange')

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=timezone.utc))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone.utc))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15, 30, 45],
                                                        tz=timezone.utc))
    axes[4].set_xlim(df['Timestamp'].iloc[0], df['Timestamp'].iloc[-1])

    def colore_per_risultato(ris):
        if ris is None:
            return 'gray'
        return ris.get('colore_cat', 'gray')

    ris_by_picco = {}
    for ris in risultati:
        if ris and ris.get('picco_dt') is not None:
            ris_by_picco[ris['picco_dt']] = ris

    if flares:
        for flare in flares:
            picco = parse_flare_datetime(flare['picco'])
            if picco is None:
                continue
            classe = flare['classe']
            ris = ris_by_picco.get(picco)
            colore = colore_per_risultato(ris)

            for ax in axes:
                ax.axvline(picco, color=colore, linestyle='--',
                           linewidth=1.2, alpha=0.7)
            axes[4].annotate(classe, xy=(picco, axes[4].get_ylim()[1]),
                             xytext=(4, -12), textcoords='offset points',
                             fontsize=10, color=colore, fontweight='bold')

    stato_filtro = "ON" if (hasattr(fig, '_fd') and fig._fd.attivo) else "OFF"
    axes[4].text(
        0.01, 0.10,
        f"[R] Refresh   [I] Cambia data   [P] Correlazione   [V] PNG eventi   "
        f"[F] Filtro [{stato_filtro}]   [T] Detrend [OFF]   "
        f"[U] Scala [{'LIN' if scala_lineare_prec else 'dB'}]   [K] Check PLL  [Q] Esci",
        transform=axes[4].transAxes,
        ha='left', va='bottom',
        fontsize=8, family='monospace',
        color='#333333',
        bbox=dict(boxstyle='round,pad=0.35',
                  facecolor='#F8F8F8',
                  edgecolor='#AAAAAA',
                  linewidth=0.8,
                  alpha=0.92)
    )

    fig.tight_layout(rect=[0, 0.02, 1, 1])

    path_png = os.path.join(CARTELLA_FLARES, f'analisi_doppler_{DATA}.png')
    return fig

# ============================================================
# PLOT PER SINGOLO EVENTO
# ============================================================
def plot_evento(df, flare_dt, risultato, out_path):
    """
    Salva un PNG con 4 pannelli per un singolo flare:
      1) Power_dB + finestre quiete/evento
      2) Doppler fase (Hz) + finestre
      3) PLL_Freq (Hz detrended) + finestre
      4) X-ray GOES (log) se disponibile
    """
    t_peak = flare_dt
    t_base_a = t_peak - timedelta(seconds=600)
    t_base_b = t_peak - timedelta(seconds=120)
    t_ev_a   = t_peak - timedelta(seconds=30)
    t_ev_b   = t_peak + timedelta(seconds=180)

    t_zoom_a = t_base_a - timedelta(minutes=3)
    t_zoom_b = t_ev_b + timedelta(minutes=5)

    mask_zoom = ((df['Timestamp'] >= t_zoom_a) &
                 (df['Timestamp'] <= t_zoom_b))
    df_z = df[mask_zoom]

    if len(df_z) < 30:
        return

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)

    # --- Pannello 1: Power ---
    axes[0].plot(df_z['Timestamp'], df_z['Power_dB'],
                 color='green', linewidth=0.7, label='Power_dB')
    axes[0].axvspan(t_base_a, t_base_b, alpha=0.15, color='green',
                    label='Baseline')
    axes[0].axvspan(t_ev_a, t_ev_b, alpha=0.15, color='red',
                    label='Evento')
    axes[0].axvline(t_peak, color='k', linestyle='--', linewidth=0.8,
                    label='Picco GOES')
    axes[0].set_ylabel('Power (dB)', color='green')
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc='upper left', fontsize=8)

    # --- Pannello 2: Doppler fase (Hz) ---
    fase_raw = df_z['Phase_Deg'].values.astype(float).copy()
    disc_z = df_z['Discontinuita'].values if 'Discontinuita' in df_z.columns else None
    if disc_z is not None:
        fase_raw[disc_z == 1] = np.nan
    fase_for_grad = pd.Series(fase_raw).interpolate(
        method='linear', limit_direction='both').bfill().ffill().values
    d_fase_hz = np.gradient(fase_for_grad) / 360.0
    d_fase_hz = d_fase_hz - np.nanmedian(d_fase_hz)
    axes[1].plot(df_z['Timestamp'], d_fase_hz,
                 color='blue', linewidth=0.7, label='Doppler fase')
    axes[1].axvspan(t_base_a, t_base_b, alpha=0.15, color='green')
    axes[1].axvspan(t_ev_a, t_ev_b, alpha=0.15, color='red')
    axes[1].axvline(t_peak, color='k', linestyle='--', linewidth=0.8)
    axes[1].set_ylabel('Doppler fase (Hz)', color='blue')
    axes[1].grid(alpha=0.3)

    # --- Pannello 3: PLL_Freq ---
    axes[2].plot(df_z['Timestamp'], df_z['PLL_Freq_Hz'],
                 color='purple', linewidth=0.7, label='PLL_Freq')
    axes[2].axvspan(t_base_a, t_base_b, alpha=0.15, color='green')
    axes[2].axvspan(t_ev_a, t_ev_b, alpha=0.15, color='red')
    axes[2].axvline(t_peak, color='k', linestyle='--', linewidth=0.8)
    axes[2].set_ylabel('PLL_Freq (Hz)', color='purple')
    axes[2].grid(alpha=0.3)

    # --- Pannello 4: X-ray (usa la variabile globale PATH_XRAY) ---
    xray_ts, xray_flux = [], []
    if os.path.exists(PATH_XRAY):
        xray_ts, xray_flux = leggi_xray_locale(PATH_XRAY)
    if xray_ts:
        t_min = df_z['Timestamp'].iloc[0]
        t_max = df_z['Timestamp'].iloc[-1]
        ts_f = [t for t in xray_ts if t_min <= t <= t_max]
        fl_f = [f for t, f in zip(xray_ts, xray_flux) if t_min <= t <= t_max]
        if ts_f:
            axes[3].plot(ts_f, fl_f, color='orange', linewidth=0.8)
            axes[3].set_yscale('log')
            axes[3].axvline(t_peak, color='k', linestyle='--', linewidth=0.8)
    axes[3].axvspan(t_base_a, t_base_b, alpha=0.15, color='green')
    axes[3].axvspan(t_ev_a, t_ev_b, alpha=0.15, color='red')
    axes[3].set_ylabel('X-ray (W/m²)', color='orange')
    axes[3].grid(alpha=0.3, which='both')
    axes[3].set_xlabel('Ora UTC')

    # Formattazione asse X
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2, tz=timezone.utc))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone.utc))

    # Titolo
    cat = risultato.get('categoria', '?')
    p_fase = risultato.get('p_fase')
    p_pll  = risultato.get('p_pll')
    titolo = (f"Flare {risultato['classe']} @ {t_peak.strftime('%H:%M:%S')} UTC  |  "
              f"Verdetto: {cat}")
    if p_fase is not None:
        titolo += f"  |  p_fase={p_fase:.3f}"
    if p_pll is not None:
        titolo += f"  |  p_pll={p_pll:.3f}"

    # --- Riga frozen v1.0 (se disponibile) ---
    verd_fz = risultato.get('frozen_verdetto')
    if verd_fz is not None:
        corr_fz = risultato.get('frozen_corr', float('nan'))
        p_A = risultato.get('frozen_p_A', float('nan'))
        p_B = risultato.get('frozen_p_B', float('nan'))
        titolo += (f"  |  FRZ: {verd_fz} "
                   f"(|c|={abs(corr_fz):.3f}, "
                   f"p_A={p_A:.3f}, p_B={p_B:.3f})")

    fig.suptitle(titolo, fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

def plot_correlazione_filtrata(df, ts_xray, flux_xray, flares, risultati,
                                fig=None, idx_flare=0):
    """
    Finestra di correlazione paginata: un flare alla volta.
    Naviga su `flares` (tutti i flare rilevati). Per quelli con un
    risultato in `risultati` mostra le info di correlazione; per gli
    altri mostra solo le curve, con dicitura "NON ANALIZZATO".
    Tasti: [A] precedente, [D] successivo, [Q] chiudi.
    """
    # --- Setup figura: registrala SEMPRE in pyplot (TkAgg) ---
    if fig is None:
        fig = plt.figure(figsize=(14, 10))
        plt.figure(fig.number)
    else:
        plt.figure(fig.number)
        fig.clf()

    # --- Ramo "nessun flare" ---
    if not flares:
        fig.suptitle('Nessun flare da mostrare', fontsize=12,
                     fontweight='bold', color='#FF6347')
        fig.text(0.5, 0.5,
                 "Nessun flare nella giornata selezionata.\n"
                 "Premi [Q] per chiudere.",
                 ha='center', va='center', fontsize=11,
                 family='monospace', color='#333')
        if not hasattr(fig, '_corr_key_connected'):
            fig.canvas.mpl_connect('key_press_event',
                                    lambda e: _on_key_correlazione(e, fig))
            fig._corr_key_connected = True
        return fig

    # --- Lookup risultato per picco_dt ---
    ris_by_pk = {}
    for r in (risultati or []):
        pk_r = r.get('picco_dt')
        if pk_r is not None:
            ris_by_pk[pk_r] = r

    # --- Clamp indice e riferimenti al flare corrente ---
    n_fl = len(flares)
    idx_flare = max(0, min(idx_flare, n_fl - 1))
    flare_corrente = flares[idx_flare]
    pk = parse_flare_datetime(flare_corrente.get('picco', ''))
    classe_fl = flare_corrente.get('classe', '?')
    ora_fl = pk.strftime('%H:%M') if pk else '??:??'
    r = ris_by_pk.get(pk, None)   # None = non analizzato

    # --- Layout: overview + pannello A + pannello B ---
    n_rows = 4
    fig.suptitle(
        f'Analisi flare {idx_flare + 1}/{n_fl}  —  '
        f'[A] precedente  [D] successivo  [Q] chiudi',
        fontsize=12, fontweight='bold'
    )

    # --- FUNZIONE DI NORMALIZZAZIONE ROBUSTA (MAD) ---
    # Ignora i picchi estremi (come il transiente iniziale delle 00:00)
    def _norm_robusta(arr):
        med = np.nanmedian(arr)
        mad = 1.4826 * np.nanmedian(np.abs(arr - med))
        if mad == 0 or not np.isfinite(mad):
            mad = 1e-12
        return (arr - med) / mad

    # ---------- Prepara Doppler filtrato ----------
    fase_raw = df['Phase_Deg'].values.astype(float).copy()
    disc_mask = (df['Discontinuita'].values == 1) \
                if 'Discontinuita' in df.columns else None
    if disc_mask is not None:
        fase_raw[disc_mask] = np.nan
    fase_interp = pd.Series(fase_raw).interpolate(
        method='linear', limit_direction='both').bfill().ffill().values
    d_fase_hz = np.gradient(fase_interp) / 360.0
    d_fase_hz = filtro_hampel(d_fase_hz, window=7, n_sigma=3.0)
    d_fase_hz = d_fase_hz - np.nanmedian(d_fase_hz)
    fase_smooth = median_filter(d_fase_hz, size=SMOOTH_FASE_SEC)

    fase_smooth = sanity_doppler_con_durata(fase_smooth, soglia_hz=0.020, min_durata_s=60)

    if disc_mask is not None:
        fase_smooth = fase_smooth.copy()
        fase_smooth[disc_mask] = np.nan

    fd = FiltroDoppler(t_lento=2400, t_veloce=120, fs=1.0)
    fd.calcola(fase_smooth)
    fase_f = (fd._filtrato.copy() if fd._filtrato is not None
              else fase_smooth.copy())
    if disc_mask is not None:
        fase_f[disc_mask] = np.nan

    # ---------- Prepara potenza detrenduta ----------
    pwr_raw = df['Power_dB'].values.astype(float).copy()
    if disc_mask is not None:
        pwr_raw[disc_mask] = np.nan
    pwr_interp = pd.Series(pwr_raw).interpolate(
        method='linear', limit_direction='both').bfill().ffill().values
    pwr_smooth = uniform_filter1d(pwr_interp, size=SMOOTH_AMPIEZZA_SEC,
                                  mode='nearest')
    pwr_det = pwr_smooth
    pwr_f = pwr_det.copy()

    if disc_mask is not None:
        pwr_f[disc_mask] = np.nan

    # ---------- Prepara fase integrata (per pannello C) ----------
    if fase_f is not None and len(fase_f) > 0:
        fase_f_clean = np.where(np.isfinite(fase_f), fase_f, 0.0)

        # AZZERA IL WARMUP INIZIALE e RIMUOVI L'OFFSET DC
        fase_f_clean[:600] = 0.0
        if disc_mask is not None:
            fase_f_clean[disc_mask] = 0.0
        fase_f_clean = fase_f_clean - np.nanmean(fase_f_clean)

        fase_integrata = np.cumsum(fase_f_clean) * 1.0     # cicli
        fase_integrata = detrend_mediano(fase_integrata, 1200)
        # Smoothing 120 s (NaN -> 0, poi smoothing, poi ri-NaN)
        fase_integrata = uniform_filter1d(
            np.where(np.isfinite(fase_integrata), fase_integrata, 0.0),
            size=120, mode='nearest')
        if disc_mask is not None:
            fase_integrata = fase_integrata.copy()
            fase_integrata[disc_mask] = np.nan
    else:
        fase_integrata = np.full(len(df), np.nan)

    # ---------- Prepara X-ray: flusso detrenduto + gradiente filtrato ----------
    xray_grad = None
    xray_flux_det = None
    if ts_xray and flux_xray:
        t_xray = np.array([t.timestamp() for t in ts_xray])
        fx = np.array(flux_xray)
        t_tr = np.array([t.timestamp() for t in df['Timestamp']])
        logflux = np.interp(t_tr, t_xray, np.log10(fx + 1e-12))
        lf_s = pd.Series(logflux).rolling(60, center=True,
                                           min_periods=1).mean().values
        xg = np.gradient(lf_s)
        if len(xg) >= 4 * 60:
            pad_x = min(len(xg) // 4, 3600)
            xp_x = np.pad(xg, pad_x, mode='edge')
            xray_grad = sosfiltfilt(
                butter(2, [1/2400, 1/120], btype='band', fs=1.0, output='sos'),
                xp_x
            )[pad_x:-pad_x]
        else:
            xray_grad = xg
        xray_flux_det = detrend_mediano(lf_s, 3600)

    # ---------- Layout ----------
    axes = fig.subplots(n_rows, 1, sharex=False,
                        gridspec_kw={'height_ratios': [1.3, 1.3, 1.0, 1.0]})

    # ---------- Overview ----------
    ax0 = axes[0]

    f_n = _norm_robusta(fase_f)
    ax0.plot(df['Timestamp'], f_n, color='blue', linewidth=0.6,
             alpha=0.7, label='Doppler (norm.)')

    p_n = _norm_robusta(pwr_f)
    ax0.plot(df['Timestamp'], p_n, color='green', linewidth=0.6,
             alpha=0.7, label='Potenza (norm.)')
    for flare in flares:
        pkk = parse_flare_datetime(flare['picco'])
        if pkk is not None:
            ax0.axvline(pkk, color='red', linestyle='--',
                        linewidth=0.7, alpha=0.5)
            ax0.annotate(flare.get('classe', '?'),
                         xy=(pkk, ax0.get_ylim()[1]),
                         xytext=(3, -10), textcoords='offset points',
                         fontsize=7, color='red', fontweight='bold',
                         rotation=90, va='top')
    ax0.set_ylabel('Norm.', fontsize=8)
    ax0.set_title('Giornata intera', fontsize=10)
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc='upper right', fontsize=7, framealpha=0.85)

    # ---------- Pannello A: Potenza + Flusso X-ray ----------
    ax_a = axes[1]
    if pk is not None:
        t_a = pk - timedelta(seconds=FINESTRA_FLARE_SEC)
        t_b = pk + timedelta(seconds=FINESTRA_FLARE_SEC)
        mask = (df['Timestamp'] >= t_a) & (df['Timestamp'] <= t_b)

        if mask.any():
            t_win = df['Timestamp'][mask]
            p_win = pwr_f[mask.values]
            p_norm = _norm_robusta(p_win)
            ax_a.plot(t_win, p_norm, color='green', linewidth=1.0,
                      label='Potenza (norm.)')
            if xray_flux_det is not None:
                x_flux = xray_flux_det[mask.values]
                xf_norm = (x_flux - np.nanmean(x_flux)) / (np.nanstd(x_flux) + 1e-12)
                ax_a.plot(t_win, xf_norm, color='orange', linewidth=1.0,
                          alpha=0.85, label='Log-flusso X-ray (norm.)')
            ax_a.axvline(pk, color='red', linestyle='--',
                         linewidth=0.9, alpha=0.8)
            ax_a.annotate(f"{classe_fl} (picco X-ray)",
                          xy=(pk, ax_a.get_ylim()[1]),
                          xytext=(5, -10), textcoords='offset points',
                          fontsize=8, color='red', fontweight='bold',
                          va='top')

    if r is not None:
        dpp = r.get('delta_pwr_picco')
        spp = r.get('snr_pwr_picco')
        info_a = (f"  picco={dpp:+.2f} dB  SNR={spp:+.1f}"
                  if dpp is not None and spp is not None else "")
        titolo_a = (f"{ora_fl}  {classe_fl}  —  "
                    f"{r.get('categoria', '?')}{info_a}")
    else:
        ok_elev, elev = flare_ha_sid_atteso(flare_corrente)
        if elev is not None and elev < 0.0:
            titolo_a = (f"{ora_fl}  {classe_fl}  —  NON ANALIZZATO "
                        f"(notte, elevazione {elev:.1f}°, D-layer assente)")
        elif not ok_elev:
            soglia_classe = soglia_elevazione_per_classe(classe_fl)
            titolo_a = (f"{ora_fl}  {classe_fl}  —  NON ANALIZZATO "
                        f"(elevazione {elev:.1f}° < {soglia_classe:.1f}°)")
        else:
            titolo_a = (f"{ora_fl}  {classe_fl}  —  NON ANALIZZATO "
                        f"(dati insufficienti o Reality Check X-ray fallito)")

    # Invece del titolo classico (che sfora in alto), scriviamo il testo DENTRO il pannello
    ax_a.text(0.01, 0.95, titolo_a, 
              transform=ax_a.transAxes, 
              ha='left', va='top', 
              fontsize=9, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#AAAAAA', alpha=0.9))
    
    ax_a.set_ylabel('Norm.', fontsize=8)
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(loc='upper right', fontsize=7, framealpha=0.85)

    # ---------- Pannello B: Doppler + Gradiente X-ray ----------
    ax_b = axes[2]
    if pk is not None:
        t_a = pk - timedelta(seconds=FINESTRA_FLARE_SEC)
        t_b = pk + timedelta(seconds=FINESTRA_FLARE_SEC)
        mask = (df['Timestamp'] >= t_a) & (df['Timestamp'] <= t_b)

        if mask.any():
            t_win = df['Timestamp'][mask]
            f_win = fase_f[mask.values]
            f_norm = _norm_robusta(f_win)
            ax_b.plot(t_win, f_norm, color='blue', linewidth=1.0,
                      label='Doppler filtrato (norm.)')
            if xray_grad is not None:
                x_grad = xray_grad[mask.values]
                xg_norm = (x_grad - np.nanmean(x_grad)) / (np.nanstd(x_grad) + 1e-12)
                ax_b.plot(t_win, xg_norm, color='orange', linewidth=1.0,
                          alpha=0.85, label='Grad. X-ray filtrato (norm.)')
            ax_b.axvline(pk, color='red', linestyle='--',
                         linewidth=0.9, alpha=0.8)
            ax_b.annotate(f"{classe_fl}",
                          xy=(pk, ax_b.get_ylim()[1]),
                          xytext=(5, -10), textcoords='offset points',
                          fontsize=8, color='red', fontweight='bold',
                          va='top')

    ax_b.set_ylabel('Norm.', fontsize=8)
    ax_b.set_xlabel('Ora UTC', fontsize=9)
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(loc='upper right', fontsize=7, framealpha=0.85)

    if r is not None:
        cf = r.get('corr_fase_filtrata')
        lf = r.get('lag_fase_filtrata')
        testo_b = ""
        if cf is not None and lf is not None:
            testo_b = f"Doppler-grad(X):  corr={cf:+.2f}  lag={lf:+d}s"

        bsnr = r.get('bump_snr_locale')
        bamp = r.get('bump_ampiezza_mhz')
        bstd = r.get('bump_std_quieta')
        bsu  = r.get('bump_forma_su')

        if bsnr is not None and bamp is not None and bstd is not None:
            testo_b += (f"\nBump locale: {bamp:.3f} mHz  "
                        f"(std_quieta={bstd:.3f} mHz)  "
                        f"SNR_locale={bsnr:.2f}")
            if bsu is not None:
                testo_b += f"  forma={'su' if bsu else 'giu'}"

            if bsnr > 3:
                 testo_b += "  [segnale forte]"
            elif bsnr > 2:
                 testo_b += "  [segnale debole]"
            else:
                 testo_b += "  [sotto soglia]"

        if testo_b:
            if bsnr is not None and bsnr > 3:
                edge_b = '#00AA00'
            elif bsnr is not None and bsnr > 2:
                edge_b = '#FFD700'
            else:
                edge_b = '#888888'
            ax_b.text(0.01, 0.95, testo_b,
                      transform=ax_b.transAxes, ha='left', va='top',
                      fontsize=7, family='monospace',
                      bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='white', edgecolor=edge_b,
                                linewidth=1.2, alpha=0.9))

    # ---------- Pannello C: Fase integrata + Flusso X-ray ----------
    ax_c = axes[3]
    if pk is not None:
        t_a = pk - timedelta(seconds=FINESTRA_FLARE_SEC)
        t_b = pk + timedelta(seconds=FINESTRA_FLARE_SEC)
        mask = (df['Timestamp'] >= t_a) & (df['Timestamp'] <= t_b)

        if mask.any():
            t_win = df['Timestamp'][mask]
            f_int_win = fase_integrata[mask.values]

            # Normalizza la fase integrata (robusta MAD)
            fi_norm = _norm_robusta(f_int_win)

            ax_c.plot(t_win, fi_norm, color='purple', linewidth=1.0,
                      label='Fase integrata (norm.)')

            if xray_flux_det is not None:
                x_flux = xray_flux_det[mask.values]
                xf_norm = _norm_robusta(x_flux)
                ax_c.plot(t_win, xf_norm, color='orange', linewidth=1.0,
                          alpha=0.85, label='Log-flusso X-ray (norm.)')

            ax_c.axvline(pk, color='red', linestyle='--',
                         linewidth=0.9, alpha=0.8)
            ax_c.annotate(f"{classe_fl}",
                          xy=(pk, ax_c.get_ylim()[1]),
                          xytext=(5, -10), textcoords='offset points',
                          fontsize=8, color='red', fontweight='bold',
                          va='top')

            if r is not None:
                snr_int = r.get('snr_fase_int')
                dfase_int_deg = r.get('delta_fase_int_deg')
                forma_int = r.get('forma_su_int')

                if snr_int is not None and dfase_int_deg is not None:
                    forma_str = 'su' if forma_int else 'giu'
                    sigma_int = r.get('std_quiete_gradi', None)
                    sigma_str = f"  σ={sigma_int:.2f}°" if sigma_int is not None else ""
                    testo_c = (f"FASE INTEGRATA: Δφ/σ={snr_int:.1f}  "
                               f"Δφ={dfase_int_deg:+.1f}°{sigma_str}  "
                               f"forma={forma_str}")

                    if snr_int > SNR_FASE_INT_MIN and forma_int:
                        edge_c = '#00AA00'
                    else:
                        edge_c = '#888888'

                    ax_c.text(0.01, 0.95, testo_c,
                              transform=ax_c.transAxes, ha='left', va='top',
                              fontsize=8, family='monospace', fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor='white', edgecolor=edge_c,
                                        linewidth=1.2, alpha=0.9))

    ax_c.set_ylabel('Norm.', fontsize=8)
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(loc='upper right', fontsize=7, framealpha=0.85)

    for ax in axes:
        ax.xaxis.set_major_locator(
            mdates.HourLocator(interval=1, tz=timezone.utc))
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter('%H:%M', tz=timezone.utc))

    # ---------- Riepilogo completo ----------
    righe = ["RIEPILOGO FLARE:"]
    for i_fl, fl in enumerate(flares):
        pk_f = parse_flare_datetime(fl['picco'])
        rf = ris_by_pk.get(pk_f, None)
        ora_f = pk_f.strftime('%H:%M') if pk_f else '??:??'
        classe_f = fl.get('classe', '?')

        if rf is not None:
            icona = {
                'SID_COMPLETO':       '✓✓',
                'SID_REALE':          '✓',
                'SID_POTENZA':        '✓',
                'SID_FASE':           '✓',
                'SID_POTENZA_DEBOLE': '◐',
                'CANDIDATO_DEBOLE':   '◐',
                'DERIVA_TID':         '◐',
                'ARTEFATTO':          '⚠',
                'NESSUN_EVENTO':      '○',
                'INCERTO':            '◑',
                'SID_DOPPLER_ABBINATO': '✓',
            }.get(rf.get('categoria', ''), '?')

            dpp_f = rf.get('delta_pwr_picco')
            spp_f = rf.get('snr_pwr_picco')
            parti = []
            if dpp_f is not None and spp_f is not None and not np.isnan(dpp_f) and not np.isnan(spp_f):
                parti.append(f"picco={dpp_f:+.2f}dB(SNR={spp_f:+.1f})")
            else:
                parti.append("picco=NaN")
            dettaglio = "  ".join(parti) if parti else "—"
            cat_f = rf.get('categoria', '?')
        else:
            ok_elev, elev = flare_ha_sid_atteso(fl)
            if elev is not None and elev < 0.0:
                icona = '🌙'
                cat_f = 'NOTTE'
                dettaglio = f"elev={elev:.1f}°"
            elif not ok_elev:
                icona = '☁'
                cat_f = 'TRAN.IONOSF.'
                dettaglio = f"elev={elev:.1f}°"
            else:
                icona = '·'
                cat_f = 'NON ANALIZZATO'
                dettaglio = "(dati insuff./reality check)"

        marker = "►" if i_fl == idx_flare else " "
        righe.append(
            f"{marker} {icona} {ora_f}  {classe_f:<6s}  "
            f"{cat_f:<20s}  {dettaglio}"
        )

    cats = [ris_by_pk.get(parse_flare_datetime(fl['picco']), {})
            .get('categoria', 'NON_ANALIZZATO') for fl in flares]
    if 'ARTEFATTO' in cats:
        edge = '#FF0000'
    elif 'SID_COMPLETO' in cats:
        edge = '#00CC00'
    elif ('SID_REALE' in cats or 'SID_POTENZA' in cats or
          'SID_FASE' in cats):
        edge = '#00AA00'
    elif 'SID_POTENZA_DEBOLE' in cats or 'CANDIDATO_DEBOLE' in cats:
        edge = '#FFD700'
    elif 'DERIVA_TID' in cats:
        edge = '#FF8C00'
    else:
        edge = '#888888'

    fig.text(
        0.02, 0.005, "\n".join(righe),
        ha='left', va='bottom',
        fontsize=8, family='monospace',
        color='#222222',
        bbox=dict(boxstyle='round,pad=0.4',
                  facecolor='#F8F8F8', edgecolor=edge,
                  linewidth=1.2, alpha=0.95)
    )

    # --- Legenda sintetica (Allineamento perfetto a rettangolo) ---
    legenda_righe = [
        ("LEGENDA (Canali di classificazione):", ""),
        ("", ""),
        ("[OK] SID_COMPLETO", "Potenza + Fase Doppler"),
        ("[OK] SID_REALE", "Doppler/PLL correlato X-ray"),
        ("[OK] SID_POTENZA", "Solo Potenza"),
        ("[OK] SID_FASE", "Solo Fase Doppler"),
        ("[OK] SID_DOPPLER_ABB.", "Matched Doppler (Lock basso)"),
        ("[?]  SID_POTENZA_DEB.", "Solo Potenza (SNR marginale)"),
        ("[?]  CANDIDATO_DEBOLE", "Bump marginale"),
        ("[?]  DERIVA_TID", "Fase senza Doppler"),
        ("[!]  ARTEFATTO", "Lock perso / dati inaffidabili"),
        ("[-]  NESSUN_EVENTO", "Nulla di misurabile"),
        ("[?]  INCERTO", "Dati ambigui"),
        ("[N]  NOTTE", "Sole sotto l'orizzonte"),
        ("[T]  TRAN.IONOSF.", "Sole < 20° (transizione)")
    ]
    
    max_left = max(len(r[0]) for r in legenda_righe[2:])
    max_right = max(len(r[1]) for r in legenda_righe[2:])
    
    righe_formattate = [legenda_righe[0][0], legenda_righe[1][0]]
    for sinistra, destra in legenda_righe[2:]:
        righe_formattate.append(f"{sinistra.ljust(max_left)} : {destra.ljust(max_right)}")
        
    legenda_cat = "\n".join(righe_formattate)
    
    fig.text(
        0.98, 0.005, legenda_cat,
        ha='right', va='bottom',
        fontsize=8, family='monospace',
        color='#333333',
        bbox=dict(boxstyle='round,pad=0.4',
                  facecolor='#F8F8F8', edgecolor='#AAAAAA',
                  linewidth=1.0, alpha=0.95)
    )

    n_fl_tot = len(flares)
    fraz_bottom = min(0.12 + 0.022 * n_fl_tot, 0.50)
    fig.subplots_adjust(left=0.08, right=0.97,
                        top=0.92, bottom=fraz_bottom,
                        hspace=0.55)

    fig._corr_state = {
        'df': df,
        'ts_xray': ts_xray,
        'flux_xray': flux_xray,
        'flares': flares,
        'risultati': risultati,
        'idx_flare': idx_flare,
    }

    if not hasattr(fig, '_corr_key_connected'):
        fig.canvas.mpl_connect('key_press_event',
                                lambda e: _on_key_correlazione(e, fig))
        fig._corr_key_connected = True

    return fig

def _on_key_correlazione(event, fig):
    import traceback
    if event.key is None:
        return
    key = event.key.lower()
    print(f"[DEBUG-CORR] tasto={key!r}")

    if key == 'q':
        plt.close(fig)
        return

    state = getattr(fig, '_corr_state', None)
    if state is None:
        print("[DEBUG-CORR] state=None, nessuno stato salvato")
        return

    n = len(state.get('flares', []))
    print(f"[DEBUG-CORR] n_flares={n} idx={state.get('idx_flare')}")
    if n == 0:
        return

    idx = state['idx_flare']
    nuovo_idx = idx
    if key == 'a':
        nuovo_idx = idx - 1
    elif key == 'd':
        nuovo_idx = idx + 1
    else:
        return

    if nuovo_idx < 0 or nuovo_idx >= n:
        print(f"[DEBUG-CORR] nuovo_idx={nuovo_idx} fuori range [0,{n-1}]")
        return

    print(f"[DEBUG-CORR] ridisegno flare {nuovo_idx + 1}/{n}")
    try:
        plot_correlazione_filtrata(
            state['df'], state['ts_xray'], state['flux_xray'],
            state['flares'], state['risultati'],
            fig=fig, idx_flare=nuovo_idx
        )
        fig.canvas.draw_idle()
        print("[DEBUG-CORR] ridisegno OK")
    except Exception as e:
        print(f"[DEBUG-CORR] ERRORE: {e}")
        traceback.print_exc()

def cache_valida(path, data_str):
    """
    Restituisce True SOLO se:
      - data_str è più vecchia di GIORNI_REFRESH_CACHE giorni
      - il file esiste
      - il file è leggibile e non vuoto

    Per date recenti (oggi + GIORNI_REFRESH_CACHE giorni fa) restituisce
    sempre False, forzando il refresh da internet. Se internet non è
    disponibile, le funzioni chiamanti (scarica_xray / scarica_flares)
    usano la cache esistente come fallback.

    Se la cache esiste ma è vuota/corrotta, la cancella.
    """
    if e_recente(data_str, GIORNI_REFRESH_CACHE):
        return False

    oggi_utc = datetime.now(timezone.utc).date()
    try:
        d_req = datetime.strptime(data_str, "%Y-%m-%d").date()
    except Exception:
        return False
    if d_req > oggi_utc:
        if os.path.exists(path):
            return _file_non_vuoto(path)
        return False

    if not os.path.exists(path):
        return False

    if _file_non_vuoto(path):
        return True
    else:
        print(f"[CACHE] File corrotto o vuoto, lo elimino: {path}")
        try:
            os.remove(path)
        except Exception as e:
            print(f"[CACHE] Impossibile eliminare: {e}")
        return False
    
def _file_non_vuoto(path):
    """Ritorna True se il file ha contenuto utile (non solo header/vuoto)."""
    try:
        with open(path, 'r') as f:
            contenuto = f.read().strip()
        righe_utili = [r for r in contenuto.splitlines()
                       if r.strip() and not r.strip().startswith('#')]
        return len(righe_utili) > 0
    except Exception:
        return False


def valuta_affidabilita(lag_sec, corr_max):
    """
    Restituisce una stringa di affidabilità basata su lag e correlazione.
    """
    if lag_sec is None or corr_max is None:
        return "N/D"
    lag_al_bordo = abs(lag_sec) >= LAG_MAX_SEC - 5
    corr_bassa = abs(corr_max) < 0.6
    if lag_al_bordo:
        return "SCARSA (lag al bordo)"
    if corr_bassa:
        return "DEBOLE (corr bassa)"
    return "OK"

def pulisci_fase(df, colonna='Phase_Deg'):
    """
    Fase con NaN nelle zone di discontinuità (inclusi i 10 min di warmup
    dopo ogni riavvio). Interpola solo buchi brevi (< 10 campioni).
    I buchi lunghi rimangono NaN: nel plot appariranno come linee interrotte.
    """
    fase = df[colonna].values.astype(float).copy()
    if 'Discontinuita' in df.columns:
        disc = df['Discontinuita'].values
        fase[disc == 1] = np.nan
    fase = pd.Series(fase).interpolate(
        method='linear', limit=10, limit_direction='both'
    ).values
    return fase

# ============================================================
# DIAGNOSTICA DETTAGLIATA PER FLARE
# ============================================================
def stampa_diagnostica_dettagliata(r):
    """
    Stampa una tabella con tutti i criteri di classificazione per un flare.
    Serve a capire PERCHÉ il classificatore ha assegnato una certa categoria.
    """
    print()
    print(f"  --- Diagnostica per {r['classe']} @ {r['ora']} UTC ---")
    print(f"  {'Criterio':<22} | {'Valore':>12} | {'Soglia':>16} | Esito")
    print(f"  {'-'*22}-+-{'-'*12}-+-{'-'*16}-+-------")

    dpm = r.get('delta_pwr_media')
    if dpm is not None:
        esito = "OK" if abs(dpm) > PWR_SID_MIN_DB else "Sotto soglia"
        print(f"  {'Δpwr_media':<22} | {dpm:>+11.2f} dB | "
              f"{'|x| > ' + str(PWR_SID_MIN_DB) + ' dB':>16} | {esito}")

    dpp = r.get('delta_pwr_picco')
    spp = r.get('snr_pwr_picco')
    soglia_din = r.get('soglia_picco_dinamica', PWR_SID_PICCO_MIN_DB)
    std_loc = r.get('std_ref_pre')
    if dpp is not None:
        esito = "OK" if abs(dpp) > soglia_din else "Sotto soglia"
        nota_soglia = f"|x| > {soglia_din:.1f} dB"
        print(f"  {'Δpwr_picco':<22} | {dpp:>+11.2f} dB | "
              f"{nota_soglia:>16} | {esito}")
    if std_loc is not None:
        print(f"  {'std rumore pre-evento':<22} | {std_loc:>+11.2f} dB | "
              f"{'(dinamica)':>16} |")

    dt_picco = r.get('dt_picco_pwr_sec')
    if dt_picco is not None:
        esito = "OK" if -600 <= dt_picco <= 600 else "Disallineato"
        print(f"  {'Δt picco-flare':<22} | {dt_picco:>+9.0f} s | "
              f"{'±10 min':>16} | {esito}")

    spm = r.get('snr_pwr_media')
    if spm is not None:
        esito = "OK" if abs(spm) > PWR_SID_MIN_SNR else "Sotto soglia"
        print(f"  {'SNR_pwr (forte)':<22} | {spm:>+12.2f} | "
              f"{'> ' + str(PWR_SID_MIN_SNR):>16} | {esito}")

    if spm is not None:
        esito = "OK" if abs(spm) > PWR_SID_WEAK_SNR else "Sotto soglia"
        print(f"  {'SNR_pwr (debole)':<22} | {spm:>+12.2f} | "
              f"{'> ' + str(PWR_SID_WEAK_SNR):>16} | {esito}")

    dpll = r.get('delta_pll')
    dfase = r.get('delta_fase')
    if dpll is not None:
        if dfase is not None and dfase > 1e-6:
            rapporto = dpll / dfase
            esito = "Rumoroso" if rapporto > DPLL_RAPPORTO_MAX else "OK"
            print(f"  {'ΔPLL':<22} | {dpll:>+11.4f} Hz | "
                  f"{'sanity check':>16} | {esito} (rapporto {rapporto:.0f}x)")
        else:
            print(f"  {'ΔPLL':<22} | {dpll:>+11.4f} Hz | "
                  f"{'sanity check':>16} | OK")

    dfase = r.get('delta_fase')
    soglia_din = r.get('d_fase_soglia', 0.001)
    range_base = r.get('range_baseline')
    std_base   = r.get('std_baseline')
    if dfase is not None:
        if dfase > DFASE_FORTE_HZ:
            esito = "Forte"
        elif dfase > soglia_din:
            esito = "Presente"
        else:
            esito = "Troppo piccolo"
        info_base = f" (soglia={soglia_din:.4f})"
        if range_base is not None:
            info_base += f" range={range_base:.4f}"
        if std_base is not None:
            info_base += f" std={std_base:.4f}"
        print(f"  {'Δfase':<22} | {dfase:>+11.4f} Hz | "
              f"{'> ' + f'{soglia_din:.4f}' + ' Hz':>16} | {esito}{info_base}")

    bump_snr = r.get('bump_snr_locale')
    bump_amp = r.get('bump_ampiezza_mhz')
    bump_std = r.get('bump_std_quieta')
    bump_su  = r.get('bump_forma_su')
    if bump_snr is not None and bump_amp is not None:
        if bump_snr > SNR_FASE_MIN:
            esito = "OK"
        elif bump_snr > SNR_FASE_MIN * 0.5:
            esito = "Marginale"
        else:
            esito = "Sotto soglia"
        forma_str = 'su' if bump_su else 'giu'
        print(f"  {'SNR_fase (bump)':<22} | {bump_snr:>+12.2f} | "
              f"{'> ' + str(SNR_FASE_MIN):>16} | {esito}")
        if bump_amp is not None:
            print(f"  {'Δp medio (fase)':<22} | {bump_amp:>+11.4f} mHz | "
                  f"{'sanity check':>16} | forma={forma_str}")
        if bump_std is not None:
            print(f"  {'std quieta (fase)':<22} | {bump_std:>+11.4f} mHz | "
                  f"{'(riferimento)':>16} |")
            
    cf = r.get('corr_fase')
    if cf is not None:
        cf_or = POLARITA_SID_DOPPLER * cf
        esito = "OK" if cf_or > CORR_MIN_SID else "Bassa"
        print(f"  {'corr_F':<22} | {cf:>+12.2f} | "
              f"{'> ' + str(CORR_MIN_SID):>16} | {esito} "
              f"(or={cf_or:+.2f})")
        
    cf_f = r.get('corr_fase_filtrata')
    lf_f = r.get('lag_fase_filtrata')

    bump_snr = r.get('bump_snr_locale')
    bump_snr_abs = abs(bump_snr) if bump_snr is not None else 0.0

    if cf_f is not None and lf_f is not None:
        cf_f_abs = abs(cf_f)
        esito_c = "OK" if cf_f_abs > 0.55 and bump_snr_abs > SNR_FASE_MIN else "Bassa/Ampia"
        esito_l = "OK" if abs(lf_f) < 240 else "Fuori range"
        print(f"  {'corr_F (filtrata)':<22} | {cf_f:>+12.2f} | "
              f"{'> 0.55 (e SNR>2.5)':>16} | {esito_c}")
        print(f"  {'lag_F (filtrata)':<22} | {lf_f:>+11d} s | "
              f"{'< 240 s':>16} | {esito_l}")

    cp = r.get('corr_pll')
    if cp is not None:
        esito = "OK" if abs(cp) > CORR_MIN_SID else "Bassa"
        print(f"  {'corr_P':<22} | {cp:>+12.2f} | "
              f"{'> ' + str(CORR_MIN_SID):>16} | {esito}")

    lf = r.get('lag_fase')
    if lf is not None:
        esito = "OK" if abs(lf) < LAG_COERENTE_SEC else "Fuori range"
        print(f"  {'lag_F':<22} | {lf:>+11d} s | "
              f"{'< ' + str(LAG_COERENTE_SEC) + ' s':>16} | {esito}")

    fi = r.get('frazione_invalida', 0.0)
    esito = "OK" if fi < FRAZIONE_INVALIDA_MAX else "Troppo alta"
    print(f"  {'frazione invalida':<22} | {fi*100:>11.1f} % | "
          f"{'< ' + str(int(FRAZIONE_INVALIDA_MAX*100)) + ' %':>16} | {esito}")

    cp = r.get('corr_pwr')
    lp = r.get('lag_pwr')
    if cp is not None and lp is not None:
        esito_c = "OK" if abs(cp) > 0.5 else "Bassa"
        esito_l = "OK" if abs(lp) < 180 else "Fuori range"
        print(f"  {'corr_pwr':<22} | {cp:>+12.2f} | "
              f"{'> 0.5':>16} | {esito_c}")
        print(f"  {'lag_pwr':<22} | {lp:>+11d} s | "
              f"{'< 180 s':>16} | {esito_l}")

    cp_f = r.get('corr_pwr_f')
    lp_f = r.get('lag_pwr_f')
    if cp_f is not None and lp_f is not None:
        cp_f_abs = abs(cp_f)
        esito_c = "OK" if cp_f_abs > 0.55 else "Bassa"
        esito_l = "OK" if abs(lp_f) < 240 else "Fuori range"
        print(f"  {'corr_pwr (filtrata)':<22} | {cp_f:>+12.2f} | "
            f"{'> 0.55':>16} | {esito_c} (|corr|={cp_f_abs:.2f})")
        print(f"  {'lag_pwr (filtrata)':<22} | {lp_f:>+11d} s | "
            f"{'< 240 s':>16} | {esito_l}")

    corr_fz = r.get('frozen_corr')
    p_A     = r.get('frozen_p_A')
    p_B     = r.get('frozen_p_B')
    verd_fz = r.get('frozen_verdetto')
    if corr_fz is not None and p_A is not None and p_B is not None:
        segno_txt = "negativo (SPA)" if corr_fz < 0 else "positivo"
        print(f"  {'--- FROZEN v1.0 ---':<22} |")
        print(f"  {'FRZ corr':<22} | {corr_fz:>+12.4f} | "
              f"{'(firmata)':>16} | {segno_txt}")
        esito_A = "OK" if p_A < 0.01 else "Sotto soglia"
        esito_B = "OK" if p_B < 0.05 else "Sotto soglia"
        print(f"  {'FRZ p_A':<22} | {p_A:>+12.4f} | "
              f"{'< 0.01':>16} | {esito_A}")
        print(f"  {'FRZ p_B':<22} | {p_B:>+12.4f} | "
              f"{'< 0.05':>16} | {esito_B}")
        print(f"  {'FRZ verdetto':<22} | {verd_fz:>12} |")

    mp = r.get('matched_doppler_pct')
    if mp is not None:
        mg = r.get('matched_doppler_n_gem', 0)
        mb = r.get('matched_doppler_bump_mhz')
        ma = r.get('matched_doppler_amp100')
        print(f"  {'--- MATCHED DOPPLER ---':<22} |")
        if mb is not None: print(f"  {'Bump evento':<22} | {mb:>+11.4f} mHz |")
        if ma is not None: print(f"  {'Amp100 evento':<22} | {ma:>+11.2f} dB  |")
        print(f"  {'N gemelle':<22} | {mg:>12d}   |")
        esito_m = "OK" if mp >= PERCENTILE_SOGLIA else "Sotto soglia"
        print(f"  {'Percentile':<22} | {mp:>+12.1f}  | {'≥ 95':>16} | {esito_m}")
        
    print(f"  {'-'*22}-+-{'-'*12}-+-{'-'*16}-+-------")
    print(f"  {'VERDETTO':<22} | {r['categoria']:>12}")
    print(f"  {r['spiegazione']}")
    print()

def converti_classe_in_flusso(classe_str):
    """Converte 'C9.3', 'M1.0', 'X2.0' in un valore numerico di flusso per il confronto."""
    if not classe_str:
        return 0.0
    c = classe_str[0].upper()
    try:
        val = float(classe_str[1:])
    except (ValueError, IndexError):
        val = 1.0
    if c == 'C': return val * 1e-6
    elif c == 'M': return val * 1e-5
    elif c == 'X': return val * 1e-4
    return 0.0

def esegui_analisi(data_utc, fig=None):
    """
    Esegue l'analisi completa per una data: legge CSV, scarica X-ray,
    rileva flare, analizza, aggiorna il plot.

    Parametri:
      - data_utc : datetime UTC (mezzanotte del giorno da analizzare)
      - fig      : figura matplotlib esistente (o None per crearne una nuova)

    Ritorna la figura matplotlib aggiornata.
    """

    global DATA, CSV_TRACKER, PATH_XRAY, PATH_FLARES

    DATA = data_utc.strftime("%Y-%m-%d")

    CSV_TRACKER = trova_csv(DATA)
    if CSV_TRACKER is None:
        CSV_TRACKER = os.path.join(CARTELLE_CSV[0],
                                    f"GAESID_NSY_{DATA}.csv")


    PATH_XRAY   = os.path.join(CARTELLA_XRAY,   f"GOES_XRAY_{DATA}.csv")
    PATH_FLARES = os.path.join(CARTELLA_FLARES, f"FLARES_{DATA}.json")

    print()
    print("=" * 60)
    print(f"Analisi Doppler flare VLF — {DATA}")
    print("=" * 60)
    print(f"CSV:    {CSV_TRACKER}")
    print(f"X-ray:  {PATH_XRAY}")
    print(f"Flares: {PATH_FLARES}")
    print("=" * 60)

    df = leggi_csv_tracker(CSV_TRACKER)

    print(f"[CSV] {len(df)} campioni, da {df['Timestamp'].iloc[0]} a {df['Timestamp'].iloc[-1]}")

    print()
    print("[STAT] Statistiche grandezze:")
    print(f"  Power_dB:    media={df['Power_dB'].mean():.2f}  std={df['Power_dB'].std():.2f}")
    print(f"  Phase_Deg:   media={df['Phase_Deg'].mean():.1f}  std={df['Phase_Deg'].std():.1f}")
    if 'PLL_Freq_Hz' in df.columns:
        print(f"  PLL_Freq_Hz: media={df['PLL_Freq_Hz'].mean():.4f}  std={df['PLL_Freq_Hz'].std():.4f}")

    ts_xray, flux_xray = scarica_xray(DATA)
    print(f"\n[XRAY] {len(ts_xray)} campioni")

    flares = scarica_flares(DATA, ts_xray=ts_xray, flux_xray=flux_xray)
    print(f"[FLARE] {len(flares)} flare totali")

    if flares and len(df) > 0:
        t_min = df['Timestamp'].iloc[0]
        t_max = df['Timestamp'].iloc[-1]
        flares_nel_range = []
        for f in flares:
            picco = parse_flare_datetime(f['picco'])
            if picco is not None and t_min <= picco <= t_max:
                flares_nel_range.append(f)
        print(f"[FLARE] {len(flares_nel_range)} flare nel range del CSV")
        flares = flares_nel_range

    flares_tutti = list(flares)

    flares_diurni = []
    
    if flares:
        print()
        print("[SOLE] Elevazione solare al punto medio del percorso "
              "(NSY→Torino) e classificazione flare:")
        print(f"{'Ora':>6} | {'Classe':>7} | {'elev°':>7} | {'SID?':>5}")
        print("-" * 40)
        flares_diurni = []
        for f in flares:
            ok, elev = flare_ha_sid_atteso(f)
            picco = parse_flare_datetime(f['picco'])
            ora = picco.strftime('%H:%M') if picco else '??:??'
            print(f"{ora:>6} | {f['classe']:>7} | {elev:>7.1f} | "
                  f"{'SI' if ok else 'no':>5}")
            if ok:
                flares_diurni.append(f)
        print(f"[FLARE] {len(flares_diurni)} flare con SID atteso "
              f"(soglia dipendente dalla classe)")
        pass

    # Rimosso il vincolo di distanza. La maschera flare_vicini()
    # protegge già le baseline.
    flares_da_analizzare = list(flares_diurni)

    print(f"[FLARE] {len(flares_tutti)} flare totali, "
          f"{len(flares_diurni)} diurni (elev > {SOGLIA_ELEVAZIONE_SOLE_DEG}°), "
          f"{len(flares_da_analizzare)} analizzabili")

    if flares:
        print()
        print("=" * 155)
        print("RISULTATI ANALISI FLARE")
        print("=" * 155)
        print(f"{'Ora':>6} | {'Classe':>7} | {'Δpwr':>6} | {'Δfase':>7} | "
              f"{'ΔPLL':>7} | {'lag_F':>6} | {'corr_F':>6} | "
              f"{'lag_P':>6} | {'corr_P':>6} | {'p_F':>6} | {'p_P':>6} | "
              f"{'inval%':>6} | {'Δpwr_m':>6} | {'SNR_p':>5} | Categoria")
        print("-" * 155)

    risultati = []
    png_salvati = []
    for f in flares_da_analizzare:
        r = analizza_flare(df, f, ts_xray, flux_xray, flares=flares_tutti)
        if r is None:
            continue
        risultati.append(r)

        lag_f_str = f"{r['lag_fase']:+d}s" if r['lag_fase'] is not None else "N/D"
        cor_f_str = f"{r['corr_fase']:+.2f}" if r['corr_fase'] is not None else "N/D"
        lag_p_str = f"{r['lag_pll']:+d}s"  if r['lag_pll']  is not None else "N/D"
        cor_p_str = f"{r['corr_pll']:+.2f}" if r['corr_pll'] is not None else "N/D"

        p_f = f"{r['p_fase']:.3f}" if r.get('p_fase') is not None else "  —  "
        p_p = f"{r['p_pll']:.3f}" if r.get('p_pll') is not None else "  —  "
        dpm = r.get('delta_pwr_media')
        spm = r.get('snr_pwr_media')
        dpm_str = f"{dpm:+.2f}" if dpm is not None else "  —  "
        spm_str = f"{spm:+.1f}" if spm is not None else "  —  "

        print(f"{r['ora']:>6} | {r['classe']:>7} | "
              f"{r['delta_pwr']:>6.2f} | {r['delta_fase']:>7.3f} | "
              f"{r['delta_pll']:>7.4f} | "
              f"{lag_f_str:>6} | {cor_f_str:>6} | "
              f"{lag_p_str:>6} | {cor_p_str:>6} | "
              f"{p_f:>6} | {p_p:>6} | "
              f"{r['frazione_invalida']*100:>5.1f}% | "
              f"{dpm_str:>6} | {spm_str:>5} | {r['categoria']}")

        stampa_diagnostica_dettagliata(r)

        try:
            picco_dt = r['picco_dt']
            classe_safe = r['classe'].replace('.', 'p')
            nome_png = (f"{DATA}_{picco_dt.strftime('%H%M%S')}_"
                        f"{classe_safe}.png")
            out_png = os.path.join(CARTELLA_PLOT_EVENTI, nome_png)
            plot_evento(df, picco_dt, r, out_png)
            png_salvati.append(nome_png)
        except Exception as e:
            print(f"  [PLOT] errore: {e}")

    if not risultati:
        print("(Nessun flare analizzabile nella finestra)")
    else:
        from collections import Counter
        cat_count = Counter(r['categoria'] for r in risultati)
        print()
        print("RIEPILOGO DIAGNOSTICO:")
        for cat, n in cat_count.most_common():
            print(f"  {cat:<25s}: {n} flare")
        print()
        print("LEGENDA:")
        print("  SID_COMPLETO       = potenza E fase entrambe significative")
        print("  SID_REALE          = Doppler correlato X-ray (lag breve, corr alta)")
        print("  SID_POTENZA        = SID in ampiezza confermato (SNR alto)")
        print("  SID_FASE           = solo canale fase significativo (forma su)")
        print("  SID_POTENZA_DEBOLE = Candidato SID in potenza (SNR marginale)")
        print("  CANDIDATO_DEBOLE   = bump in potenza sotto le soglie standard")
        print("  DERIVA_TID         = variazione fase senza Doppler -> NON e' SID")
        print("  ARTEFATTO          = reset/slip PLL o finestra degradata")
        print("  NESSUN_EVENTO      = nulla di misurabile")
        print("  INCERTO            = dati ambigui")
        print("  SID_DOPPLER_ABBINATO = evento ARTEFATTO primario ma")
        print("                        significativo vs pool Doppler")
        print("                        abbinato (percentile ≥ 95)")
        print()

    if png_salvati:
        print(f"[PLOT] {len(png_salvati)} PNG evento salvati in:")
        print(f"       {CARTELLA_PLOT_EVENTI}")
        for nome in png_salvati:
            print(f"       - {nome}")
        print()

    print("[PLOT] Generazione grafico...")
    fig = plot_riassuntivo(df, ts_xray, flux_xray, flares_tutti, risultati, fig=fig)

    fig._df          = df
    fig._ts_xray     = ts_xray
    fig._flux_xray   = flux_xray
    fig._flares      = flares_tutti
    fig._risultati   = risultati

    if not risultati:
        motivo = (
            "Nessun flare rilevato nella giornata selezionata."
            if not flares
            else "Flare trovati ma nessuno analizzabile "
                 "(finestra dati insufficiente o fuori range)."
        )
        mostra_messaggio_overlay(
            fig, f"Nessun evento — {DATA}", motivo,
            "Verifica i dati X-ray GOES o prova un'altra data con [I]."
        )
    else:
        rimuovi_overlay(fig)

    return fig

# ============================================================
# KEY HANDLER (R = refresh, I = cambia data, Q = esci)
# ============================================================
def on_key_analisi(event, stato):
    if event.key is None:
        return
    key = event.key.lower()

    if key == 'q':
        print("\n[GUI] Chiusura.")
        plt.close('all')
        return

    if key == 'r':
        print("\n[GUI] Refresh dati...")
        try:
            stato['fig'] = esegui_analisi(stato['data'], fig=stato['fig'])
            stato['fig'].canvas.draw_idle()
            aggiorna_titolo_finestra(stato)
        except Exception as e:
            print(f"[GUI] Errore refresh: {e}")
        return

    if key == 'i':
        print("\n[GUI] Selezione nuova data...")
        nuova = scegli_data_da_calendario(
            data_default=stato['data'].strftime("%Y-%m-%d")
        )
        if nuova is not None:
            stato['data'] = nuova
            try:
                stato['fig'] = esegui_analisi(nuova, fig=stato['fig'])
                stato['fig'].canvas.draw_idle()
                aggiorna_titolo_finestra(stato)
            except Exception as e:
                print(f"[GUI] Errore cambio data: {e}")
        return
        
    if key == 'f':
        fig = stato.get('fig')
        if fig is None or not hasattr(fig, '_fd'):
            print("[GUI] Filtro non disponibile (nessun dato caricato)")
            return
        fd = fig._fd
        linea = fig._linea_doppler
        ax = fig._ax_doppler
        y = fd.toggle()
        linea.set_ydata(y)
        ax.relim()
        ax.autoscale_view()
        label = 'Doppler fase (Hz)'
        if fd.attivo:
            label += '  [filtrato 2–40 min]'
        ax.set_ylabel(label, color='blue')

        for txt in fig.axes[4].texts:
            if 'Refresh' in txt.get_text():
                txt.set_text(
                    f"[R] Refresh   [I] Cambia data   "
                    f"[P] Correlazione   [V] PNG eventi   "
                    f"[F] Filtro [{'ON' if fd.attivo else 'OFF'}]   "
                    f"[K] Check PLL   [Q] Esci"
                )
                break

        fig.canvas.draw_idle()
        print(f"[GUI] Filtro Doppler: {'ON' if fd.attivo else 'OFF'}")
        return
        
    if key == 'u':
        fig = stato.get('fig')
        if fig is None or not hasattr(fig, '_pwr_data'):
            return

        if getattr(fig, '_detrend_attivo', False):
            fig._detrend_attivo = False
            l_raw, l_smooth = fig._pwr_lines
            ax = fig._ax_pwr
            lineare_prec = getattr(fig, '_pwr_scala_lineare', False)
            l_raw.set_ydata(fig._pwr_data['lin_raw'] if lineare_prec else fig._pwr_data['db_raw'])
            l_smooth.set_ydata(fig._pwr_data['lin_smooth'] if lineare_prec else fig._pwr_data['db_smooth'])
            
            if hasattr(fig, '_amp100_lines'):
                l_amp_raw, l_amp_smooth = fig._amp100_lines
                ax_amp = fig._ax_amp100
                l_amp_raw.set_ydata(fig._amp100_data['raw'])
                l_amp_smooth.set_ydata(fig._amp100_data['smooth'])
                ax_amp.set_ylabel('Amp 100 Hz (dB)', color='#00BFFF')
                if fig._linea_soglia_amp100 is not None:
                    fig._linea_soglia_amp100.set_visible(True)
            
            print("[GUI] Detrend disattivato automaticamente per cambio scala.")

        lineare = not getattr(fig, '_pwr_scala_lineare', False)
        fig._pwr_scala_lineare = lineare

        dati = fig._pwr_data
        l_raw, l_smooth = fig._pwr_lines
        ax = fig._ax_pwr

        if lineare:
            l_raw.set_ydata(dati['lin_raw'])
            l_smooth.set_ydata(dati['lin_smooth'])
            ax.set_ylabel('Power (lin.)', color='green')
            dati_plot = dati['lin_smooth']
        else:
            l_raw.set_ydata(dati['db_raw'])
            l_smooth.set_ydata(dati['db_smooth'])
            ax.set_ylabel('Power (dB)', color='green')
            dati_plot = dati['db_smooth']

        validi = dati_plot[np.isfinite(dati_plot)]
        if len(validi) > 0:
            v_min, v_max = np.min(validi), np.max(validi)
            margine = (v_max - v_min) * 0.1
            if margine == 0: margine = 1.0
            ax.set_ylim(v_min - margine, v_max + margine)
        else:
            ax.relim()
            ax.autoscale_view()

        fig.canvas.draw_idle()

        for txt in fig.axes[4].texts:
            if 'Refresh' in txt.get_text():
                stato_filtro = "ON" if (hasattr(fig, '_fd') and fig._fd.attivo) else "OFF"
                scala_str = 'LIN' if lineare else 'dB'
                txt.set_text(
                    f"[R] Refresh   [I] Cambia data   [P] Correlazione   [V] PNG eventi   "
                    f"[F] Filtro [{stato_filtro}]   [T] Detrend [OFF]   "
                    f"[U] Scala [{scala_str}]   [K] Check PLL  [Q] Esci"
                )
                break
        return

    if key == 't':
        fig = stato.get('fig')
        if fig is None or not hasattr(fig, '_pwr_data'):
            return

        fig._detrend_attivo = not getattr(fig, '_detrend_attivo', False)
        attivo = fig._detrend_attivo

        def _aggiorna_ylim(ax, dati, margine_perc=0.2):
            validi = dati[np.isfinite(dati)]
            if len(validi) > 0:
                v_min = float(np.min(validi))
                v_max = float(np.max(validi))
                margine = max((v_max - v_min) * margine_perc, 0.5)
                ax.set_ylim(v_min - margine, v_max + margine)

        l_raw, l_smooth = fig._pwr_lines
        ax_pwr = fig._ax_pwr
        if attivo:
            l_raw.set_ydata(fig._pwr_data['det_raw'])
            l_smooth.set_ydata(fig._pwr_data['det_smooth'])
            ax_pwr.set_ylabel('Power (Detrend dB)', color='green')
            _aggiorna_ylim(ax_pwr, fig._pwr_data['det_smooth'])
        else:
            lineare = getattr(fig, '_pwr_scala_lineare', False)
            l_raw.set_ydata(fig._pwr_data['lin_raw'] if lineare else fig._pwr_data['db_raw'])
            l_smooth.set_ydata(fig._pwr_data['lin_smooth'] if lineare else fig._pwr_data['db_smooth'])
            ax_pwr.set_ylabel('Power (lin.)' if lineare else 'Power (dB)', color='green')
            _aggiorna_ylim(ax_pwr, fig._pwr_data['lin_smooth'] if lineare else fig._pwr_data['db_smooth'])

        if hasattr(fig, '_amp100_lines'):
            l_amp_raw, l_amp_smooth = fig._amp100_lines
            ax_amp = fig._ax_amp100
            if attivo:
                l_amp_raw.set_ydata(fig._amp100_data['det_raw'])
                l_amp_smooth.set_ydata(fig._amp100_data['det_smooth'])
                ax_amp.set_ylabel('Amp 100 Hz (Detrend dB)', color='#00BFFF')
                if fig._linea_soglia_amp100 is not None:
                    fig._linea_soglia_amp100.set_visible(False)
                _aggiorna_ylim(ax_amp, fig._amp100_data['det_smooth'])
            else:
                l_amp_raw.set_ydata(fig._amp100_data['raw'])
                l_amp_smooth.set_ydata(fig._amp100_data['smooth'])
                ax_amp.set_ylabel('Amp 100 Hz (dB)', color='#00BFFF')
                if fig._linea_soglia_amp100 is not None:
                    fig._linea_soglia_amp100.set_visible(True)
                _aggiorna_ylim(ax_amp, fig._amp100_data['smooth'])

        fig.canvas.draw_idle()
        print(f"[GUI] Detrend: {'ON' if attivo else 'OFF'}")

        for txt in fig.axes[4].texts:
            if 'Refresh' in txt.get_text():
                stato_filtro = "ON" if (hasattr(fig, '_fd') and fig._fd.attivo) else "OFF"
                scala_str = 'LIN' if getattr(fig, '_pwr_scala_lineare', False) else 'dB'
                txt.set_text(
                    f"[R] Refresh   [I] Cambia data   [P] Correlazione   [V] PNG eventi   "
                    f"[F] Filtro [{stato_filtro}]   [T] Detrend [{'ON' if attivo else 'OFF'}]   "
                    f"[U] Scala [{scala_str}]   [K] Check PLL  [Q] Esci"
                )
                break
        return
            
    if key == 'p':
        fig = stato.get('fig')
        if fig is None or not hasattr(fig, '_risultati'):
            return
        if not getattr(fig, '_flares', []):
            print("[GUI] Nessun flare da mostrare")
            return

        print("[GUI] Apro finestra correlazione filtrata...")
        try:
            fig_corr = plt.figure(figsize=(14, 10))
            plt.figure(fig_corr.number)

            plot_correlazione_filtrata(
                fig._df, fig._ts_xray, fig._flux_xray,
                fig._flares, fig._risultati,
                fig=fig_corr, idx_flare=0
            )

            try:
                fig_corr.canvas.manager.set_window_title(
                    f"Correlazione filtrata — "
                    f"{stato['data'].strftime('%Y-%m-%d')}"
                )
            except Exception:
                pass

            try:
                fig_corr.canvas.manager.show()
            except Exception:
                pass

            try:
                win = fig_corr.canvas.manager.window
                win.lift()
                win.attributes('-topmost', True)
                win.after(1000, lambda: (
                    win.attributes('-topmost', False)
                    if win.winfo_exists() else None))
                win.focus_force()
            except Exception as e:
                print(f"[GUI] lift correlazione: {e}")

            fig_corr.canvas.draw_idle()
        except Exception as e:
            import traceback
            print(f"[GUI] Errore apertura finestra correlazione: {e}")
            traceback.print_exc()
        return

    if key == 'k':
        fig = stato.get('fig')
        if fig is None or not hasattr(fig, '_df'):
            print("[GUI] Dati non disponibili per il Check PLL.")
            return
        
        print("[GUI] Esecuzione Sanity Check PLL...")
        try:
            mostra_finestra_sanity_check(fig._df, fig_principale=fig)
        except Exception as e:
            import traceback
            print(f"[GUI] Errore durante il Check PLL: {e}")
            traceback.print_exc()
        return

    if key == 'v':
        fig = stato.get('fig')
        if fig is None:
            return
        print("[GUI] Apro visualizzatore PNG eventi...")
        try:
            mostra_png_eventi(stato['data'])
        except Exception as e:
            import traceback
            print(f"[GUI] Errore visualizzatore PNG: {e}")
            traceback.print_exc()
        return
    
def aggiorna_titolo_finestra(stato):
    """Aggiorna il titolo della finestra con data e scorciatoie."""
    if stato.get('fig') is None:
        return
    try:
        data_str = stato['data'].strftime('%Y-%m-%d')
        titolo = (f"Analisi Doppler — {data_str}  "
                  f"[R] Refresh  |  [I] Cambia data  |  "
                  f"[P] Correlazione  |  [F] Filtro  |  [K] Check PLL |  [Q] Esci")
        if hasattr(stato['fig'].canvas.manager, 'set_window_title'):
            stato['fig'].canvas.manager.set_window_title(titolo)
    except Exception:
        pass

# ============================================================
# VISUALIZZATORE PNG EVENTI
# ============================================================
_PNG_VIEWER = {'fig': None, 'idx': 0, 'files': []}


def _raccogli_png_eventi(data_str):
    """Restituisce la lista ordinata dei PNG evento per la data."""
    if not os.path.isdir(CARTELLA_PLOT_EVENTI):
        return []
    prefisso = f"{data_str}_"
    files = []
    for f in os.listdir(CARTELLA_PLOT_EVENTI):
        if f.startswith(prefisso) and f.lower().endswith('.png'):
            files.append(f)
    files.sort()
    return files


def mostra_png_eventi(data_utc, fig_esistente=None):
    """
    Apre (o riusa) una finestra matplotlib che mostra i PNG degli
    eventi salvati per la giornata. Navigazione: [A] prev, [D] next,
    [Q] chiude.
    """
    import matplotlib.pyplot as plt

    data_str = data_utc.strftime("%Y-%m-%d")
    files = _raccogli_png_eventi(data_str)

    if not files:
        print(f"[PNG] Nessun PNG evento per {data_str} in "
              f"{CARTELLA_PLOT_EVENTI}")
        return None

    if _PNG_VIEWER['fig'] is not None:
        try:
            plt.close(_PNG_VIEWER['fig'])
        except Exception:
            pass
        _PNG_VIEWER['fig'] = None

    fig = plt.figure(figsize=(14, 10))
    plt.figure(fig.number)

    _PNG_VIEWER['fig'] = fig
    _PNG_VIEWER['files'] = files
    _PNG_VIEWER['idx'] = 0
    _PNG_VIEWER['data_str'] = data_str

    _disegna_png_corrente()

    fig.canvas.mpl_connect(
        'key_press_event',
        lambda e: _on_key_png_viewer(e)
    )

    try:
        fig.canvas.manager.set_window_title(
            f"Eventi PNG — {data_str}  "
            f"[A] prev  [D] next  [Q] chiudi"
        )
    except Exception:
        pass

    try:
        fig.canvas.manager.show()
    except Exception as e:
        print(f"[PNG] manager.show() errore: {e}")

    try:
        fig.canvas.draw()
    except Exception as e:
        print(f"[PNG] canvas.draw() errore: {e}")

    try:
        win = fig.canvas.manager.window
        win.lift()
        win.attributes('-topmost', True)
        win.after(1000, lambda: (win.attributes('-topmost', False)
                                  if win.winfo_exists() else None))
        win.focus_force()
    except Exception as e:
        print(f"[PNG] lift errore: {e}")

    print(f"[PNG] Visualizzatore aperto: {len(files)} PNG per {data_str}")
    return fig

def _disegna_png_corrente():
    """Ridisegna il PNG corrente nella figura del viewer."""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    fig = _PNG_VIEWER.get('fig')
    if fig is None:
        return
    files = _PNG_VIEWER.get('files', [])
    idx = _PNG_VIEWER.get('idx', 0)
    data_str = _PNG_VIEWER.get('data_str', '?')

    if not files or idx < 0 or idx >= len(files):
        return

    nome = files[idx]
    path = os.path.join(CARTELLA_PLOT_EVENTI, nome)

    fig.clf()
    ax = fig.add_subplot(111)
    try:
        img = mpimg.imread(path)
        ax.imshow(img)
    except Exception as e:
        ax.text(0.5, 0.5, f"Errore lettura PNG:\n{path}\n\n{e}",
                ha='center', va='center', fontsize=11,
                color='red', transform=ax.transAxes)
    ax.set_axis_off()

    fig.suptitle(
        f"[{idx + 1}/{len(files)}]  {nome}\n"
        f"Data: {data_str}   |   [A] precedente   [D] successivo   [Q] chiudi",
        fontsize=10, fontweight='bold'
    )

    try:
        fig.canvas.draw_idle()
    except Exception:
        pass


def _on_key_png_viewer(event):
    import matplotlib.pyplot as plt

    if event.key is None:
        return
    key = event.key.lower()

    fig = _PNG_VIEWER.get('fig')
    if fig is None:
        return

    files = _PNG_VIEWER.get('files', [])
    idx = _PNG_VIEWER.get('idx', 0)
    n = len(files)
    if n == 0:
        return

    if key == 'q':
        try:
            plt.close(fig)
        except Exception:
            pass
        _PNG_VIEWER['fig'] = None
        return

    if key == 'a':
        nuovo = idx - 1
    elif key == 'd':
        nuovo = idx + 1
    else:
        return

    if nuovo < 0 or nuovo >= n:
        print(f"[PNG] già al limite (indice {nuovo})")
        return

    _PNG_VIEWER['idx'] = nuovo
    _disegna_png_corrente()
    print(f"[PNG] evento {nuovo + 1}/{n}: {files[nuovo]}")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("Analisi Doppler flare VLF")
    print("=" * 60)

    oggi_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    data_scelta = scegli_data_da_calendario(data_default=oggi_str)

    if data_scelta is not None:
        data_utc = data_scelta
        print(f"[GUI] Data selezionata: {data_utc.strftime('%Y-%m-%d')}")
    else:
        data_utc = datetime.now(timezone.utc)
        print(f"[GUI] Selezione annullata → uso oggi: {data_utc.strftime('%Y-%m-%d')}")

    stato = {
        'data': data_utc,
        'fig': None,
    }

    stato['fig'] = esegui_analisi(data_utc, fig=None)

    aggiorna_titolo_finestra(stato)

    print()
    print("=" * 60)
    print("TASTI ATTIVI NELLA FINESTRA:")
    print("  [R] Refresh      → ricarica e rianalizza la data corrente")
    print("  [I] Cambia data  → apre il calendario per scegliere un'altra data")
    print("  [Q] Esci         → chiude la finestra")
    print("  [F] Filtro       → mostra/nascondi il passa-banda 2–40 min sul Doppler")
    print("  [T] Detrend      → appiattisce la linea di base per evidenziare gli SFE deboli")
    print("  [U] Scala        → passa la potenza tra scala logaritmica (dB) e lineare")
    print("  [P] Correlazione → apre la finestra con la correlazione filtrata")
    print("  [V] PNG eventi   → apre il visualizzatore dei PNG salvati per la giornata")
    print("  [K] Check PLL    → apre la schermata per il controllo dello stato del PLL durante la giornata")
    print("=" * 60)

    stato['fig'].canvas.mpl_connect(
        'key_press_event',
        lambda e: on_key_analisi(e, stato)
    )

    plt.show()

if __name__ == "__main__":
    main()