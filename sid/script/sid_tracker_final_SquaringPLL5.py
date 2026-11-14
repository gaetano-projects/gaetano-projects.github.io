# ============================================================
# sid_tracker_final6.py
#
# Monitoraggio SID continuo di NSY 45.905 kHz
#
# Fix inclusi in questa versione:
#   - Tasti G/Q/Z/+/-/A/B funzionanti sia su finestra principale
#     che su finestra FFT (binding duplicati)
#   - Tasti A/B come alternativa a +/- (tastiera italiana)
#   - Binding matplotlib sul canvas per intercettare i tasti
#   - Forced focus sulla finestra FFT all'apertura
#   - FFT su s2 (20 kHz), smoothing stile SDRuno, zoom Y
#   - Rotazione giornaliera del CSV
#   - Continuita' filtro con zi
#   - Recovery errori + Reset_Flag
#
# ============================================================

import os
import sys
import time
import csv
import datetime
import queue
import threading
import numpy as np
import scipy.signal as signal
from collections import deque

# --- Tkinter / matplotlib per la GUI ---
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Percorso pysdrplay
PYSDRPLAY_DIR = r"D:\Gaetano\SID\pysdrplay"
if os.path.exists(PYSDRPLAY_DIR):
    sys.path.insert(0, PYSDRPLAY_DIR)

from pysdrplay import SDRPlayRadio, Event, PowerOverloadCbEventId, Bw


# ============================================================
# CONFIGURAZIONE
# ============================================================
TARGET_FREQ = 45_905.0
OFFSET_HZ = 20000.0
OFFSET_HZ_CORRECTION = 0.0    # <-- correzione offset misurata (es. +5.5)

TUNE_FREQ = TARGET_FREQ + OFFSET_HZ
MIX_SIGN = +1.0

HW_SAMPLE_RATE = 2_000_000
# Banda IF dell'RSP1B. Default è BW_1_536 (1.536 MHz) — troppo larga,
# porta dentro molto rumore fuori banda che consuma margine dell'ADC.
# Il segnale utile è a 45.9 kHz, quindi 200 kHz è più che sufficiente.
IF_BANDWIDTH = Bw.BW_0_200   # 200 kHz
DECIM_STAGE1 = 10
DECIM_STAGE2 = 10
DECIM_STAGE3 = 4
FINAL_SR = HW_SAMPLE_RATE / (DECIM_STAGE1 * DECIM_STAGE2 * DECIM_STAGE3)  # 5 kHz

# Frequenza di campionamento del segnale s2 (usato per la FFT)
FS_FFT = HW_SAMPLE_RATE / (DECIM_STAGE1 * DECIM_STAGE2)   # 20 kHz

FIR_CUTOFF_HZ = 45.0
FIR_TAPS = 1001

# Cartella output e naming giornaliero
CARTELLA_OUTPUT = r"D:\Gaetano\SID\dati\Sdruno"
PREFISSO_FILE = "GAESID_NSY"

# Metadati della stazione (scritti in testa a ogni nuovo CSV)
SITE_NAME        = "Torino"
SITE_LAT         = 45.07
SITE_LON         = 7.68
MONITOR_ID       = "GAESID"
STATION_ID       = "NSY"
FREQ_PORTANTE_HZ = 45900.0     # frequenza reale della portante NSY
LOG_INTERVAL_S   = 1           # intervallo di log in secondi

MEASUREMENT_INTERVAL_S = 1.0
MAX_CONSECUTIVE_ERRORS = 10
TIMEOUT_PER_CHUNK_S = 1.0

# FFT: parametri di visualizzazione
FFT_N = 16384              # punti FFT da mostrare

# Zoom asse X (banda visualizzata): lista di semi-ampiezze in Hz.
#   A = zoom in  (scala verso l'indice precedente, banda più stretta)
#   D = zoom out (scala verso l'indice successivo, banda più larga)
FFT_X_ZOOM_LEVELS_HZ = [50, 100, 200, 500, 1000, 2000, 5000]
FFT_X_ZOOM_INDEX = 4       # default: ±1000 Hz
FFT_XMIN_HZ = -FFT_X_ZOOM_LEVELS_HZ[FFT_X_ZOOM_INDEX]
FFT_XMAX_HZ = +FFT_X_ZOOM_LEVELS_HZ[FFT_X_ZOOM_INDEX]

# Smoothing FFT (stile SDRuno)
FFT_SMOOTH_ALPHA = 0.15    # 0.02 = molto liscio, 1.0 = nessuno smoothing
FFT_BIN_GROUP = 8          # raggruppa N bin FFT in 1 (riduce rumore di √N)

# Zoom Y: ciclato col tasto Z
FFT_Y_ZOOM_LEVELS = [(-80.0, 0.0), (-60.0, 0.0), (-40.0, 0.0), (-20.0, 0.0)]
FFT_Y_ZOOM_INDEX = 2
FFT_YMIN_DB, FFT_YMAX_DB = FFT_Y_ZOOM_LEVELS[FFT_Y_ZOOM_INDEX]

# Costante di smoothing della fase (filtro IIR del primo ordine)
# PHASE_SMOOTH_ALPHA = 1/tau, dove tau è la costante di tempo in secondi.
#   0.05 → tau = 20 s  (consigliato)
#   0.033 → tau = 30 s
#   0.02 → tau = 50 s
PHASE_SMOOTH_ALPHA = 0.05


# Offset noto della portante NSY rispetto alla sintonia nominale.
# Serve a inizializzare il VCO del PLL vicino alla riga MSK vera,
# evitando che debba spostarsi di diversi Hz dopo un reset (operazione
# che richiederebbe decine di minuti con la banda di tracking stretta).
# Valore in Hz, riferito alla PORTANTE (non alla riga a 100 Hz).
OFFSET_PLL_DEFAULT_HZ = -4.7


# Debug prints in trova_offset_msk (spam console).
# Mettere a True solo per debug manuale.
DEBUG_AUTO_RICERCA = False

# ============================================================
# CODA IQ
# ============================================================
iq_queue = queue.Queue(maxsize=10000)
# Contatori di campioni persi (thread-safe tramite lock)
dropped_lock = threading.Lock()
dropped_queue_full = 0        # chunk scartati perché la coda era piena
dropped_excess_samples = 0    # campioni in eccesso tagliati da [:target_samples]
drop_flag_callback = False    # segnalato dal callback IQ quando la coda è piena

# --- Debug prestazioni (attivare solo per diagnosi) ---
DEBUG_PERF = False

debug_chunk_sizes = deque(maxlen=200)   # dimensioni degli ultimi 200 chunk

def iq_callback(xi, xq, params, num_samples, reset, ctx):
    global dropped_queue_full, drop_flag_callback
    try:
        # Una sola allocazione: il buffer complex64 di output.
        samples = np.empty(num_samples, dtype=np.complex64)

        # Viste zero-copy sui buffer C di input (int16)
        i_raw = np.ctypeslib.as_array(xi, shape=(num_samples,))
        q_raw = np.ctypeslib.as_array(xq, shape=(num_samples,))

        # Scrittura diretta nelle parti real/imag.
        # NumPy converte int16 -> float32 internamente, senza
        # creare array temporanei espliciti come prima.
        samples.real = i_raw
        samples.imag = q_raw

        if DEBUG_PERF:
            debug_chunk_sizes.append(int(num_samples))
        try:
            iq_queue.put_nowait(samples)
        except queue.Full:
            with dropped_lock:
                dropped_queue_full += 1
                drop_flag_callback = True
    except Exception as e:
        print("[CALLBACK ERROR]", e)

# ============================================================
# OVERLOAD ADC — stato condiviso
# ============================================================
# L'evento di overload arriva dal wrapper via callback `on_event`.
# La callback è chiamata da un thread interno SDRplay e NON deve
# chiamare funzioni SDRplay (rischio deadlock). Quindi:
#   - la callback setta solo dei flag
#   - il thread di acquisizione legge i flag e invia l'ack
overload_lock = threading.Lock()
overload_attivo = False        # True tra DETECTED e CORRECTED
overload_da_ackare = False     # evento da confermare al prossimo giro
overload_visto_nel_s = 0       # contatore eventi nel secondo corrente
overload_totali = 0            # contatore cumulativo
overload_ultimo_evento_ts = 0.0   # timestamp dell'ultimo evento ricevuto
eventi_log = deque(maxlen=200)   # log eventi per diagnostica

def event_callback(event_id, tuner, params, ctx):
    global overload_attivo, overload_da_ackare, overload_visto_nel_s
    global overload_totali, overload_ultimo_evento_ts
    try:
        if event_id == Event.POWER_OVERLOAD_CHANGE:
            tipo = params.contents.powerOverloadParams.powerOverloadChangeType
            with overload_lock:
                overload_ultimo_evento_ts = time.time()
                if tipo == PowerOverloadCbEventId.OVERLOAD_DETECTED:
                    overload_attivo = True
                    overload_visto_nel_s += 1
                    overload_totali += 1
                    eventi_log.append((time.time(), "DETECTED"))
                else:
                    overload_attivo = False
                    eventi_log.append((time.time(), "CORRECTED"))
                overload_da_ackare = True
    except Exception as e:
        print(f"[EVENT] Errore callback: {e}")

# ============================================================
# CSV MANAGER (rotazione giornaliera)
# ============================================================
def nome_file_giornaliero(data_utc, cartella=CARTELLA_OUTPUT, prefisso=PREFISSO_FILE):
    data_str = data_utc.strftime("%Y-%m-%d")
    return os.path.join(cartella, f"{prefisso}_{data_str}.csv")


class CSVManager:
    def __init__(self, cartella=CARTELLA_OUTPUT, prefisso=PREFISSO_FILE):
        self.cartella = cartella
        self.prefisso = prefisso
        self.giorno_corrente = None
        self.file_handle = None
        self.writer = None
        self.path_corrente = None
        os.makedirs(self.cartella, exist_ok=True)

    def _apri_file(self, data_utc):
        path = nome_file_giornaliero(data_utc, self.cartella, self.prefisso)
        esiste = os.path.exists(path) and os.path.getsize(path) > 0

        self.file_handle = open(path, mode="a", newline="")
        self.writer = csv.writer(self.file_handle)
        self.path_corrente = path

        if not esiste:
            # ---- Header stile supersid (righe di commento con #) ----
            data_inizio = data_utc.strftime("%Y-%m-%d %H:%M:%S")
            header_lines = [
                f"# Site = {SITE_NAME}",
                f"# Longitude = {SITE_LON}",
                f"# Latitude = {SITE_LAT}",
                "#",
                f"# UTC_StartTime = {data_inizio}",
                f"# LogInterval = {LOG_INTERVAL_S}",
                "# LogType = raw",
                f"# MonitorID = {MONITOR_ID}",
                f"# StationID = {STATION_ID}",
                f"# Frequency = {int(FREQ_PORTANTE_HZ)}",
            ]
            for linea in header_lines:
                self.file_handle.write(linea + "\n")

            # ---- Riga di intestazione dei dati (per pandas read_csv) ----
            self.writer.writerow(["Timestamp_UTC", "Power_dB", "Phase_Deg",
                                  "PLL_Freq_Hz", "Reset_Flag", "Amp_100Hz_dB",
                                  "ADC_Overload"])
            self.file_handle.flush()
            print(f"[CSV] Nuovo file creato: {path}")
        else:
            print(f"[CSV] File esistente, aggiungo in coda: {path}")

    def _chiudi_file(self):
        if self.file_handle is not None:
            try:
                self.file_handle.flush()
                self.file_handle.close()
                print(f"[CSV] File chiuso: {self.path_corrente}")
            except Exception as e:
                print(f"[CSV] Errore chiusura file: {e}")
            self.file_handle = None
            self.writer = None
            self.path_corrente = None

    def aggiorna(self, data_utc):
        giorno = data_utc.date()
        if giorno != self.giorno_corrente:
            self._chiudi_file()
            self._apri_file(data_utc)
            self.giorno_corrente = giorno
            return True
        return False

    def scrivi(self, timestamp_str, pwr, phase, pll_freq_hz, reset_flag,
               amp_100hz_db=0.0, adc_overload=0):
        if self.writer is None:
            return
        self.writer.writerow([timestamp_str, f"{pwr:.3f}", f"{phase:.3f}",
                                f"{pll_freq_hz:.6f}", reset_flag,
                                f"{amp_100hz_db:.3f}", adc_overload])
        # NIENTE flush qui: viene chiamato da AsyncCSVWriter ogni 30 righe

    def chiudi(self):
        self._chiudi_file()

# ============================================================
# ASYNC CSV WRITER
# ============================================================
class AsyncCSVWriter(threading.Thread):
    """
    Scrivo le righe CSV in un thread separato, così il thread di
    acquisizione (che contiene il DSP e riceve i callback IQ) non
    viene mai bloccato dall'I/O su disco.

    Se il disco è lento (flush del filesystem, antivirus, ecc.), la
    coda interna assorbe il ritardo e il DSP continua a girare.
    Solo se la coda si riempie (disco bloccato per >5 minuti) le
    righe CSV vengono perse — ma il flusso IQ resta intatto.
    """
    def __init__(self, csv_manager, state, maxsize=300):
        super().__init__(daemon=True)
        self.csv_manager = csv_manager
        self.state = state
        self.row_queue = queue.Queue(maxsize=maxsize)
        self.stop_event = threading.Event()
        self.righe_scritte = 0
        self.righe_droppate = 0
        self._righe_da_flush = 0

    def put(self, record):
        """Inserisce una riga in coda senza mai bloccare il chiamante."""
        try:
            self.row_queue.put_nowait(record)
        except queue.Full:
            self.righe_droppate += 1

    def run(self):
        print("[CSV_WRITER] Thread avviato.")
        while not self.stop_event.is_set():
            try:
                record = self.row_queue.get(timeout=1.0)
            except queue.Empty:
                # Flush periodico anche quando la coda è vuota
                self._flush_periodico()
                continue

            (timestamp_str, timestamp_dt, pwr, phase, pll_freq_hz,
             reset_flag, amp_100hz_db, adc_overload) = record

            # Cambio giorno UTC (raro): gestito qui, non nel DSP
            try:
                if self.csv_manager.aggiorna(timestamp_dt):
                    self.state["path_csv"] = self.csv_manager.path_corrente
            except Exception as e:
                print(f"[CSV_WRITER] Errore aggiorna(): {e}")

            # Scrittura effettiva
            try:
                self.csv_manager.scrivi(
                    timestamp_str, pwr, phase, pll_freq_hz, reset_flag,
                    amp_100hz_db=amp_100hz_db,
                    adc_overload=adc_overload
                )
                self.righe_scritte += 1
                self._righe_da_flush += 1
            except Exception as e:
                print(f"[CSV_WRITER] Errore scrivi(): {e}")

            # Flush ogni 30 righe (~30 s)
            if self._righe_da_flush >= 30:
                self._flush_periodico()

        # Flush finale prima di uscire
        self._flush_periodico()
        #print(f"[CSV_WRITER] Terminato. "
        #      f"Scritte={self.righe_scritte}, Droppate={self.righe_droppate}")

    def _flush_periodico(self):
        if self._righe_da_flush <= 0:
            return
        try:
            if self.csv_manager.file_handle is not None:
                self.csv_manager.file_handle.flush()
        except Exception:
            pass
        self._righe_da_flush = 0

    def stop(self):
        self.stop_event.set()

# ============================================================
# TRACKER (DSP)
# ============================================================
class SquaringPLL:
    """
    Squaring + PLL a banda stretta per estrarre la fase della portante MSK.

    1. s²(t) → riga a R/2 (25 Hz per baud 50, 100 Hz per baud 200)
    2. PLL insegue la riga a R/2 (con frequenza nominale!)
    3. Fase portante = (fase_VCO - avanzamento_nominale) / 2
    """
    def __init__(self, baud_rate=200.0, sample_rate=5000.0, loop_bw_hz=1.0):
        self.fs = sample_rate
        self.f_nominal = baud_rate / 2.0                            # R/2
        self.omega_nominal = 2 * np.pi * self.f_nominal / self.fs   # rad/campione

        # Limite di banda sul VCO: il PLL non può allontanarsi più di
        # MAX_FREQ_DEV_HZ dal nominale. Impedisce al VCO di inseguire DC
        # o spurie lontane quando la portante NSY è assente.
        # Valore fisico: la portante VLF può spostarsi di ±5-10 Hz per
        # Doppler/TID, e fino a ±20 Hz in tempeste geomagnetiche forti.
        MAX_FREQ_DEV_HZ = 25.0
        self.max_freq_error = 2 * np.pi * MAX_FREQ_DEV_HZ / self.fs  # rad/sample

        wn = 2 * np.pi * loop_bw_hz
        zeta = 0.707
        self.alpha = 2 * zeta * wn / self.fs
        self.beta = (wn / self.fs) ** 2

        self.vco_phase = 0.0
        self.integrator = 0.0
        self.total_nominal_phase = 0.0
        self.freq_offset_hz = 0.0
        self.omega_offset = 0.0

    def process(self, samples):
        squared = samples ** 2

        # Accumulatore per l'ampiezza della riga a R/2 (correlazione coerente)
        correlatore = 0.0 + 0.0j

        # Accumulatore per la frequenza MEDIA sul blocco
        freq_acc = 0.0

        for s in squared:
            prod = s * np.exp(-1j * self.vco_phase)
            error = np.angle(prod)
            correlatore += prod
            self.integrator += self.beta * error
            freq_error = self.alpha * error + self.integrator

            # Clamp: il VCO non può deviare oltre ±MAX_FREQ_DEV_HZ dal nominale
            if freq_error > self.max_freq_error:
                freq_error = self.max_freq_error
                self.integrator = self.max_freq_error - self.alpha * error
            elif freq_error < -self.max_freq_error:
                freq_error = -self.max_freq_error
                self.integrator = -self.max_freq_error - self.alpha * error

            self.vco_phase += self.omega_nominal + freq_error
            freq_acc += freq_error

        # Ampiezza della riga a R/2 = modulo della correlazione mediata
        amp_r2 = np.abs(correlatore) / len(squared)
        amp_r2_db = 20.0 * np.log10(amp_r2 + 1e-12)

        # Accumula fase nominale per il blocco corrente
        self.total_nominal_phase += self.omega_nominal * len(squared)

        # Fase in eccesso rispetto al nominale = fase della riga MSK
        excess_phase = self.vco_phase - self.total_nominal_phase
        carrier_rad = excess_phase / 2.0

        # Frequenza MEDIA sul blocco (era: ultimo campione)
        freq_media = freq_acc / len(squared)
        self.freq_offset_hz = freq_media * self.fs / (2 * np.pi) / 2.0

        # Indicatore diretto di lock (fase media del correlatore)
        self.fase_err_deg = float(np.degrees(np.angle(correlatore)))

        return np.degrees(carrier_rad), self.freq_offset_hz, amp_r2_db
    
    def reset(self, offset_hz=0.0):
        """
        Azzera lo stato del PLL.

        Parametro offset_hz (opzionale): offset noto della portante rispetto
        alla sintonia nominale, in Hz. Se non zero, il VCO viene inizializzato
        direttamente a (f_nominal + 2·offset_hz), così il PLL cerca la riga
        MSK nella posizione corretta fin dal primo campione. Il fattore 2
        tiene conto del fatto che la riga MSK è al doppio della frequenza
        della portante.

        Esempio: portante a -4.5 Hz → VCO inizializzato a 100 - 9 = 91 Hz.
        """
        self.vco_phase = 0.0
        self.total_nominal_phase = 0.0
        self.freq_offset_hz = offset_hz
        # L'integrator definisce freq_error a regime. Impostandolo a
        # 2π·(2·offset)/fs, il VCO parte già all'offset giusto.
        self.omega_offset = 2 * np.pi * (2.0 * offset_hz) / self.fs
        self.integrator = self.omega_offset

class SIDTracker:
    def __init__(self):
        self.sample_counter = 0
        self.last_phase = 0.0
        self.pll_contatore = 0   # secondi dall'avvio

        self.fir_coeff = signal.firwin(
            FIR_TAPS, FIR_CUTOFF_HZ, fs=FINAL_SR, window=("kaiser", 8.0)
        )
        self.filter_zi = signal.lfilter_zi(self.fir_coeff, 1.0) * 0.0

        # Filtro LARGO per la FASE MSK (copre R/2 = 100 Hz)
        FIR_CUTOFF_FASE_HZ = 150.0
        self.fir_coeff_fase = signal.firwin(
            FIR_TAPS, FIR_CUTOFF_FASE_HZ, fs=FINAL_SR, window=("kaiser", 8.0)
        )
        self.filter_zi_fase = signal.lfilter_zi(self.fir_coeff_fase, 1.0) * 0.0

        self.warmed_up = False
        self.reset_occurred = False

        # Stato del filtro IIR di smoothing sulla fase unwrapped
        self.phase_smooth = 0.0

        # Stato per unwrap fase PLL tra blocchi consecutivi
        self.last_pll_phase_rad = None
        self.pll_phase_unwrapped = 0.0

        # PLL per la fase della portante MSK
        self.pll = SquaringPLL(baud_rate=200.0, sample_rate=FINAL_SR, loop_bw_hz=1.0)

        # Buffer circolare di blocchi squadrati per l'auto-ricerca.
        # Serve a mediare il rumore impulsivo: un singolo blocco può
        # contenere picchi spuri che l'auto-ricerca scambierebbe per NSY.
        self.buffer_sq = deque(maxlen=20)
        
    def process(self, iq_samples):
        """
        Ritorna (power_db, phase_deg, filtered, s2).
        - s2 : segnale a 20 kHz (per FFT diagnostica)

        La fase è stimata su una media complessa di 1 secondo.
        Lo smoothing è un filtro IIR del primo ordine sulla fase unwrapped.
        """
        num_samples = len(iq_samples)

        # --- Mixing a banda base ---
        n = np.arange(num_samples, dtype=np.float64) + self.sample_counter
        t = n / HW_SAMPLE_RATE
        effective_offset = OFFSET_HZ + OFFSET_HZ_CORRECTION
        baseband = iq_samples * np.exp(MIX_SIGN * 1j * 2 * np.pi * effective_offset * t)
        self.sample_counter = (self.sample_counter + num_samples) % int(HW_SAMPLE_RATE * 3600)

        # --- Decimazioni ---
        s1 = signal.decimate(baseband, DECIM_STAGE1, ftype="fir", zero_phase=True)
        s2 = signal.decimate(s1, DECIM_STAGE2, ftype="fir", zero_phase=True)
        s3 = signal.decimate(s2, DECIM_STAGE3, ftype="fir", zero_phase=True)

        # --- Filtro FIR STRETTO (per potenza) ---
        filtered, self.filter_zi = signal.lfilter(
            self.fir_coeff, 1.0, s3, zi=self.filter_zi
        )

        # --- Filtro FIR LARGO (per fase MSK) ---
        filtered_fase, self.filter_zi_fase = signal.lfilter(
            self.fir_coeff_fase, 1.0, s3, zi=self.filter_zi_fase
        )

        # Salva il segnale filtrato e il suo quadrato (s²) come attributi
        # della classe, così il chiamante può passarli alla FFT.
        self.filtered_fase = filtered_fase
        self.filtered_fase_squared = filtered_fase ** 2

        # Aggiungi al buffer per l'auto-ricerca (solo se non è warm-up)
        if self.warmed_up:
            self.buffer_sq.append(self.filtered_fase_squared.copy())

        if not self.warmed_up:
            self.warmed_up = True
            return None, None, 0.0, 0.0, filtered, s2

        # --- Potenza: dal filtro STRETTO ---
        power_db = 10 * np.log10(np.mean(np.abs(filtered) ** 2) + 1e-12)

        # --- Fase: dal filtro LARGO (copre la riga MSK a 100 Hz) ---
        phase_pll_deg, freq_offset_hz, amp_100hz_db = self.pll.process(filtered_fase)

        # Cambio banda PLL: acquisizione veloce, poi tracking stretto
        self.pll_contatore += 1
        if self.pll_contatore == 30:   # dopo 30 secondi
            wn = 2 * np.pi * 0.05
            zeta = 0.707
            self.pll.alpha = 2 * zeta * wn / self.pll.fs
            self.pll.beta = (wn / self.pll.fs) ** 2
            print("[PLL] Banda ridotta a 0.05 Hz (tracking stretto)")

        #if self.sample_counter % 10 == 0:
        #    print(f"[PLL] freq_offset = {freq_offset_hz:+.3f} Hz")

        # --- Fase PLL già continua: nessun secondo unwrap ---
        # Il PLL accumula la fase in modo continuo (vco_phase non è mai
        # avvolta). Il vecchio unwrap causava aliasing quando la fase
        # variava > 180°/s, cioè per offset di frequenza > 0.5 Hz
        # (tipico di flare M alti e X). Ora la fase passa diretta.
        phase_pll_rad = np.radians(phase_pll_deg)
        self.pll_phase_unwrapped = phase_pll_rad

        # --- Smoothing IIR sulla fase unwrapped ---
        self.phase_smooth = (PHASE_SMOOTH_ALPHA * self.pll_phase_unwrapped +
                             (1 - PHASE_SMOOTH_ALPHA) * self.phase_smooth)

        return (power_db, np.degrees(self.phase_smooth),
                freq_offset_hz, amp_100hz_db, filtered, s2)


    def trova_offset_msk(self):
        """
        Cerca la riga MSK della portante analizzando la FFT del segnale
        squadrato (self.filtered_fase_squared).

        Cerca un picco nell'intervallo +80 / +120 Hz (dove sta la riga
        MSK per una portante entro ±10 Hz dalla sintonia nominale).

        Restituisce una tupla (offset_hz, freq_picco, amp_picco, rumore)
        dove offset_hz è l'offset della PORTANTE in Hz, calcolato come:
            offset = (freq_picco - 100) / 2

        Restituisce None se:
          - il segnale non è disponibile
          - nessun picco nell'intervallo
          - il picco è troppo debole (SNR < 5)
        """
        if not hasattr(self, 'buffer_sq') or len(self.buffer_sq) < 3:
            return None

        # Accumula i blocchi del buffer in un unico array reale.
        # Più blocchi = risoluzione migliore + rumore impulsivo mediato.
        # FFT per blocco (Welch-like), poi media in potenza.
        # NON concatenare i blocchi complessi: la fase è discontinua
        # tra un blocco e l'altro e la concatenazione distrugge lo spettro.
        spettri = []
        freqs = None
        for sq in self.buffer_sq:
            if sq is None or len(sq) < 512:
                continue
            sq_centered = sq - np.mean(sq)
            n_b = len(sq_centered)
            window = np.hanning(n_b)
            sq_win = sq_centered * window

            # FFT complessa con fftshift (spettro da -fs/2 a +fs/2)
            spettro_b = np.abs(np.fft.fftshift(np.fft.fft(sq_win)))
            if freqs is None:
                fs = self.pll.fs
                freqs = np.fft.fftshift(np.fft.fftfreq(n_b, d=1.0 / fs))
            spettri.append(spettro_b)

        if not spettri:
            return None

        # Media degli spettri (riduce rumore impulsivo)
        spettro = np.mean(spettri, axis=0)
        n = len(spettro)

        # Cerca il picco nell'intervallo +80 / +120 Hz
        mask = (freqs >= 80.0) & (freqs <= 120.0)
        if not np.any(mask):
            return None

        spettro_banda = spettro[mask]

        # DEBUG esteso (disattivabile con DEBUG_AUTO_RICERCA)
        if DEBUG_AUTO_RICERCA:
            mask_debug = (freqs >= 80.0) & (freqs <= 100.0)
            spettro_debug = spettro[mask_debug]
            freqs_debug = freqs[mask_debug]
            idx_max = np.argmax(spettro_debug)
            print(f"[DEBUG] Spettro 80-100 Hz (max a +{freqs_debug[idx_max]:.2f}):")
            for i in range(0, len(freqs_debug), 2):
                barra = "#" * int(40 * spettro_debug[i] / spettro_debug[idx_max])
                print(f"  +{freqs_debug[i]:6.2f} Hz {spettro_debug[i]:12.0f} {barra}")

        freqs_banda = freqs[mask]

        idx_picco = int(np.argmax(spettro_banda))
        freq_picco = float(freqs_banda[idx_picco])
        amp_picco = float(spettro_banda[idx_picco])

        # Interpolazione parabolica per precisione sub-bin.
        # Usa i 3 bin attorno al picco per stimare la posizione reale
        # del massimo con precisione ~0.1 Hz (invece di ±1 Hz).
        if 0 < idx_picco < len(spettro_banda) - 1:
            y0 = spettro_banda[idx_picco - 1]
            y1 = spettro_banda[idx_picco]
            y2 = spettro_banda[idx_picco + 1]
            denom = 2 * y1 - y0 - y2
            if denom > 0:
                delta = (y2 - y0) / (2 * denom)
                bin_width = float(freqs_banda[1] - freqs_banda[0])
                freq_picco = freq_picco + delta * bin_width

        rumore_mask = ((np.abs(freqs) > 130.0) & (np.abs(freqs) < 200.0))
        if np.any(rumore_mask):
            rumore = float(np.median(spettro[rumore_mask]))
        else:
            rumore = float(np.median(spettro))

        # Soglia: picco almeno 5× il rumore
        if rumore <= 0 or amp_picco < 5.0 * rumore:
            return None

        # Offset della portante (in Hz)
        offset_hz = (freq_picco - self.pll.f_nominal) / 2.0

        # DEBUG (disattivabile con DEBUG_AUTO_RICERCA)
        if DEBUG_AUTO_RICERCA:
            print(f"[DEBUG] Buffer FFT: {len(spettri)} blocchi, "
                  f"{n} campioni totali, RBW={fs/n:.2f} Hz")

            idx_ordinati = np.argsort(spettro_banda)[::-1][:5]
            print("[DEBUG] Top 5 picchi in banda 80-120 Hz:")
            for i, idx in enumerate(idx_ordinati):
                print(f"  {i+1}. +{freqs_banda[idx]:.2f} Hz  "
                      f"(amp={spettro_banda[idx]:.2f}, "
                      f"SNR={spettro_banda[idx]/rumore:.1f})")
                   
        return (offset_hz, freq_picco, amp_picco, rumore)


    def reset(self):
        self.sample_counter = 0
        self.last_phase = 0.0
        self.phase_smooth = 0.0
        self.pll_contatore = 0
        
        self.filter_zi = signal.lfilter_zi(self.fir_coeff, 1.0) * 0.0
        self.filter_zi_fase = signal.lfilter_zi(self.fir_coeff_fase, 1.0) * 0.0
        self.warmed_up = False
        self.reset_occurred = True

        # Reset del PLL e dello stato unwrap
        self.pll.reset(offset_hz=OFFSET_PLL_DEFAULT_HZ)
        self.last_pll_phase_rad = None
        self.pll_phase_unwrapped = 0.0

# ============================================================
# ACQUISIZIONE (thread)
# ============================================================
class AcquisitoreSID(threading.Thread):
    def __init__(self, state):
        super().__init__(daemon=True)
        self.state = state
        self.stop_event = threading.Event()
        self.radio = None
        self.tracker = None
        self.csv_manager = None
        self.async_writer = None

        self.letture_totali = 0
        self.errori_consecutivi = 0
        self.ultimo_reset = None
        self.warmup_counter = 0

        # Buffer circolare per diagnostica PLL in tempo reale
        self.pll_history = deque(maxlen=300)

        # Buffer circolare per la qualità del lock del PLL
        # (std dell'ampiezza della riga a 100 Hz sugli ultimi 30 campioni)
        self.amp100_history = deque(maxlen=30)
        # Storico amp100 per media mobile 60s (per reset PLL robusto)
        self.amp100_media_storico = deque(maxlen=60)

        # Contatore di secondi consecutivi con lock perso.
        # Se supera la soglia, il PLL viene resettato automaticamente
        # per permettere il ri-aggancio quando la portante NSY ritorna.
        self.lock_perso_da_s = 0

        # --- Macchina a stati "segnale presente/assente" basata su Power_dB ---
        # Soglie con isteresi: discesa a 28 dB, risalita a 35 dB.
        # Debounce: 10 secondi consecutivi sotto/sopra soglia.
        # NOTA: soglie provvisorie, da calibrare sui dati reali.
        self.POTENZA_SOGLIA_BASSA = 20.0    # dB — sotto → segnale assente
        self.POTENZA_SOGLIA_ALTA  = 33.0    # dB — sopra → segnale presente

        # 10 minuti: evita che il dip di propagazione all'alba (potenza
        # che scende a 23 dB per 5-8 min con TX attivo) venga interpretato
        # come spegnimento del trasmettitore. Uno spegnimento vero viene
        # riconosciuto in 10 minuti, tempo accettabile per il monitoraggio.
        self.DEBOUNCE_POTENZA_S = 600

        # NUOVO: Buffer per la media mobile della potenza (60 secondi)
        self.pwr_media_storico = deque(maxlen=60)

        self.segnale_presente = None        # None = stato iniziale
        self.potenza_sotto_soglia_da_s = 0  # contatore debounce discesa
        self.potenza_sopra_soglia_da_s = 0  # contatore debounce risalita

        # Flag: auto-ricerca iniziale già eseguita?
        self.auto_ricerca_iniziale_fatta = False
        self.auto_ricerca_tentativi = 0

        # Buffer di riporto: campioni in eccesso dal secondo precedente.
        # Vengono anteposti al prossimo accumulo per evitare discontinuità
        # di fase nel mixer software (che assume campioni contigui).
        self.carryover = np.array([], dtype=np.complex64)
        # Contatore per stampa diagnostica ogni 60 secondi
        self._debug_counter = 0

    def stop(self):
        self.stop_event.set()

    def run(self):
        global overload_da_ackare, overload_visto_nel_s, overload_totali
        global overload_attivo, overload_ultimo_evento_ts

        amp_100hz_db = 0.0    # valore di default per il check drop al primo ciclo

        print("[ACQ] Thread acquisizione avviato.")
        self.tracker = SIDTracker()
        self.csv_manager = CSVManager()
        self.async_writer = AsyncCSVWriter(self.csv_manager, self.state, maxsize=300)
        self.async_writer.start()

        try:
            self.radio = SDRPlayRadio()
            self.radio.select_device(self.radio.devices[0])
            self.radio.set_frequency(TUNE_FREQ)
            self.radio.set_sample_rate(HW_SAMPLE_RATE)

            # Restringe la banda IF a 200 kHz (default: 1536 kHz).
            # Riduce l'energia fuori banda che satura il front-end RF.
            try:
                self.radio.set_bandwidth(IF_BANDWIDTH)
                print(f"[ACQ] Banda IF impostata a {IF_BANDWIDTH}")
            except Exception as e:
                print(f"[ACQ] Errore set_bandwidth: {e}")

            self.radio.set_agc(False)   
            print("[ACQ] AGC disabilitato")

            GAIN_LNA = 0
            GAIN_IF_DB = 30
            self.radio.set_gain(lna_state=GAIN_LNA, gr_db=GAIN_IF_DB)
            self.state["gain_lna"] = GAIN_LNA
            self.state["gain_if"] = GAIN_IF_DB
            print(f"[ACQ] Gain impostato: LNA={GAIN_LNA}, IF_GR={GAIN_IF_DB} dB")

            self.radio.set_dc_offset(dc_enable=True, iq_enable=True)
            print("[ACQ] Correzione DC offset e IQ attivata")

            self.radio.start(iq_callback, on_event=event_callback)
            self.state["radio_pronta"] = True
            print("[ACQ] Radio pronta.")

            print("[ACQ] Apro CSV per il giorno corrente...")
            ora_utc = datetime.datetime.now(datetime.timezone.utc)
            self.csv_manager.aggiorna(ora_utc)
            print(f"[ACQ] CSV aperto: {self.csv_manager.path_corrente}")
            self.state["path_csv"] = self.csv_manager.path_corrente

            while not self.stop_event.is_set():
                t_start = time.time()
                timestamp_dt = datetime.datetime.now(datetime.timezone.utc)
                timestamp_str = timestamp_dt.strftime("%Y-%m-%d %H:%M:%S")

                # --- Applica eventuali richieste di cambio gain dalla GUI ---
                if self.state.get("gain_change_pending", False):
                    try:
                        nuovo_lna = self.state.get("gain_richiesto_lna")
                        nuovo_if = self.state.get("gain_richiesto_if")
                        if nuovo_lna is not None and nuovo_if is not None:
                            self.radio.set_gain(lna_state=int(nuovo_lna),
                                                gr_db=int(nuovo_if))
                            self.state["gain_lna"] = int(nuovo_lna)
                            self.state["gain_if"] = int(nuovo_if)
                            print(f"[ACQ] Gain aggiornato: LNA={nuovo_lna}, "
                                  f"IF_GR={nuovo_if} dB")
                    except Exception as e:
                        print(f"[ACQ] Errore cambio gain: {e}")
                    finally:
                        self.state["gain_change_pending"] = False

                # Il cambio giorno UTC è gestito da AsyncCSVWriter,
                # non più nel thread di acquisizione.
                # if self.csv_manager.aggiorna(timestamp_dt):
                #     self.state["path_csv"] = self.csv_manager.path_corrente

                target_samples = HW_SAMPLE_RATE
                accumulated = []
                total = 0
                got_data = False

                # Anteponi i campioni in eccesso dal secondo precedente.
                # Questo garantisce che i campioni passati al mixer siano
                # sempre contigui (nessun buco temporale tra un secondo e il
                # successivo). Senza questo, ogni secondo si creerebbe una
                # discontinuità di fase di ~127 rad nel residuo baseband.
                if len(self.carryover) > 0:
                    accumulated.append(self.carryover)
                    total += len(self.carryover)
                    self.carryover = np.array([], dtype=np.complex64)

                while total < target_samples and not self.stop_event.is_set():
                    try:
                        chunk = iq_queue.get(timeout=TIMEOUT_PER_CHUNK_S)
                        accumulated.append(chunk)
                        total += len(chunk)
                        self.errori_consecutivi = 0
                        got_data = True
                    except queue.Empty:
                        self.errori_consecutivi += 1
                        if self.errori_consecutivi > MAX_CONSECUTIVE_ERRORS:
                            print(f"[ACQ] {self.errori_consecutivi} errori consecutivi. Reset stream...")
                            try:
                                self.radio.stop()
                            except Exception:
                                pass
                            time.sleep(1.0)
                            try:
                                self.radio.start(iq_callback, on_event=event_callback)
                            except Exception as e:
                                print(f"[ACQ] Reset fallito: {e}")
                                continue
                            self.tracker.reset()
                            self.carryover = np.array([], dtype=np.complex64)
                            self.pll_history.clear()
                            self.amp100_history.clear()
                            self.state["amp100_std"] = None
                            self.state["amp100_mean"] = None
                            self.state["amp100_stato"] = "ACQUISIZIONE"
                            self.state["pll_std_30"] = None
                            self.state["pll_mean_30"] = None
                            self.state["pll_stato"] = "ACQUISIZIONE"
                            self.errori_consecutivi = 0
                            self.ultimo_reset = timestamp_str
                            accumulated = []
                            total = 0
                            break
                        time.sleep(0.05)

                if not got_data or not accumulated:
                    continue

                concatenated = np.concatenate(accumulated)

                # Salva l'eccesso per il secondo successivo, invece di
                # scartarlo. Evita la discontinuità di fase del mixer.
                if len(concatenated) > target_samples:
                    self.carryover = concatenated[target_samples:].copy()
                else:
                    self.carryover = np.array([], dtype=np.complex64)

                iq_data = concatenated[:target_samples]

                # --- Controllo drop IQ (coda piena) ---
                # Se il callback IQ ha scartato un chunk, il PLL ha perso
                # continuità temporale. Forza un reset: il CSV scriverà
                # Reset_Flag=1 e analisi_doppler2.py escluderà il warmup
                # successivo (600 campioni) da eventuali correlazioni con flare.
                global drop_flag_callback
                with dropped_lock:
                    drop_rilevato = drop_flag_callback
                    drop_flag_callback = False

                if drop_rilevato:
                    # Reset del tracker solo se il lock è davvero debole.
                    # Un drop di 1 secondo con Amp_100Hz forte (> 60 dB)
                    # non giustifica un reset completo: basta perdere
                    # quel secondo, la fase resta continua.
                    SOGLIA_RESET_DROP_DB = 45.0

                    if amp_100hz_db < SOGLIA_RESET_DROP_DB:
                        print(f"[ACQ] Drop IQ rilevato (Amp={amp_100hz_db:.1f} dB "
                              f"< {SOGLIA_RESET_DROP_DB:.0f} dB) → reset tracker")
                        # Prova auto-ricerca della riga MSK prima del reset
                        offset_auto = self.tracker.trova_offset_msk()
                        if offset_auto is not None:
                            offset_hz, freq_picco, amp, rumore = offset_auto
                            snr = amp / rumore if rumore > 0 else 0
                            print(f"[PLL] Auto-ricerca post-drop OK: picco a +{freq_picco:.1f} Hz "
                                  f"(SNR={snr:.1f}) → offset {offset_hz:+.3f} Hz")
                            self.tracker.reset()
                            self.tracker.pll.reset(offset_hz=offset_hz)
                        else:
                            print(f"[PLL] Auto-ricerca post-drop fallita, uso default "
                                  f"{OFFSET_PLL_DEFAULT_HZ:+.2f} Hz")
                            self.tracker.reset()
                        self.tracker.reset_occurred = True
                    else:
                        print(f"[ACQ] Drop IQ rilevato (Amp={amp_100hz_db:.1f} dB, "
                              f"lock forte) → nessun reset")

                # Diagnostica: picco ADC (per verifica saturazione)
                picco_i = float(np.max(np.abs(iq_data.real)))
                picco_q = float(np.max(np.abs(iq_data.imag)))

                with overload_lock:
                    da_ackare = overload_da_ackare
                    overload_da_ackare = False

                    n_det = overload_visto_nel_s
                    ovl_flag = 1 if n_det > 0 else 0
                    overload_visto_nel_s = 0

                if da_ackare:
                    try:
                        self.radio.acknowledge_power_overload()
                    except Exception as e:
                        print(f"[ACQ] Errore ack overload: {e}")

                # Diagnostica: stampa eventi ogni 10 secondi
                if int(time.time()) % 60 == 0:
                    with overload_lock:
                        if len(eventi_log) > 0:
                            copia = list(eventi_log)
                            eventi_log.clear()
                        else:
                            copia = []
                    for ts, tipo in copia:
                        ora = time.strftime('%H:%M:%S', time.localtime(ts))
                        print(f"[EVENT_LOG] {ora}  {tipo}")
                if int(time.time()) % 60 == 0:
                    #print(f"[ADC_PEAK] I={picco_i:.0f}  Q={picco_q:.0f}  "
                    #      f"tot_eventi={overload_totali}")
                    
                    #if self.async_writer is not None:
                    #    print(f"[CSV_WRITER] scritte={self.async_writer.righe_scritte} "
                    #          f"droppate={self.async_writer.righe_droppate} "
                    #          f"coda={self.async_writer.row_queue.qsize()}")

                    qs = iq_queue.qsize()
                    qs_sec = qs * 1008 / HW_SAMPLE_RATE
                    #print(f"[QUEUE_IQ] qsize={qs} chunk (~{qs_sec:.2f} s di ritardo)")

                self.state["adc_overload"] = bool(ovl_flag)
                self.state["adc_overload_tot"] = overload_totali

                # Se il segnale è assente, forza il PLL a restare in
                # acquisizione veloce (banda larga). Senza questo, dopo
                # 30 s il PLL passa a tracking stretto (0.05 Hz) e
                # aggancia solo rumore/spurie. Quando il segnale torna,
                # la macchina a stati fa il reset e il conteggio riparte.
                if self.segnale_presente is False:
                    self.tracker.pll_contatore = 0

                t_dsp_start = time.time()
                try:
                    pwr, phase, pll_freq_hz, amp_100hz_db, filtered, s2 = self.tracker.process(iq_data)
                except Exception as e:
                    print(f"[ACQ] DSP error: {e}")
                    continue
                t_dsp = time.time() - t_dsp_start

                if DEBUG_PERF:
                    self._debug_counter += 1
                    if self._debug_counter % 60 == 0:
                        sizes = list(debug_chunk_sizes)
                        if sizes:
                            arr = np.array(sizes)
                            print(f"[DSP_TIME] {t_dsp*1000:6.1f} ms | "
                                  f"chunk: min={arr.min():6d} "
                                  f"med={int(np.median(arr)):6d} "
                                  f"max={arr.max():6d}")

                if pwr is None:
                    self.warmup_counter += 1
                    print(f"[ACQ] Warm-up... ({self.warmup_counter})")
                    continue

                # --- Auto-ricerca iniziale ---
                # Esegui solo se il segnale è PRESENTE (o se lo stato è
                # ancora sconosciuto, cioè al primo avvio in attesa della
                # macchina a stati). Se il segnale è assente, non cercare
                # la riga MSK: non c'è.
                if (not self.auto_ricerca_iniziale_fatta and
                        self.segnale_presente is not False):
                    # Aspetta che il buffer abbia almeno 12 blocchi (12 secondi)
                    if len(self.tracker.buffer_sq) >= 12:
                        # Incrementa il contatore SOLO quando si effettua
                        # un vero tentativo di ricerca (non ogni secondo).
                        self.auto_ricerca_tentativi += 1

                        offset_auto = self.tracker.trova_offset_msk()
                        if offset_auto is not None:
                            offset_hz, freq_picco, amp, rumore = offset_auto
                            snr = amp / rumore if rumore > 0 else 0

                            print(f"[PLL] Auto-ricerca OK (tentativo "
                                f"{self.auto_ricerca_tentativi}): "
                                f"picco a +{freq_picco:.2f} Hz (SNR={snr:.1f}) "
                                f"→ offset portante {offset_hz:+.3f} Hz")

                            self.tracker.pll.reset(offset_hz=offset_hz)
                            self.tracker.reset_occurred = True   # ← Reset_Flag nel CSV

                            # Lascia l'acquisizione a banda larga (1 Hz) per
                            # 30 secondi: l'auto-ricerca ha RBW 1 Hz, quindi
                            # il picco trovato può essere a 1-3 Hz dal vero
                            # massimo della riga MSK. Solo con banda larga
                            # il PLL si sposta verso il centro del picco.
                            # Dopo 30s passa automaticamente a tracking
                            # stretto (gestito in SIDTracker.process).
                            self.tracker.pll_contatore = 0
                            print(f"[PLL] Acquisizione 30s per centratura fine")

                            self.auto_ricerca_iniziale_fatta = True

                        elif self.auto_ricerca_tentativi > 5:
                            # Dopo 5 tentativi falliti, usa il default
                            print(f"[PLL] Auto-ricerca fallita dopo "
                                  f"{self.auto_ricerca_tentativi} tentativi, "
                                  f"uso default {OFFSET_PLL_DEFAULT_HZ:+.2f} Hz")
                            self.tracker.pll.reset(offset_hz=OFFSET_PLL_DEFAULT_HZ)
                            self.tracker.reset_occurred = True   # ← Reset_Flag nel CSV
                            self.auto_ricerca_iniziale_fatta = True
        

                # --- Reset dell'ampiezza a 100 Hz alla transizione ---
                # (fatto qui perché amp100_history e state appartengono
                # ad AcquisitoreSID, non a SIDTracker)
                if self.tracker.pll_contatore == 30:
                    self.amp100_history.clear()
                    self.state["amp100_stato"] = "ACQUISIZIONE"

                # --- Diagnostica PLL in tempo reale ---
                if self.tracker.pll_contatore > 30:
                    self.state["pll_modo"] = "TRACKING"
                    self.pll_history.append(pll_freq_hz)
                    if len(self.pll_history) >= 30:
                        pll_recent = list(self.pll_history)[-30:]
                        pll_std_30 = float(np.std(pll_recent))
                        pll_mean_30 = float(np.mean(pll_recent))
                        self.state["pll_std_30"] = pll_std_30
                        self.state["pll_mean_30"] = pll_mean_30

                        # Priorità 1: se il segnale è assente, il PLL
                        # è "stabile" solo perché aggancia rumore/spurie.
                        # Non mostrare "STABILE" in questo caso.
                        # Priorità 1: se la macchina a stati ha rilevato
                        # che il segnale è assente, il PLL NON è "stabile"
                        # (aggancia solo rumore). Lo stato riflette questo.
                        if self.segnale_presente is False:
                            self.state["pll_stato"] = "SEGNALE_ASSENTE"
                        elif pll_std_30 < 0.05:
                            self.state["pll_stato"] = "STABILE"
                        elif pll_std_30 < 0.5:
                            self.state["pll_stato"] = "ACCETTABILE"
                        else:
                            self.state["pll_stato"] = "INSTABILE"
                else:
                    self.state["pll_modo"] = "ACQUISIZIONE"
                    self.state["pll_stato"] = "ACQUISIZIONE"

                reset_flag = 1 if self.tracker.reset_occurred else 0
                self.async_writer.put((
                    timestamp_str, timestamp_dt, pwr, phase, pll_freq_hz,
                    reset_flag, amp_100hz_db, ovl_flag
                ))
                self.tracker.reset_occurred = False

                self.state["pwr_corrente"] = pwr
                self.state["phase_corrente"] = phase
                self.state["pll_freq_corrente"] = pll_freq_hz   # <-- AGGIUNTA
                self.state["amp_100hz_corrente"] = amp_100hz_db

                # --- Qualità del lock del PLL ---
                # Finché il PLL è in acquisizione veloce (primi 3 minuti),
                # lo stato del lock rimane "ACQUISIZIONE" per non confondere
                # l'utente con un segnale "stabile" che non è ancora quello
                # definitivo del tracking stretto.
                if self.tracker.pll_contatore > 30:
                    self.amp100_history.append(amp_100hz_db)
                    if len(self.amp100_history) >= 30:
                        amp100_std = float(np.std(self.amp100_history))
                        amp100_mean = float(np.mean(self.amp100_history))
                        self.state["amp100_std"] = amp100_std
                        self.state["amp100_mean"] = amp100_mean

                        # Priorità 1: segnale ASSENTE (media bassa).
                        # Con il TX spento, Amp_100Hz scende a 30-40 dB
                        # e il PLL si aggancia a spurie di rumore. Lo
                        # stato deve riflettere l'assenza del SEGNALE,
                        # non la "stabilità" del PLL.
                        # Lo stato "SEGNALE_ASSENTE" ora dipende dalla
                        # macchina a stati su Power_dB, non da Amp_100Hz.
                        if self.segnale_presente is False:
                            self.state["amp100_stato"] = "SEGNALE_ASSENTE"
                        elif amp_100hz_db < amp100_mean - 12.0:
                            self.state["amp100_stato"] = "PERSO"
                        elif amp100_std < 4.0:
                            self.state["amp100_stato"] = "STABILE"
                        elif amp100_std < 10.0:
                            self.state["amp100_stato"] = "AL_LIMITE"
                        else:
                            self.state["amp100_stato"] = "INSTABILE"
                    else:
                        self.state["amp100_stato"] = "ACQUISIZIONE"
                else:
                    self.state["amp100_stato"] = "ACQUISIZIONE"


                # ====================================================
                # RESET AUTOMATICO DEL PLL SU LOCK PERSO PROLUNGATO
                # ====================================================
                # Soglia sul valore MEDIO degli ultimi 60 s, non sul
                # valore istantaneo. Il valore istantaneo fluttua con
                # std ~8 dB: la sua coda supera spesso la soglia 50 dB
                # e azzera il contatore a ogni oscillazione, impedendo
                # il reset anche in assenza di segnale.

                # Aggiorna la media mobile a 60 s
                self.amp100_media_storico.append(amp_100hz_db)
                if len(self.amp100_media_storico) >= 30:
                    amp100_media_60s = float(np.mean(self.amp100_media_storico))
                else:
                    amp100_media_60s = None

                # ====================================================
                # MACCHINA A STATI "SEGNALE PRESENTE / ASSENTE"
                # basata su Power_dB (istantanea, non su Amp_100Hz).
                # Il salto di potenza TX-acceso ↔ TX-spento è di 15-20 dB,
                # inconfondibile. Amp_100Hz viene usata solo come
                # controllo di secondo livello per lo stato GUI.
                # ====================================================
                if pwr is not None:
                    # 1. Aggiorna il buffer e calcola la media mobile a 60s
                    self.pwr_media_storico.append(pwr)
                    if len(self.pwr_media_storico) >= 30:
                        pwr_smooth = float(np.mean(self.pwr_media_storico))
                    else:
                        pwr_smooth = pwr

                    # 2. INIZIALIZZAZIONE IMMEDIATA (evita il reset a 10 minuti)
                    if self.segnale_presente is None:
                        if pwr_smooth > self.POTENZA_SOGLIA_ALTA:
                            self.segnale_presente = True
                            print(f"[{timestamp_str}] [PLL] Stato iniziale: Segnale PRESENTE (Pwr_media={pwr_smooth:.1f} dB).")
                        elif pwr_smooth < self.POTENZA_SOGLIA_BASSA:
                            self.segnale_presente = False
                            print(f"[{timestamp_str}] [PLL] Stato iniziale: Segnale ASSENTE (Pwr_media={pwr_smooth:.1f} dB). PLL in attesa.")
                        # Se è nella zona grigia, rimane None e aspetta il prossimo ciclo

                    # 3. MACCHINA A STATI NORMALE (solo se lo stato è già True o False)
                    else:
                        if pwr_smooth < self.POTENZA_SOGLIA_BASSA:
                            self.potenza_sotto_soglia_da_s += 1
                            self.potenza_sopra_soglia_da_s = 0
                        elif pwr_smooth > self.POTENZA_SOGLIA_ALTA:
                            self.potenza_sopra_soglia_da_s += 1
                            self.potenza_sotto_soglia_da_s = 0
                        else:
                            self.potenza_sotto_soglia_da_s = 0
                            self.potenza_sopra_soglia_da_s = 0

                        # Transizione PRESENTE → ASSENTE
                        if (self.segnale_presente is True and
                                self.potenza_sotto_soglia_da_s >= self.DEBOUNCE_POTENZA_S):
                            print(f"[{timestamp_str}] [PLL] Segnale ASSENTE (Pwr_media={pwr_smooth:.1f} dB < {self.POTENZA_SOGLIA_BASSA} dB per {self.potenza_sotto_soglia_da_s}s). PLL in attesa.")
                            self.segnale_presente = False

                        # Transizione ASSENTE → PRESENTE
                        elif (self.segnale_presente is False and
                                self.potenza_sopra_soglia_da_s >= self.DEBOUNCE_POTENZA_S):
                            print(f"[{timestamp_str}] [PLL] Segnale PRESENTE (Pwr_media={pwr_smooth:.1f} dB > {self.POTENZA_SOGLIA_ALTA} dB per {self.potenza_sopra_soglia_da_s}s). Reset PLL + auto-ricerca.")
                            self.segnale_presente = True

                            # Reset del PLL + auto-ricerca
                            offset_auto = self.tracker.trova_offset_msk()
                            if offset_auto is not None:
                                offset_hz, freq_picco, amp_picco, rumore = offset_auto
                                snr = amp_picco / rumore if rumore > 0 else 0
                                print(f"[PLL] Auto-ricerca OK: picco a +{freq_picco:.1f} Hz (SNR={snr:.1f}) → offset {offset_hz:+.2f} Hz")
                                self.tracker.pll.reset(offset_hz=offset_hz)
                            else:
                                print(f"[PLL] Auto-ricerca fallita, uso default {OFFSET_PLL_DEFAULT_HZ:+.2f} Hz")
                                self.tracker.pll.reset(offset_hz=OFFSET_PLL_DEFAULT_HZ)
                            self.tracker.reset_occurred = True
                            self.tracker.pll_contatore = 0
                            self.pll_history.clear()
                            self.amp100_history.clear()
                            self.amp100_media_storico.clear()
                            self.auto_ricerca_iniziale_fatta = True

                    
                self.state["timestamp_corrente"] = timestamp_str
                self.state["letture_totali"] = self.letture_totali + 1
                self.state["ultimo_reset"] = self.ultimo_reset
                self.letture_totali += 1
                with dropped_lock:
                    self.state["dropped_queue_full"] = dropped_queue_full
                    self.state["dropped_excess_samples"] = dropped_excess_samples

                if self.state.get("fft_aperta", False):
                    with self.state["fft_lock"]:
                        n_fft_max = 32768
                        # Segnale normale (s2 a 20 kHz)
                        if len(s2) > n_fft_max:
                            self.state["fft_segnale"] = s2[-n_fft_max:].copy()
                        else:
                            self.state["fft_segnale"] = s2.copy()
                        # Segnale squadrato (filtered_fase² a 5 kHz)
                        sq = getattr(self.tracker, 'filtered_fase_squared', None)
                        if sq is not None:
                            if len(sq) > n_fft_max:
                                self.state["fft_segnale_sq"] = sq[-n_fft_max:].copy()
                            else:
                                self.state["fft_segnale_sq"] = sq.copy()
                    self.state["fft_nuovo_dato"] = True

                # Rimosso: il pacing è già garantito dalla coda IQ.
                # Il ciclo consuma esattamente 2.000.000 campioni, e
                # iq_queue.get() con timeout=1 s attende naturalmente
                # se la coda è vuota. Il time.sleep() aggiungeva un
                # overshoot di ~1 ms per ciclo che faceva crescere la
                # coda di ~1 chunk/s (riempimento in ~3 ore).
                # elapsed = time.time() - t_start
                # if elapsed < MEASUREMENT_INTERVAL_S:
                #     time.sleep(MEASUREMENT_INTERVAL_S - elapsed)

        except Exception as e:
            print(f"[ACQ] Errore critico: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Ferma il writer PRIMA di chiudere il CSV, così l'ultima
            # parte della coda viene svuotata correttamente.
            try:
                if self.async_writer:
                    self.async_writer.stop()
                    self.async_writer.join(timeout=5.0)
            except Exception:
                pass
            try:
                if self.csv_manager:
                    self.csv_manager.chiudi()
            except Exception:
                pass
            try:
                if self.radio:
                    self.radio.stop()
            except Exception:
                pass
            print("[ACQ] Thread acquisizione terminato.")
            self.state["radio_pronta"] = False

# ============================================================
# GUI TKINTER
# ============================================================
class SIDGui:
    def __init__(self, state, acquisitore=None):
        self.state = state
        self.acquisitore = acquisitore
        self.root = tk.Tk()

        self.root.title("SID Tracker NSY - Monitor 45.905 kHz")
        self.root.geometry("900x600")
        self.root.configure(bg='#0d0d1a')

        font_titolo = ('Consolas', 11, 'bold')
        font_label = ('Consolas', 9)
        font_valore = ('Consolas', 11, 'bold')
        font_hint = ('Consolas', 8)

        # === SEZIONE SUPERIORE ===
        frame_top = tk.Frame(self.root, bg='#0d0d1a')
        frame_top.pack(fill='x', padx=15, pady=(15, 5))

        tk.Label(frame_top, text="SID Tracker - NSY 45.905 kHz",
                 bg='#0d0d1a', fg='#00FF99', font=font_titolo).pack(anchor='w')

        self.label_stato = tk.Label(frame_top, text="Inizializzazione...",
                                     bg='#0d0d1a', fg='#FFD700', font=font_label)
        self.label_stato.pack(anchor='w', pady=(5, 0))

        self.label_file = tk.Label(frame_top, text="",
                                    bg='#0d0d1a', fg='#888888', font=font_hint)
        self.label_file.pack(anchor='w')

        # === SEZIONE CENTRALE ===
        frame_center = tk.Frame(self.root, bg='#0d0d1a')
        frame_center.pack(fill='both', expand=True, padx=15, pady=10)

        frame_pwr = tk.LabelFrame(frame_center, text=" Potenza NSY ",
                                   bg='#0d0d1a', fg='#00FF99',
                                   font=font_label, labelanchor='nw')
        frame_pwr.pack(side='left', fill='both', expand=True, padx=(0, 8))
        self.label_pwr = tk.Label(frame_pwr, text="--- dB",
                                   bg='#0d0d1a', fg='#00FF99', font=font_valore)
        self.label_pwr.pack(expand=True, pady=8)


        frame_phase = tk.LabelFrame(frame_center, text=" Fase NSY ",
                                     bg='#0d0d1a', fg='#FF6B9D',
                                     font=font_label, labelanchor='nw')
        frame_phase.pack(side='left', fill='both', expand=True, padx=(8, 0))
        self.label_phase = tk.Label(frame_phase, text="--- °",
                                     bg='#0d0d1a', fg='#FF6B9D', font=font_valore)
        self.label_phase.pack(expand=True, pady=8)

        # === SEZIONE INFERIORE ===
        frame_bot = tk.Frame(self.root, bg='#0d0d1a')
        frame_bot.pack(fill='x', padx=15, pady=(0, 15))

        self.label_stats = tk.Label(frame_bot, text="Letture: 0",
                                     bg='#0d0d1a', fg='#888888', font=font_hint)
        self.label_stats.pack(anchor='w')

        self.label_gain = tk.Label(frame_bot, text="Gain RF: ---  |  Gain IF: ---",
                                    bg='#0d0d1a', fg='#FFA500', font=font_hint)
        self.label_gain.pack(anchor='w', pady=(2, 0))

        self.label_drops = tk.Label(frame_bot, text="Campioni persi: coda=0  ecc=0",
                                     bg='#0d0d1a', fg='#FF6347', font=font_hint)
        self.label_drops.pack(anchor='w', pady=(2, 0))

        self.label_amp100 = tk.Label(frame_bot, text="Amp riga 100 Hz: --- dB",
                                      bg='#0d0d1a', fg='#00BFFF', font=font_hint)
        self.label_amp100.pack(anchor='w', pady=(2, 0))

        self.label_amp100_stato = tk.Label(frame_bot, text="Lock PLL: ---",
                                            bg='#0d0d1a', fg='#888888',
                                            font=('Consolas', 10, 'bold'))
        self.label_amp100_stato.pack(anchor='w', pady=(2, 0))

        self.label_amp100_legenda = tk.Label(
            frame_bot,
            text="< 2 dB = stabile  |  2–5 dB = al limite  |  "
                 "> 5 dB o crollo = lock perso  |  "
                 "media < 45 dB = segnale assente",
            bg='#0d0d1a', fg='#666666', font=('Consolas', 9))
        
        self.label_amp100_legenda.pack(anchor='w', pady=(0, 2))
        
        self.label_pll = tk.Label(frame_bot, text="PLL: acquisizione...",
                                   bg='#0d0d1a', fg='#888888', font=font_hint)
        self.label_pll.pack(anchor='w', pady=(5, 0))

        # Label modo PLL in basso a destra (sopra freq_offset)
        self.label_pll_modo = tk.Label(frame_bot,
                                        text="PLL modo: acquisizione veloce",
                                        bg='#0d0d1a', fg='#FFD700',
                                        font=('Consolas', 10, 'bold'))
        self.label_pll_modo.pack(side='right', anchor='e', padx=(0, 5))

        # Label freq_offset in basso a destra
        self.label_freq_offset = tk.Label(frame_bot,
                                           text="freq_offset: --- Hz",
                                           bg='#0d0d1a', fg='#00BFFF',
                                           font=('Consolas', 11, 'bold'))
        self.label_freq_offset.pack(side='right', anchor='e', padx=(0, 5))


        self.label_hint = tk.Label(frame_bot,
            text="[G] FFT  |  [A/B] Smoothing  |  [Z] Zoom Y  |  [Q] Esci",
            bg='#0d0d1a', fg='#666666', font=font_hint)
        self.label_hint.pack(anchor='w', pady=(5, 0))

        self.label_hint_gain = tk.Label(frame_bot,
            text="Gain:  0 (LNA=0 - IF=30 dB) |  6 (LNA=6 - IF=59 dB)  |  1-5=intermedi  |  7-9=MIN",
            bg='#0d0d1a', fg='#FFA500', font=font_hint)
        self.label_hint_gain.pack(anchor='w', pady=(2, 0))

        # Finestra FFT
        self.fft_window = None
        self.fft_canvas = None
        self.fft_ax = None
        self.fft_line = None
        self.fft_testo_picco = None
        self.fft_testo_snr = None
        self.fft_spettro_smoothed = None
        self.fft_freqs_smoothed = None

        # === BINDING TASTIERA SULLA FINESTRA PRINCIPALE ===
        self._bind_all_keys(self.root)

        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.root.focus_set()

        self.aggiorna_etichette()

    def _bind_all_keys(self, widget):
        """Registra tutti i binding dei tasti su un widget (root o Toplevel)."""
        widget.bind('<KeyPress-g>', self.toggle_fft)
        widget.bind('<KeyPress-G>', self.toggle_fft)
        widget.bind('<KeyPress-q>', self.quit_app)
        widget.bind('<KeyPress-Q>', self.quit_app)
        # Tasti +/- (multipli, per compatibilità internazionale)
        widget.bind('<KeyPress-plus>', self.aumenta_smoothing)
        widget.bind('<KeyPress-equal>', self.aumenta_smoothing)
        widget.bind('<KeyPress-KP_Add>', self.aumenta_smoothing)
        widget.bind('<KeyPress-minus>', self.diminuisci_smoothing)
        widget.bind('<KeyPress-underscore>', self.diminuisci_smoothing)
        widget.bind('<KeyPress-KP_Subtract>', self.diminuisci_smoothing)
        # Tasti A/D: zoom banda FFT (A = zoom in, D = zoom out)
        widget.bind('<KeyPress-a>', self.fft_zoom_in)
        widget.bind('<KeyPress-A>', self.fft_zoom_in)
        widget.bind('<KeyPress-d>', self.fft_zoom_out)
        widget.bind('<KeyPress-D>', self.fft_zoom_out)
        # Tasto B: diminuisci smoothing
        widget.bind('<KeyPress-b>', self.diminuisci_smoothing)
        widget.bind('<KeyPress-B>', self.diminuisci_smoothing)
        # Zoom Y
        widget.bind('<KeyPress-z>', self.cicla_zoom_y)
        widget.bind('<KeyPress-Z>', self.cicla_zoom_y)
        # Tasto M: commuta FFT normale / squadrato
        #widget.bind('<KeyPress-m>', self.toggle_fft_mode)
        #widget.bind('<KeyPress-M>', self.toggle_fft_mode)
        # Tasti 0-9 per il guadagno LNA/IF
        for n in range(10):
            widget.bind(f'<KeyPress-{n}>', lambda e, k=n: self.cambia_gain(k))

    def aggiorna_etichette(self):
        pwr = self.state.get("pwr_corrente")
        phase = self.state.get("phase_corrente")
        ts = self.state.get("timestamp_corrente", "")
        letture = self.state.get("letture_totali", 0)
        path_csv = self.state.get("path_csv", "")
        radio_pronta = self.state.get("radio_pronta", False)
        ultimo_reset = self.state.get("ultimo_reset")

        if radio_pronta:
            if self.state.get("adc_overload"):
                self.label_stato.config(
                    text=f"⚠ ADC OVERLOAD (tot={self.state.get('adc_overload_tot', 0)})  |  Ultimo dato: {ts}",
                    fg='#FF3030')
            else:
                self.label_stato.config(text=f"Radio attiva  |  Ultimo dato: {ts}",
                                         fg='#00FF99')
        else:
            self.label_stato.config(text="Inizializzazione radio...", fg='#FFD700')

        if pwr is not None:
            self.label_pwr.config(text=f"{pwr:+.2f} dB")
        if phase is not None:
            self.label_phase.config(text=f"{phase:+.2f}°")

        stats_text = f"Letture: {letture}"
        if ultimo_reset:
            stats_text += f"  |  Ultimo reset: {ultimo_reset}"
        self.label_stats.config(text=stats_text)

        # --- Gain RF e IF (con valori di riferimento) ---
        gain_lna = self.state.get("gain_lna")
        gain_if  = self.state.get("gain_if")
        if gain_lna is not None and gain_if is not None:
            testo_gain = (
                f"Gain RF: LNA={gain_lna} [0=max, 6=min]  |  "
                f"Gain IF: {gain_if} dB [20=max, 59=min]"
            )
        else:
            testo_gain = (
                "Gain RF: LNA=--- [0=max, 6=min]  |  "
                "Gain IF: --- dB [20=max, 59=min]"
            )

        self.label_gain.config(text=testo_gain)
        # --- Contatori campioni persi ---
        dq = self.state.get("dropped_queue_full", 0)
        de = self.state.get("dropped_excess_samples", 0)
        self.label_drops.config(text=f"Campioni persi: coda={dq}  ecc={de}")

        # --- Ampiezza riga 100 Hz (qualità lock PLL) ---
        amp100 = self.state.get("amp_100hz_corrente")
        if amp100 is not None:
            self.label_amp100.config(text=f"Amp riga 100 Hz: {amp100:+.2f} dB")
        else:
            self.label_amp100.config(text="Amp riga 100 Hz: --- dB")

        # --- Stato del lock PLL ---
        stato_amp100 = self.state.get("amp100_stato")
        std_amp100   = self.state.get("amp100_std")
        mean_amp100  = self.state.get("amp100_mean")

        # Priorità assoluta: se il segnale NSY è assente (TX spento,
        # propagazione pessima, antenna scollegata), il PLL "stabile"
        # è solo un aggancio al rumore. Mostriamo SEGNALE ASSENTE
        # anche se std/mean sembrano normali.
        if stato_amp100 == "SEGNALE_ASSENTE":
            testo = (f"Lock PLL: ⊘ SEGNALE ASSENTE (attesa TX)  |  "
                     f"media={mean_amp100:+.1f} dB")
            colore = '#6666FF'
        elif stato_amp100 == "STABILE" and std_amp100 is not None:
            testo = f"Lock PLL: ✓ STABILE  |  std={std_amp100:.2f} dB  media={mean_amp100:+.1f} dB"
            colore = '#00FF99'
        elif stato_amp100 == "AL_LIMITE" and std_amp100 is not None:
            testo = f"Lock PLL: ⚠ AL LIMITE  |  std={std_amp100:.2f} dB  media={mean_amp100:+.1f} dB"
            colore = '#FFD700'
        elif stato_amp100 == "INSTABILE" and std_amp100 is not None:
            testo = f"Lock PLL: ✗ INSTABILE  |  std={std_amp100:.2f} dB"
            colore = '#FF6347'
        elif stato_amp100 == "PERSO":
            testo = "Lock PLL: ✗ LOCK PERSO (valore molto sotto la media)"
            colore = '#FF0000'
        elif stato_amp100 == "ACQUISIZIONE":
            testo = "Lock PLL: ⏳ in acquisizione..."
            colore = '#888888'
        else:
            testo = "Lock PLL: ---"
            colore = '#888888'

        self.label_amp100_stato.config(text=testo, fg=colore)

        self.label_file.config(text=f"File: {path_csv}")

        # --- Aggiorna label PLL ---
        pll_stato = self.state.get("pll_stato", None)
        pll_std = self.state.get("pll_std_30", None)
        pll_mean = self.state.get("pll_mean_30", None)

        if pll_stato == "SEGNALE_ASSENTE":
            testo_pll = "⊘ PLL IN ATTESA (segnale assente)"
            colore_pll = '#6666FF'
        elif pll_stato == "STABILE" and pll_std is not None:
            testo_pll = (f"✓ PLL STABILE  |  media={pll_mean:+.4f} Hz  "
                         f"std={pll_std:.4f} Hz")
            colore_pll = '#00FF99'
        elif pll_stato == "ACCETTABILE" and pll_std is not None:
            testo_pll = (f"⚠ PLL ACCETTABILE  |  media={pll_mean:+.4f} Hz  "
                         f"std={pll_std:.4f} Hz")
            colore_pll = '#FFD700'
        elif pll_stato == "INSTABILE" and pll_std is not None:
            testo_pll = (f"✗ PLL INSTABILE  |  media={pll_mean:+.4f} Hz  "
                         f"std={pll_std:.4f} Hz")
            colore_pll = '#FF6347'
        elif pll_stato == "ACQUISIZIONE":
            testo_pll = "⏳ PLL in acquisizione (attendi 3 minuti)..."
            colore_pll = '#888888'
        else:
            testo_pll = "PLL: in attesa..."
            colore_pll = '#888888'

        self.label_pll.config(text=testo_pll, fg=colore_pll)
        # --- Aggiorna label freq_offset ---
        pll_freq_corrente = self.state.get("pll_freq_corrente", None)
        if pll_freq_corrente is not None:
            self.label_freq_offset.config(text=f"freq_offset: {pll_freq_corrente:+.4f} Hz")
        else:
            self.label_freq_offset.config(text="freq_offset: --- Hz")

        # --- Aggiorna label modo PLL ---
        pll_modo = self.state.get("pll_modo", "ACQUISIZIONE")
        if pll_modo == "TRACKING":
            self.label_pll_modo.config(
                text="PLL modo: tracking stretto (oscillazioni ridotte)",
                fg='#00FF99'
            )
        else:
            self.label_pll_modo.config(
                text="PLL modo: acquisizione veloce",
                fg='#FFD700'
            )

        if self.state.get("fft_aperta", False):
            self.aggiorna_fft()

        # Aggiorna ogni 1000 ms (era 500) per ridurre la contesa del GIL
        # col thread di acquisizione: ogni update GUI blocca il DSP per
        # ~20-50 ms. Dimezzando la frequenza si recupera ~15 ms/s di CPU.
        self.root.after(1000, self.aggiorna_etichette)

    # ========================================================
    # FINESTRA FFT
    # ========================================================
    def toggle_fft(self, event=None):
        if self.state.get("fft_aperta", False):
            self.chiudi_fft()
        else:
            self.apri_fft()

    def apri_fft(self):
        print("[GUI] Apertura finestra FFT...")
        self.state["fft_aperta"] = True
        with self.state["fft_lock"]:
            self.state["fft_segnale"] = None

        # Reset dello smoothing quando apri la finestra
        self.fft_spettro_smoothed = None
        self.fft_freqs_smoothed = None

        self.fft_window = tk.Toplevel(self.root)
        self.fft_window.title("FFT in tempo reale - NSY 45.905 kHz")
        self.fft_window.geometry("1100x550")
        self.fft_window.configure(bg='#0d0d1a')
        self.fft_window.protocol("WM_DELETE_WINDOW", self.chiudi_fft)

        fig = Figure(figsize=(11, 5), dpi=100, facecolor='#0d0d1a')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#0d0d1a')
        ax.tick_params(colors='white', labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor('#444444')
        ax.grid(True, alpha=0.15, color='white')
        ax.set_xlim(FFT_XMIN_HZ, FFT_XMAX_HZ)
        ax.set_ylim(FFT_YMIN_DB, FFT_YMAX_DB)
        ax.set_xlabel('Frequenza (Hz) rispetto a NSY', color='white', fontsize=10)
        ax.set_ylabel('Ampiezza (dB)', color='white', fontsize=10)
        ax.set_title('FFT — G chiude  |  A/D zoom banda  |  +/- smoothing  |  Z zoom Y',
                     color='#00FF99', fontsize=11, fontweight='bold')

        # Linea di riferimento NSY (0 Hz)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.0, alpha=0.7)

        # Bordi del filtro FIR (±70 Hz)
        ax.axvline(-FIR_CUTOFF_HZ, color='#FFA500',
                   linestyle=':', linewidth=0.8, alpha=0.6)
        ax.axvline(+FIR_CUTOFF_HZ, color='#FFA500',
                   linestyle=':', linewidth=0.8, alpha=0.6)

        # Legenda
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color='red', linestyle='--', label='NSY (0 Hz)'),
            Line2D([0], [0], color='#FFA500', linestyle=':',
                   label=f'Banda FIR (±{FIR_CUTOFF_HZ:.0f} Hz)'),
        ]
        ax.legend(handles=handles, loc='upper right', facecolor='#1a1a2e',
                  edgecolor='#444444', labelcolor='white', fontsize=9)

        # --- Traccia vuota dello spettro (verrà riempita da aggiorna_fft) ---
        linea, = ax.plot([], [], color='#00FF99', linewidth=0.9)

        # --- Testi diagnostici in basso a destra ---
        testo_picco = ax.text(0.98, 0.02, 'Attendo dati...',
                              transform=ax.transAxes, color='#00FF99',
                              fontsize=10, va='bottom', ha='right',
                              bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor='#1a1a2e', edgecolor='#00FF99'))
        testo_snr = ax.text(0.98, 0.12, '', transform=ax.transAxes,
                            color='#FFD700', fontsize=9, va='bottom', ha='right')

        # --- Hint funzioni tasti in basso a sinistra ---
        ax.text(0.02, 0.02, "A = zoom in banda  |  D = zoom out banda",
                transform=ax.transAxes, color='#00FF99',
                fontsize=9, fontweight='bold',
                va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='#1a1a2e', edgecolor='#00FF99', alpha=0.8))

        self.fft_ax = ax
        self.fft_line = linea
        self.fft_testo_picco = testo_picco
        self.fft_testo_snr = testo_snr

        self.fft_canvas = FigureCanvasTkAgg(fig, master=self.fft_window)
        self.fft_canvas.draw()
        self.fft_canvas.get_tk_widget().pack(fill='both', expand=True)

        # === BINDING SULLA FINESTRA FFT ===
        self._bind_all_keys(self.fft_window)

        # === BINDING MATPLOTLIB SUL CANVAS ===
        self.fft_canvas.mpl_connect('key_press_event', self._on_key_matplotlib)

        # === FORCED FOCUS ===
        self.fft_window.focus_set()
        self.fft_window.lift()
        self.fft_canvas.get_tk_widget().focus_set()

        print("[GUI] Finestra FFT aperta. Clicca dentro per attivare i tasti.")

    def _on_key_matplotlib(self, event):
        """Gestisce i tasti ricevuti dal canvas matplotlib."""
        if event.key is None:
            return
        key = event.key
        if key in ('g', 'G'):
            self.toggle_fft()
        elif key in ('q', 'Q'):
            self.quit_app()
        elif key in ('a', 'A'):
            self.fft_zoom_in()
        elif key in ('d', 'D'):
            self.fft_zoom_out()
        elif key in ('+', '='):
            self.aumenta_smoothing()
        elif key in ('-', '_', 'b', 'B'):
            self.diminuisci_smoothing()
        elif key in ('z', 'Z'):
            self.cicla_zoom_y()
        elif key in ('m', 'M'):
            self.toggle_fft_mode()

    def aggiorna_fft(self):
        """Aggiorna la FFT con s2 (modo normale) o filtered_fase² (modo squadrato)."""
        if not self.state.get("fft_aperta", False):
            return
        if not self.state.get("fft_nuovo_dato", False):
            return

        # ---- Selezione del segnale in base al modo ----
        modo = self.state.get("fft_modo", "normale")
        with self.state["fft_lock"]:
            if modo == "squadrato":
                segnale = self.state.get("fft_segnale_sq")
            else:
                segnale = self.state.get("fft_segnale")
            self.state["fft_nuovo_dato"] = False

        if segnale is None or len(segnale) < 512:
            return

        # Sample rate dipendente dal modo:
        #   - normale:   s2 è a 20 kHz
        #   - squadrato: filtered_fase² è a 5 kHz
        if modo == "squadrato":
            fs_corrente = FINAL_SR
        else:
            fs_corrente = FS_FFT

        # ---- FFT grezza ----
        n = min(len(segnale), FFT_N)
        segmento = segnale[-n:].copy()
        segmento = segmento - np.mean(segmento)

        window = signal.windows.hann(n, sym=False)
        spec = np.fft.fftshift(np.fft.fft(segmento * window))
        freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs_corrente))

        amp = np.abs(spec) / n

        # ---- Raggruppamento in bin (RBW) ----
        if FFT_BIN_GROUP > 1:
            n_gruppi = len(amp) // FFT_BIN_GROUP
            if n_gruppi < 4:
                freqs_grupped = freqs
                power_db_grupped = 20 * np.log10(amp + 1e-12)
            else:
                amp_grupped = amp[:n_gruppi * FFT_BIN_GROUP].reshape(n_gruppi, FFT_BIN_GROUP).mean(axis=1)
                freqs_grupped = freqs[:n_gruppi * FFT_BIN_GROUP].reshape(n_gruppi, FFT_BIN_GROUP).mean(axis=1)
                power_db_grupped = 20 * np.log10(amp_grupped + 1e-12)
        else:
            freqs_grupped = freqs
            power_db_grupped = 20 * np.log10(amp + 1e-12)

        # Normalizza al massimo
        power_db_grupped -= np.max(power_db_grupped)

        # ---- Smoothing esponenziale tra frame ----
        if self.fft_spettro_smoothed is None or len(self.fft_spettro_smoothed) != len(power_db_grupped):
            self.fft_spettro_smoothed = power_db_grupped.copy()
            self.fft_freqs_smoothed = freqs_grupped.copy()
        else:
            self.fft_spettro_smoothed = (
                FFT_SMOOTH_ALPHA * power_db_grupped +
                (1.0 - FFT_SMOOTH_ALPHA) * self.fft_spettro_smoothed
            )

        # ---- Aggiorna grafico ----
        self.fft_line.set_data(self.fft_freqs_smoothed, self.fft_spettro_smoothed)

        # RBW effettivo (dipende dal modo)
        rbw_hz = fs_corrente / (n / FFT_BIN_GROUP)

        # ============================================================
        # INFO FFT: testo diagnostico dipendente dal modo
        # ============================================================
        if modo == "squadrato":
            # Nel modo squadrato cerchiamo la riga MSK a ~+100 Hz
            idx_picco = int(np.argmax(self.fft_spettro_smoothed))
            freq_picco = float(self.fft_freqs_smoothed[idx_picco])
            picco_db = float(self.fft_spettro_smoothed[idx_picco])
            rumore = float(np.median(self.fft_spettro_smoothed))

            self.fft_testo_picco.set_text(
                f"MODO SQUADRATO | Picco: {picco_db:+.1f} dB "
                f"@ {freq_picco:+.1f} Hz  |  RBW≈{rbw_hz:.1f} Hz"
            )
            self.fft_testo_snr.set_text(
                f"Riga MSK attesa a ~+100 Hz (baud 200)  |  "
                f"Rumore: {rumore:+.1f} dB  |  α={FFT_SMOOTH_ALPHA:.2f}"
            )
        else:
            # Modo normale: stesso comportamento di prima
            mask_banda = np.abs(self.fft_freqs_smoothed) <= FIR_CUTOFF_HZ
            power_integrata_db = None
            picco_db = None
            freq_picco = None

            if np.any(mask_banda):
                power_banda = self.fft_spettro_smoothed[mask_banda]
                freqs_banda = self.fft_freqs_smoothed[mask_banda]
                media_lineare = np.mean(10 ** (power_banda / 10.0))
                power_integrata_db = 10 * np.log10(media_lineare + 1e-12)
                idx_picco_locale = int(np.argmax(power_banda))
                picco_db = float(power_banda[idx_picco_locale])
                freq_picco = float(freqs_banda[idx_picco_locale])

            mask_rumore = ((np.abs(self.fft_freqs_smoothed) > 500) &
                           (np.abs(self.fft_freqs_smoothed) < 5000))
            rumore = None
            if np.any(mask_rumore):
                rumore = float(np.median(self.fft_spettro_smoothed[mask_rumore]))

            if power_integrata_db is not None:
                self.fft_testo_picco.set_text(
                    f"Potenza ±{FIR_CUTOFF_HZ:.0f} Hz: {power_integrata_db:+.1f} dB  |  "
                    f"Picco: {picco_db:+.1f} dB @ {freq_picco:+.1f} Hz  |  "
                    f"RBW≈{rbw_hz:.1f} Hz"
                )
            else:
                self.fft_testo_picco.set_text(
                    f"Nessun bin in ±{FIR_CUTOFF_HZ:.0f} Hz  |  RBW≈{rbw_hz:.1f} Hz"
                )

            if rumore is not None:
                parti = []
                if picco_db is not None:
                    parti.append(f"SNR(picco): {picco_db - rumore:.1f} dB")
                if power_integrata_db is not None:
                    parti.append(f"SNR(banda): {power_integrata_db - rumore:.1f} dB")
                parti.append(f"Rumore: {rumore:.1f} dB")
                parti.append(f"α={FFT_SMOOTH_ALPHA:.2f}")
                self.fft_testo_snr.set_text("  |  ".join(parti))

        # ---- Aggiorna il canvas ----
        try:
            self.fft_canvas.draw_idle()
        except Exception:
            pass

    def chiudi_fft(self):
        print("[GUI] Chiusura finestra FFT...")
        self.state["fft_aperta"] = False
        with self.state["fft_lock"]:
            self.state["fft_segnale"] = None
        if self.fft_window is not None:
            try:
                self.fft_window.destroy()
            except Exception:
                pass
        self.fft_window = None
        self.fft_canvas = None
        self.fft_ax = None
        self.fft_line = None
        self.fft_testo_picco = None
        self.fft_testo_snr = None
        self.fft_spettro_smoothed = None
        self.fft_freqs_smoothed = None
        # Riporta il focus alla finestra principale
        try:
            self.root.focus_set()
        except Exception:
            pass
        print("[GUI] Finestra FFT chiusa.")

    # ========================================================
    # CONTROLLI SMOOTHING E ZOOM
    # ========================================================
    def aumenta_smoothing(self, event=None):
        """Diminuisce alpha (curva più liscia)."""
        global FFT_SMOOTH_ALPHA
        FFT_SMOOTH_ALPHA = max(0.02, FFT_SMOOTH_ALPHA - 0.03)
        print(f"[FFT] Smoothing α={FFT_SMOOTH_ALPHA:.2f} (più liscio)")

    def diminuisci_smoothing(self, event=None):
        """Aumenta alpha (curva più reattiva)."""
        global FFT_SMOOTH_ALPHA
        FFT_SMOOTH_ALPHA = min(1.0, FFT_SMOOTH_ALPHA + 0.03)
        print(f"[FFT] Smoothing α={FFT_SMOOTH_ALPHA:.2f} (più reattivo)")

    def cicla_zoom_y(self, event=None):
        """Cicla tra i livelli di zoom dell'asse Y."""
        global FFT_Y_ZOOM_INDEX, FFT_YMIN_DB, FFT_YMAX_DB
        FFT_Y_ZOOM_INDEX = (FFT_Y_ZOOM_INDEX + 1) % len(FFT_Y_ZOOM_LEVELS)
        FFT_YMIN_DB, FFT_YMAX_DB = FFT_Y_ZOOM_LEVELS[FFT_Y_ZOOM_INDEX]
        if self.fft_ax is not None:
            try:
                self.fft_ax.set_ylim(FFT_YMIN_DB, FFT_YMAX_DB)
            except Exception:
                pass
        print(f"[FFT] Zoom Y: [{FFT_YMIN_DB:.0f}, {FFT_YMAX_DB:.0f}] dB")

    def cambia_gain(self, n):
        """
        Cambia il gain RF del ricevitore (tasti 0-9).

        Mapping:
          - n=0  → LNA=0 (max), IF_GR=30 dB (configurazione di lavoro)
          - n=9  → LNA=9 (min), IF_GR=59 dB (gain minimo, segnale assente)
          - n intermedi → interpolati linearmente

        L'applicazione effettiva avviene nel thread di acquisizione
        (la GUI non può chiamare radio.set_gain() direttamente).
        """
        if n < 0 or n > 9:
            return

        # LNA ha 7 livelli validi (0-6) per RSP1A.
        # Tasti 0-6 → LNA 0-6 direttamente. Tasti 7-9 → clamp a 6.
        lna = min(n, 6)
        # IF_GR interpolato da 30 (tasto 0) a 59 (tasto 6), poi resta 59
        if n <= 6:
            if_gr = int(30 + (n / 6.0) * (59 - 30))
        else:
            if_gr = 59

        self.state["gain_richiesto_lna"] = lna
        self.state["gain_richiesto_if"] = if_gr
        self.state["gain_change_pending"] = True

        # Aggiorna subito l'etichetta nella GUI (feedback visivo)
        self.label_gain.config(
            text=f"Gain RF: LNA={lna} [0=max, 9=min]  |  "
                 f"Gain IF: {if_gr} dB [20=max, 59=min]  "
                 f"(in applicazione...)",
            fg='#FFD700'
        )

        print(f"[GUI] Richiesta cambio gain: tasto {n} → LNA={lna}, IF_GR={if_gr} dB")
        
    def toggle_fft_mode(self, event=None):
        """Commuta la FFT tra segnale normale (s2) e segnale squadrato (fase²)."""
        modo_attuale = self.state.get("fft_modo", "normale")
        nuovo_modo = "squadrato" if modo_attuale == "normale" else "normale"
        self.state["fft_modo"] = nuovo_modo

        # Reset dello smoothing: i due modi hanno lunghezze diverse
        self.fft_spettro_smoothed = None
        self.fft_freqs_smoothed = None

        if self.fft_ax is not None:
            if nuovo_modo == "squadrato":
                self.fft_ax.set_title(
                    'FFT MODO SQUADRATO (fase²) — M torna a normale  |  '
                    'A/D zoom  |  +/- smoothing  |  Z zoom Y',
                    color='#00BFFF', fontsize=11, fontweight='bold')
                # Nel modo squadrato la riga MSK attesa è a ~+100 Hz:
                # conviene una finestra più stretta.
                self.fft_ax.set_xlim(-500, +500)
            else:
                self.fft_ax.set_title(
                    'FFT MODO NORMALE (s2) — M passa a squadrato  |  '
                    'A/D zoom  |  +/- smoothing  |  Z zoom Y',
                    color='#00FF99', fontsize=11, fontweight='bold')
                self.fft_ax.set_xlim(FFT_XMIN_HZ, FFT_XMAX_HZ)
            try:
                self.fft_canvas.draw_idle()
            except Exception:
                pass
        print(f"[FFT] Modo: {nuovo_modo}")

    def fft_zoom_in(self, event=None):
        """Riduce la banda visualizzata della FFT (zoom in)."""
        global FFT_X_ZOOM_INDEX, FFT_XMIN_HZ, FFT_XMAX_HZ
        if FFT_X_ZOOM_INDEX > 0:
            FFT_X_ZOOM_INDEX -= 1
            half = FFT_X_ZOOM_LEVELS_HZ[FFT_X_ZOOM_INDEX]
            FFT_XMIN_HZ = -half
            FFT_XMAX_HZ = +half
            if self.fft_ax is not None:
                try:
                    self.fft_ax.set_xlim(FFT_XMIN_HZ, FFT_XMAX_HZ)
                    self.fft_canvas.draw_idle()
                except Exception:
                    pass
            print(f"[FFT] Zoom in: banda ±{half} Hz")
        else:
            print(f"[FFT] Zoom in: già al massimo (±{FFT_X_ZOOM_LEVELS_HZ[0]} Hz)")

    def fft_zoom_out(self, event=None):
        """Aumenta la banda visualizzata della FFT (zoom out)."""
        global FFT_X_ZOOM_INDEX, FFT_XMIN_HZ, FFT_XMAX_HZ
        if FFT_X_ZOOM_INDEX < len(FFT_X_ZOOM_LEVELS_HZ) - 1:
            FFT_X_ZOOM_INDEX += 1
            half = FFT_X_ZOOM_LEVELS_HZ[FFT_X_ZOOM_INDEX]
            FFT_XMIN_HZ = -half
            FFT_XMAX_HZ = +half
            if self.fft_ax is not None:
                try:
                    self.fft_ax.set_xlim(FFT_XMIN_HZ, FFT_XMAX_HZ)
                    self.fft_canvas.draw_idle()
                except Exception:
                    pass
            print(f"[FFT] Zoom out: banda ±{half} Hz")
        else:
            print(f"[FFT] Zoom out: già al minimo (±{FFT_X_ZOOM_LEVELS_HZ[-1]} Hz)")

    def quit_app(self, event=None):
        print("\n[GUI] Chiusura applicazione...")
        self.state["quit"] = True
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        self.root.mainloop()

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("SID Tracker - NSY 45.905 kHz (GUI)")
    print("=" * 60)
    print(f"Sintonia hardware : {TUNE_FREQ:.3f} Hz")
    print(f"Offset mixdown    : {MIX_SIGN * OFFSET_HZ:+.3f} Hz")
    print(f"Correzione offset : {OFFSET_HZ_CORRECTION:+.6f} Hz")
    print(f"Sample rate DSP   : {FINAL_SR/1000:.1f} kHz")
    print(f"Sample rate FFT   : {FS_FFT/1000:.1f} kHz")
    print(f"FIR               : +/-{FIR_CUTOFF_HZ} Hz, {FIR_TAPS} taps")
    print(f"Cartella output   : {CARTELLA_OUTPUT}")
    print(f"Prefisso file     : {PREFISSO_FILE}_YYYY-MM-DD.csv")
    print("=" * 60 + "\n")

    state = {
        "radio_pronta": False,
        "adc_overload": False,
        "adc_overload_tot": 0,
        "pwr_corrente": None,
        "phase_corrente": None,
        "timestamp_corrente": "",
        "letture_totali": 0,
        "ultimo_reset": None,
        "path_csv": "",
        "quit": False,
        "fft_aperta": False,
        "fft_segnale": None,
        "fft_segnale_sq": None,       # segnale squadrato per FFT in modalità MSK
        "fft_modo": "normale",        # "normale" o "squadrato"
        "fft_nuovo_dato": False,
        "fft_lock": threading.Lock(),
        # Diagnostica PLL
        "pll_std_30": None,
        "pll_mean_30": None,
        "pll_stato": None,      
        "pll_freq_corrente": None,   
        "pll_modo": "ACQUISIZIONE",
        "amp_100hz_corrente": None,
        # Gain radio
        "gain_lna": None,
        "gain_if": None,
        # Diagnostica lock PLL (riga 100 Hz)
        "amp100_std": None,
        "amp100_mean": None,
        "amp100_stato": None,
        # Contatori campioni persi
        "dropped_queue_full": 0,
        "dropped_excess_samples": 0,
        "gain_lna": None,
        "gain_if": None,
        "gain_richiesto_lna": None,   # comando in attesa dalla GUI
        "gain_richiesto_if": None,
        "gain_change_pending": False,
    }

    acquisitore = AcquisitoreSID(state)
    acquisitore.start()

    gui = SIDGui(state, acquisitore=acquisitore)
    try:
        gui.run()
    except KeyboardInterrupt:
        print("\n[MAIN] CTRL+C ricevuto.")
    finally:
        acquisitore.stop()
        acquisitore.join(timeout=3.0)
        print("[MAIN] Chiusura completata.")


if __name__ == "__main__":
    main()