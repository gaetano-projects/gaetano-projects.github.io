# ============================================================
# sid_tracker_duale.py
# Tracker Multi-PLL V1 + V2 + BW02 + BW01 + PESATO (v5.0)
# ============================================================

import os
import sys
import time
import csv
import datetime
import queue
import threading
import logging
import numpy as np
import scipy.signal as signal
from collections import deque

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

PYSDRPLAY_DIR = r"D:\Gaetano\SID\pysdrplay"
CARTELLA_LOG = r"D:\Gaetano\SID\dati\Sdruno\Log"
if os.path.exists(PYSDRPLAY_DIR):
    sys.path.insert(0, PYSDRPLAY_DIR)

from pysdrplay import SDRPlayRadio, Event, PowerOverloadCbEventId, Bw

# ============================================================
# LOGGING (patch diagnostica - non tocca DSP/soglie/logica)
# Log giornaliero UTC: tracker_YYYY-MM-DD.log
# ============================================================
LOGGER_NAME = "sid_tracker"
logger = logging.getLogger(LOGGER_NAME)


class DailyUTCFileHandler(logging.Handler):
    """
    File handler che scrive su <base>_YYYY-MM-DD.log (UTC).
    Ruota automaticamente a mezzanotte UTC, in modo lazy: il cambio
    avviene alla prima scrittura dopo le 00:00 UTC. Nessun thread.
    Apre in append: se il tracker viene riavviato lo stesso giorno,
    il log continua nello stesso file (i riavvii si vedono come
    nuove righe 'Avvio sessione').
    """
    def __init__(self, log_dir, base_name="tracker", encoding="utf-8"):
        super().__init__()
        self.log_dir = log_dir
        self.base_name = base_name
        self.encoding = encoding
        self._current_date = None
        self._stream = None
        self._ensure_open()

    @staticmethod
    def _today_utc():
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

    def _path_for(self, date_str):
        return os.path.join(self.log_dir, f"{self.base_name}_{date_str}.log")

    def _ensure_open(self):
        today = self._today_utc()
        if today == self._current_date and self._stream is not None:
            return
        try:
            if self._stream is not None:
                self._stream.close()
        except Exception:
            pass
        self._stream = None
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except Exception:
            pass
        try:
            self._stream = open(self._path_for(today), "a", encoding=self.encoding)
            self._current_date = today
        except Exception:
            self._stream = None
            self._current_date = None

    def emit(self, record):
        try:
            self._ensure_open()
            if self._stream is None:
                return
            self._stream.write(self.format(record) + "\n")
            self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        try:
            if self._stream is not None:
                self._stream.close()
        except Exception:
            pass
        self._stream = None
        super().close()


class _StreamToLogger:
    """File-like: scrive sia sullo stream originale sia sul logger."""
    def __init__(self, original_stream, log_method):
        self._original = original_stream
        self._log = log_method
        self._buffer = ""

    def write(self, msg):
        if self._original is not None:
            try:
                self._original.write(msg)
                self._original.flush()
            except Exception:
                pass
        self._buffer += msg
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                try:
                    self._log(line)
                except Exception:
                    pass

    def flush(self):
        if self._original is not None:
            try:
                self._original.flush()
            except Exception:
                pass
        if self._buffer.strip():
            try:
                self._log(self._buffer)
            except Exception:
                pass
            self._buffer = ""

    def isatty(self):
        return self._original.isatty() if self._original is not None else False


def _install_excepthooks():
    """Installa excepthook per main thread e per i thread: logga traceback completi."""
    def _log_uncaught(exc_type, exc_value, exc_tb, source):
        try:
            logger.critical(
                "Eccezione non gestita (%s): %s: %s",
                source, exc_type.__name__, exc_value,
                exc_info=(exc_type, exc_value, exc_tb),
            )
        except Exception:
            pass

    def _sys_excepthook(exc_type, exc_value, exc_tb):
        _log_uncaught(exc_type, exc_value, exc_tb, "main")

    def _thread_excepthook(args):
        _log_uncaught(
            args.exc_type, args.exc_value, args.exc_traceback,
            f"thread:{args.thread.name if args.thread else '?'}",
        )

    sys.excepthook = _sys_excepthook
    threading.excepthook = _thread_excepthook


def setup_logging():
    """Configura logging su file giornaliero UTC. Idempotente. Da chiamare in main()."""
    if logger.handlers:
        return

    logger.setLevel(logging.DEBUG)

    try:
        os.makedirs(CARTELLA_LOG, exist_ok=True)
    except Exception:
        pass

    fh = DailyUTCFileHandler(CARTELLA_LOG, base_name="tracker", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(threadName)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False

    _install_excepthooks()

    # Redirect "tee": console + file. Tutti i print() esistenti finiscono nel log.
    sys.stdout = _StreamToLogger(sys.stdout, logger.info)
    sys.stderr = _StreamToLogger(sys.stderr, logger.error)

    today = DailyUTCFileHandler._today_utc()
    logger.info("=" * 60)
    logger.info("Avvio sessione tracker (log giornaliero: %s)", os.path.join(CARTELLA_LOG, f"tracker_{today}.log"))
    logger.info("=" * 60)

# ============================================================
# CONFIGURAZIONE
# ============================================================
TARGET_FREQ = 45_905.0
OFFSET_HZ = 20000.0
OFFSET_HZ_CORRECTION = 0.0
TUNE_FREQ = TARGET_FREQ + OFFSET_HZ
MIX_SIGN = +1.0

HW_SAMPLE_RATE = 2_000_000
IF_BANDWIDTH = Bw.BW_0_200
DECIM_STAGE1 = 10
DECIM_STAGE2 = 10
DECIM_STAGE3 = 4
FINAL_SR = HW_SAMPLE_RATE / (DECIM_STAGE1 * DECIM_STAGE2 * DECIM_STAGE3)  # 5 kHz
FS_FFT = HW_SAMPLE_RATE / (DECIM_STAGE1 * DECIM_STAGE2)

FIR_CUTOFF_HZ = 45.0
FIR_CUTOFF_NOISE_GUARD_HZ = 250.0
FIR_TAPS = 1001

CARTELLA_OUTPUT = r"D:\Gaetano\SID\dati\Sdruno"
PREFISSI = {
    "v1": "GAESID_NSY_V1",
    "v2": "GAESID_NSY",
    "bw02": "GAESID_NSY_BW02",
    "bw01": "GAESID_NSY_BW01",
    "pesato": "GAESID_NSY_PESATO"
}

BLOCK_SIZE_MS = 10
BLOCK_SIZE_SAMPLES = int(FINAL_SR * BLOCK_SIZE_MS / 1000)
FS_LOOP = FINAL_SR / BLOCK_SIZE_SAMPLES
LOOP_BW_HZ_INIZIALE = 1.0
LOOP_BW_HZ_TRACKING = 0.05
ZETA = 0.707

SITE_NAME = "Torino"
SITE_LAT = 45.07
SITE_LON = 7.68
MONITOR_ID = "GAESID"
FREQ_PORTANTE_HZ = 45900.0
LOG_INTERVAL_S = 1

MEASUREMENT_INTERVAL_S = 1.0
MAX_CONSECUTIVE_ERRORS = 10
TIMEOUT_PER_CHUNK_S = 1.0

FFT_N = 16384
FFT_X_ZOOM_LEVELS_HZ = [50, 100, 200, 500, 1000, 2000, 5000]
FFT_X_ZOOM_INDEX = 4
FFT_XMIN_HZ = -FFT_X_ZOOM_LEVELS_HZ[FFT_X_ZOOM_INDEX]
FFT_XMAX_HZ = +FFT_X_ZOOM_LEVELS_HZ[FFT_X_ZOOM_INDEX]
FFT_SMOOTH_ALPHA = 0.15
FFT_BIN_GROUP = 8
FFT_Y_ZOOM_LEVELS = [(-80.0, 0.0), (-60.0, 0.0), (-40.0, 0.0), (-20.0, 0.0)]
FFT_Y_ZOOM_INDEX = 2
FFT_YMIN_DB, FFT_YMAX_DB = FFT_Y_ZOOM_LEVELS[FFT_Y_ZOOM_INDEX]

PHASE_SMOOTH_ALPHA = 0.05
OFFSET_PLL_DEFAULT_HZ = -4.7

# ============================================================
# CODA IQ E CALLBACK
# ============================================================
iq_queue = queue.Queue(maxsize=10000)
dropped_lock = threading.Lock()
dropped_queue_full = 0
drop_flag_callback = False

def iq_callback(xi, xq, params, num_samples, reset, ctx):
    global dropped_queue_full, drop_flag_callback
    try:
        samples = np.empty(num_samples, dtype=np.complex64)
        i_raw = np.ctypeslib.as_array(xi, shape=(num_samples,))
        q_raw = np.ctypeslib.as_array(xq, shape=(num_samples,))
        samples.real = i_raw
        samples.imag = q_raw
        try:
            iq_queue.put_nowait(samples)
        except queue.Full:
            with dropped_lock:
                dropped_queue_full += 1
                drop_flag_callback = True
    except Exception as e:
        print("[CALLBACK ERROR]", e)

overload_lock = threading.Lock()
overload_da_ackare = False
overload_visto_nel_s = 0
overload_totali = 0

def event_callback(event_id, tuner, params, ctx):
    global overload_da_ackare, overload_visto_nel_s, overload_totali
    try:
        if event_id == Event.POWER_OVERLOAD_CHANGE:
            tipo = params.contents.powerOverloadParams.powerOverloadChangeType
            with overload_lock:
                if tipo == PowerOverloadCbEventId.OVERLOAD_DETECTED:
                    overload_visto_nel_s += 1
                    overload_totali += 1
                overload_da_ackare = True
    except Exception as e:
        print(f"[EVENT] Errore callback: {e}")

# ============================================================
# CSV MANAGER
# ============================================================
def nome_file_giornaliero(data_utc, prefisso):
    data_str = data_utc.strftime("%Y-%m-%d")
    return os.path.join(CARTELLA_OUTPUT, f"{prefisso}_{data_str}.csv")

class CSVManager:
    def __init__(self, prefisso, station_id, is_v2=True):
        self.prefisso = prefisso
        self.station_id = station_id
        self.is_v2 = is_v2
        self.giorno_corrente = None
        self.file_handle = None
        self.writer = None
        self.path_corrente = None
        os.makedirs(CARTELLA_OUTPUT, exist_ok=True)

    def _apri_file(self, data_utc):
        path = nome_file_giornaliero(data_utc, self.prefisso)
        esiste = os.path.exists(path) and os.path.getsize(path) > 0
        self.file_handle = open(path, mode="a", newline="")
        self.writer = csv.writer(self.file_handle)
        self.path_corrente = path

        if not esiste:
            data_inizio = data_utc.strftime("%Y-%m-%d %H:%M:%S")
            header_lines = [
                f"# Site = {SITE_NAME}", f"# Longitude = {SITE_LON}", f"# Latitude = {SITE_LAT}", "#",
                f"# UTC_StartTime = {data_inizio}", f"# LogInterval = {LOG_INTERVAL_S}", "# LogType = raw",
                f"# MonitorID = {MONITOR_ID}", f"# StationID = {self.station_id}",
                f"# Frequency = {int(FREQ_PORTANTE_HZ)}",
            ]
            if self.is_v2:
                header_lines.append(f"# PLL_Fs_loop = {FS_LOOP}")
                header_lines.append(f"# PLL_Block_ms = {BLOCK_SIZE_MS}")
                # Identifica la banda in base al prefisso del file
                if "BW02" in self.prefisso: bw_str = "0.02 Hz"
                elif "BW01" in self.prefisso: bw_str = "0.01 Hz"
                elif "PESATO" in self.prefisso: bw_str = "0.05 Hz (Pesato)"
                else: bw_str = "0.05 Hz"
                header_lines.append(f"# PLL_Target_BW = {bw_str}")
            
            for linea in header_lines:
                self.file_handle.write(linea + "\n")

            if self.is_v2:
                self.writer.writerow(["Timestamp_UTC", "Power_dB", "Phase_Deg", "PLL_Freq_Hz", 
                                      "Reset_Flag", "Amp_100Hz_dB", "ADC_Overload", 
                                      "Phase_Resid_Deg", "Corr_Mag"])
            else:
                self.writer.writerow(["Timestamp_UTC", "Power_dB", "Phase_Deg", "PLL_Freq_Hz", 
                                      "Reset_Flag", "Amp_100Hz_dB", "ADC_Overload"])
            self.file_handle.flush()

    def _chiudi_file(self):
        if self.file_handle is not None:
            try:
                self.file_handle.flush()
                self.file_handle.close()
            except: pass
            self.file_handle = None
            self.writer = None

    def aggiorna(self, data_utc):
        giorno = data_utc.date()
        if giorno != self.giorno_corrente:
            self._chiudi_file()
            self._apri_file(data_utc)
            self.giorno_corrente = giorno

    def scrivi(self, row_data):
        if self.writer is not None:
            self.writer.writerow(row_data)

    def chiudi(self):
        self._chiudi_file()

class AsyncCSVWriter(threading.Thread):
    def __init__(self, maxsize=3600):
        super().__init__(daemon=True)
        self.managers = {
            "v1": CSVManager(PREFISSI["v1"], "NSY_V1", is_v2=False),
            "v2": CSVManager(PREFISSI["v2"], "NSY_V2", is_v2=True),
        }
        self.row_queue = queue.Queue(maxsize=maxsize)
        self.stop_event = threading.Event()
        self._righe_da_flush = 0
        self.dropped_rows = 0

    def put(self, record):
        try: 
            self.row_queue.put_nowait(record)
        except queue.Full: 
            self.dropped_rows += 1

    def run(self):
        while not self.stop_event.is_set():
            try:
                timestamp_dt, rows_dict = self.row_queue.get(timeout=1.0)
            except queue.Empty:
                self._flush()
                continue
            
            for key, manager in self.managers.items():
                try:
                    manager.aggiorna(timestamp_dt)
                    manager.scrivi(rows_dict[key])
                except Exception as e:
                    print(f"[CSV_WRITER] Errore critico sul file {key}: {e}")
                    logger.exception("[CSV_WRITER] Errore critico sul file %s", key)
                    try: manager._chiudi_file()
                    except: pass
                    manager.giorno_corrente = None # Forza la riapertura alla riga successiva
            
            self._righe_da_flush += 1
            if self._righe_da_flush >= 30:
                self._flush()
        
        while not self.row_queue.empty():
            try:
                timestamp_dt, rows_dict = self.row_queue.get_nowait()
                for key, manager in self.managers.items():
                    manager.aggiorna(timestamp_dt)
                    manager.scrivi(rows_dict[key])
            except queue.Empty:
                break
        self._flush()

    def _flush(self):
        if self._righe_da_flush > 0:
            for manager in self.managers.values():
                if manager.file_handle:
                    try: manager.file_handle.flush()
                    except: pass
            self._righe_da_flush = 0

    def stop(self):
        self.stop_event.set()

class SNRLogger:
    """Log leggero dell'SNR di banda, stato NSY e ADC: una riga al minuto, file separato."""
    def __init__(self, cartella, prefisso="snr_log"):
        self.cartella = cartella
        self.prefisso = prefisso
        self.giorno_corrente = None
        self.file_handle = None
        os.makedirs(cartella, exist_ok=True)

    def _apri(self, data_utc):
        path = os.path.join(self.cartella, f"{self.prefisso}_{data_utc.strftime('%Y-%m-%d')}.csv")
        esiste = os.path.exists(path) and os.path.getsize(path) > 0
        self.file_handle = open(path, "a", newline="")
        if not esiste:
            self.file_handle.write("# Log SNR NSY - una riga al minuto\n")
            self.file_handle.write("Timestamp_UTC,SNR_band_dB,Power_dB,Noise_dB,Stato_NSY_Num,ADC_Overload\n")
            self.file_handle.flush()
            print(f"[SNR_LOG] Nuovo file creato: {path}")

    def scrivi(self, timestamp_dt, snr_band_db, segnale_presente, pwr, noise_db, adc_overload):
        giorno = timestamp_dt.date()
        if giorno != self.giorno_corrente:
            if self.file_handle is not None:
                self.file_handle.close()
            self._apri(timestamp_dt)
            self.giorno_corrente = giorno

        # Mappatura numerica: 1 = PRESENTE, 0 = ASSENTE, 2 = ATTESA (o -1)
        if segnale_presente is True:
            stato_str = "1"
        elif segnale_presente is False:
            stato_str = "0"
        else:
            stato_str = "-1"   # oppure "2" se preferisci

        ovl_flag = 1 if adc_overload else 0
        self.file_handle.write(
            f"{timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')},"
            f"{snr_band_db:.2f},{pwr:.2f},{noise_db:.2f},{stato_str},{ovl_flag}\n"
        )
        self.file_handle.flush()

    def chiudi(self):
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None

# ============================================================
# DSP
# ============================================================
class StatefulDecimator:
    def __init__(self, factor, fs_in, cutoff_hz, numtaps=101):
        self.factor = int(factor)
        self.taps = signal.firwin(numtaps, cutoff_hz, fs=fs_in)
        self.zi = np.zeros(len(self.taps) - 1)
        self.samples_seen = 0

    def process(self, data):
        filtered, self.zi = signal.lfilter(self.taps, 1.0, data, zi=self.zi)
        indices = np.arange(len(data))
        keep = ((self.samples_seen + indices) % self.factor) == 0
        output = filtered[keep]
        self.samples_seen += len(data)
        return output

class SquaringPLL_v1:
    def __init__(self, baud_rate=200.0, sample_rate=5000.0, loop_bw_hz=1.0):
        self.fs = sample_rate
        self.f_nominal = baud_rate / 2.0
        self.omega_nominal = 2 * np.pi * self.f_nominal / self.fs
        MAX_FREQ_DEV_HZ = 25.0
        self.max_freq_error = 2 * np.pi * MAX_FREQ_DEV_HZ / self.fs
        self.set_bandwidth(loop_bw_hz)
        self.vco_phase = 0.0
        self.integrator = 0.0
        self.total_nominal_phase = 0.0
        self.freq_offset_hz = 0.0

    def set_bandwidth(self, bw_hz):
        wn = 2 * np.pi * bw_hz
        zeta = 0.707
        self.alpha = 2 * zeta * wn / self.fs
        self.beta = (wn / self.fs) ** 2

    def process(self, samples):
        squared = samples ** 2
        correlatore = 0.0 + 0.0j
        freq_acc = 0.0
        for s in squared:
            prod = s * np.exp(-1j * self.vco_phase)
            error = np.angle(prod)
            correlatore += prod
            self.integrator += self.beta * error
            freq_error = self.alpha * error + self.integrator
            if freq_error > self.max_freq_error:
                freq_error = self.max_freq_error
                self.integrator = self.max_freq_error - self.alpha * error
            elif freq_error < -self.max_freq_error:
                freq_error = -self.max_freq_error
                self.integrator = -self.max_freq_error - self.alpha * error
            self.vco_phase += self.omega_nominal + freq_error
            freq_acc += freq_error

        amp_r2 = np.abs(correlatore) / len(squared) if len(squared) > 0 else 0.0
        amp_r2_db = 20.0 * np.log10(amp_r2 + 1e-12)
        self.total_nominal_phase += self.omega_nominal * len(squared)
        excess_phase = self.vco_phase - self.total_nominal_phase
        carrier_rad = excess_phase / 2.0
        freq_media = freq_acc / len(squared) if len(squared) > 0 else 0.0
        self.freq_offset_hz = freq_media * self.fs / (2 * np.pi) / 2.0
        return np.degrees(carrier_rad), self.freq_offset_hz, amp_r2_db

    def reset(self, offset_hz=0.0):
        self.vco_phase = 0.0
        self.total_nominal_phase = 0.0
        self.freq_offset_hz = offset_hz
        self.integrator = 2 * np.pi * (2.0 * offset_hz) / self.fs

class SquaringPLL_v2:
    def __init__(self, baud_rate=200.0, sample_rate=5000.0, loop_bw_hz=1.0):
        self.fs = sample_rate
        self.f_nominal = baud_rate / 2.0
        self.omega_nominal = 2 * np.pi * self.f_nominal / self.fs
        MAX_FREQ_DEV_HZ = 25.0
        self.max_freq_error = 2 * np.pi * MAX_FREQ_DEV_HZ / FS_LOOP
        self.set_bandwidth(loop_bw_hz)
        self.last_freq_error = 0.0
        self.vco_phase = 0.0
        self.integrator = 0.0
        self.total_nominal_phase = 0.0
        self.freq_offset_hz = 0.0
        self.resto_campioni = np.array([], dtype=np.float64)

    def set_bandwidth(self, bw_hz):
        omega_n = 2 * np.pi * bw_hz
        Ts = 1.0 / FS_LOOP
        self.alpha = 2 * ZETA * omega_n * Ts
        self.beta = (omega_n * Ts) ** 2

    def process(self, samples):
        samples_full = np.concatenate((self.resto_campioni, samples))
        n_blocks = len(samples_full) // BLOCK_SIZE_SAMPLES
        
        if n_blocks == 0:
            return (np.degrees((self.vco_phase - self.total_nominal_phase) / 2.0), 
                    self.freq_offset_hz, 0.0, 0.0, 0.0)

        self.resto_campioni = samples_full[n_blocks * BLOCK_SIZE_SAMPLES:].copy()
        squared = samples_full[:n_blocks * BLOCK_SIZE_SAMPLES] ** 2
        
        freq_acc = 0.0
        mags = []
        coherences = []
        errors_deg = []

        for i in range(n_blocks):
            block = squared[i*BLOCK_SIZE_SAMPLES : (i+1)*BLOCK_SIZE_SAMPLES]
            vco_freq = self.omega_nominal + (self.last_freq_error / BLOCK_SIZE_SAMPLES)
            vco_ref = np.exp(-1j * (self.vco_phase + np.arange(len(block)) * vco_freq))
            
            z = block * vco_ref
            correlator = np.sum(z)
            error = np.angle(correlator)
            
            mags.append(np.abs(correlator))
            errors_deg.append(np.degrees(error))
            coherences.append(np.abs(correlator) / (np.sum(np.abs(z)) + 1e-12))

            proposed = self.alpha * error + self.integrator + self.beta * error
            if (proposed > self.max_freq_error and error > 0) or (proposed < -self.max_freq_error and error < 0):
                pass 
            else:
                self.integrator += self.beta * error
            
            self.last_freq_error = np.clip(self.alpha * error + self.integrator, -self.max_freq_error, self.max_freq_error)
            freq_acc += (vco_freq - self.omega_nominal) 
            self.vco_phase += vco_freq * BLOCK_SIZE_SAMPLES

        amp_r2 = np.mean(mags) / BLOCK_SIZE_SAMPLES if mags else 0.0
        amp_r2_db = 20.0 * np.log10(amp_r2 + 1e-12)
        self.total_nominal_phase += self.omega_nominal * (n_blocks * BLOCK_SIZE_SAMPLES)
        excess_phase = self.vco_phase - self.total_nominal_phase
        carrier_rad = excess_phase / 2.0
        
        freq_media = freq_acc / n_blocks if n_blocks > 0 else 0.0
        self.freq_offset_hz = (freq_media * self.fs) / (2 * np.pi) / 2.0
        
        phase_resid_deg = (errors_deg[-1] / 2.0) if errors_deg else 0.0
        corr_mag = float(np.mean(coherences)) if coherences else 0.0

        return np.degrees(carrier_rad), self.freq_offset_hz, amp_r2_db, phase_resid_deg, corr_mag

    def reset(self, offset_hz=0.0):
        self.vco_phase = 0.0
        self.total_nominal_phase = 0.0
        self.freq_offset_hz = offset_hz
        self.last_freq_error = 2 * np.pi * (2.0 * offset_hz) / FS_LOOP
        self.integrator = self.last_freq_error
        self.resto_campioni = np.array([], dtype=np.float64)

class SquaringPLL_v2_pesato(SquaringPLL_v2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_amp = None
        self.alpha_ref = 1.0 / (30 * FS_LOOP)
        self.max_err_rad = 0.5

    def process(self, samples):
        samples_full = np.concatenate((self.resto_campioni, samples))
        n_blocks = len(samples_full) // BLOCK_SIZE_SAMPLES
        
        if n_blocks == 0:
            return super().process(samples)

        self.resto_campioni = samples_full[n_blocks * BLOCK_SIZE_SAMPLES:].copy()
        squared = samples_full[:n_blocks * BLOCK_SIZE_SAMPLES] ** 2
        
        freq_acc = 0.0
        mags = []
        coherences = []
        errors_deg = []

        for i in range(n_blocks):
            block = squared[i*BLOCK_SIZE_SAMPLES : (i+1)*BLOCK_SIZE_SAMPLES]
            vco_freq = self.omega_nominal + (self.last_freq_error / BLOCK_SIZE_SAMPLES)
            vco_ref = np.exp(-1j * (self.vco_phase + np.arange(len(block)) * vco_freq))
            
            z = block * vco_ref
            correlator = np.sum(z)
            
            c_amp = np.abs(correlator)
            if self.ref_amp is None:
                self.ref_amp = c_amp
            else:
                self.ref_amp = (1 - self.alpha_ref) * self.ref_amp + self.alpha_ref * c_amp
            
            error_raw = np.imag(correlator) / (self.ref_amp + 1e-12)
            error = np.clip(error_raw, -self.max_err_rad, self.max_err_rad)
            
            mags.append(c_amp)
            errors_deg.append(np.degrees(error))
            coherences.append(c_amp / (np.sum(np.abs(z)) + 1e-12))

            proposed = self.alpha * error + self.integrator + self.beta * error
            if (proposed > self.max_freq_error and error > 0) or (proposed < -self.max_freq_error and error < 0):
                pass 
            else:
                self.integrator += self.beta * error
            
            self.last_freq_error = np.clip(self.alpha * error + self.integrator, -self.max_freq_error, self.max_freq_error)
            freq_acc += (vco_freq - self.omega_nominal) 
            self.vco_phase += vco_freq * BLOCK_SIZE_SAMPLES

        amp_r2 = np.mean(mags) / BLOCK_SIZE_SAMPLES if mags else 0.0
        amp_r2_db = 20.0 * np.log10(amp_r2 + 1e-12)
        self.total_nominal_phase += self.omega_nominal * (n_blocks * BLOCK_SIZE_SAMPLES)
        excess_phase = self.vco_phase - self.total_nominal_phase
        carrier_rad = excess_phase / 2.0
        
        freq_media = freq_acc / n_blocks if n_blocks > 0 else 0.0
        self.freq_offset_hz = (freq_media * self.fs) / (2 * np.pi) / 2.0
        
        phase_resid_deg = (errors_deg[-1] / 2.0) if errors_deg else 0.0
        corr_mag = float(np.mean(coherences)) if coherences else 0.0

        return np.degrees(carrier_rad), self.freq_offset_hz, amp_r2_db, phase_resid_deg, corr_mag

    def reset(self, offset_hz=0.0):
        super().reset(offset_hz)
        self.ref_amp = None

class SIDTrackerDuale:
    def __init__(self):
        self.sample_counter = 0
        self.pll_contatore = 0
        self.decim1 = StatefulDecimator(DECIM_STAGE1, HW_SAMPLE_RATE, 45000, 101)
        self.decim2 = StatefulDecimator(DECIM_STAGE2, HW_SAMPLE_RATE/DECIM_STAGE1, 4500, 101)
        self.decim3 = StatefulDecimator(DECIM_STAGE3, HW_SAMPLE_RATE/(DECIM_STAGE1*DECIM_STAGE2), 450, 101)
        self.fir_coeff = signal.firwin(FIR_TAPS, FIR_CUTOFF_HZ, fs=FINAL_SR, window=("kaiser", 8.0))
        self.filter_zi = signal.lfilter_zi(self.fir_coeff, 1.0) * 0.0
        FIR_CUTOFF_FASE_HZ = 150.0
        self.fir_coeff_fase = signal.firwin(FIR_TAPS, FIR_CUTOFF_FASE_HZ, fs=FINAL_SR, window=("kaiser", 8.0))
        self.filter_zi_fase = signal.lfilter_zi(self.fir_coeff_fase, 1.0) * 0.0

        # Filtro bandpass per la stima del rumore (250-2500 Hz su s2)
        self.fir_coeff_noise_bp = signal.firwin(
            FIR_TAPS, 
            [FIR_CUTOFF_NOISE_GUARD_HZ, 2500.0], 
            pass_zero=False, 
            fs=FS_FFT, 
            window=("kaiser", 8.0))
        self.filter_zi_noise_bp = signal.lfilter_zi(self.fir_coeff_noise_bp, 1.0) * 0.0

        self.warmed_up = False
        self.reset_occurred = False
        
        self.pll_v1 = SquaringPLL_v1(loop_bw_hz=LOOP_BW_HZ_INIZIALE)
        self.pll_v2 = SquaringPLL_v2(loop_bw_hz=LOOP_BW_HZ_INIZIALE)
        self.pll_bw02 = SquaringPLL_v2(loop_bw_hz=LOOP_BW_HZ_INIZIALE)
        self.pll_bw01 = SquaringPLL_v2(loop_bw_hz=LOOP_BW_HZ_INIZIALE)
        self.pll_pesato = SquaringPLL_v2_pesato(loop_bw_hz=LOOP_BW_HZ_INIZIALE)

        self.phase_smooth_v1 = 0.0
        self.phase_smooth_v2 = 0.0
        self.phase_smooth_bw02 = 0.0
        self.phase_smooth_bw01 = 0.0
        self.phase_smooth_pesato = 0.0
        self.buffer_sq = deque(maxlen=20)

    def process(self, iq_samples):
        num_samples = len(iq_samples)
        n = np.arange(num_samples, dtype=np.float64) + self.sample_counter
        t = n / HW_SAMPLE_RATE
        effective_offset = OFFSET_HZ + OFFSET_HZ_CORRECTION
        baseband = iq_samples * np.exp(MIX_SIGN * 1j * 2 * np.pi * effective_offset * t)
        self.sample_counter = (self.sample_counter + num_samples) % int(HW_SAMPLE_RATE * 3600)
        
        s1 = self.decim1.process(baseband)
        s2 = self.decim2.process(s1)
        s3 = self.decim3.process(s2)
        filtered, self.filter_zi = signal.lfilter(self.fir_coeff, 1.0, s3, zi=self.filter_zi)
        filtered_fase, self.filter_zi_fase = signal.lfilter(self.fir_coeff_fase, 1.0, s3, zi=self.filter_zi_fase)
        # Filtro per il rumore su s2 (banda 250-2500 Hz)
        filtered_noise_bp, self.filter_zi_noise_bp = signal.lfilter(
            self.fir_coeff_noise_bp, 1.0, s2, zi=self.filter_zi_noise_bp)   
        self.filtered_fase = filtered_fase
        self.filtered_fase_squared = filtered_fase ** 2

        if self.warmed_up:
            self.buffer_sq.append(self.filtered_fase_squared.copy())
        
        if not self.warmed_up:
            self.warmed_up = True
            return None, None, None, None, None, None, None, None, filtered, s2
        
        power_db = 10 * np.log10(np.mean(np.abs(filtered) ** 2) + 1e-12)

        # --- SNR di banda (segnale ±45 Hz vs rumore, banda 550 - 1500 Hz su s2) ---
        # --- STIMA SPETTRALE ROBUSTA DEL RUMORE ---
        # Usiamo una FFT su s2 per evitare che le righe a banda stretta falsino il rumore.
        n_fft = 2048
        bw_signal = 2 * FIR_CUTOFF_HZ  # 90 Hz
        
        if len(s2) >= n_fft:
            # Prendiamo gli ultimi n_fft campioni e applichiamo la finestra di Hann
            window = np.hanning(n_fft)
            s2_fft = s2[-n_fft:] * window
            spettro = np.abs(np.fft.fft(s2_fft)) ** 2
            freqs = np.fft.fftfreq(n_fft, d=1.0/FS_FFT)
            
            # BANDA RISTRETTA: 550 - 1500 Hz per evitare le righe RFI
            mask_noise = (freqs > 550.0) & (freqs < 1500.0)
            
            if np.any(mask_noise):
                # Calcolo della potenza assoluta del rumore (Parseval normalizzato)
                sum_w2 = np.sum(window ** 2)
                mediana_bin = np.median(spettro[mask_noise])
                
                # Formula normalizzata per ottenere la potenza media in banda
                p_noise_in_signal = mediana_bin * bw_signal / (FS_FFT * sum_w2)
            else:
                p_noise_in_signal = 1e-12
        else:
            p_noise_in_signal = 1e-12

        # --- SNR di banda ---
        # Potenza del segnale (da s3, filtrata a ±45 Hz)
        p_in = np.mean(np.abs(filtered) ** 2)
        
        # Pavimento di sicurezza: se il segnale è sotto il rumore, l'SNR non scende sotto -30 dB
        p_signal = max(p_in - p_noise_in_signal, p_noise_in_signal * 1e-3)
        
        self.snr_band_lin = p_signal / p_noise_in_signal
        self.snr_band_db = 10 * np.log10(self.snr_band_lin + 1e-12)
        self.noise_level_db = 10 * np.log10(p_noise_in_signal + 1e-12)

        res_v1 = self.pll_v1.process(filtered_fase)
        res_v2 = self.pll_v2.process(filtered_fase)
        res_bw02 = self.pll_bw02.process(filtered_fase)
        res_bw01 = self.pll_bw01.process(filtered_fase)
        res_pesato = self.pll_pesato.process(filtered_fase)
        
        self.phase_smooth_v1 = (PHASE_SMOOTH_ALPHA * np.radians(res_v1[0]) + (1 - PHASE_SMOOTH_ALPHA) * self.phase_smooth_v1)
        self.phase_smooth_v2 = (PHASE_SMOOTH_ALPHA * np.radians(res_v2[0]) + (1 - PHASE_SMOOTH_ALPHA) * self.phase_smooth_v2)
        self.phase_smooth_bw02 = (PHASE_SMOOTH_ALPHA * np.radians(res_bw02[0]) + (1 - PHASE_SMOOTH_ALPHA) * self.phase_smooth_bw02)
        self.phase_smooth_bw01 = (PHASE_SMOOTH_ALPHA * np.radians(res_bw01[0]) + (1 - PHASE_SMOOTH_ALPHA) * self.phase_smooth_bw01)
        self.phase_smooth_pesato = (PHASE_SMOOTH_ALPHA * np.radians(res_pesato[0]) + (1 - PHASE_SMOOTH_ALPHA) * self.phase_smooth_pesato)

        self.pll_contatore += 1
        
        # 1. V1, V2 e PESATO: transizione diretta a 0.05 Hz (come prima)
        if self.pll_contatore == 30:
            self.pll_v1.set_bandwidth(LOOP_BW_HZ_TRACKING)
            self.pll_v2.set_bandwidth(LOOP_BW_HZ_TRACKING)
            self.pll_pesato.set_bandwidth(LOOP_BW_HZ_TRACKING)
            print("[PLL] Bande ridotte: V1/V2/Pesato=0.05 Hz")

        # 2. BW02 e BW01: transizione graduale (1.0 -> 0.3 -> 0.1 -> 0.03 -> target)
        if self.pll_contatore == 60:
            self.pll_bw02.set_bandwidth(0.3)
            self.pll_bw01.set_bandwidth(0.3)
        elif self.pll_contatore == 90:
            self.pll_bw02.set_bandwidth(0.1)
            self.pll_bw01.set_bandwidth(0.1)
        elif self.pll_contatore == 120:
            self.pll_bw02.set_bandwidth(0.03)
            self.pll_bw01.set_bandwidth(0.03)
        elif self.pll_contatore == 150:
            self.pll_bw02.set_bandwidth(0.02)
            self.pll_bw01.set_bandwidth(0.01)
            print("[PLL] Bande sperimentali ridotte: BW02=0.02 Hz, BW01=0.01 Hz")

        return (power_db, 
                np.degrees(self.phase_smooth_v1), res_v1[1], res_v1[2],
                np.degrees(self.phase_smooth_v2), res_v2[1], res_v2[2], res_v2[3], res_v2[4],
                np.degrees(self.phase_smooth_bw02), res_bw02[1], res_bw02[2], res_bw02[3], res_bw02[4],
                np.degrees(self.phase_smooth_bw01), res_bw01[1], res_bw01[2], res_bw01[3], res_bw01[4],
                np.degrees(self.phase_smooth_pesato), res_pesato[1], res_pesato[2], res_pesato[3], res_pesato[4],
                filtered, s2)

    def trova_offset_msk(self):
        if not hasattr(self, 'buffer_sq') or len(self.buffer_sq) < 3: return None
        spettri = []
        freqs = None
        for sq in self.buffer_sq:
            if sq is None or len(sq) < 512: continue
            sq_centered = sq - np.mean(sq)
            n_b = len(sq_centered)
            window = np.hanning(n_b)
            spettro_b = np.abs(np.fft.fftshift(np.fft.fft(sq_centered * window)))
            if freqs is None:
                freqs = np.fft.fftshift(np.fft.fftfreq(n_b, d=1.0 / self.pll_v2.fs))
            spettri.append(spettro_b)
        if not spettri: return None
        spettro = np.mean(spettri, axis=0)
        mask = (freqs >= 80.0) & (freqs <= 120.0)
        if not np.any(mask): return None
        spettro_banda = spettro[mask]
        freqs_banda = freqs[mask]
        idx_picco = int(np.argmax(spettro_banda))
        freq_picco = float(freqs_banda[idx_picco])
        amp_picco = float(spettro_banda[idx_picco])
        if 0 < idx_picco < len(spettro_banda) - 1:
            y0 = spettro_banda[idx_picco - 1]; y1 = spettro_banda[idx_picco]; y2 = spettro_banda[idx_picco + 1]
            denom = 2 * y1 - y0 - y2
            if denom > 0:
                delta = (y2 - y0) / (2 * denom)
                bin_width = float(freqs_banda[1] - freqs_banda[0])
                freq_picco = freq_picco + delta * bin_width
        rumore_mask = ((np.abs(freqs) > 130.0) & (np.abs(freqs) < 200.0))
        rumore = float(np.median(spettro[rumore_mask])) if np.any(rumore_mask) else float(np.median(spettro))
        if rumore <= 0 or amp_picco < 5.0 * rumore: return None
        offset_hz = (freq_picco - self.pll_v2.f_nominal) / 2.0
        return (offset_hz, freq_picco, amp_picco, rumore)

    def reset(self):
        self.sample_counter = 0
        self.phase_smooth_v1 = 0.0
        self.phase_smooth_v2 = 0.0
        self.phase_smooth_bw02 = 0.0
        self.phase_smooth_bw01 = 0.0
        self.phase_smooth_pesato = 0.0
        self.pll_contatore = 0
        self.filter_zi = signal.lfilter_zi(self.fir_coeff, 1.0) * 0.0
        self.filter_zi_fase = signal.lfilter_zi(self.fir_coeff_fase, 1.0) * 0.0
        self.filter_zi_noise_bp = signal.lfilter_zi(self.fir_coeff_noise_bp, 1.0) * 0.0
        self.warmed_up = False
        self.reset_occurred = True
        self.decim1.zi = np.zeros(len(self.decim1.taps) - 1); self.decim1.samples_seen = 0
        self.decim2.zi = np.zeros(len(self.decim2.taps) - 1); self.decim2.samples_seen = 0
        self.decim3.zi = np.zeros(len(self.decim3.taps) - 1); self.decim3.samples_seen = 0
        
        self.pll_v1.reset(offset_hz=OFFSET_PLL_DEFAULT_HZ); self.pll_v1.set_bandwidth(LOOP_BW_HZ_INIZIALE)
        self.pll_v2.reset(offset_hz=OFFSET_PLL_DEFAULT_HZ); self.pll_v2.set_bandwidth(LOOP_BW_HZ_INIZIALE)
        self.pll_bw02.reset(offset_hz=OFFSET_PLL_DEFAULT_HZ); self.pll_bw02.set_bandwidth(LOOP_BW_HZ_INIZIALE)
        self.pll_bw01.reset(offset_hz=OFFSET_PLL_DEFAULT_HZ); self.pll_bw01.set_bandwidth(LOOP_BW_HZ_INIZIALE)
        self.pll_pesato.reset(offset_hz=OFFSET_PLL_DEFAULT_HZ); self.pll_pesato.set_bandwidth(LOOP_BW_HZ_INIZIALE)

# ============================================================
# ACQUISIZIONE
# ============================================================
class AcquisitoreSID(threading.Thread):
    def __init__(self, state):
        super().__init__(daemon=True)
        self.state = state
        self.stop_event = threading.Event()
        self.radio = None
        self.tracker = None
        self.writer = AsyncCSVWriter()
        self.snr_logger = SNRLogger(CARTELLA_OUTPUT)        
        self.letture_totali = 0
        self.errori_consecutivi = 0
        self.ultimo_reset = None
        self.warmup_counter = 0
        self.pll_history = deque(maxlen=300)
        self.amp100_history = deque(maxlen=30)
        self.amp100_media_storico = deque(maxlen=60)
        self.pwr_media_storico = deque(maxlen=60)
        self.POTENZA_SOGLIA_BASSA = 24.0
        self.POTENZA_SOGLIA_ALTA  = 24.0
        self.DEBOUNCE_POTENZA_S = 600
        self.segnale_presente = None
        self.potenza_sotto_soglia_da_s = 0
        self.potenza_sopra_soglia_da_s = 0
        self.auto_ricerca_iniziale_fatta = False
        self.auto_ricerca_tentativi = 0
        self.carryover = np.array([], dtype=np.complex64)
        self.buffer_iq = deque(maxlen=int(FINAL_SR * 900))  # 15 minuti a 5 kHz
        self.ultimo_salvataggio_auto = 0
        # Soglie per la presenza del segnale basate su SNR di banda (S/N puro)
        # Nota: 4.8 dB di S/N corrispondono a ~6.0 dB di (S+N)/N (vecchia metrica)
        self.SNR_BANDA_SOGLIA_ALTA = 4.8
        self.SNR_BANDA_SOGLIA_BASSA = 4.8
        self.snr_band_storico = deque(maxlen=60) # media su 60 secondi per stabilizzare

    def stop(self):
        self.stop_event.set()

    def run(self):
        global overload_da_ackare, overload_visto_nel_s, overload_totali
        print("[ACQ] Thread acquisizione avviato (5 PLL paralleli).")
        self.tracker = SIDTrackerDuale()
        self.writer.start()

        try:
            self.radio = SDRPlayRadio()
            self.radio.select_device(self.radio.devices[0])
            self.radio.set_frequency(TUNE_FREQ)
            self.radio.set_sample_rate(HW_SAMPLE_RATE)
            self.radio.set_bandwidth(IF_BANDWIDTH)
            self.radio.set_agc(False)   
            GAIN_LNA = 0
            GAIN_IF_DB = 35
            self.radio.set_gain(lna_state=GAIN_LNA, gr_db=GAIN_IF_DB)
            self.state["gain_lna"] = GAIN_LNA
            self.state["gain_if"] = GAIN_IF_DB
            self.radio.set_dc_offset(dc_enable=True, iq_enable=True)
            self.radio.start(iq_callback, on_event=event_callback)
            self.state["radio_pronta"] = True
            print("[ACQ] Radio pronta.")

            while not self.stop_event.is_set():
                timestamp_dt = datetime.datetime.now(datetime.timezone.utc)
                timestamp_str = timestamp_dt.strftime("%Y-%m-%d %H:%M:%S")

                if self.state.get("gain_change_pending", False):
                    try:
                        nuovo_lna = self.state.get("gain_richiesto_lna")
                        nuovo_if = self.state.get("gain_richiesto_if")
                        if nuovo_lna is not None and nuovo_if is not None:
                            self.radio.set_gain(lna_state=int(nuovo_lna), gr_db=int(nuovo_if))
                            self.state["gain_lna"] = int(nuovo_lna)
                            self.state["gain_if"] = int(nuovo_if)
                            print(f"[ACQ] Gain aggiornato: LNA={nuovo_lna}, IF_GR={nuovo_if} dB")
                    except Exception as e: print(f"[ACQ] Errore cambio gain: {e}")
                    finally: self.state["gain_change_pending"] = False

                target_samples = HW_SAMPLE_RATE
                accumulated = []
                total = 0
                got_data = False

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
                            try: self.radio.stop()
                            except: pass
                            time.sleep(1.0)
                            try: self.radio.start(iq_callback, on_event=event_callback)
                            except: pass
                            self.tracker.reset()
                            self.carryover = np.array([], dtype=np.complex64)
                            self.pll_history.clear()
                            self.amp100_history.clear()
                            self.state["amp100_stato"] = "ACQUISIZIONE"
                            self.state["pll_stato"] = "ACQUISIZIONE"
                            self.errori_consecutivi = 0
                            self.ultimo_reset = timestamp_str
                            self.auto_ricerca_iniziale_fatta = False
                            accumulated = []
                            total = 0
                            break
                        time.sleep(0.05)

                if not got_data or not accumulated: continue

                concatenated = np.concatenate(accumulated)
                if len(concatenated) > target_samples:
                    self.carryover = concatenated[target_samples:].copy()
                else:
                    self.carryover = np.array([], dtype=np.complex64)

                iq_data = concatenated[:target_samples]

                global drop_flag_callback
                with dropped_lock:
                    drop_rilevato = drop_flag_callback
                    drop_flag_callback = False

                if drop_rilevato:
                    SOGLIA_RESET_DROP_DB = 45.0
                    amp_now = self.state.get("amp_100hz_corrente") or 0.0
                    if amp_now < SOGLIA_RESET_DROP_DB:
                        print(f"[ACQ] Drop IQ (Amp={amp_now:.1f} dB) → reset")
                        self.tracker.reset()
                        self.tracker.reset_occurred = True

                with overload_lock:
                    da_ackare = overload_da_ackare
                    overload_da_ackare = False
                    n_det = overload_visto_nel_s
                    ovl_flag = 1 if n_det > 0 else 0
                    overload_visto_nel_s = 0

                if da_ackare:
                    try: self.radio.acknowledge_power_overload()
                    except: pass

                self.state["adc_overload"] = bool(ovl_flag)
                self.state["adc_overload_tot"] = overload_totali

                if self.segnale_presente is False:
                    self.tracker.pll_contatore = 0

                try:
                    res = self.tracker.process(iq_data)
                except Exception as e:
                    print(f"[ACQ] DSP error: {e}")
                    logger.exception("[ACQ] DSP error in tracker.process()")
                    continue

                if res is None or res[0] is None:
                    self.warmup_counter += 1
                    print(f"[ACQ] Warm-up... ({self.warmup_counter})")
                    continue

                try:
                    (pwr, 
                     phase_v1, freq_v1, amp100_v1,
                     phase_v2, freq_v2, amp100_v2, resid_v2, mag_v2,
                     phase_bw02, freq_bw02, amp100_bw02, resid_bw02, mag_bw02,
                     phase_bw01, freq_bw01, amp100_bw01, resid_bw01, mag_bw01,
                     phase_pesato, freq_pesato, amp100_pesato, resid_pesato, mag_pesato,
                     filtered, s2) = res
                except Exception as e:
                    logger.exception(
                        "[ACQ] Unpacking di res fallito (len=%s, tipo=%s)",
                        len(res) if res is not None else "None",
                        type(res).__name__,
                    )
                    continue

                if (not self.auto_ricerca_iniziale_fatta and self.segnale_presente is not False):
                    if len(self.tracker.buffer_sq) >= 12:
                        self.auto_ricerca_tentativi += 1
                        try:
                            offset_auto = self.tracker.trova_offset_msk()
                        except Exception:
                            logger.exception("[ACQ] trova_offset_msk() fallito in init")
                            offset_auto = None
                        if offset_auto is not None:
                            offset_hz, freq_picco, amp, rumore = offset_auto
                            self.tracker.pll_v1.reset(offset_hz=offset_hz)
                            self.tracker.pll_v2.reset(offset_hz=offset_hz)
                            self.tracker.pll_bw02.reset(offset_hz=offset_hz)
                            self.tracker.pll_bw01.reset(offset_hz=offset_hz)
                            self.tracker.pll_pesato.reset(offset_hz=offset_hz)
                            self.tracker.reset_occurred = True
                            self.tracker.pll_contatore = 0
                            self.auto_ricerca_iniziale_fatta = True
                        elif self.auto_ricerca_tentativi > 5:
                            self.tracker.pll_v1.reset(offset_hz=OFFSET_PLL_DEFAULT_HZ)
                            self.tracker.pll_v2.reset(offset_hz=OFFSET_PLL_DEFAULT_HZ)
                            self.tracker.pll_bw02.reset(offset_hz=OFFSET_PLL_DEFAULT_HZ)
                            self.tracker.pll_bw01.reset(offset_hz=OFFSET_PLL_DEFAULT_HZ)
                            self.tracker.pll_pesato.reset(offset_hz=OFFSET_PLL_DEFAULT_HZ)
                            self.tracker.reset_occurred = True
                            self.auto_ricerca_iniziale_fatta = True

                reset_flag = 1 if self.tracker.reset_occurred else 0
                
                rows = {
                    "v1": [timestamp_str, f"{pwr:.3f}", f"{phase_v1:.3f}", f"{freq_v1:.6f}", reset_flag, f"{amp100_v1:.3f}", ovl_flag],
                    "v2": [timestamp_str, f"{pwr:.3f}", f"{phase_v2:.3f}", f"{freq_v2:.6f}", reset_flag, f"{amp100_v2:.3f}", ovl_flag, f"{resid_v2:.3f}", f"{mag_v2:.6f}"]
                }
                
                self.writer.put((timestamp_dt, rows))
                self.tracker.reset_occurred = False

                self.state["pwr_corrente"] = pwr
                self.state["snr_band_db"] = self.tracker.snr_band_db
                self.state["noise_level_db"] = self.tracker.noise_level_db
                self.state["phase_corrente"] = phase_v2
                self.state["phase_v1_corrente"] = phase_v1
                self.state["pll_freq_corrente"] = freq_v2
                self.state["amp_100hz_corrente"] = amp100_v2
                self.state["phase_resid_corrente"] = resid_v2
                self.state["corr_mag_corrente"] = mag_v2
                self.state["timestamp_corrente"] = timestamp_str

                # --- AGGIORNAMENTO STATO PLL E LOCK ---
                if self.tracker.pll_contatore > 30:
                    # 1. Aggiorna pll_modo e pll_stato
                    self.state["pll_modo"] = "TRACKING"
                    self.pll_history.append(freq_v2)
                    if len(self.pll_history) >= 30:
                        pll_recent = list(self.pll_history)[-30:]
                        pll_recent_smoothed = np.convolve(pll_recent, np.ones(5)/5, mode='valid')
                        pll_std_30 = float(np.std(pll_recent_smoothed))
                        pll_mean_30 = float(np.mean(pll_recent_smoothed))
                        self.state["pll_std_30"] = pll_std_30
                        self.state["pll_mean_30"] = pll_mean_30
                        if self.segnale_presente is False: self.state["pll_stato"] = "SEGNALE_ASSENTE"
                        elif pll_std_30 < 0.001: self.state["pll_stato"] = "STABILE"      # < 1 mHz
                        elif pll_std_30 < 0.005: self.state["pll_stato"] = "ACCETTABILE"  # < 5 mHz
                        else: self.state["pll_stato"] = "INSTABILE"
                    
                    # 2. Aggiorna amp100_stato (Lock PLL)
                    self.amp100_history.append(amp100_v2)
                    if len(self.amp100_history) >= 30:
                        amp100_std = float(np.std(self.amp100_history))
                        amp100_mean = float(np.mean(self.amp100_history))
                        self.state["amp100_std"] = amp100_std
                        self.state["amp100_mean"] = amp100_mean
                        
                        if self.segnale_presente is False:
                            self.state["amp100_stato"] = "SEGNALE_ASSENTE"
                        elif amp100_v2 < amp100_mean - 12.0:
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
                    self.state["pll_modo"] = "ACQUISIZIONE"
                    self.state["pll_stato"] = "ACQUISIZIONE"
                    self.state["amp100_stato"] = "ACQUISIZIONE"
                # --------------------------------------

                self.amp100_media_storico.append(amp100_v2)

                if pwr is not None:
                    # 1. Aggiorna il buffer dello SNR e calcola la media mobile a 60s
                    self.snr_band_storico.append(self.tracker.snr_band_db)
                    if len(self.snr_band_storico) >= 30:
                        snr_smooth = float(np.mean(self.snr_band_storico))
                    else:
                        snr_smooth = self.tracker.snr_band_db

                    # 2. INIZIALIZZAZIONE IMMEDIATA
                    if self.segnale_presente is None:
                        if snr_smooth > self.SNR_BANDA_SOGLIA_ALTA:
                            self.segnale_presente = True
                            print(f"[{timestamp_str}] [STATO] Segnale PRESENTE (SNR={snr_smooth:.1f} dB).")
                        elif snr_smooth < self.SNR_BANDA_SOGLIA_BASSA:
                            self.segnale_presente = False
                            print(f"[{timestamp_str}] [STATO] Segnale ASSENTE (SNR={snr_smooth:.1f} dB). PLL in attesa.")

                    # 3. MACCHINA A STATI NORMALE (basata solo su SNR)
                    else:
                        if snr_smooth < self.SNR_BANDA_SOGLIA_BASSA:
                            self.potenza_sotto_soglia_da_s += 1
                            self.potenza_sopra_soglia_da_s = 0
                        elif snr_smooth > self.SNR_BANDA_SOGLIA_ALTA:
                            self.potenza_sopra_soglia_da_s += 1
                            self.potenza_sotto_soglia_da_s = 0
                        else:
                            # Zona grigia: resetta i contatori per evitare falsi cambi
                            self.potenza_sotto_soglia_da_s = 0
                            self.potenza_sopra_soglia_da_s = 0

                        # Transizione PRESENTE → ASSENTE
                        if (self.segnale_presente is True and
                                self.potenza_sotto_soglia_da_s >= self.DEBOUNCE_POTENZA_S):
                            print(f"[{timestamp_str}] [STATO] Segnale ASSENTE (SNR={snr_smooth:.1f} dB per {self.potenza_sotto_soglia_da_s}s). PLL in attesa.")
                            self.segnale_presente = False

                        # Transizione ASSENTE → PRESENTE
                        elif (self.segnale_presente is False and
                                self.potenza_sopra_soglia_da_s >= self.DEBOUNCE_POTENZA_S):
                            print(f"[{timestamp_str}] [STATO] Segnale PRESENTE (SNR={snr_smooth:.1f} dB per {self.potenza_sopra_soglia_da_s}s). Reset PLL.")
                            self.segnale_presente = True

                            # Reset PLL + auto-ricerca (codice esistente)
                            try:
                                offset_auto = self.tracker.trova_offset_msk()
                            except Exception:
                                logger.exception("[ACQ] trova_offset_msk() fallito in ASSENTE->PRESENTE")
                                offset_auto = None
                            if offset_auto is not None:
                                offset_hz, freq_picco, amp_picco, rumore = offset_auto
                                self.tracker.pll_v1.reset(offset_hz=offset_hz)
                                self.tracker.pll_v2.reset(offset_hz=offset_hz)
                                self.tracker.pll_bw02.reset(offset_hz=offset_hz)
                                self.tracker.pll_bw01.reset(offset_hz=offset_hz)
                                self.tracker.pll_pesato.reset(offset_hz=offset_hz)
                            else:
                                self.tracker.pll_v1.reset(offset_hz=OFFSET_PLL_DEFAULT_HZ)
                                self.tracker.pll_v2.reset(offset_hz=OFFSET_PLL_DEFAULT_HZ)
                                self.tracker.pll_bw02.reset(offset_hz=OFFSET_PLL_DEFAULT_HZ)
                                self.tracker.pll_bw01.reset(offset_hz=OFFSET_PLL_DEFAULT_HZ)
                                self.tracker.pll_pesato.reset(offset_hz=OFFSET_PLL_DEFAULT_HZ)

                            self.tracker.reset_occurred = True
                            self.tracker.pll_contatore = 0
                            self.pll_history.clear()
                            self.amp100_history.clear()
                            self.amp100_media_storico.clear()
                            self.auto_ricerca_iniziale_fatta = True

                    # 4. Salva lo stato per la GUI
                    self.state["segnale_nsy_presente"] = self.segnale_presente

                    # 5. Log SNR + stato + ADC ogni 1 secondo (1 Hz)
                    if not hasattr(self, '_ultimo_log_snr_s'):
                        self._ultimo_log_snr_s = 0.0
                    ora_corrente_log = time.time()
                    if ora_corrente_log - self._ultimo_log_snr_s >= 1.0:
                        try:
                            self.snr_logger.scrivi(
                                timestamp_dt,
                                self.tracker.snr_band_db,
                                self.segnale_presente,
                                pwr,
                                self.tracker.noise_level_db,
                                ovl_flag
                            )
                            self._ultimo_log_snr_s = ora_corrente_log
                        except Exception as e:
                            print(f"[SNR_LOG] Errore scrittura: {e}")

                self.state["letture_totali"] = self.letture_totali + 1
                self.state["ultimo_reset"] = self.ultimo_reset
                self.letture_totali += 1
                with dropped_lock:
                    self.state["dropped_queue_full"] = dropped_queue_full
                    self.state["dropped_rows_v1"] = self.writer.dropped_rows  

                # --- HEARTBEAT (una riga al minuto) ---
                if not hasattr(self, '_ultimo_heartbeat_s'):
                    self._ultimo_heartbeat_s = 0.0
                _now_hb = time.time()
                if _now_hb - self._ultimo_heartbeat_s >= 60.0:
                    try:
                        logger.info(
                            "[HEARTBEAT] letture=%d snr=%.1fdB segnale=%s adc_ovl=%s "
                            "dropped_q=%d dropped_csv=%d reset_flag=%s pll_modo=%s",
                            self.letture_totali,
                            getattr(self.tracker, 'snr_band_db', float('nan')),
                            self.segnale_presente,
                            self.state.get("adc_overload", False),
                            self.state.get("dropped_queue_full", 0),
                            self.state.get("dropped_rows_v1", 0),
                            reset_flag,
                            self.state.get("pll_modo", "?"),
                        )
                    except Exception:
                        pass
                    self._ultimo_heartbeat_s = _now_hb

                # --- GESTIONE SALVATAGGIO IQ ---

                # Aggiungi il segnale filtrato al buffer circolare
                if hasattr(self.tracker, 'filtered_fase'):
                    self.buffer_iq.extend(self.tracker.filtered_fase)

                    # DEBUG: stampa solo se il buffer non è ancora pieno
                    max_len = int(FINAL_SR * 900)
                    if len(self.buffer_iq) < max_len and len(self.buffer_iq) % (FINAL_SR * 60) < FINAL_SR:
                        print(f"[DEBUG] Buffer IQ: {len(self.buffer_iq)/FINAL_SR:.0f}s")

                # Salvataggio manuale (tasto S)
                if self.state.get("salva_iq_richiesto", False):
                    if len(self.buffer_iq) > 0:
                        # Forza il salvataggio nella cartella corretta
                        filename = os.path.join(PYSDRPLAY_DIR, f"iq_manuale_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npy")
                        try:
                            np.save(filename, np.array(self.buffer_iq))
                            print(f"[ACQ] IQ salvato manualmente: {filename} ({len(self.buffer_iq)/FINAL_SR:.1f}s)")
                            logger.info("[ACQ] IQ manuale salvato: %s (%.1fs)", filename, len(self.buffer_iq)/FINAL_SR)
                        except Exception:
                            logger.exception("[ACQ] Salvataggio IQ manuale fallito: %s", filename)
                            print(f"[ACQ] ERRORE salvataggio IQ manuale: {filename}")
                    self.state["salva_iq_richiesto"] = False
                
                # Trigger automatico: V2 > 20 mHz mentre V1 < 5 mHz
                if self.state.get("trigger_auto_iq", False) and self.tracker.pll_contatore > 60:
                    # Calcola l'ampiezza dei salti nell'ultimo secondo
                    if len(self.pll_history) >= 2:
                        v2_salto = abs(freq_v2 - self.pll_history[-2]) * 1000
                        v1_salto = abs(freq_v1 - self.state.get("pll_freq_v1_prev", freq_v1)) * 1000
                        
                        # Condizione di trigger
                        if v2_salto > 20.0 and v1_salto < 5.0:
                            # Evita salvataggi multipli troppo ravvicinati
                            ora_corrente = time.time()
                            if ora_corrente - self.ultimo_salvataggio_auto > 300:  # 5 minuti di pausa
                                # Forza il salvataggio nella cartella corretta
                                filename = os.path.join(PYSDRPLAY_DIR, f"iq_trigger_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npy")
                                try:
                                    np.save(filename, np.array(self.buffer_iq))
                                    print(f"[ACQ] IQ salvato da trigger (V2={v2_salto:.1f} mHz, V1={v1_salto:.1f} mHz): {filename} ({len(self.buffer_iq)/FINAL_SR:.1f}s)")
                                    logger.info("[ACQ] IQ trigger salvato: %s (%.1fs, V2=%.1fmHz, V1=%.1fmHz)",
                                                filename, len(self.buffer_iq)/FINAL_SR, v2_salto, v1_salto)
                                except Exception:
                                    logger.exception("[ACQ] Salvataggio IQ trigger fallito: %s", filename)
                                self.ultimo_salvataggio_auto = ora_corrente
                    
                    # Aggiorna il valore precedente di V1 per il prossimo ciclo
                    self.state["pll_freq_v1_prev"] = freq_v1
                # ---------------------------------

                if self.state.get("fft_aperta", False):
                    with self.state["fft_lock"]:
                        n_fft_max = 32768
                        if len(s2) > n_fft_max: self.state["fft_segnale"] = s2[-n_fft_max:].copy()
                        else: self.state["fft_segnale"] = s2.copy()

                        sq = getattr(self.tracker, 'filtered_fase_squared', None)
                        if sq is not None:
                            if len(sq) > n_fft_max: self.state["fft_segnale_sq"] = sq[-n_fft_max:].copy()
                            else: self.state["fft_segnale_sq"] = sq.copy()
                    self.state["fft_nuovo_dato"] = True

                # --- AUTO-BASELINE (salvataggio automatico periodico) ---
                if self.state.get("auto_baseline_attivo", False):
                    # Inizializza contatori se non esistono
                    if not hasattr(self, '_contatore_auto_baseline'):
                        self._contatore_auto_baseline = 0
                    if not hasattr(self, '_cicli_auto_baseline'):
                        self._cicli_auto_baseline = 0
                    
                    self._cicli_auto_baseline += 1

                    if self._cicli_auto_baseline % 60 == 0:
                        print(f"[DEBUG_AUTO] Cicli: {self._cicli_auto_baseline}, buffer: {len(self.buffer_iq)/FINAL_SR:.0f}s")

                    # Salva ogni 900 cicli (~15 minuti, dato che ogni ciclo ~1s)
                    if self._cicli_auto_baseline >= 900:
                        min_campioni = int(FINAL_SR * 300)  # almeno 5 minuti
                        if len(self.buffer_iq) >= min_campioni:
                            # Forza il salvataggio nella cartella corretta
                            filename = os.path.join(PYSDRPLAY_DIR, f"iq_baseline_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npy")
                            try:
                                np.save(filename, np.array(self.buffer_iq))
                                print(f"[ACQ] Auto-Baseline salvata: {filename} ({len(self.buffer_iq)/FINAL_SR:.1f}s)")
                                logger.info("[ACQ] Auto-Baseline salvata: %s (%.1fs)", filename, len(self.buffer_iq)/FINAL_SR)
                                self._cicli_auto_baseline = 0
                                self._contatore_auto_baseline += 1
                            except Exception:
                                logger.exception("[ACQ] Salvataggio Auto-Baseline fallito: %s", filename)
                                # Non azzerare il contatore: si ritenta al prossimo ciclo
                            
                            # Controlla se ha raggiunto il numero massimo di file
                            max_file = int(self.state.get("auto_baseline_ore", 4.0) * 4)  # 4 file/ora
                            if self._contatore_auto_baseline >= max_file:
                                self.state["auto_baseline_attivo"] = False
                                print(f"[ACQ] Auto-Baseline terminato (raggiunti {self._contatore_auto_baseline} file).")
                        else:
                            if self._cicli_auto_baseline % 60 == 0:
                                print(f"[ACQ] Auto-Baseline: buffer troppo piccolo ({len(self.buffer_iq)/FINAL_SR:.0f}s), attendo...")
                            # Non azzerare il contatore: riprova al prossimo ciclo
                # ------------------------------------------------

        except Exception as e:
            print(f"[ACQ] Errore critico: {e}")
            logger.exception("[ACQ] Errore critico nel thread di acquisizione")
            import traceback
            traceback.print_exc()
        finally:
            self.writer.stop()
            self.writer.join(timeout=5.0)
            for m in self.writer.managers.values():
                m.chiudi()
            try:
                self.snr_logger.chiudi()
            except Exception:
                pass
            if self.radio:
                try: self.radio.stop()
                except: pass
            self.state["radio_pronta"] = False

# ============================================================
# GUI
# ============================================================
class SIDGui:
    def __init__(self, state, acquisitore):
        self.state = state
        self.acquisitore = acquisitore
        self.root = tk.Tk()
        self.root.title("SID Tracker MULTI-PLL (V1 + V2 + Varianti) - NSY")
        self.root.geometry("950x700")
        self.root.configure(bg='#0d0d1a')

        font_titolo = ('Consolas', 11, 'bold')
        font_label = ('Consolas', 9)
        font_valore = ('Consolas', 11, 'bold')
        font_hint = ('Consolas', 8)

        frame_top = tk.Frame(self.root, bg='#0d0d1a')
        frame_top.pack(fill='x', padx=15, pady=(15, 5))
        tk.Label(frame_top, text="SID Tracker MULTI-PLL - NSY 45.905 kHz", bg='#0d0d1a', fg='#00FF99', font=font_titolo).pack(anchor='w')
        self.label_stato = tk.Label(frame_top, text="Inizializzazione...", bg='#0d0d1a', fg='#FFD700', font=font_label)
        self.label_stato.pack(anchor='w', pady=(5, 0))
        self.label_segnale_nsy = tk.Label(frame_top, text="Segnale NSY: ---", bg='#0d0d1a', fg='#888888', font=font_label)
        self.label_segnale_nsy.pack(anchor='w', pady=(2, 0))
        self.label_snr_band = tk.Label(frame_top, text="SNR banda ±45 Hz: --- dB",
                                       bg='#0d0d1a', fg='#888888', font=font_label)
        self.label_snr_band.pack(anchor='w', pady=(2, 0))
        self.label_noise_level = tk.Label(frame_top, text="Rumore (banda SID): --- dB",
                                         bg='#0d0d1a', fg='#888888', font=font_label)
        self.label_noise_level.pack(anchor='w', pady=(2, 0))
        self.label_file_v2 = tk.Label(frame_top, text="File V2: ---", bg='#0d0d1a', fg='#888888', font=font_hint)
        self.label_file_v2.pack(anchor='w')
        self.label_file_v1 = tk.Label(frame_top, text="File V1: ---", bg='#0d0d1a', fg='#888888', font=font_hint)
        self.label_file_v1.pack(anchor='w')

        frame_center = tk.Frame(self.root, bg='#0d0d1a')
        frame_center.pack(fill='both', expand=True, padx=15, pady=10)

        frame_pwr = tk.LabelFrame(frame_center, text=" Potenza NSY ", bg='#0d0d1a', fg='#00FF99', font=font_label, labelanchor='nw')
        frame_pwr.pack(side='left', fill='both', expand=True, padx=(0, 8))
        self.label_pwr = tk.Label(frame_pwr, text="--- dB", bg='#0d0d1a', fg='#00FF99', font=font_valore)
        self.label_pwr.pack(expand=True, pady=8)

        frame_phase = tk.LabelFrame(frame_center, text=" Fase (V2) ", bg='#0d0d1a', fg='#FF6B9D', font=font_label, labelanchor='nw')
        frame_phase.pack(side='left', fill='both', expand=True, padx=(8, 0))
        self.label_phase_v2 = tk.Label(frame_phase, text="--- °", bg='#0d0d1a', fg='#FF6B9D', font=font_valore)
        self.label_phase_v2.pack(expand=True, pady=(8, 2))
        self.label_phase_v1 = tk.Label(frame_phase, text="V1: --- °", bg='#0d0d1a', fg='#888888', font=font_hint)
        self.label_phase_v1.pack(expand=True, pady=(2, 8))

        frame_bot = tk.Frame(self.root, bg='#0d0d1a')
        frame_bot.pack(fill='x', padx=15, pady=(0, 15))

        self.label_stats = tk.Label(frame_bot, text="Letture: 0", bg='#0d0d1a', fg='#888888', font=font_hint)
        self.label_stats.pack(anchor='w')
        self.label_gain = tk.Label(frame_bot, text="Gain RF: ---", bg='#0d0d1a', fg='#FFA500', font=font_hint)
        self.label_gain.pack(anchor='w', pady=(2, 0))
        self.label_drops = tk.Label(frame_bot, text="Campioni persi: coda=0", bg='#0d0d1a', fg='#FF6347', font=font_hint)
        self.label_drops.pack(anchor='w', pady=(2, 0))
        self.label_amp100 = tk.Label(frame_bot, text="Amp riga 100 Hz: --- dB", bg='#0d0d1a', fg='#00BFFF', font=font_hint)
        self.label_amp100.pack(anchor='w', pady=(2, 0))
        self.label_amp100_stato = tk.Label(frame_bot, text="Lock PLL: ---", bg='#0d0d1a', fg='#888888', font=('Consolas', 10, 'bold'))
        self.label_amp100_stato.pack(anchor='w', pady=(2, 0))
        
        self.label_resid = tk.Label(frame_bot, text="Lock Quality: Resid=---° | Mag=---", bg='#0d0d1a', fg='#FFA500', font=font_hint)
        self.label_resid.pack(anchor='w', pady=(2, 0))

        self.label_pll = tk.Label(frame_bot, text="PLL: acquisizione...", bg='#0d0d1a', fg='#888888', font=font_hint)
        self.label_pll.pack(anchor='w', pady=(5, 0))
        self.label_pll_modo = tk.Label(frame_bot, text="PLL modo: acquisizione", bg='#0d0d1a', fg='#FFD700', font=('Consolas', 10, 'bold'))
        self.label_pll_modo.pack(side='right', anchor='e', padx=(0, 5))
        self.label_freq_offset = tk.Label(frame_bot, text="freq_offset: --- Hz", bg='#0d0d1a', fg='#00BFFF', font=('Consolas', 11, 'bold'))
        self.label_freq_offset.pack(side='right', anchor='e', padx=(0, 5))

        self.label_hint = tk.Label(frame_bot, text="[G] FFT  |  [S] Salva IQ  |  [T] Trigger  |  [L] Auto-Baseline  |  [Q] Esci", bg='#0d0d1a', fg='#666666', font=font_hint)
        self.label_hint.pack(anchor='w', pady=(5, 0))
        self.label_hint_gain = tk.Label(frame_bot, text="Gain:  0 (LNA=0 - IF=30 dB) |  6 (LNA=6 - IF=59 dB)", bg='#0d0d1a', fg='#FFA500', font=font_hint)
        self.label_hint_gain.pack(anchor='w', pady=(2, 0))

        # Etichetta per lo stato del trigger IQ
        self.label_iq_trigger = tk.Label(frame_bot, text="IQ Trigger: ATTIVO (salvataggio auto)", bg='#0d0d1a', fg='#00FF99', font=font_hint)
        self.label_iq_trigger.pack(anchor='w', pady=(2, 0))

        # Etichetta per lo stato dell'Auto-Baseline
        self.label_auto_baseline = tk.Label(frame_bot, text="Auto-Baseline: DISATTIVATO", bg='#0d0d1a', fg='#FF6347', font=font_hint)
        self.label_auto_baseline.pack(anchor='w', pady=(2, 0))
        
        self.fft_window = None
        self.fft_canvas = None
        self.fft_ax = None
        self.fft_line = None
        self.fft_testo_picco = None
        self.fft_testo_snr = None
        self.fft_spettro_smoothed = None
        self.fft_freqs_smoothed = None

        self._bind_all_keys()
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.root.focus_set()
        self.aggiorna_etichette()

    def _bind_all_keys(self):
        self.root.bind_all('<KeyPress-g>', self.toggle_fft)
        self.root.bind_all('<KeyPress-G>', self.toggle_fft)
        self.root.bind_all('<KeyPress-q>', self.quit_app)
        self.root.bind_all('<KeyPress-Q>', self.quit_app)
        self.root.bind_all('<KeyPress-plus>', self.aumenta_smoothing)
        self.root.bind_all('<KeyPress-equal>', self.aumenta_smoothing)
        self.root.bind_all('<KeyPress-KP_Add>', self.aumenta_smoothing)
        self.root.bind_all('<KeyPress-minus>', self.diminuisci_smoothing)
        self.root.bind_all('<KeyPress-underscore>', self.diminuisci_smoothing)
        self.root.bind_all('<KeyPress-KP_Subtract>', self.diminuisci_smoothing)
        self.root.bind_all('<KeyPress-a>', self.fft_zoom_in)
        self.root.bind_all('<KeyPress-A>', self.fft_zoom_in)
        self.root.bind_all('<KeyPress-d>', self.fft_zoom_out)
        self.root.bind_all('<KeyPress-D>', self.fft_zoom_out)
        self.root.bind_all('<KeyPress-b>', self.diminuisci_smoothing)
        self.root.bind_all('<KeyPress-B>', self.diminuisci_smoothing)
        self.root.bind_all('<KeyPress-z>', self.cicla_zoom_y)
        self.root.bind_all('<KeyPress-Z>', self.cicla_zoom_y)
        self.root.bind_all('<KeyPress-s>', self.salva_iq)
        self.root.bind_all('<KeyPress-S>', self.salva_iq)
        self.root.bind_all('<KeyPress-t>', self.toggle_trigger_iq)
        self.root.bind_all('<KeyPress-T>', self.toggle_trigger_iq)
        self.root.bind_all('<KeyPress-l>', self.toggle_auto_baseline)
        self.root.bind_all('<KeyPress-L>', self.toggle_auto_baseline)
        self.root.bind_all('<KeyPress-m>', self.toggle_fft_mode)
        self.root.bind_all('<KeyPress-M>', self.toggle_fft_mode)

    def salva_iq(self, event=None):
        self.state["salva_iq_richiesto"] = True
        print("[GUI] Richiesto salvataggio IQ. Attendi la fine del secondo...")

    def toggle_trigger_iq(self, event=None):
        stato_attuale = self.state.get("trigger_auto_iq", True)
        self.state["trigger_auto_iq"] = not stato_attuale
        nuovo_stato = "ATTIVATO" if not stato_attuale else "DISATTIVATO"
        print(f"[GUI] IQ Trigger {nuovo_stato}")

    def toggle_auto_baseline(self, event=None):
        stato = self.state.get("auto_baseline_attivo", False)
        if not stato:
            # Attiva
            self.state["auto_baseline_attivo"] = True
            self.state["auto_baseline_inizio"] = time.time()
            self.state["auto_baseline_ultimo_save"] = time.time()
            print(f"[GUI] Auto-Baseline ATTIVATO (durata: {self.state['auto_baseline_ore']} ore)")
        else:
            # Disattiva
            self.state["auto_baseline_attivo"] = False
            print("[GUI] Auto-Baseline DISATTIVATO")
    def aggiorna_etichette(self):
        # --- 1. Stato Radio (Pronta / Overload / Errore) ---
        if not self.acquisitore.is_alive() and not self.state.get("quit", False):
            self.label_stato.config(text="⚠ ACQUISIZIONE FERMA! Controlla il log.", fg='#FF0000')
        elif self.state.get("radio_pronta"):
            if self.state.get("adc_overload"):
                self.label_stato.config(text=f"⚠ ADC OVERLOAD  |  Ultimo: {self.state.get('timestamp_corrente', '')}", fg='#FF3030')
            else:
                self.label_stato.config(text=f"Radio attiva  |  Ultimo: {self.state.get('timestamp_corrente', '')}", fg='#00FF99')
        else:
            self.label_stato.config(text="Inizializzazione radio...", fg='#FFD700')

        # --- 2. Stato Segnale NSY (basato su SNR) ---
        if hasattr(self, 'label_segnale_nsy'):
            segnale = self.state.get("segnale_nsy_presente", None)
            if segnale is True:
                self.label_segnale_nsy.config(text="Segnale NSY: PRESENTE", fg='#00FF99')
            elif segnale is False:
                self.label_segnale_nsy.config(text="Segnale NSY: ASSENTE (TX spento?)", fg='#FF3030')
            else:
                self.label_segnale_nsy.config(text="Segnale NSY: IN ATTESA...", fg='#FFD700')

        # --- 3. Valore SNR di banda ---
        if hasattr(self, 'label_snr_band'):
            snr_band = self.state.get("snr_band_db")
            if snr_band is not None:
                # Soglie unificate a 6.0. Verde se >6, giallo tra 3 e 6, rosso se <3.
                if snr_band > 4.8:
                    colore_snr = '#00FF99'
                elif snr_band > 2.0:
                    colore_snr = '#FFD700'
                else:
                    colore_snr = '#FF6347'
                self.label_snr_band.config(text=f"SNR banda ±45 Hz: {snr_band:+.1f} dB", fg=colore_snr)
            else:
                self.label_snr_band.config(text="SNR banda ±45 Hz: --- dB", fg='#888888')

        # --- 3b. Livello di rumore ---
        if hasattr(self, 'label_noise_level'):
            noise_db = self.state.get("noise_level_db")
            if noise_db is not None:
                self.label_noise_level.config(text=f"Rumore (banda SID): {noise_db:+.1f} dB", fg='#FFA500')
            else:
                self.label_noise_level.config(text="Rumore (banda SID): --- dB", fg='#888888')

        # --- 4. Potenza ---
        pwr = self.state.get("pwr_corrente")
        if pwr is not None: 
            self.label_pwr.config(text=f"{pwr:+.2f} dB")
        
        # --- 5. Fase ---
        ph2 = self.state.get("phase_corrente")
        if ph2 is not None: self.label_phase_v2.config(text=f"{ph2:+.2f}°")
        ph1 = self.state.get("phase_v1_corrente")
        if ph1 is not None: self.label_phase_v1.config(text=f"V1: {ph1:+.2f}°")

        # --- 6. Statistiche Letture ---
        stats_text = f"Letture: {self.state.get('letture_totali', 0)}"
        if self.state.get("ultimo_reset"): 
            stats_text += f"  |  Ultimo reset: {self.state.get('ultimo_reset')}"
        self.label_stats.config(text=stats_text)

        # --- 7. Gain e Drops ---
        gain_lna = self.state.get("gain_lna")
        gain_if  = self.state.get("gain_if")
        if gain_lna is not None and gain_if is not None:
            self.label_gain.config(text=f"Gain RF: LNA={gain_lna} | IF={gain_if} dB")

        self.label_drops.config(text=f"Coda IQ: {self.state.get('dropped_queue_full', 0)} | CSV: {self.state.get('dropped_rows_v1', 0)}")

        # --- 8. Amp 100 Hz e Lock PLL ---
        amp100 = self.state.get("amp_100hz_corrente")
        if amp100 is not None: 
            self.label_amp100.config(text=f"Amp riga 100 Hz: {amp100:+.2f} dB")

        stato_amp100 = self.state.get("amp100_stato")
        std_amp100   = self.state.get("amp100_std")
        mean_amp100  = self.state.get("amp100_mean")

        if stato_amp100 == "SEGNALE_ASSENTE": testo, colore = f"Lock PLL: ⊘ SEGNALE ASSENTE  |  media={mean_amp100:+.1f} dB", '#6666FF'
        elif stato_amp100 == "STABILE": testo, colore = f"Lock PLL: ✓ STABILE  |  std={std_amp100:.2f} dB  media={mean_amp100:+.1f} dB", '#00FF99'
        elif stato_amp100 == "AL_LIMITE": testo, colore = f"Lock PLL: ⚠ AL LIMITE  |  std={std_amp100:.2f} dB  media={mean_amp100:+.1f} dB", '#FFD700'
        elif stato_amp100 == "INSTABILE": testo, colore = f"Lock PLL: ✗ INSTABILE  |  std={std_amp100:.2f} dB", '#FF6347'
        elif stato_amp100 == "PERSO": testo, colore = "Lock PLL: ✗ LOCK PERSO", '#FF0000'
        else: testo, colore = "Lock PLL: ⏳ in acquisizione...", '#888888'
        self.label_amp100_stato.config(text=testo, fg=colore)

        # --- 9. Lock Quality (Resid / Mag) ---
        resid = self.state.get("phase_resid_corrente")
        mag = self.state.get("corr_mag_corrente")
        if resid is not None and mag is not None:
            self.label_resid.config(text=f"Lock Quality: Resid={resid:+.2f}° | Mag={mag:.4f}")

        # --- 10. Path File CSV ---
        self.label_file_v2.config(text=f"File V2: {self.state.get('path_csv_v2', '')}")
        self.label_file_v1.config(text=f"File V1: {self.state.get('path_csv_v1', '')}")

        # --- 11. Stato PLL (Frequenza) ---
        pll_stato = self.state.get("pll_stato")
        pll_std = self.state.get("pll_std_30")
        pll_mean = self.state.get("pll_mean_30")
        if pll_stato == "SEGNALE_ASSENTE": testo_pll, colore_pll = "⊘ PLL IN ATTESA", '#6666FF'
        elif pll_stato == "STABILE": testo_pll, colore_pll = f"✓ PLL STABILE  |  media={pll_mean:+.4f} Hz  std={pll_std*1000:.3f} mHz", '#00FF99'
        elif pll_stato == "ACCETTABILE": testo_pll, colore_pll = f"⚠ PLL ACCETTABILE  |  media={pll_mean:+.4f} Hz  std={pll_std*1000:.3f} mHz", '#FFD700'
        elif pll_stato == "INSTABILE": testo_pll, colore_pll = f"✗ PLL INSTABILE  |  media={pll_mean:+.4f} Hz  std={pll_std*1000:.3f} mHz", '#FF6347'
        else: testo_pll, colore_pll = "⏳ PLL in acquisizione...", '#888888'
        self.label_pll.config(text=testo_pll, fg=colore_pll)

        pll_freq_corrente = self.state.get("pll_freq_corrente")
        if pll_freq_corrente is not None: 
            self.label_freq_offset.config(text=f"freq_offset: {pll_freq_corrente:+.4f} Hz")

        if self.state.get("pll_modo") == "TRACKING": 
            self.label_pll_modo.config(text="PLL modo: tracking stretto", fg='#00FF99')
        else: 
            self.label_pll_modo.config(text="PLL modo: acquisizione veloce", fg='#FFD700')

        # --- 12. FFT e Trigger IQ ---
        if self.state.get("fft_aperta", False): self.aggiorna_fft()

        if self.state.get("trigger_auto_iq", False):
            self.label_iq_trigger.config(text="IQ Trigger: ATTIVO (salvataggio auto)", fg='#00FF99')
        else:
            self.label_iq_trigger.config(text="IQ Trigger: DISATTIVATO", fg='#FF6347')
            
        if self.state.get("auto_baseline_attivo", False):
            self.label_auto_baseline.config(text=f"Auto-Baseline: ATTIVO ({self.state.get('auto_baseline_ore', 4.0)} ore)", fg='#00FF99')
        else:
            self.label_auto_baseline.config(text="Auto-Baseline: DISATTIVATO", fg='#FF6347')

        # Richiama se stesso dopo 1 secondo
        self.root.after(1000, self.aggiorna_etichette)

    def toggle_fft(self, event=None):
        if self.state.get("fft_aperta", False): self.chiudi_fft()
        else: self.apri_fft()

    def apri_fft(self):
        self.state["fft_aperta"] = True
        with self.state["fft_lock"]: self.state["fft_segnale"] = None
        self.fft_spettro_smoothed = None; self.fft_freqs_smoothed = None
        self.fft_window = tk.Toplevel(self.root)
        self.fft_window.title("FFT - NSY")
        self.fft_window.geometry("1100x550")
        self.fft_window.configure(bg='#0d0d1a')
        self.fft_window.protocol("WM_DELETE_WINDOW", self.chiudi_fft)
        fig = Figure(figsize=(11, 5), dpi=100, facecolor='#0d0d1a')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#0d0d1a')
        ax.tick_params(colors='white', labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor('#444444')
        ax.grid(True, alpha=0.15, color='white')
        ax.set_xlim(FFT_XMIN_HZ, FFT_XMAX_HZ); ax.set_ylim(FFT_YMIN_DB, FFT_YMAX_DB)
        ax.set_title('FFT', color='#00FF99', fontsize=11, fontweight='bold')
        ax.axvline(0, color='red', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.axvline(-FIR_CUTOFF_HZ, color='#FFA500', linestyle=':', linewidth=0.8, alpha=0.6)
        ax.axvline(+FIR_CUTOFF_HZ, color='#FFA500', linestyle=':', linewidth=0.8, alpha=0.6)
        # Evidenzia la banda usata per il calcolo del rumore (solo parte positiva)
        ax.axvspan(500, 1500, color='#00FF99', alpha=0.1, label='Banda Rumore')
        linea, = ax.plot([], [], color='#00FF99', linewidth=0.9)
        testo_picco = ax.text(0.98, 0.02, 'Attendo...', transform=ax.transAxes, color='#00FF99', fontsize=10, va='bottom', ha='right')
        testo_snr = ax.text(0.98, 0.12, '', transform=ax.transAxes, color='#FFD700', fontsize=9, va='bottom', ha='right')
        self.fft_ax = ax; self.fft_line = linea; self.fft_testo_picco = testo_picco; self.fft_testo_snr = testo_snr
        self.fft_canvas = FigureCanvasTkAgg(fig, master=self.fft_window)
        self.fft_canvas.draw()
        self.fft_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.fft_canvas.mpl_connect('key_press_event', self._on_key_matplotlib)
        self.fft_window.focus_set(); self.fft_window.lift()

    def _on_key_matplotlib(self, event):
        if event.key is None: return
        if event.key in ('g', 'G'): self.toggle_fft()
        elif event.key in ('q', 'Q'): self.quit_app()
        elif event.key in ('a', 'A'): self.fft_zoom_in()
        elif event.key in ('d', 'D'): self.fft_zoom_out()
        elif event.key in ('+', '='): self.aumenta_smoothing()
        elif event.key in ('-', '_', 'b', 'B'): self.diminuisci_smoothing()
        elif event.key in ('z', 'Z'): self.cicla_zoom_y()

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

        # Sample rate dipendente dal modo
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
                f"Riga MSK attesa a ~+100 Hz  |  "
                f"Rumore: {rumore:+.1f} dB  |  α={FFT_SMOOTH_ALPHA:.2f}"
            )
        else:
            # Modo normale
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

            # --- CALCOLO SNR DI BANDA ROBUSTO (mediana, allineato al DSP) ---
            # Usiamo la banda di guardia a 250 Hz e la mediana dei bin per ignorare le righe di rumore
            mask_rumore_fft = (np.abs(self.fft_freqs_smoothed) > FIR_CUTOFF_NOISE_GUARD_HZ) & \
                              (np.abs(self.fft_freqs_smoothed) <= FINAL_SR / 2.0)
            snr_banda_db = 0.0
            
            if np.any(mask_banda) and np.any(mask_rumore_fft):
                p_lin = 10 ** (self.fft_spettro_smoothed / 10.0)
                
                # Usiamo la MEDIANA invece della somma
                p_in = np.median(p_lin[mask_banda])
                p_out = np.median(p_lin[mask_rumore_fft])
                
                if p_out > 0:
                    snr_banda_db = 10 * np.log10(p_in / p_out + 1e-12)

            # --- AGGIORNAMENTO TESTI ---
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

            # Calcolo del rumore per la sola visualizzazione (mediana)
            mask_rumore_vis = ((np.abs(self.fft_freqs_smoothed) > 500) &
                               (np.abs(self.fft_freqs_smoothed) < 5000))
            rumore_vis = None
            if np.any(mask_rumore_vis):
                rumore_vis = float(np.median(self.fft_spettro_smoothed[mask_rumore_vis]))

            if rumore_vis is not None:
                parti = []
                if picco_db is not None:
                    parti.append(f"SNR(picco): {picco_db - rumore_vis:.1f} dB")
                
                parti.append(f"SNR(banda): {snr_banda_db:.1f} dB")
                
                parti.append(f"Rumore: {rumore_vis:.1f} dB")
                parti.append(f"α={FFT_SMOOTH_ALPHA:.2f}")
                self.fft_testo_snr.set_text("  |  ".join(parti))

        # ---- Aggiorna il canvas ----
        try:
            self.fft_canvas.draw_idle()
        except Exception:
            pass

    def chiudi_fft(self):
        self.state["fft_aperta"] = False
        if self.fft_window is not None:
            try: self.fft_window.destroy()
            except: pass
        self.fft_window = None
        try: self.root.focus_set()
        except: pass

    def aumenta_smoothing(self, event=None):
        global FFT_SMOOTH_ALPHA; FFT_SMOOTH_ALPHA = max(0.02, FFT_SMOOTH_ALPHA - 0.03)
    def diminuisci_smoothing(self, event=None):
        global FFT_SMOOTH_ALPHA; FFT_SMOOTH_ALPHA = min(1.0, FFT_SMOOTH_ALPHA + 0.03)
    def cicla_zoom_y(self, event=None):
        global FFT_Y_ZOOM_INDEX, FFT_YMIN_DB, FFT_YMAX_DB
        FFT_Y_ZOOM_INDEX = (FFT_Y_ZOOM_INDEX + 1) % len(FFT_Y_ZOOM_LEVELS)
        FFT_YMIN_DB, FFT_YMAX_DB = FFT_Y_ZOOM_LEVELS[FFT_Y_ZOOM_INDEX]
        if self.fft_ax is not None:
            try: self.fft_ax.set_ylim(FFT_YMIN_DB, FFT_YMAX_DB)
            except: pass

    def toggle_fft_mode(self, event=None):
        """Commuta la FFT tra segnale normale (s2) e segnale squadrato (fase²)."""
        modo_attuale = self.state.get("fft_modo", "normale")
        nuovo_modo = "squadrato" if modo_attuale == "normale" else "normale"
        self.state["fft_modo"] = nuovo_modo

        # Reset dello smoothing
        self.fft_spettro_smoothed = None
        self.fft_freqs_smoothed = None

        if self.fft_ax is not None:
            if nuovo_modo == "squadrato":
                self.fft_ax.set_title(
                    'FFT MODO SQUADRATO (fase²) — M torna a normale  |  '
                    'A/D zoom  |  +/- smoothing  |  Z zoom Y',
                    color='#00BFFF', fontsize=11, fontweight='bold')
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
        global FFT_X_ZOOM_INDEX, FFT_XMIN_HZ, FFT_XMAX_HZ
        if FFT_X_ZOOM_INDEX > 0:
            FFT_X_ZOOM_INDEX -= 1; half = FFT_X_ZOOM_LEVELS_HZ[FFT_X_ZOOM_INDEX]
            FFT_XMIN_HZ, FFT_XMAX_HZ = -half, +half
            if self.fft_ax is not None:
                try: self.fft_ax.set_xlim(FFT_XMIN_HZ, FFT_XMAX_HZ); self.fft_canvas.draw_idle()
                except: pass

    def fft_zoom_out(self, event=None):
        global FFT_X_ZOOM_INDEX, FFT_XMIN_HZ, FFT_XMAX_HZ
        if FFT_X_ZOOM_INDEX < len(FFT_X_ZOOM_LEVELS_HZ) - 1:
            FFT_X_ZOOM_INDEX += 1; half = FFT_X_ZOOM_LEVELS_HZ[FFT_X_ZOOM_INDEX]
            FFT_XMIN_HZ, FFT_XMAX_HZ = -half, +half
            if self.fft_ax is not None:
                try: self.fft_ax.set_xlim(FFT_XMIN_HZ, FFT_XMAX_HZ); self.fft_canvas.draw_idle()
                except: pass

    def quit_app(self, event=None):
        if messagebox.askyesno("Conferma", "Sei sicuro di voler interrompere l'acquisizione?"):
            self.state["quit"] = True
            try: self.root.destroy()
            except: pass

    def run(self):
        self.root.mainloop()

def main():
    setup_logging()
    print("=" * 60)
    print("SID Tracker MULTI-PLL (v5.0) - V1, V2, BW02, BW01, PESATO")
    print("=" * 60)
    print(f"File V2 (ufficiale): {PREFISSI['v2']}_YYYY-MM-DD.csv (9 col)")
    print(f"File V1 (controllo): {PREFISSI['v1']}_YYYY-MM-DD.csv (7 col)")
    print(f"File sperimentali : {PREFISSI['bw02']}, {PREFISSI['bw01']}, {PREFISSI['pesato']}")
    print("=" * 60 + "\n")

    state = {
        "radio_pronta": False, "adc_overload": False, "adc_overload_tot": 0,
        "pwr_corrente": None, "phase_corrente": None, "timestamp_corrente": "",
        "letture_totali": 0, "ultimo_reset": None, "path_csv_v1": "", "path_csv_v2": "", "quit": False,
        "fft_aperta": False, "fft_segnale": None, "fft_segnale_sq": None,
        "fft_modo": "normale", "fft_nuovo_dato": False, "fft_lock": threading.Lock(),
        "fft_segnale_sq": None,
        "pll_std_30": None, "pll_mean_30": None, "pll_stato": None,      
        "pll_freq_corrente": None, "pll_modo": "ACQUISIZIONE",
        "amp_100hz_corrente": None, "gain_lna": None, "gain_if": None,
        "amp100_std": None, "amp100_mean": None, "amp100_stato": None,
        "dropped_queue_full": 0, "dropped_excess_samples": 0,
        "dropped_rows_v1": 0, "dropped_rows_v2": 0,
        "gain_richiesto_lna": None, "gain_richiesto_if": None, "gain_change_pending": False,
        "phase_resid_corrente": None, "corr_mag_corrente": None,
        "salva_iq_richiesto": False,
        "trigger_auto_iq": False,
        "auto_baseline_attivo": False,
        "auto_baseline_ore": 4.0,
        "auto_baseline_inizio": 0.0,
        "auto_baseline_ultimo_save": 0.0,
        "segnale_nsy_presente": None,
        "snr_band_db": None,     
    }

    acquisitore = AcquisitoreSID(state)
    acquisitore.start()
    gui = SIDGui(state, acquisitore)
    
    def update_paths():
        state["path_csv_v2"] = acquisitore.writer.managers["v2"].path_corrente or "In attesa..."
        state["path_csv_v1"] = acquisitore.writer.managers["v1"].path_corrente or "In attesa..."
        gui.root.after(2000, update_paths)
        
    gui.root.after(2000, update_paths)

    try: 
        gui.run()
    except KeyboardInterrupt: 
        pass
    finally:
        acquisitore.stop()
        acquisitore.join(timeout=3.0)
        print("[MAIN] Chiusura completata.")

if __name__ == "__main__":
    main()