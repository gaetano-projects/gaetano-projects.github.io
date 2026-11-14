#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualizza_tempesta.py (v33 - Fase per segmenti + detrend spline per segmento)

Novita' rispetto a v32:
  - Nessun "stitching" con previsione lineare attraverso i buchi lunghi
    (l'ambiguita' sul numero di cicli persi e' irrecuperabile).
  - Segmentazione su Corr_Mag / Reset_Flag / gap > MAX_BRIDGE_S.
  - Unwrap solo dentro ciascun segmento.
  - Detrend con spline ai minimi quadrati a nodi radi, per segmento.
  - I buchi restano NaN: la linea si interrompe.
"""

import sys
import os
import glob
import argparse
import calendar
import traceback
import datetime as dt

import numpy as np
import pandas as pd
from scipy.interpolate import LSQUnivariateSpline
from scipy.ndimage import median_filter, uniform_filter1d

try:
    import tkinter as tk
    _HAS_TK = True
except ImportError:
    tk = None
    _HAS_TK = False

# ----------------------- CONFIG -----------------------
DATA_DIR       = r"D:\Gaetano\SID\dati\Sdruno"
STORICO_DIR    = os.path.join(DATA_DIR, "Storico_NSY")
SEARCH_DIRS    = [(DATA_DIR, False), (STORICO_DIR, True)]

TARGET_FREQ    = 45905.0
CANON_COLS = [
    "Timestamp_UTC", "Power_dB", "Phase_Deg", "PLL_Freq_Hz",
    "Reset_Flag", "Amp_100Hz_dB", "ADC_Overload", "Phase_Resid_Deg", "Corr_Mag",
]

DETREND_WINDOW_MIN     = 60
DOPPLER_RMS_WINDOW_MIN = 5
CORR_LOW_THRESH        = 0.4

# --- Parametri fase ---
MAX_BRIDGE_S   = 120     # gap massimo (s) attraversabile senza spezzare il segmento
                         # (con la fase de-ruotata la deriva e' lenta: 120 s restano ben sotto i 180 gradi)
MIN_SEG_S      = 600     # segmenti piu' corti di cosi' (s) -> NaN
KNOT_SLOW_S    = 21600   # trend lento (6 h) usato solo per il confronto col tasto [u]
KNOT_S         = 1800   # un nodo della spline ogni 30 min (piu' alto = piu' lento il trend rimosso)
EDGE_TRIM_S    = 900     # secondi mascherati attorno a ogni interruzione/riavvio (non a inizio/fine giornata)
PHASE_MEDIAN_N = 0       # >0 (dispari, es. 5): filtro mediano su sin/cos prima dell'unwrap
PHASE_AVG_N    = 1       # media circolare su N campioni (solo dopo la de-rotazione); 1 = off
DOPPLER_GLITCH_SIGMA = 6  # scarta la fase dove il Doppler dista > N sigma robuste dalla mediana (0 = off)
DOPPLER_GLITCH_PAD   = 2  # campioni scartati attorno a ogni outlier del Doppler
GLITCH_DEG     = 35     # scarta campioni che si discostano > N gradi dalla mediana locale (0 = off)
GLITCH_N       = 31      # finestra (campioni) della mediana locale per il rilevamento glitch
DEROTATE       = True   # toglie la rotazione costante della fase (battimento) prima dell'unwrap

ITALIAN_MONTHS = ["Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno", "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"]
WEEKDAY_LABELS_IT = ["Lu", "Ma", "Me", "Gi", "Ve", "Sa", "Do"]

_TK_ROOT = None

def get_tk_root():
    global _TK_ROOT
    if _TK_ROOT is None or not _TK_ROOT.winfo_exists():
        _TK_ROOT = tk.Tk()
        _TK_ROOT.withdraw()
    return _TK_ROOT

# ----------------------- RICERCA FILE -----------------------
def _iter_v2_files_for_ymd(ymd_compact, ymd_dash):
    seen = set()
    for base, recursive in SEARCH_DIRS:
        if not os.path.isdir(base): continue
        for pat in (f"GAESID_NSY_{ymd_compact}*.csv", f"GAESID_NSY_{ymd_dash}*.csv"):
            full = os.path.join(base, "**", pat) if recursive else os.path.join(base, pat)
            for f in glob.glob(full, recursive=recursive):
                rp = os.path.realpath(f)
                if rp not in seen:
                    seen.add(rp); yield f

def find_v2_files(date):
    return sorted(_iter_v2_files_for_ymd(date.strftime("%Y%m%d"), date.strftime("%Y-%m-%d")))

def _scan_month(year=None, month=None):
    days = set()
    for base, recursive in SEARCH_DIRS:
        if not os.path.isdir(base): continue
        pattern = os.path.join(base, "**", "GAESID_NSY_*.csv") if recursive else os.path.join(base, "GAESID_NSY_*.csv")
        try:
            for f in glob.glob(pattern, recursive=recursive):
                name = os.path.basename(f)
                if not name.startswith("GAESID_NSY_"): continue
                datepart = name[len("GAESID_NSY_"):].split("_")[0].split(".")[0].replace("-", "")
                if len(datepart) >= 8 and datepart[:8].isdigit():
                    y, m, d = int(datepart[:4]), int(datepart[4:6]), int(datepart[6:8])
                    if y == year and m == month: days.add(d)
        except Exception: pass
    return days

# ----------------------- LETTORE CSV -----------------------
def _map_columns(df, path):
    n = len(df.columns)
    if n < 2: return None
    if n > len(CANON_COLS):
        df = df.iloc[:, :len(CANON_COLS)].copy(); n = len(df.columns)
    df = df.copy(); df.columns = CANON_COLS[:n]
    return df

def _has_usable(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return len(s) > 0 and (s != 0).any()

def _read_csv_robust(path):
    attempts = []
    for sep in (',', ';', '\t', None):
        for comment in ('#', None):
            kw = {'comment': comment}
            if sep: kw['sep'] = sep
            else: kw['engine'] = 'python'
            attempts.append(kw)

    for kw in attempts:
        try:
            df_test = pd.read_csv(path, nrows=5, **kw)
            if 2 <= len(df_test.columns) <= len(CANON_COLS):
                df = pd.read_csv(path, **kw)
                if len(df) > 0:
                    out = _map_columns(df, path)
                    if out is not None: return out
        except Exception: pass
    return None

def load_day(date):
    files = find_v2_files(date)
    if not files: return None

    frames = [_read_csv_robust(f) for f in files]
    frames = [d for d in frames if d is not None and not d.empty]
    if not frames: return None

    df = pd.concat(frames, ignore_index=True, sort=False)
    for c in CANON_COLS:
        if c not in df.columns: df[c] = np.nan
    df = df[CANON_COLS].copy()

    df["Timestamp_UTC"] = pd.to_datetime(df["Timestamp_UTC"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["Timestamp_UTC"]).sort_values("Timestamp_UTC").drop_duplicates("Timestamp_UTC").reset_index(drop=True)

    for c in CANON_COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    avail = {
        "power": _has_usable(df["Power_dB"]),
        "phase": _has_usable(df["Phase_Deg"]),
        "doppler": _has_usable(df["PLL_Freq_Hz"]),
        "corr": _has_usable(df["Corr_Mag"])
    }
    df.attrs["avail"] = avail
    return df

# ----------------------- FASE: SEGMENTI + DETREND -----------------------
def _spline_trend(ts, u, knot_s):
    """Trend lento: spline cubica LSQ con un nodo ogni knot_s secondi (fallback polinomiale)."""
    T = ts[-1] - ts[0]
    nk = int(T // knot_s)
    if nk >= 1:
        knots = np.linspace(ts[0], ts[-1], nk + 2)[1:-1]
        try:
            return LSQUnivariateSpline(ts, u, knots, k=3)(ts)
        except Exception:
            pass
    deg = 2 if T > 3600 else 1
    return np.polyval(np.polyfit(ts - ts[0], u, deg), ts - ts[0])


def phase_segments_detrend(t_sec, p_deg, corr, reset, bad=None):
    """
    Restituisce (unwrap_rel, detrended), entrambi NaN fuori dai segmenti validi.

    - Un segmento e' una sequenza di campioni validi (corr alta, no reset)
      senza gap > MAX_BRIDGE_S e senza Reset_Flag nel mezzo.
    - L'unwrap e' fatto solo dentro il segmento (nessuna estrapolazione).
    - unwrap_rel e' la fase srotolata relativa al primo campione del segmento.
    - detrended = fase srotolata - spline LSQ a nodi radi (ciclo diurno lento).
    """
    n = len(p_deg)
    unwrap = np.full(n, np.nan)
    detr   = np.full(n, np.nan)
    trend_out = np.full(n, np.nan)   # trend stimato, nello stesso riferimento di unwrap_rel
    slow_out  = np.full(n, np.nan)   # fase detrended con trend LENTO (KNOT_SLOW_S): per confronto

    valid = (corr > CORR_LOW_THRESH) & (reset == 0) & np.isfinite(p_deg)
    if bad is not None:
        valid &= ~bad                # campioni scartati a monte (es. outlier del Doppler)
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return unwrap, detr, trend_out, slow_out

    reset_cum = np.cumsum(reset > 0)
    brk = (np.diff(t_sec[idx]) > MAX_BRIDGE_S) | (np.diff(reset_cum[idx]) > 0)
    segs = np.split(idx, np.flatnonzero(brk) + 1)

    # Velocita' di rotazione costante della fase (battimento tra segnale e
    # oscillatore di riferimento), in gradi/secondo: mediana dei salti avvolti.
    omega = 0.0
    if DEROTATE:
        dts = np.diff(t_sec)
        pair = valid[1:] & valid[:-1] & (dts > 0) & (dts <= 2)
        if pair.sum() > 100:
            step = ((np.diff(p_deg) + 180.0) % 360.0 - 180.0)[pair] / dts[pair]
            omega = float(np.median(step))
    phase_segments_detrend.last_omega = omega

    for k_seg, seg in enumerate(segs):
        ts = t_sec[seg]
        T = ts[-1] - ts[0]
        if len(seg) < 10 or T < MIN_SEG_S:
            continue

        # De-rotazione: tolgo la rampa lineare omega*t PRIMA di mediare/unwrappare
        p_seg = p_deg[seg]
        if omega != 0.0:
            p_seg = (p_seg - omega * (ts - ts[0]) + 180.0) % 360.0 - 180.0
        ph = np.deg2rad(p_seg)
        if PHASE_MEDIAN_N and PHASE_MEDIAN_N > 1 and len(seg) > PHASE_MEDIAN_N:
            s = median_filter(np.sin(ph), size=PHASE_MEDIAN_N, mode="nearest")
            c = median_filter(np.cos(ph), size=PHASE_MEDIAN_N, mode="nearest")
            ph = np.arctan2(s, c)
        if PHASE_AVG_N and PHASE_AVG_N > 1 and len(seg) > PHASE_AVG_N:
            # Media circolare mobile pesata da Corr_Mag: riduce il rumore di fase
            # sotto la soglia oltre cui np.unwrap accumula errori di 360 gradi.
            w = np.clip(corr[seg], 0.0, 1.0)
            s = uniform_filter1d(w * np.sin(ph), PHASE_AVG_N, mode="nearest")
            c = uniform_filter1d(w * np.cos(ph), PHASE_AVG_N, mode="nearest")
            ph = np.arctan2(s, c)
        # Glitch isolati (slip del PLL, campioni con timestamp sfalsato): scarto i campioni
        # che si discostano troppo dalla mediana locale della fase de-ruotata,
        # PRIMA dell'unwrap, cosi' non lo corrompono.
        if GLITCH_DEG > 0 and len(seg) > GLITCH_N:
            s = median_filter(np.sin(ph), size=GLITCH_N, mode="nearest")
            c = median_filter(np.cos(ph), size=GLITCH_N, mode="nearest")
            dev = np.abs(np.rad2deg(np.angle(np.exp(1j * (ph - np.arctan2(s, c))))))
            good = dev <= GLITCH_DEG
            if good.sum() < 10:
                continue
            seg, ts, ph = seg[good], ts[good], ph[good]
            T = ts[-1] - ts[0]
            if T < MIN_SEG_S:
                continue
        u = np.rad2deg(np.unwrap(ph))

        # Trend: spline cubica ai minimi quadrati con nodi interni radi
        nk = int(T // KNOT_S)
        trend = None
        if nk >= 1:
            knots = np.linspace(ts[0], ts[-1], nk + 2)[1:-1]
            try:
                trend = LSQUnivariateSpline(ts, u, knots, k=3)(ts)
            except Exception:
                trend = None
        if trend is None:
            deg = 2 if T > 3600 else 1
            cf = np.polyfit(ts - ts[0], u, deg)
            trend = np.polyval(cf, ts - ts[0])

        d = u - trend
        r = u - u[0]

        # Maschera i bordi interni (dopo/prima di un'interruzione o un riavvio), dove la spline e'
        # meno vincolata e il PLL puo' avere un transitorio. Non maschera l'inizio della giornata
        # ne' la fine (gli ultimi minuti servono per seguire un evento in corso).
        lo = EDGE_TRIM_S if k_seg > 0 else 0
        hi = EDGE_TRIM_S if k_seg < len(segs) - 1 else 0
        edge = np.zeros(len(ts), bool)
        if (lo or hi) and T > lo + hi + 300:
            edge = (ts < ts[0] + lo) | (ts > ts[-1] - hi)
            d = d.copy(); d[edge] = np.nan
            r = r.copy(); r[edge] = np.nan

        unwrap[seg] = r
        detr[seg]   = d
        trend_out[seg] = trend - u[0]

        sl = u - _spline_trend(ts, u, KNOT_SLOW_S)
        sl[edge] = np.nan
        slow_out[seg] = sl

    return unwrap, detr, trend_out, slow_out

# ----------------------- CALCOLO CANALI (V33) -----------------------
def compute_channels(df):
    out = df.copy()
    avail = out.attrs.get("avail", {})
    idx = out.set_index("Timestamp_UTC")
    win = f"{DETREND_WINDOW_MIN}min"
    rms_win = f"{DOPPLER_RMS_WINDOW_MIN}min"

    # --- POTENZA ---
    if avail.get("power", False):
        p = idx["Power_dB"].astype(float).where(idx["Power_dB"] > -80)
        out["Power_detr"] = (p - p.rolling(win, center=True, min_periods=5).median()).values
    else: out["Power_detr"] = np.nan

    # --- FASE (segmenti + spline per segmento) ---
    if avail.get("phase", False):
        t_sec = (idx.index - idx.index[0]).total_seconds().values

        # Maschera dai campioni in cui il Doppler e' un outlier (glitch del PLL):
        # |Doppler - mediana| > DOPPLER_GLITCH_SIGMA * sigma robusta, allargata di +-DOPPLER_GLITCH_PAD campioni
        bad = None
        if avail.get("doppler", False) and DOPPLER_GLITCH_SIGMA > 0:
            f = idx["PLL_Freq_Hz"].astype(float).replace(0, np.nan).to_numpy()
            dm = (f - np.nanmedian(f)) * 1000.0
            med = np.nanmedian(dm)
            sig = 1.4826 * np.nanmedian(np.abs(dm - med))
            bad = ~np.isfinite(dm)
            if np.isfinite(sig) and sig > 0:
                bad |= np.abs(dm - med) > DOPPLER_GLITCH_SIGMA * sig
            if DOPPLER_GLITCH_PAD > 0 and bad.any():
                k = 2 * DOPPLER_GLITCH_PAD + 1
                bad = np.convolve(bad.astype(float), np.ones(k), mode="same") > 0
            out.attrs["n_doppler_glitch"] = int(bad.sum())

        pu, pdt, ptr, psl = phase_segments_detrend(
            t_sec,
            idx["Phase_Deg"].astype(float).to_numpy(),
            idx["Corr_Mag"].fillna(0).to_numpy(),
            idx["Reset_Flag"].fillna(0).to_numpy(),
            bad=bad,
        )
        out["Phase_unwrap"] = pu
        out["Phase_detr"]   = pdt
        out["Phase_trend"]  = ptr
        out["Phase_slow"]   = psl
    else:
        out["Phase_unwrap"] = np.nan
        out["Phase_detr"]   = np.nan
        out["Phase_trend"]  = np.nan
        out["Phase_slow"]   = np.nan

    # --- DOPPLER ---
    if avail.get("doppler", False):
        f = idx["PLL_Freq_Hz"].astype(float).replace(0, np.nan)
        median_f = f.dropna().median()
        if pd.isna(median_f): median_f = TARGET_FREQ

        tol = 100.0 if median_f > 1000 else 0.1
        f = f.where((f > median_f - tol) & (f < median_f + tol))

        d = (f - median_f) * 1000.0
        out["Doppler_mHz"] = d.values

        d_series = pd.Series(d.values, index=idx.index)
        out["Doppler_rms_mHz"] = np.sqrt(d_series.pow(2).rolling(rms_win, center=True, min_periods=3).mean()).values
    else:
        out["Doppler_mHz"] = np.nan
        out["Doppler_rms_mHz"] = np.nan

    return out

# ----------------------- CALENDARIO -----------------------
class CalendarDialog:
    def __init__(self, parent, initial_date=None):
        self.result = None
        base = initial_date or dt.date.today()
        self.year, self.month, self.selected = base.year, base.month, base
        self._data_cache = {}

        self.top = tk.Toplevel(parent)
        self.top.title("Seleziona data")
        self.top.resizable(False, False)
        self.top.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self._build_static()
        self._render_month()

        self.top.update_idletasks()
        w, h = max(280, self.top.winfo_reqwidth()), self.top.winfo_reqheight()
        self.top.geometry(f"{w}x{h}+{(self.top.winfo_screenwidth()-w)//2}+{(self.top.winfo_screenheight()-h)//2}")

        self.top.lift()
        self.top.attributes("-topmost", True)
        self.top.after(200, lambda: self.top.attributes("-topmost", False))
        self.top.focus_force()
        self.top.grab_set()
        self.top.wait_window()

        try: self.top.grab_release()
        except Exception: pass

    def _build_static(self):
        nav = tk.Frame(self.top, padx=8, pady=6); nav.pack(fill="x")
        tk.Button(nav, text="<<", width=3, command=self._prev_month).pack(side="left")
        self.lbl_month = tk.Label(nav, text="", font=("TkDefaultFont", 10, "bold"))
        self.lbl_month.pack(side="left", expand=True)
        tk.Button(nav, text=">>", width=3, command=self._next_month).pack(side="right")
        hdr = tk.Frame(self.top, padx=4); hdr.pack()
        for i, lab in enumerate(WEEKDAY_LABELS_IT):
            tk.Label(hdr, text=lab, width=3, font=("TkDefaultFont", 9, "bold")).grid(row=0, column=i)
        self.days_frame = tk.Frame(self.top, padx=4); self.days_frame.pack()
        bot = tk.Frame(self.top, padx=8, pady=6); bot.pack(fill="x")
        tk.Button(bot, text="Oggi", width=8, command=self._go_today).pack(side="left")
        tk.Button(bot, text="Annulla", width=8, command=self._on_cancel).pack(side="right")
        tk.Button(bot, text="OK", width=8, command=self._on_ok).pack(side="right", padx=4)

    def _render_month(self):
        self.lbl_month.config(text=f"{ITALIAN_MONTHS[self.month - 1]} {self.year}")
        for w in self.days_frame.winfo_children(): w.destroy()
        key = (self.year, self.month)
        if key not in self._data_cache: self._data_cache[key] = _scan_month(self.year, self.month)
        days_with_data = self._data_cache[key]
        weeks = calendar.Calendar(0).monthdayscalendar(self.year, self.month)
        while len(weeks) < 6: weeks.append([0] * 7)
        for r, week in enumerate(weeks):
            for c, day in enumerate(week):
                if day == 0:
                    tk.Label(self.days_frame, text="", width=3).grid(row=r, column=c); continue
                d = dt.date(self.year, self.month, day)
                has_data = day in days_with_data; is_sel = (d == self.selected)
                bg, fg = ("#4a90d9", "white") if is_sel else (("#ffffff", "black") if has_data else ("#f0f0f0", "#9a9a9a"))
                b = tk.Button(self.days_frame, text=str(day), width=3, bg=bg, fg=fg, relief="flat", command=lambda dd=day: self._on_click(dd))
                b.grid(row=r, column=c, padx=1, pady=1)
                b.bind("<Double-Button-1>", lambda e, dd=day: self._on_double(dd))

    def _prev_month(self):
        self.month, self.year = (12, self.year-1) if self.month == 1 else (self.month-1, self.year); self._render_month()
    def _next_month(self):
        self.month, self.year = (1, self.year+1) if self.month == 12 else (self.month+1, self.year); self._render_month()
    def _go_today(self):
        t = dt.date.today(); self.year, self.month, self.selected = t.year, t.month, t; self._render_month()
    def _on_click(self, day): self.selected = dt.date(self.year, self.month, day); self._render_month()
    def _on_double(self, day): self.selected = dt.date(self.year, self.month, day); self._on_ok()
    def _on_ok(self): self.result = self.selected; self.top.destroy()
    def _on_cancel(self): self.result = None; self.top.destroy()

# ----------------------- MATPLOTLIB -----------------------
def import_pyplot():
    import matplotlib
    if matplotlib.get_backend().lower() in ("agg", "pdf", "ps", "svg", "cairo"):
        try: matplotlib.use("TkAgg")
        except Exception: pass
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec
    return plt, mdates, GridSpec

class StormViewer:
    def __init__(self, date, plt, mdates, GridSpec):
        self.date, self.plt, self.mdates, self._GridSpec = date, plt, mdates, GridSpec
        self.show_unwrap = False
        self._build_figure()
        self._load_and_plot()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _style_axes(self):
        for ax in self.axes.values():
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(self.mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(self.mdates.HourLocator(interval=2))
            ax.ticklabel_format(useOffset=False, style='plain', axis='y')
            ax.xaxis.offsetText.set_visible(False)

        for k in ("power", "phase", "doppler"):
            self.plt.setp(self.axes[k].get_xticklabels(), visible=False)

    def _build_figure(self):
        self.fig = self.plt.figure(figsize=(14, 10))
        gs = self._GridSpec(4, 1, figure=self.fig, hspace=0.35, left=0.08, right=0.92, top=0.94, bottom=0.06)
        ax_p = self.fig.add_subplot(gs[0])
        self.ax_power_raw = ax_p.twinx(); self.ax_power_raw.set_visible(False)
        ax_f = self.fig.add_subplot(gs[1], sharex=ax_p)
        self.ax_phase_raw = ax_f.twinx(); self.ax_phase_raw.set_visible(False)
        ax_d = self.fig.add_subplot(gs[2], sharex=ax_p)
        ax_q = self.fig.add_subplot(gs[3], sharex=ax_p)
        self.axes = {"power": ax_p, "phase": ax_f, "doppler": ax_d, "quality": ax_q}
        self.title = self.fig.suptitle("", fontsize=12)
        self._style_axes()

    def _smart_ylim(self, ax, data_list, margin=0.1):
        for data in data_list:
            s = pd.Series(data).dropna()
            s = s[np.isfinite(s)]
            if len(s) > 10:
                p1, p99 = np.percentile(s, 1), np.percentile(s, 99)
                if not np.isfinite(p1) or not np.isfinite(p99): continue
                if p99 == p1: p99 += 1.0; p1 -= 1.0
                diff = p99 - p1
                ax.set_ylim(p1 - diff * margin, p99 + diff * margin)
                return

    def _load_and_plot(self):
        try:
            for ax in self.axes.values(): ax.clear()
            if hasattr(self, "ax_power_raw"): self.ax_power_raw.clear(); self.ax_power_raw.set_visible(False)
            if hasattr(self, "ax_phase_raw"): self.ax_phase_raw.clear(); self.ax_phase_raw.set_visible(False)
            # clear() su un asse gemello riporta ticks ed etichetta a sinistra: li rimetto a destra
            for _a in (self.ax_power_raw, self.ax_phase_raw):
                _a.yaxis.tick_right(); _a.yaxis.set_label_position("right")

            df = load_day(self.date)
            if df is None or df.empty:
                self.title.set_text(f"Nessun dato per {self.date.isoformat()}")
                self._style_axes()
                self.fig.canvas.draw_idle(); return

            df = compute_channels(df)
            avail = df.attrs.get("avail", {})
            t = df["Timestamp_UTC"]

            if len(t) > 0:
                x_min, x_max = t.iloc[0], t.iloc[-1]
                for ax in self.axes.values(): ax.set_xlim(x_min, x_max)

            # POTENZA
            ax = self.axes["power"]
            if avail.get("power", False):
                ax.plot(t, df["Power_detr"], lw=1.4, color="C0", label="Power detrended (sx)")
                ax.axhline(0, color="k", lw=0.5, alpha=0.3)
                self._smart_ylim(ax, [df["Power_detr"]])
                self.ax_power_raw.set_visible(True)
                self.ax_power_raw.plot(t, df["Power_dB"], lw=0.6, color="0.55", alpha=0.6, label="Power raw (dx)")
                self._smart_ylim(self.ax_power_raw, [df["Power_dB"]])
                ax.set_ylabel("Detrended (dB)", color="C0")
                self.ax_power_raw.set_ylabel("Raw (dB)", color="0.55")
                ax.set_title("CANALE POTENZA", fontsize=10, loc="left")

            # FASE: solo la curva detrended; i buchi restano vuoti (NaN)
            ax = self.axes["phase"]
            if avail.get("phase", False) and df["Phase_detr"].notna().any():
                ax.plot(t, df["Phase_detr"], lw=1.2, color="C2", label="Fase Detrended (°)")
                ax.axhline(0, color="k", lw=0.5, alpha=0.5)
                self._smart_ylim(ax, [df["Phase_detr"]])
                ax.legend(loc="upper right", fontsize=8)
                ax.set_title("FASE DETRENDED (segmenti + spline)   [u] confronta con detrend lento", fontsize=10, loc="left")
                if self.show_unwrap and df["Phase_unwrap"].notna().any():
                    # Fase de-ruotata PRIMA del detrend (relativa all'inizio di ciascun segmento) e trend,
                    # su asse gemello a destra e dietro la curva verde: serve a distinguere
                    # un gradino permanente (strumentale) da un impulso che rientra (fisico).
                    axr = self.ax_phase_raw
                    axr.set_visible(True)
                    axr.set_zorder(ax.get_zorder() - 1)
                    ax.patch.set_visible(False)
                    axr.plot(t, df["Phase_slow"], lw=1.0, color="C1", alpha=0.8,
                             label=f"Detrend lento {KNOT_SLOW_S // 3600} h (dx)")
                    axr.axhline(0, color="C1", lw=0.5, alpha=0.4)
                    self._smart_ylim(axr, [df["Phase_slow"]])
                    axr.set_ylabel(f"Detrend lento {KNOT_SLOW_S // 3600} h (°)", color="C1")
                    h1, l1 = ax.get_legend_handles_labels()
                    h2, l2 = axr.get_legend_handles_labels()
                    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
            else:
                ax.text(0.5, 0.5, "N/D", transform=ax.transAxes, ha="center", va="center", color="gray")
            ax.set_ylabel("Fase (°)")

            # DOPPLER
            ax = self.axes["doppler"]
            if avail.get("doppler", False):
                ax.plot(t, df["Doppler_mHz"], lw=0.5, color="0.6", label="Doppler (mHz)")
                ax.plot(t, df["Doppler_rms_mHz"], lw=1.4, color="C3", label=f"RMS {DOPPLER_RMS_WINDOW_MIN} min")
                ax.axhline(0, color="k", lw=0.5, alpha=0.5)

                s = df["Doppler_mHz"].dropna()
                if len(s) > 0:
                    med, std = s.median(), s.std()
                    if np.isfinite(med) and np.isfinite(std) and std > 0:
                        ax.set_ylim(med - 5*std, med + 5*std)
                    else:
                        self._smart_ylim(ax, [s])
                ax.legend(loc="upper right", fontsize=8)
                ax.set_title("DOPPLER", fontsize=10, loc="left")
            else:
                ax.text(0.5, 0.5, "N/D", transform=ax.transAxes, ha="center", va="center", color="gray")
            ax.set_ylabel("Doppler (mHz)")

            # QUALITÀ
            ax = self.axes["quality"]
            if avail.get("corr", False):
                ax.plot(t, df["Corr_Mag"], lw=1.0, color="C4", label="Corr_Mag")
                ax.axhline(CORR_LOW_THRESH, color="red", lw=0.8, ls="--", label="soglia")
                ax.legend(loc="upper right", fontsize=8)
            else:
                ax.text(0.5, 0.5, "N/D", transform=ax.transAxes, ha="center", va="center", color="gray")
            ax.set_ylabel("Corr_Mag"); ax.set_xlabel("UTC")
            ax.set_title("QUALITÀ", fontsize=10, loc="left")

            self.title.set_text(f"Tempesta geomagnetica — {self.date.isoformat()} — {len(df)} campioni")
            self._style_axes()
            self.fig.canvas.draw_idle()

        except Exception as e:
            print(f"[ERRORE PLOT] {e}")
            traceback.print_exc()
            self.title.set_text(f"Errore: {e}")
            self._style_axes()
            self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == 'i':
            if _HAS_TK:
                root = get_tk_root()
                d = CalendarDialog(root, self.date).result
                if d and d != self.date: self.date = d; self._load_and_plot()
        elif event.key == 'r': self._load_and_plot()
        elif event.key == 'u':
            self.show_unwrap = not self.show_unwrap
            self._load_and_plot()
        elif event.key == 'q': self.plt.close(self.fig)

    def show(self): self.plt.show()

# ----------------------- MAIN -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true")
    parser.add_argument("--date", type=str)
    args = parser.parse_args()

    initial = dt.date.fromisoformat(args.date) if args.date else None
    print("[INFO] Apertura...")

    if args.cli or not _HAS_TK:
        s = input(f"Data (YYYY-MM-DD) [{(initial or dt.date.today()).isoformat()}]: ").strip()
        date = dt.date.fromisoformat(s or (initial or dt.date.today()).isoformat())
    else:
        root = get_tk_root()
        dlg = CalendarDialog(root, initial)
        date = dlg.result

    if date is None:
        print("[INFO] Nessuna data selezionata. Uscita."); return

    print(f"[INFO] Data selezionata: {date.isoformat()}")
    plt, mdates, GridSpec = import_pyplot()
    StormViewer(date, plt, mdates, GridSpec).show()

if __name__ == "__main__":
    try: main()
    except Exception:
        traceback.print_exc()
        try: input("\nErrore - premi INVIO per chiudere...")
        except EOFError: pass
