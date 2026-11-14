# ============================================================
# frozen_method.py
#
# Metodo congelato v1.0 (2026-10-09) — correlazione Doppler vs
# d(log X)/dt con doppio null empirico.
#
# Validato su M6.7 (p_A=0.002, p_B=0.016).
# Controllo negativo superato su C1.5, C3.6, C5.3.
#
# NON MODIFICARE senza nuova sessione di validazione.
# ============================================================

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from dsp_utils import filtro_hampel, sanity_doppler_con_durata

# ---- Configurazione congelata ----
DOPPLER_BANDPASS_MIN = (5.0, 60.0)
FILTER_EDGE_PAD_S    = 1800.0
XRAY_SMOOTH_MIN      = 3.0
XRAY_FLUX_MIN        = 1e-10
XRAY_DLOG_DT_MAX     = 0.5
FINESTRA_MIN         = 90.0
FS                   = 1.0

NULL_A_N             = 500
NULL_A_ESCL_MIN      = 45.0
NULL_A_MASK_MIN      = 0.6
NULL_B_N             = 500
NULL_B_ESCL_MIN      = 90.0
NULL_SEED            = 42

SOGLIA_P_NULL_A      = 0.01
SOGLIA_P_NULL_B      = 0.05

VERDETTO_SIG         = "SIGNIFICATIVO"
VERDETTO_PARZIALE    = "PARZIALE"
VERDETTO_NS          = "NON SIGNIFICATIVO"


def _passa_banda(serie, banda_min, fs=FS, pad_s=FILTER_EDGE_PAD_S):
    n = len(serie)
    if n < 300:
        return serie - np.median(serie)
    s = serie - np.median(serie)
    pad_len = min(int(pad_s * fs), n // 2)
    s_pad = np.concatenate([s[:pad_len][::-1], s, s[-pad_len:][::-1]])
    f_lo = 1.0 / (banda_min[1] * 60.0)
    f_hi = 1.0 / (banda_min[0] * 60.0)
    nyq = 0.5 * fs
    f_lo_n = max(f_lo / nyq, 1e-4)
    f_hi_n = min(f_hi / nyq, 0.99)
    if f_lo_n >= f_hi_n:
        return s
    sos = butter(2, [f_lo_n, f_hi_n], btype='bandpass', output='sos')
    return sosfiltfilt(sos, s_pad)[pad_len:pad_len + n]


def _smooth_moving(serie, win_s):
    if win_s <= 1:
        return serie
    return pd.Series(serie).rolling(int(win_s), center=True,
                                     min_periods=1).mean().values


def _corr_mascherata(a, b, mask):
    m = mask & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return np.nan
    a_m = a[m]
    b_m = b[m]
    # Protezione esplicita contro stddev = 0. Senza questa,
    # np.corrcoef genera NaN e RuntimeWarning su serie costanti.
    std_a = float(np.std(a_m))
    std_b = float(np.std(b_m))
    if std_a < 1e-12 or std_b < 1e-12:
        return np.nan
    az = (a_m - a_m.mean()) / std_a
    bz = (b_m - b_m.mean()) / std_b
    with np.errstate(invalid='ignore', divide='ignore'):
        return float(np.corrcoef(az, bz)[0, 1])


def _p_empirico(val_oss, null_vals):
    null_arr = np.asarray(null_vals)
    N = len(null_arr)
    if N == 0:
        return 1.0, 0, 0
    k = int((np.abs(null_arr) >= abs(val_oss)).sum())
    return (k + 1) / (N + 1), k, N


def _null_A(dop_win, dlog_full, mask_xray_full, idx_pk, n_half):
    np.random.seed(NULL_SEED)
    margine = n_half
    idx_min = margine
    idx_max = len(dlog_full) - margine - 1
    esclusi = int(NULL_A_ESCL_MIN * 60)

    null_vals, tent = [], 0
    while len(null_vals) < NULL_A_N and tent < NULL_A_N * 6:
        tent += 1
        center = np.random.randint(idx_min, idx_max)
        if abs(center - idx_pk) < esclusi:
            continue
        idx_s = np.arange(center - n_half, center + n_half + 1)
        dlog_s = dlog_full[idx_s]
        mask_s = mask_xray_full[idx_s]
        if mask_s.mean() < NULL_A_MASK_MIN:
            continue
        c = _corr_mascherata(dop_win, dlog_s, mask_s)
        if np.isfinite(c):
            null_vals.append(c)
    return np.array(null_vals)


def _null_B(doppler_full, dlog_win, mask_win, idx_pk, n_half):
    np.random.seed(NULL_SEED)
    margine = n_half
    idx_min = margine
    idx_max = len(doppler_full) - margine - 1
    esclusi = int(NULL_B_ESCL_MIN * 60)

    null_vals, tent = [], 0
    while len(null_vals) < NULL_B_N and tent < NULL_B_N * 6:
        tent += 1
        center = np.random.randint(idx_min, idx_max)
        if abs(center - idx_pk) < esclusi:
            continue
        idx_s = np.arange(center - n_half, center + n_half + 1)
        dop_s = doppler_full[idx_s]
        c = _corr_mascherata(dop_s, dlog_win, mask_win)
        if np.isfinite(c):
            null_vals.append(c)
    return np.array(null_vals)


def valuta_frozen_da_df(df, ts_xray, flux_xray, t_pk):
    """
    Wrapper che prende il df del tracker (intera giornata), i dati
    X-ray (ts_xray, flux_xray da analisi_doppler2), e il timestamp
    del picco X-ray.

    Ritorna un dict con:
      frozen_corr, frozen_p_A, frozen_k_A, frozen_N_A, frozen_p95_A,
      frozen_p_B, frozen_k_B, frozen_N_B, frozen_p95_B,
      frozen_verdetto
    oppure None se i dati non bastano.
    """
    if ts_xray is None or not len(ts_xray) or flux_xray is None:
        return None
    if df is None or len(df) < 200:
        return None

    # --- Doppler lungo la giornata ---
    phase = df['Phase_Deg'].values.astype(float).copy()
    if 'Discontinuita' in df.columns:
        phase[df['Discontinuita'].values == 1] = np.nan
    phase = pd.Series(phase).interpolate(
        limit_direction='both').bfill().ffill().values
    doppler = np.gradient(phase, FS) / 360.0

    # --- Pre-processing del Doppler (2026-10-09) ---
    # 1) Filtro Hampel: rimuove glitch impulsivi (1-2 campioni) senza
    #    toccare transienti reali (>7 campioni).
    # 2) Sanity con durata: mette a NaN i campioni oltre 20 mHz che
    #    durano < 60 s. Preserva i transienti fisici lunghi (X-class).
    doppler = filtro_hampel(doppler, window=7, n_sigma=3.0)
    doppler = sanity_doppler_con_durata(doppler, soglia_hz=0.020,
                                         min_durata_s=60)

    doppler_bp = _passa_banda(doppler, DOPPLER_BANDPASS_MIN)

    # --- X-ray ricampionato a 1 Hz sulla griglia del tracker ---
    t_trk = df['Timestamp'].values.astype('datetime64[s]').astype(np.int64)
    t_x = np.array([t.timestamp() for t in ts_xray], dtype=np.int64)
    fx = np.array(flux_xray, dtype=float)

    # Se il tracker ha gap, la griglia non è uniforme: interpolo su
    # t_trk che è la griglia reale del tracker.
    flux_i = np.interp(t_trk, t_x, fx)
    mask_flux = flux_i > XRAY_FLUX_MIN
    log_flux = np.log10(np.clip(flux_i, 1e-12, None))
    log_sm = _smooth_moving(log_flux, XRAY_SMOOTH_MIN * 60)
    dlog = np.gradient(log_sm, FS)
    mask_xray = mask_flux & (np.abs(dlog) < XRAY_DLOG_DT_MAX)

    # --- Localizza il picco nella griglia del tracker ---
    t_pk_int = int(pd.Timestamp(t_pk).timestamp())
    idx_pk = int(np.argmin(np.abs(t_trk - t_pk_int)))

    n_half = int(FINESTRA_MIN * 60)
    if idx_pk < n_half or idx_pk >= len(dlog) - n_half:
        return None

    idx_win = np.arange(idx_pk - n_half, idx_pk + n_half + 1)
    dop_win = doppler_bp[idx_win]
    dlog_win = dlog[idx_win]
    mask_win = mask_xray[idx_win]

    if mask_win.mean() < 0.5:
        return None

    corr_oss = _corr_mascherata(dop_win, dlog_win, mask_win)
    if not np.isfinite(corr_oss):
        return None

    null_A = _null_A(dop_win, dlog, mask_xray, idx_pk, n_half)
    p_A, k_A, N_A = _p_empirico(corr_oss, null_A)
    p95_A = float(np.percentile(np.abs(null_A), 95)) if N_A > 0 else np.nan

    null_B = _null_B(doppler_bp, dlog_win, mask_win, idx_pk, n_half)
    p_B, k_B, N_B = _p_empirico(corr_oss, null_B)
    p95_B = float(np.percentile(np.abs(null_B), 95)) if N_B > 0 else np.nan

    if p_A < SOGLIA_P_NULL_A and p_B < SOGLIA_P_NULL_B:
        verdetto = VERDETTO_SIG
    elif p_A < SOGLIA_P_NULL_A or p_B < SOGLIA_P_NULL_B:
        verdetto = VERDETTO_PARZIALE
    else:
        verdetto = VERDETTO_NS

    return {
        'frozen_corr':    corr_oss,
        'frozen_p_A':     p_A,
        'frozen_k_A':     k_A,
        'frozen_N_A':     N_A,
        'frozen_p95_A':   p95_A,
        'frozen_p_B':     p_B,
        'frozen_k_B':     k_B,
        'frozen_N_B':     N_B,
        'frozen_p95_B':   p95_B,
        'frozen_verdetto': verdetto,
    }