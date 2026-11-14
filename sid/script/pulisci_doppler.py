# ============================================================
# pulisci_doppler.py  (v3 — interattivo)
#
# Confronta 5 strategie di pulizia del Doppler fase, per evidenziare
# la "V" dell'SFE che è sepolta nel rumore ad alta frequenza.
#
# 1. Baseline (median 30s, come analisi_doppler2)
# 2. + Hampel 7pt
# 3. + Low-pass 5 min
# 4. + Bandpass 3-60 min
# 5. PIPELINE FROZEN UFFICIALE:
#    Hampel 7pt -> sanity(20 mHz / 60 s) -> BP 5-60 min (reflect)
#    Niente median 30s (non fa parte della pipeline di frozen_method.py).
#    E' il riferimento per validare frozen_method: se il pannello 5
#    mostra una V, e' la V che vede il classificatore ufficiale.
#
# Tasti:
#   [I] cambia data (calendario)
#   [A] flare precedente
#   [D] flare successivo
#   [Q] chiudi
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

# Override keymap matplotlib (indipendente da analisi_doppler2).
# Necessario perché 'f' e 'p' sono usati come tasti applicativi.
# Prima questi override venivano da analisi_doppler2 a import-time
# (fragile: se qualcuno li sposta sotto __main__, i tasti qui
# smettono di funzionare silenziosamente).
plt.rcParams['keymap.fullscreen'] = []   # libera 'f'
plt.rcParams['keymap.save']       = []   # libera 's'
plt.rcParams['keymap.grid']       = []   # libera 'g'
plt.rcParams['keymap.xscale']     = []   # libera 'k'
plt.rcParams['keymap.yscale']     = []   # libera 'l'
plt.rcParams['keymap.pan']        = []   # libera 'p'
plt.rcParams['keymap.zoom']       = []   # libera 'o'

from analisi_doppler2 import (
    scegli_data_da_calendario,
    scarica_xray,
    scarica_flares,
    rileva_flare_da_xray,
    parse_flare_datetime,
)
from dsp_utils import filtro_hampel, sanity_doppler_con_durata

CARTELLE_CSV = [
    r"D:\Gaetano\SID\dati\Sdruno",
    r"D:\Gaetano\SID\dati\Sdruno\Storico_NSY",
]
CARTELLA_XRAY = r"D:\Gaetano\SID\pysdrplay\Dati\Storico_NSY\XRAY"

FINESTRA_MIN      = 90.0
FINESTRA_ZOOM_MIN = (-60.0, 90.0)
SMOOTH_FASE_SEC   = 30

# Filtro per i flare mostrati nel visualizzatore
CLASSI_UTILI = ('M', 'X', 'C')   # classi GOES da includere
SOGLIA_FLUSSO_MIN = 1e-7         # B1.0: sotto questa soglia non è un flare
FLUSSO_MIN_UFFICIALE = 1e-7      # idem per gli ufficiali
TOLLERANZA_DUP_SEC = 900         # 15 min: due picchi entro 15 min sono lo stesso flare

# ============================================================
# STATO GLOBALE
# ============================================================
_STATO = {
    'fig': None,
    'data': None,
    'idx_flare': 0,
    'flares': [],
    'ts_xray': None,
    'flux_xray': None,
    'df': None,
    'mask_disc': None,
    'doppler': None,   # tupla (dop_1..dop_5)
}


# ============================================================
# I/O E UTILITY
# ============================================================
def trova_csv(date_str):
    for c in CARTELLE_CSV:
        p = os.path.join(c, f"GAESID_NSY_{date_str}.csv")
        if os.path.exists(p):
            return p
    return None


def carica_xray(data_str):
    """
    Carica X-ray via scarica_xray (analisi_doppler2), che gestisce
    nell'ordine:
      1. cache locale in CARTELLA_XRAY
      2. download online da SWPC (se data <= 7 giorni fa)
      3. fallback su cache esistente se il download fallisce

    Restituisce (ts_series, flux_array) o (None, None) se non
    disponibile.
    """
    try:
        ts_list, flux_list = scarica_xray(data_str)
    except Exception as e:
        print(f"[XRAY] errore scarica_xray: {e}")
        return None, None

    if not ts_list:
        print(f"[XRAY] nessun dato disponibile per {data_str} "
              f"(né cache né online)")
        return None, None

    ts_series = pd.Series(pd.to_datetime(ts_list, utc=True))
    flux_arr = np.asarray(flux_list, dtype=float)
    print(f"[XRAY] {len(ts_series)} campioni caricati")
    return ts_series, flux_arr


def classifica_flusso_xray(flux):
    if flux >= 1e-4:
        return f"X{flux/1e-4:.1f}"
    elif flux >= 1e-5:
        return f"M{flux/1e-5:.1f}"
    elif flux >= 1e-6:
        return f"C{flux/1e-6:.1f}"
    elif flux >= 1e-7:
        return f"B{flux/1e-7:.1f}"
    return f"A{flux/1e-8:.1f}"


def ricrea_maschera_disc(df):
    mask_reset = (df['Reset_Flag'].values == 1)
    dt = df['Timestamp'].diff().dt.total_seconds().to_numpy(copy=True)
    dt[0] = 0.0
    mask_gap = (dt > 5.0)
    fase_vals = df['Phase_Deg'].values.astype(float)
    d_fase = np.abs(np.diff(fase_vals, prepend=fase_vals[0]))
    d_fase_mediana = np.median(d_fase[np.isfinite(d_fase)])
    soglia_salto = max(10.0 * d_fase_mediana, 1000.0)
    mask_salto = (d_fase > soglia_salto)
    mask_disc = mask_reset | mask_gap | mask_salto
    WARMUP_S = 600
    idx_disc = np.where(mask_disc)[0]
    for idx in idx_disc:
        end = min(len(mask_disc), idx + WARMUP_S + 1)
        mask_disc[idx:end] = True

    # Warmup iniziale proporzionale (allineato a leggi_csv_tracker
    # di analisi_doppler2). Se il file è corto, un warmup fisso di
    # 600 righe coprirebbe l'intero CSV: riduciamo a 30% del file.
    WARMUP_INIZIALE_S = 600
    n = len(mask_disc)
    if n < WARMUP_INIZIALE_S * 1.5:
        WARMUP_INIZIALE_S = max(int(n * 0.30), 30)
        print(f"[WARN] File corto ({n} righe): "
              f"warmup ridotto a {WARMUP_INIZIALE_S} righe")
    mask_disc[0:min(WARMUP_INIZIALE_S, n)] = True
    return mask_disc


def _filtra_con_pad(x, sos, mode='edge', pad_sec=3600):
    n = len(x)
    if n < 300:
        return x.copy()
    pad = int(min(n // 4, pad_sec))
    xp = np.pad(x, pad, mode=mode)

    mask_nan = ~np.isfinite(xp)
    if mask_nan.any() and (~mask_nan).any():
        idx = np.arange(len(xp))
        xp[mask_nan] = np.interp(idx[mask_nan], idx[~mask_nan], xp[~mask_nan])
    y = sosfiltfilt(sos, xp)[pad:-pad]
    y[~np.isfinite(x)] = np.nan
    return y

def lowpass(x, cutoff_min, fs=1.0, pad_mode='edge', pad_sec=3600):
    if len(x) < 300:
        return x.copy()
    fc_n = (1.0 / (cutoff_min * 60.0)) / (0.5 * fs)
    if fc_n >= 0.99:
        return x.copy()
    sos = butter(2, fc_n, btype='low', output='sos')
    return _filtra_con_pad(x, sos, mode=pad_mode, pad_sec=pad_sec)

def bandpass(x, banda_min, fs=1.0, pad_mode='edge', pad_sec=3600):
    if len(x) < 300:
        return x.copy()
    f_lo = 1.0 / (banda_min[1] * 60.0)
    f_hi = 1.0 / (banda_min[0] * 60.0)
    nyq = 0.5 * fs
    f_lo_n = max(f_lo / nyq, 1e-4)
    f_hi_n = min(f_hi / nyq, 0.99)
    if f_lo_n >= f_hi_n:
        return x.copy()
    sos = butter(2, [f_lo_n, f_hi_n], btype='band', output='sos')
    return _filtra_con_pad(x, sos, mode=pad_mode, pad_sec=pad_sec)

# ============================================================
# CARICAMENTO FLARE
# ============================================================
def carica_flare_del_giorno(data_str, ts_xray, flux_xray):
    """
    Ritorna la lista di flare del giorno.

    Priorità:
      1. Flare ufficiali (SWPC/DONKI). Se presenti, si usa SOLO
         questa fonte: il catalogo ufficiale è autorevole.
      2. Se non ci sono ufficiali, fallback su auto-detect dal
         gradiente X-ray.
      3. Se anche l'auto-detect non trova nulla, usa il picco
         massimo del flusso X-ray come unico flare.

    Questo evita di mescolare decine di micro-eventi auto con
    i pochi eventi reali ufficiali.
    """
    ts_list = [t.to_pydatetime() for t in ts_xray]
    flux_list = [float(f) for f in flux_xray]
    t_arr = np.array([t.timestamp() for t in ts_list])

    # 1. Ufficiali
    try:
        ufficiali = scarica_flares(data_str,
                                    ts_xray=ts_list,
                                    flux_xray=flux_list)
    except Exception as e:
        print(f"[FLARE] errore scarica ufficiali: {e}")
        ufficiali = []

    # Costruisci lista ufficiali (con flusso al picco)
    flares = []
    for u in ufficiali:
        pk = parse_flare_datetime(u.get('picco', ''))
        if pk is None:
            continue
        t_int = int(pk.timestamp())
        idx_flux = int(np.argmin(np.abs(t_arr - t_int)))
        flux_pk = float(flux_list[idx_flux])
        if flux_pk < FLUSSO_MIN_UFFICIALE:
            continue
        flares.append({
            'picco': pk,
            'classe': u.get('classe',
                             classifica_flusso_xray(flux_pk)),
            'flusso': flux_pk,
            'fonte': u.get('_fonte', 'ufficiale'),
        })

    # Deduplica tra ufficiali (a volte lo stesso evento è doppio
    # tra primary e secondary SWPC)
    flares.sort(key=lambda f: f['picco'])
    dedup = []
    for f in flares:
        if dedup and abs((f['picco'] - dedup[-1]['picco']
                           ).total_seconds()) < TOLLERANZA_DUP_SEC:
            # Tieni quello con classe più alta
            classe_ord = {'X': 5, 'M': 4, 'C': 3, 'B': 2, 'A': 1}
            c1 = classe_ord.get(dedup[-1]['classe'][0], 0)
            c2 = classe_ord.get(f['classe'][0], 0)
            if c2 > c1:
                dedup[-1] = f
            continue
        dedup.append(f)
    flares = dedup

    print(f"[FLARE] Ufficiali: {len(ufficiali)} pre-filtro → "
          f"{len(flares)} dopo filtro e deduplica")

    # 2. Se ci sono ufficiali, fermati qui
    if flares:
        for f in flares:
            print(f"  - {f['classe']:>6s} @ {f['picco'].strftime('%H:%M')} "
                  f"({f['flusso']:.2e} W/m², {f['fonte']})")
        return flares

    # 3. Fallback: auto-detect dal gradiente
    print("[FLARE] Nessun ufficiale, provo auto-detect dal X-ray")
    try:
        auto = rileva_flare_da_xray(ts_list, flux_list)
    except Exception as e:
        print(f"[FLARE] errore auto-detect: {e}")
        auto = []

    auto_accettati = []
    for a in auto:
        pk = parse_flare_datetime(a.get('picco', ''))
        if pk is None:
            continue
        t_int = int(pk.timestamp())
        idx_flux = int(np.argmin(np.abs(t_arr - t_int)))
        flux_pk = float(flux_list[idx_flux])
        if flux_pk < SOGLIA_FLUSSO_MIN:
            continue
        # Duplicato?
        dup = False
        for f in auto_accettati:
            if abs((f['picco'] - pk).total_seconds()) < TOLLERANZA_DUP_SEC:
                dup = True
                break
        if dup:
            continue
        auto_accettati.append({
            'picco': pk,
            'classe': a.get('classe', classifica_flusso_xray(flux_pk)),
            'flusso': flux_pk,
            'fonte': 'auto',
        })

    print(f"[FLARE] Auto: {len(auto)} pre-filtro → "
          f"{len(auto_accettati)} dopo filtro e deduplica")

    # 4. Se neanche auto, usa il picco massimo
    if not auto_accettati:
        idx_max = int(np.argmax(flux_list))
        pk = pd.Timestamp(ts_list[idx_max])
        flux_pk = float(flux_list[idx_max])
        auto_accettati.append({
            'picco': pk,
            'classe': classifica_flusso_xray(flux_pk),
            'flusso': flux_pk,
            'fonte': 'max-assoluto',
        })

    auto_accettati.sort(key=lambda f: f['picco'])
    for f in auto_accettati:
        print(f"  - {f['classe']:>6s} @ {f['picco'].strftime('%H:%M')} "
              f"({f['flusso']:.2e} W/m², {f['fonte']})")
    return auto_accettati

# ============================================================
# CALCOLO DOPPLER
# ============================================================
def calcola_doppler(df, mask_disc):
    fase = df['Phase_Deg'].values.astype(float).copy()
    fase[mask_disc] = np.nan
    fase = pd.Series(fase).interpolate(
        limit_direction='both').bfill().ffill().values
    d_fase_hz = np.gradient(fase) / 360.0
    offset = np.nanmedian(d_fase_hz)
    d_fase_hz = d_fase_hz - offset

    # --- Pannelli 1-4: pipeline esplorative (invariate) ---
    dop_1 = median_filter(d_fase_hz, size=SMOOTH_FASE_SEC)
    dop_1 = dop_1.copy(); dop_1[mask_disc] = np.nan

    dop_2 = filtro_hampel(d_fase_hz, window=7, n_sigma=3.0)
    dop_2 = median_filter(dop_2, size=SMOOTH_FASE_SEC)
    dop_2 = dop_2.copy(); dop_2[mask_disc] = np.nan

    dop_3 = lowpass(dop_2, cutoff_min=5.0)
    dop_4 = bandpass(dop_2, banda_min=(3.0, 60.0))

    # --- Pannello 5: FROZEN ufficiale ---
    # Pipeline identica a frozen_method.py:
    #   Hampel (7, 3.0) -> sanity (20 mHz / 60 s) -> BP 5-60 min (reflect)
    # NIENTE median 30 s: non fa parte della pipeline ufficiale.
    # Il padding 'reflect' e' quello usato da frozen_method, e cambia
    # il comportamento del filtro ai bordi della finestra (non e' una
    # scelta puramente grafica).
    dop_5 = filtro_hampel(d_fase_hz, window=7, n_sigma=3.0)
    dop_5 = sanity_doppler_con_durata(
        dop_5, soglia_hz=0.020, min_durata_s=60
    )
    dop_5 = dop_5.copy(); dop_5[mask_disc] = np.nan
    dop_5 = bandpass(dop_5, banda_min=(5.0, 60.0),
                     pad_mode='reflect', pad_sec=1800)

    return (dop_1, dop_2, dop_3, dop_4, dop_5)

# ============================================================
# DISEGNO
# ============================================================
def disegna():
    """Ridisegna la figura per il flare corrente nello stato."""
    fig = _STATO['fig']
    if fig is None:
        return
    df = _STATO['df']
    ts_xray = _STATO['ts_xray']
    flux_xray = _STATO['flux_xray']
    flares = _STATO['flares']
    idx = _STATO['idx_flare']
    data_str = _STATO['data']
    doppler = _STATO['doppler']

    if not flares or doppler is None:
        return
    idx = max(0, min(idx, len(flares) - 1))
    _STATO['idx_flare'] = idx
    flare = flares[idx]
    t_pk = flare['picco']
    classe = flare['classe']
    flux_pk = flare['flusso']

    dop_1, dop_2, dop_3, dop_4, dop_5 = doppler

    # Finestre
    t_a_zoom = t_pk + pd.Timedelta(minutes=FINESTRA_ZOOM_MIN[0])
    t_b_zoom = t_pk + pd.Timedelta(minutes=FINESTRA_ZOOM_MIN[1])
    mask = (df['Timestamp'] >= t_a_zoom) & (df['Timestamp'] <= t_b_zoom)
    t_str = df.loc[mask, 'Timestamp'].values

    ts_xr = pd.to_datetime(ts_xray)
    mask_xr = (ts_xr >= t_a_zoom) & (ts_xr <= t_b_zoom)
    ts_xr_win = ts_xr[mask_xr]
    flux_xr_win = flux_xray[mask_xr]

    # --- Disegno ---
    fig.clf()
    fig.suptitle(
        f'Pulizia Doppler — {data_str}   |   '
        f'Flare {idx + 1}/{len(flares)}: {classe} @ '
        f'{t_pk.strftime("%H:%M:%S")} UTC   |   '
        f'flusso = {flux_pk:.2e} W/m²   |   '
        f'[I] data  [A] prec  [D] succ  [Q] chiudi\n'
        f'Pannelli 1-4: esplorativi (edge)    '
        f'Pannello 5: FROZEN ufficiale (reflect + sanity)',
        fontsize=10, fontweight='bold'
    )
    axes = fig.subplots(6, 1, sharex=True)

    # 0. X-ray con legenda
    if len(ts_xr_win) > 0:
        axes[0].plot(ts_xr_win, flux_xr_win, color='orange',
                     linewidth=1.0,
                     label=f'{classe} @ {t_pk.strftime("%H:%M")} UTC')
        axes[0].set_yscale('log')
    axes[0].axvline(t_pk, color='red', linestyle='--', alpha=0.6)
    axes[0].set_ylabel('Flusso X-ray\n(W/m²)', color='orange')
    axes[0].grid(True, alpha=0.3, which='both')
    axes[0].legend(loc='lower right', fontsize=8, framealpha=0.9)

    def _plot_panel(ax, y_data, colore, ylabel, titolo):
        y = y_data[mask.values] * 1000
        ax.plot(t_str, y, color=colore, linewidth=0.9)
        ax.axvline(t_pk, color='red', linestyle='--', alpha=0.6)
        ax.set_ylabel(ylabel, color=colore)
        ax.grid(True, alpha=0.3)
        ax.set_title(titolo, fontsize=9)

    _plot_panel(axes[1], dop_1, 'gray', 'Baseline\n(median 30s) mHz',
                'Solo median 30s (come analisi_doppler2)')
    _plot_panel(axes[2], dop_2, 'steelblue', '+ Hampel\nmHz',
                'Hampel 7pt + median 30s')
    _plot_panel(axes[3], dop_3, 'green', '+ LP 5 min\nmHz',
                'Hampel + Low-pass 5 min')
    _plot_panel(axes[4], dop_4, 'purple', '+ BP 3-60 min\nmHz',
                'Hampel + Bandpass 3-60 min (pipeline esplorativa)')
    _plot_panel(axes[5], dop_5, 'darkred', 'FROZEN\nufficiale mHz',
                'FROZEN ufficiale: Hampel + sanity(20 mHz/60 s) '
                '+ BP 5-60 min (reflect)')

    axes[5].set_xlabel('Ora UTC')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.canvas.draw_idle()


# ============================================================
# CARICAMENTO GIORNO (chiamato all'avvio e su [I])
# ============================================================
def carica_giorno(data_str, idx_flare=0):
    """Carica CSV, X-ray, flare e Doppler. Riempie _STATO."""
    csv_path = trova_csv(data_str)
    if csv_path is None:
        print(f"[SKIP] CSV non trovato per {data_str}")
        return False

    df = pd.read_csv(csv_path, comment='#')
    df['Timestamp'] = pd.to_datetime(df['Timestamp_UTC'], utc=True)
    df = df.sort_values('Timestamp').reset_index(drop=True)
    print(f"[CSV] {len(df)} righe da {csv_path}")

    ts_xray, flux_xray = carica_xray(data_str)
    if ts_xray is None:
        print("[SKIP] X-ray non disponibile")
        return False

    mask_disc = ricrea_maschera_disc(df)
    doppler = calcola_doppler(df, mask_disc)
    flares = carica_flare_del_giorno(data_str, ts_xray, flux_xray)

    _STATO['data'] = data_str
    _STATO['df'] = df
    _STATO['ts_xray'] = ts_xray
    _STATO['flux_xray'] = flux_xray
    _STATO['mask_disc'] = mask_disc
    _STATO['doppler'] = doppler
    _STATO['flares'] = flares
    _STATO['idx_flare'] = max(0, min(idx_flare, len(flares) - 1))
    return True

# ============================================================
# KEY HANDLER
# ============================================================
def _cambia_flare(delta):
    """Sposta l'indice del flare corrente di `delta` (mod len) e ridisegna."""
    flares = _STATO['flares']
    if not flares:
        print("[GUI] Nessun flare da mostrare.")
        return
    nuovo_idx = (_STATO['idx_flare'] + delta) % len(flares)
    _STATO['idx_flare'] = nuovo_idx
    f = flares[nuovo_idx]
    print(f"[GUI] Flare {nuovo_idx + 1}/{len(flares)}: "
          f"{f['classe']} @ {f['picco'].strftime('%H:%M')} UTC")
    disegna()


# ============================================================
# KEY HANDLER
# ============================================================
def on_key(event):
    if event.key is None:
        return
    key = event.key.lower()

    if key == 'q':
        print("\n[GUI] Chiusura.")
        plt.close('all')
        return

    if key == 'i':
        nuova = scegli_data_da_calendario(data_default=None)
        if nuova is None:
            print("[GUI] Selezione annullata.")
            return
        data_str = nuova.strftime("%Y-%m-%d")
        print(f"\n[GUI] Nuova data: {data_str}")
        if carica_giorno(data_str, idx_flare=0):
            disegna()
        return

    if key == 'd':
        _cambia_flare(+1)
        return

    if key == 'a':
        _cambia_flare(-1)
        return


# ============================================================
# MAIN
# ============================================================
def main():
    # Selezione iniziale: calendario su oggi
    data_scelta = scegli_data_da_calendario(data_default=None)
    if data_scelta is None:
        print("Selezione annullata. Uscita.")
        return
    data_str = data_scelta.strftime("%Y-%m-%d")
    print(f"[GUI] Data selezionata: {data_str}")

    if not carica_giorno(data_str, idx_flare=0):
        return

    # Crea figura e stato
    fig = plt.figure(figsize=(14, 14))
    _STATO['fig'] = fig
    fig.canvas.mpl_connect('key_press_event', on_key)
    try:
        fig.canvas.manager.set_window_title(
            f"Pulizia Doppler — {data_str}  "
            f"[I] data  [A] prec  [D] succ  [Q] chiudi"
        )
    except Exception:
        pass

    disegna()
    plt.show()


if __name__ == "__main__":
    main()