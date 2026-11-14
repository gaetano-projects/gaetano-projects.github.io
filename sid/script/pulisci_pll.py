# ============================================================
# pulisci_pll.py
#
# Visualizzatore interattivo per la frequenza PLL (PLL_Freq_Hz).
# Mostra la risposta del PLL agli eventi solari (flare), filtrata
# con una media mobile (60s) per evidenziare la "V" o lo spike
# senza distruggere il transiente con filtri troppo aggressivi.
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
from scipy.ndimage import uniform_filter1d, median_filter

# Override keymap matplotlib
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
from dsp_utils import filtro_hampel

CARTELLE_CSV = [
    r"D:\Gaetano\SID\dati\Sdruno",
    r"D:\Gaetano\SID\dati\Sdruno\Storico_NSY",
]
CARTELLA_XRAY = r"D:\Gaetano\SID\pysdrplay\Dati\Storico_NSY\XRAY"

FINESTRA_ZOOM_MIN = (-30.0, 60.0) # Finestra intorno al flare (-30 min / +60 min)
SMOOTH_SEC = 60.0                 # Media mobile per pulire il rumore bianco

# Filtri per la selezione dei flare
CLASSI_UTILI = ('M', 'X', 'C')
SOGLIA_FLUSSO_MIN = 1e-7
FLUSSO_MIN_UFFICIALE = 1e-7
TOLLERANZA_DUP_SEC = 900

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
    'pll_smoothed': None,
    'pll_raw': None,
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
    try:
        ts_list, flux_list = scarica_xray(data_str)
    except Exception as e:
        print(f"[XRAY] errore scarica_xray: {e}")
        return None, None

    if not ts_list:
        print(f"[XRAY] nessun dato disponibile per {data_str}")
        return None, None

    ts_series = pd.Series(pd.to_datetime(ts_list, utc=True))
    flux_arr = np.asarray(flux_list, dtype=float)
    return ts_series, flux_arr

def classifica_flusso_xray(flux):
    if flux >= 1e-4: return f"X{flux/1e-4:.1f}"
    elif flux >= 1e-5: return f"M{flux/1e-5:.1f}"
    elif flux >= 1e-6: return f"C{flux/1e-6:.1f}"
    elif flux >= 1e-7: return f"B{flux/1e-7:.1f}"
    return f"A{flux/1e-8:.1f}"

def ricrea_maschera_disc(df):
    mask_reset = (df['Reset_Flag'].values == 1)
    dt = df['Timestamp'].diff().dt.total_seconds().to_numpy(copy=True)
    dt[0] = 0.0
    mask_gap = (dt > 5.0)
    mask_disc = mask_reset | mask_gap
    
    # Warmup
    WARMUP_INIZIALE_S = 600
    n = len(mask_disc)
    if n < WARMUP_INIZIALE_S * 1.5:
        WARMUP_INIZIALE_S = max(int(n * 0.30), 30)
    mask_disc[0:min(WARMUP_INIZIALE_S, n)] = True
    
    # Maschera anche i salti enormi di frequenza (glitch)
    fase_vals = df['PLL_Freq_Hz'].values.astype(float)
    d_fase = np.abs(np.diff(fase_vals, prepend=fase_vals[0]))
    d_fase_mediana = np.median(d_fase[np.isfinite(d_fase)])
    soglia_salto = max(10.0 * d_fase_mediana, 0.5) # 0.5 Hz è un salto enorme
    mask_salto = (d_fase > soglia_salto)
    mask_disc = mask_disc | mask_salto
    
    return mask_disc

def carica_flare_del_giorno(data_str, ts_xray, flux_xray):
    ts_list = [t.to_pydatetime() for t in ts_xray]
    flux_list = [float(f) for f in flux_xray]
    t_arr = np.array([t.timestamp() for t in ts_list])

    try:
        ufficiali = scarica_flares(data_str, ts_xray=ts_list, flux_xray=flux_list)
    except Exception:
        ufficiali = []

    flares = []
    for u in ufficiali:
        pk = parse_flare_datetime(u.get('picco', ''))
        if pk is None: continue
        t_int = int(pk.timestamp())
        idx_flux = int(np.argmin(np.abs(t_arr - t_int)))
        flux_pk = float(flux_list[idx_flux])
        if flux_pk < FLUSSO_MIN_UFFICIALE: continue
        flares.append({'picco': pk, 'classe': u.get('classe', classifica_flusso_xray(flux_pk)), 
                       'flusso': flux_pk, 'fonte': u.get('_fonte', 'ufficiale')})

    flares.sort(key=lambda f: f['picco'])
    dedup = []
    for f in flares:
        if dedup and abs((f['picco'] - dedup[-1]['picco']).total_seconds()) < TOLLERANZA_DUP_SEC:
            classe_ord = {'X': 5, 'M': 4, 'C': 3, 'B': 2, 'A': 1}
            c1 = classe_ord.get(dedup[-1]['classe'][0], 0)
            c2 = classe_ord.get(f['classe'][0], 0)
            if c2 > c1: dedup[-1] = f
            continue
        dedup.append(f)
    flares = dedup

    if flares: return flares

    # Fallback auto-detect
    try:
        auto = rileva_flare_da_xray(ts_list, flux_list)
    except Exception:
        auto = []

    auto_accettati = []
    for a in auto:
        pk = parse_flare_datetime(a.get('picco', ''))
        if pk is None: continue
        t_int = int(pk.timestamp())
        idx_flux = int(np.argmin(np.abs(t_arr - t_int)))
        flux_pk = float(flux_list[idx_flux])
        if flux_pk < SOGLIA_FLUSSO_MIN: continue
        
        dup = False
        for f in auto_accettati:
            if abs((f['picco'] - pk).total_seconds()) < TOLLERANZA_DUP_SEC:
                dup = True; break
        if not dup:
            auto_accettati.append({'picco': pk, 'classe': a.get('classe', classifica_flusso_xray(flux_pk)),
                                   'flusso': flux_pk, 'fonte': 'auto'})

    if not auto_accettati:
        idx_max = int(np.argmax(flux_list))
        pk = pd.Timestamp(ts_list[idx_max])
        flux_pk = float(flux_list[idx_max])
        auto_accettati.append({'picco': pk, 'classe': classifica_flusso_xray(flux_pk),
                               'flusso': flux_pk, 'fonte': 'max-assoluto'})

    auto_accettati.sort(key=lambda f: f['picco'])
    return auto_accettati

# ============================================================
# CALCOLO PLL
# ============================================================
def calcola_pll(df, mask_disc):
    # Estraiamo la colonna PLL_Freq_Hz (colonna indice 3)
    pll_hz = df['PLL_Freq_Hz'].values.astype(float).copy()
    pll_hz[mask_disc] = np.nan
    
    # Interpoliamo i buchi
    pll_hz = pd.Series(pll_hz).interpolate(limit_direction='both').bfill().ffill().values
    
    # Convertiamo in mHz e togliamo l'offset mediano
    pll_mhz = (pll_hz - np.nanmedian(pll_hz)) * 1000.0
    
    # Applicazione Hampel per glitch estremi
    pll_hampel = filtro_hampel(pll_mhz, window=7, n_sigma=3.0)
    pll_hampel = np.where(mask_disc, np.nan, pll_hampel)
    
    # Smoothing (Media mobile) per evidenziare il transiente
    # Usiamo uniform_filter1d che è una media mobile semplice
    fs = 1.0 # Assumiamo 1 Hz (1 campione al secondo)
    n_samples = int(SMOOTH_SEC * fs)
    pll_smooth = uniform_filter1d(np.nan_to_num(pll_hampel), size=n_samples, mode='nearest')
    pll_smooth = np.where(mask_disc, np.nan, pll_smooth)
    
    return pll_mhz, pll_smooth

# ============================================================
# DISEGNO
# ============================================================
def disegna():
    fig = _STATO['fig']
    if fig is None: return
    df = _STATO['df']
    ts_xray = _STATO['ts_xray']
    flux_xray = _STATO['flux_xray']
    flares = _STATO['flares']
    idx = _STATO['idx_flare']
    data_str = _STATO['data']
    pll_mhz = _STATO['pll_raw']
    pll_smooth = _STATO['pll_smoothed']

    if not flares or pll_mhz is None: return
    
    idx = max(0, min(idx, len(flares) - 1))
    _STATO['idx_flare'] = idx
    flare = flares[idx]
    t_pk = flare['picco']
    classe = flare['classe']
    flux_pk = flare['flusso']

    # Finestra Zoom
    t_a_zoom = t_pk + pd.Timedelta(minutes=FINESTRA_ZOOM_MIN[0])
    t_b_zoom = t_pk + pd.Timedelta(minutes=FINESTRA_ZOOM_MIN[1])
    
    mask = (df['Timestamp'] >= t_a_zoom) & (df['Timestamp'] <= t_b_zoom)
    t_str = df.loc[mask, 'Timestamp'].values

    ts_xr = pd.to_datetime(ts_xray)
    mask_xr = (ts_xr >= t_a_zoom) & (ts_xr <= t_b_zoom)
    ts_xr_win = ts_xr[mask_xr]
    flux_xr_win = flux_xray[mask_xr]

    fig.clf()
    fig.suptitle(
        f'Pulizia PLL — {data_str}   |   '
        f'Flare {idx + 1}/{len(flares)}: {classe} @ '
        f'{t_pk.strftime("%H:%M:%S")} UTC   |   '
        f'[I] data  [A] prec  [D] succ  [Q] chiudi\n'
        f'Pannello 2: PLL Smooth (60s)    '
        f'Pannello 3: Zoom sul transiente',
        fontsize=10, fontweight='bold'
    )
    axes = fig.subplots(3, 1, sharex=True)

    # 0. X-ray
    if len(ts_xr_win) > 0:
        axes[0].plot(ts_xr_win, flux_xr_win, color='orange', linewidth=1.0,
                     label=f'{classe} @ {t_pk.strftime("%H:%M")} UTC')
        axes[0].set_yscale('log')
    axes[0].axvline(t_pk, color='red', linestyle='--', alpha=0.6)
    axes[0].set_ylabel('Flusso X-ray\n(W/m²)', color='orange')
    axes[0].grid(True, alpha=0.3, which='both')
    axes[0].legend(loc='lower right', fontsize=8)

    # 1. PLL Intera Finestra
    y_raw = pll_mhz[mask.values]
    y_smooth = pll_smooth[mask.values]
    
    axes[1].plot(t_str, y_raw, color='purple', alpha=0.2, label='PLL Raw (mHz)')
    axes[1].plot(t_str, y_smooth, color='darkviolet', linewidth=1.5, label=f'PLL Smooth ({int(SMOOTH_SEC)}s)')
    axes[1].axvline(t_pk, color='red', linestyle='--', alpha=0.6)
    axes[1].set_ylabel('PLL Freq (mHz)', color='purple')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=8)
    
    # 2. Zoom
    zoom_start = t_pk - pd.Timedelta(minutes=20)
    zoom_end = t_pk + pd.Timedelta(minutes=30)
    mask_zoom = (df['Timestamp'] >= zoom_start) & (df['Timestamp'] <= zoom_end)
    
    t_zoom = df.loc[mask_zoom, 'Timestamp'].values
    y_raw_zoom = pll_mhz[mask_zoom.values]
    y_smooth_zoom = pll_smooth[mask_zoom.values]
    
    axes[2].plot(t_zoom, y_raw_zoom, color='purple', alpha=0.2)
    axes[2].plot(t_zoom, y_smooth_zoom, color='darkviolet', linewidth=2)
    axes[2].axvline(t_pk, color='red', linestyle='--', alpha=0.6)
    axes[2].axhline(0, color='gray', linestyle='-', alpha=0.5)
    axes[2].set_ylabel('PLL Zoom (mHz)', color='purple')
    axes[2].set_xlabel('Ora UTC')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title(f'Zoom sull\'evento ({classe})', fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.canvas.draw_idle()

# ============================================================
# CARICAMENTO
# ============================================================
def carica_giorno(data_str, idx_flare=0):
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
    pll_raw, pll_smoothed = calcola_pll(df, mask_disc)
    flares = carica_flare_del_giorno(data_str, ts_xray, flux_xray)

    _STATO['data'] = data_str
    _STATO['df'] = df
    _STATO['ts_xray'] = ts_xray
    _STATO['flux_xray'] = flux_xray
    _STATO['pll_raw'] = pll_raw
    _STATO['pll_smoothed'] = pll_smoothed
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
    if event.key is None: return
    key = event.key.lower()

    if key == 'q':
        print("\n[GUI] Chiusura.")
        plt.close('all')
        return

    if key == 'i':
        nuova = scegli_data_da_calendario(data_default=None)
        if nuova is None: return
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
    data_scelta = scegli_data_da_calendario(data_default=None)
    if data_scelta is None:
        print("Selezione annullata. Uscita.")
        return
    data_str = data_scelta.strftime("%Y-%m-%d")
    print(f"[GUI] Data selezionata: {data_str}")

    if not carica_giorno(data_str, idx_flare=0):
        return

    fig = plt.figure(figsize=(14, 10))
    _STATO['fig'] = fig
    fig.canvas.mpl_connect('key_press_event', on_key)
    try:
        fig.canvas.manager.set_window_title(
            f"Pulizia PLL — {data_str}  "
            f"[I] data  [A] prec  [D] succ  [Q] chiudi"
        )
    except Exception:
        pass

    disegna()
    plt.show()

if __name__ == "__main__":
    main()