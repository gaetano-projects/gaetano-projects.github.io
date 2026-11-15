import sid_ai
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime, timezone, timedelta
import re
import os
import json
import ephem
from scipy.signal import savgol_filter, find_peaks, medfilt
import tkinter as tk
from tkinter import ttk, simpledialog
from tkcalendar import Calendar
import tkinter.simpledialog as tksd

# --- Import aggiuntivi per modulo GRB ---
import xml.etree.ElementTree as ET
import math

from scipy.ndimage import median_filter, uniform_filter

import pandas as pd
import glob

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['toolbar'] = 'toolbar2'

# =============================================================
# CONFIGURAZIONE
# =============================================================

CARTELLA_GIORNALIERA = r"D:\Gaetano\SID\dati\Sdruno"
CARTELLA_STORICO     = r"D:\Gaetano\SID\dati\Sdruno\Storico_NSY"
CARTELLA_XRAY        = r"D:\Gaetano\SID\dati\Sdruno\Storico_NSY\XRAY"
CARTELLA_FLARES      = r"D:\Gaetano\SID\dati\Sdruno\Storico_NSY\Flares"
FILE_FLARE_MANUALI   = os.path.join(CARTELLA_STORICO, "flare_manuali.json")
CARTELLA_RISPOSTE = r"D:\Gaetano\SID\dati\Sdruno\Storico_NSY\Training_v3"
os.makedirs(CARTELLA_RISPOSTE, exist_ok=True)
CARTELLA_GEOMAG      = os.path.join(CARTELLA_STORICO, "GEOMAG")
os.makedirs(CARTELLA_GEOMAG, exist_ok=True)

COLORI_FLARE = {
    'X': '#FF0000',
    'M': '#FF8C00',
    'C': '#FFD700',
    'B': '#00BFFF',
    'A': '#90EE90',
}
COLORE_LIMB  = '#FF00FF'
SIMBOLO_LIMB = '★ BTL'

SOGLIE_CLASSE_XRAY = {
    'A': (1e-8,  1e-7),
    'B': (1e-7,  1e-6),
    'C': (1e-6,  1e-5),
    'M': (1e-5,  1e-4),
    'X': (1e-4,  1e-2),
}
COLORI_FASCE_XRAY = {
    'A': '#90EE90',
    'B': '#00BFFF',
    'C': '#FFD700',
    'M': '#FF8C00',
    'X': '#FF0000',
}

# ============================================================
# GRB — costanti e stato globale
# ============================================================

_NSY_LAT,    _NSY_LON    =  37.08,  14.44
_GAESID_LAT, _GAESID_LON =  45.07,   7.68
_PATH_MID_LAT = (_NSY_LAT + _GAESID_LAT) / 2
_PATH_MID_LON = (_NSY_LON + _GAESID_LON) / 2

_FLUENZA_ECCEZIONALE = 1.0e-4
_FLUENZA_FORTE       = 1.0e-5
_FLUENZA_DEBOLE      = 1.0e-6
_GRB_ELEV_MIN = -5.0
_ELEV_NOTTE   = -6.0

_GRB_CACHE_DIR = os.path.join(CARTELLA_STORICO, 'grb_cache')

# DONKI: nuovo base URL da ottobre 2026.
# Il vecchio kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/ e api.nasa.gov/DONKI/
# restituiscono ora la pagina HTML di annuncio CCMC.
DONKI_BASE = "https://ccmc.gsfc.nasa.gov/DONKI-API/get"

os.makedirs(_GRB_CACHE_DIR, exist_ok=True)

_grb_all         = []
_grb_ril         = []
_grb_visible     = False
_flare_visible   = True
_grb_artists     = []
_grb_status_text = None

SOGLIE_KP_TEMPESTA = {5: 'G1 (minima)', 6: 'G2 (moderata)',
                      7: 'G3 (forte)', 8: 'G4 (severa)', 9: 'G5 (estrema)'}
modello_ai = None

# X-ray: entro questo numero di giorni dal giorno corrente, riscarica sempre.
# Oltre questa soglia, il file locale è considerato definitivo.
_XRAY_GIORNI_RIFRESH = 2

# =============================================================
# SMOOTHING (usato nel grafico)
# =============================================================
def applica_smoothing(valori, finestra):
    if not valori or len(valori) == 0:
        return valori

    dati = np.array(valori, dtype=float)
    n = len(dati)
    soglia_hampel = 2.5
    finestra_spike = 7

    finestra_spike = min(finestra_spike, n)
    if finestra_spike >= 3 and n > 10:
        meta = finestra_spike // 2
        mediana = median_filter(dati, size=finestra_spike, mode='nearest')
        mean_filtered = uniform_filter(dati, size=finestra_spike, mode='nearest')
        mean_sq_filtered = uniform_filter(dati**2, size=finestra_spike, mode='nearest')
        varianza = mean_sq_filtered - mean_filtered**2
        std_dev = np.sqrt(np.maximum(varianza, 0))
        mask_outlier = np.abs(dati - mediana) > soglia_hampel * std_dev
        mask_outlier[0:meta] = False
        mask_outlier[-meta:] = False

        num_rimossi = np.sum(mask_outlier)
        if num_rimossi > 0:
            print(f"🔧 Hampel: rimossi {num_rimossi} outlier su {n} campioni")

        dati[mask_outlier] = mediana[mask_outlier]

    # Savitzky-Golay
    if finestra > 1:
        finestra_sg = finestra if finestra % 2 == 1 else finestra + 1
        if finestra_sg < 5:
            finestra_sg = 5
        polyordine = min(2, finestra_sg - 1)
        try:
            dati = savgol_filter(dati, window_length=finestra_sg, polyorder=polyordine)
        except Exception:
            kernel = np.ones(finestra_sg) / finestra_sg
            dati = np.convolve(dati, kernel, mode='same')
        meta = finestra_sg // 2
        if len(dati) > meta * 2:
            dati[:meta] = dati[meta]
            dati[-meta:] = dati[-meta - 1]

    return dati.tolist()

# =============================================================
# FLARE MANUALI
# =============================================================
def carica_flare_manuali():
    if not os.path.exists(FILE_FLARE_MANUALI):
        return []
    try:
        with open(FILE_FLARE_MANUALI, 'r', encoding='utf-8') as f:
            dati = json.load(f)
        flares = []
        for e in dati:
            try:
                inizio = datetime.strptime(e['inizio'], "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
                picco  = datetime.strptime(e['picco'],  "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
                fine   = datetime.strptime(e['fine'],   "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
                flares.append({
                    'inizio':  inizio,
                    'picco':   picco,
                    'fine':    fine,
                    'classe':  e['classe'],
                    'tipo':    e['classe'][0].upper(),
                    'limb':    True,
                    'manuale': True,
                    'nota':    e.get('nota', '')
                })
            except Exception as ex:
                print(f"Errore lettura flare manuale: {ex}")
        return flares
    except Exception as e:
        print(f"Errore lettura file flare manuali: {e}")
        return []

def salva_flare_manuale(data_str, classe, ora_inizio, ora_picco, ora_fine, nota=""):
    nuovo = {
        'inizio': f"{data_str}T{ora_inizio}Z",
        'picco':  f"{data_str}T{ora_picco}Z",
        'fine':   f"{data_str}T{ora_fine}Z",
        'classe': classe.upper(),
        'nota':   nota
    }
    dati = []
    if os.path.exists(FILE_FLARE_MANUALI):
        try:
            with open(FILE_FLARE_MANUALI, 'r', encoding='utf-8') as f:
                dati = json.load(f)
        except Exception:
            dati = []
    dati.append(nuovo)
    os.makedirs(os.path.dirname(FILE_FLARE_MANUALI), exist_ok=True)
    with open(FILE_FLARE_MANUALI, 'w', encoding='utf-8') as f:
        json.dump(dati, f, indent=2, ensure_ascii=False)
    print(f"Flare manuale salvato: {classe} il {data_str} picco {ora_picco} UTC")

def trova_csv_per_data(data_utc):
    data_str  = data_utc.strftime("%Y-%m-%d")
    nome_file = f"GAESID_NSY_{data_str}.csv"
    for cartella in [CARTELLA_GIORNALIERA, CARTELLA_STORICO]:
        path = os.path.join(cartella, nome_file)
        if os.path.exists(path):
            return path
    return None

# =============================================================
# X-RAY GOES
# =============================================================
def path_csv_xray(data_utc):
    os.makedirs(CARTELLA_XRAY, exist_ok=True)
    data_str = data_utc.strftime("%Y-%m-%d")
    return os.path.join(CARTELLA_XRAY, f"GOES_XRAY_{data_str}.csv")

def leggi_xray_locale(data_utc):
    path = path_csv_xray(data_utc)
    if not os.path.exists(path):
        return [], []
    timestamps, flux = [], []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for riga in f:
                riga = riga.strip()
                if riga.startswith('#') or not riga:
                    continue
                parti = riga.split(',')
                if len(parti) != 2:
                    continue
                try:
                    dt  = datetime.strptime(parti[0].strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    val = float(parti[1].strip())
                    if val > 0:
                        timestamps.append(dt)
                        flux.append(val)
                except Exception:
                    continue
    except Exception as e:
        print(f"Errore lettura CSV X-ray locale: {e}")
    return timestamps, flux

def salva_xray_locale(data_utc, timestamps, flux):
    path = path_csv_xray(data_utc)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# GOES X-ray flux (0.1-0.8 nm, canale B)\n")
            f.write("# timestamp_utc,flux_wm2\n")
            for dt, val in zip(timestamps, flux):
                f.write(f"{dt.strftime('%Y-%m-%d %H:%M:%S')},{val:.6e}\n")
        print(f"X-ray salvato: {path} ({len(timestamps)} campioni)")
    except Exception as e:
        print(f"Errore salvataggio CSV X-ray: {e}")

def _parse_xray_json_rt(testo, data_utc):
    timestamps, flux = [], []
    try:
        dati = json.loads(testo)
        for campione in dati:
            try:
                energy = campione.get('energy', '')
                if '0.1-0.8' not in energy:
                    continue
                time_tag = campione.get('time_tag', '').replace('T', ' ').replace('Z', '').strip()
                dt  = datetime.strptime(time_tag, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                if dt.date() != data_utc.date():
                    continue
                val = float(campione.get('flux', -1))
                if val > 0:
                    timestamps.append(dt)
                    flux.append(val)
            except Exception:
                continue
    except Exception as e:
        print(f"Errore parsing JSON X-ray real-time: {e}")
    return timestamps, flux

def _donki_str_to_dt(s):
    if not s:
        return None
    s = s.strip().rstrip('Z')
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None

def _ricostruisci_xray_da_flare(flares_donki_raw, data_utc):
    CLASSE_BASE = {'C': 1e-6, 'M': 1e-5, 'X': 1e-4}
    BACKGROUND = 5e-8

    inizio_giorno = datetime.combine(data_utc.date(), datetime.min.time()).replace(tzinfo=timezone.utc)
    n_minuti   = 24 * 60
    ts_griglia = [inizio_giorno + timedelta(minutes=i) for i in range(n_minuti)]
    fl_griglia = [BACKGROUND] * n_minuti

    n_ok = 0
    for flare in flares_donki_raw:
        try:
            classe = (flare.get('classType') or '').strip().upper()
            if not classe or classe[0] not in CLASSE_BASE:
                continue

            tipo = classe[0]
            try:
                numero = float(re.sub(r'[^\d.]', '', classe[1:])) if len(classe) > 1 else 1.0
                if numero <= 0:
                    numero = 1.0
            except ValueError:
                numero = 1.0

            pi = flare.get('peakIntensity')
            if isinstance(pi, dict) and pi.get('value') and pi.get('unit', '') in ('W/m^2', 'W/m²', ''):
                try:
                    flux_picco = float(pi['value'])
                    if flux_picco <= 0:
                        raise ValueError
                except (ValueError, TypeError):
                    flux_picco = CLASSE_BASE[tipo] * numero
            else:
                flux_picco = CLASSE_BASE[tipo] * numero

            t_inizio = _donki_str_to_dt(flare.get('beginTime'))
            t_picco  = _donki_str_to_dt(flare.get('peakTime'))
            t_fine   = _donki_str_to_dt(flare.get('endTime'))

            if t_picco is None:
                continue
            if t_inizio is None:
                t_inizio = t_picco - timedelta(minutes=max(int(numero * 5), 5))
            if t_fine is None:
                t_fine = t_picco + timedelta(minutes=max(int(numero * 15), 10))

            print(f"  Flare {classe}: picco {t_picco.strftime('%H:%M')} UTC, flux={flux_picco:.2e} W/m²")
            n_ok += 1

            durata_rise  = max((t_picco - t_inizio).total_seconds(), 60.0)
            durata_decay = max((t_fine - t_picco).total_seconds(), 60.0)

            for i, t in enumerate(ts_griglia):
                if t < t_inizio or t > t_fine:
                    continue
                if t <= t_picco:
                    frac = (t - t_inizio).total_seconds() / durata_rise
                    val  = BACKGROUND + (flux_picco - BACKGROUND) * frac
                else:
                    frac = (t - t_picco).total_seconds() / durata_decay
                    val  = BACKGROUND + (flux_picco - BACKGROUND) * np.exp(-3.0 * frac)
                fl_griglia[i] = max(fl_griglia[i], val)

        except Exception as ex:
            print(f"  [xray ricostruzione] errore: {ex}")
            continue

    print(f"  Flare utilizzati per la curva: {n_ok}/{len(flares_donki_raw)}")
    return ts_griglia, fl_griglia

def _xray_locale_completo(data_utc, ora_minima=23):
    """
    Verifica se il file X-ray locale copre almeno fino all'ora `ora_minima` UTC
    del giorno `data_utc`. Se il file non esiste o si ferma prima, restituisce False.
    Serve a intercettare file tronchi anche per giorni vecchi (oltre la soglia
    _XRAY_GIORNI_RIFRESH), evitando di considerarli definitivi per errore.
    """
    path = path_csv_xray(data_utc)
    if not os.path.exists(path):
        return False
    try:
        ts, _ = leggi_xray_locale(data_utc)
        if not ts:
            return False
        ultimo = max(ts)
        return ultimo.hour >= ora_minima
    except Exception:
        return False


def _scarica_xray_swpc(data_utc):
    """ Prova a scaricare i dati X-ray real-time da SWPC (NOAA).
    Restituisce (timestamps, flux) per la data richiesta, oppure ([], []) se fallisce.
    Copre solo gli ultimi ~6-7 giorni.
    """
    for sat in ("primary", "secondary"):
        url = f"https://services.swpc.noaa.gov/json/goes/{sat}/xrays-7-day.json"
        print(f"  Download X-ray da: {url}")
        try:
            r = requests.get(url, timeout=(8, 20))
            r.raise_for_status()
            timestamps, flux = _parse_xray_json_rt(r.text, data_utc)
            if timestamps:
                print(f"  X-ray scaricato: {len(timestamps)} campioni (json_rt/{sat})")
                return timestamps, flux
            print(f"  Nessun campione per {data_utc.strftime('%Y-%m-%d')} da {sat}")
        except requests.exceptions.HTTPError as e:
            print(f"  Errore HTTP {e.response.status_code} ({sat}), passo al prossimo...")
        except Exception as e:
            print(f"  Errore ({sat}): {e}")
    return [], []


def _scarica_xray_donki_flare(data_utc):
    """
    Fallback: ricostruisce la curva X-ray a partire dai flare DONKI del giorno.
    Restituisce (timestamps, flux) o ([], []) se non riesce.
    """
    data_str = data_utc.strftime("%Y-%m-%d")

    url_donki = f"{DONKI_BASE}/FLR?startDate={data_str}&endDate={data_str}"
    try:
        r = requests.get(url_donki, timeout=(8, 15))
        r.raise_for_status()
        flares_donki = r.json()
        if not isinstance(flares_donki, list):
            print(f"  DONKI FLR: risposta inattesa ({type(flares_donki).__name__})")
            return [], []
        flares_giorno = [f for f in flares_donki
                         if f.get('peakTime', '').startswith(data_str) or
                            f.get('beginTime', '').startswith(data_str)]
        print(f"  {len(flares_giorno)} flare DONKI trovati per la ricostruzione X-ray")
        timestamps, flux = _ricostruisci_xray_da_flare(flares_giorno, data_utc)
        if timestamps:
            print(f"  Curva X-ray ricostruita: {len(timestamps)} campioni")
            return timestamps, flux
    except Exception as e:
        print(f"  Errore DONKI per ricostruzione X-ray: {e}")
    return [], []

def scarica_xray(data_utc):
    """
    Scarica o carica i dati X-ray GOES per la data richiesta.

    Strategia:
      - Se il giorno è recente (entro _XRAY_GIORNI_RIFRESH giorni dal giorno corrente):
        riscarica SEMPRE da SWPC, ignorando la cache locale, perché il satellite
        potrebbe aver pubblicato nuovi campioni rispetto all'ultimo download.
      - Se il giorno è vecchio (oltre la soglia):
        usa il file locale come definitivo, MA solo se è completo (copre almeno
        fino alle 23:00 UTC). Se è tronco, forza comunque un refresh.
      - Fallback: se SWPC non risponde o non ha i dati, usa DONKI (ricostruzione
        da flare) o l'ultimo file locale disponibile.
    """
    oggi_utc = datetime.now(timezone.utc).date()
    delta    = (oggi_utc - data_utc.date()).days

    # --- Giorno recente: riscarica sempre ---
    if delta <= _XRAY_GIORNI_RIFRESH:
        print(f"X-ray: giorno recente (delta={delta} ≤ {_XRAY_GIORNI_RIFRESH}), riscarico da SWPC...")
        ts_new, fl_new = _scarica_xray_swpc(data_utc)
        if ts_new:
            # Unisci eventualmente con il file locale se questo ha più campioni
            ts_loc, fl_loc = leggi_xray_locale(data_utc)
            if ts_loc and len(ts_loc) > len(ts_new):
                mappa = {t: v for t, v in zip(ts_loc, fl_loc)}
                for t, v in zip(ts_new, fl_new):
                    mappa[t] = v
                ts_new = sorted(mappa.keys())
                fl_new = [mappa[t] for t in ts_new]
                print(f"X-ray: unito con locale → {len(ts_new)} campioni totali")
            salva_xray_locale(data_utc, ts_new, fl_new)
            return ts_new, fl_new
        # Se SWPC non risponde, prova comunque il file locale
        ts_loc, fl_loc = leggi_xray_locale(data_utc)
        if ts_loc:
            print(f"X-ray: SWPC non disponibile, uso file locale ({len(ts_loc)} campioni)")
            return ts_loc, fl_loc
        print(">>> X-ray: nessun dato disponibile (giorno recente)")
        return [], []

    # --- Giorno vecchio: usa cache se completa, altrimenti prova a riscaricare ---
    if _xray_locale_completo(data_utc, ora_minima=23):
        ts, fl = leggi_xray_locale(data_utc)
        if ts:
            print(f"X-ray: file locale completo ({len(ts)} campioni)")
            return ts, fl

    # Cache assente o tronca
    ts_loc, fl_loc = leggi_xray_locale(data_utc)
    if ts_loc:
        print(f"X-ray: file locale tronco ({len(ts_loc)} campioni), provo refresh...")
    else:
        print(f"X-ray: file locale assente, provo a scaricare...")

    # Prova prima il real-time (funziona per gli ultimi 6-7 giorni)
    ts_new, fl_new = _scarica_xray_swpc(data_utc)
    if ts_new:
        # Unisci con il locale se ha più campioni
        if ts_loc and len(ts_loc) > len(ts_new):
            mappa = {t: v for t, v in zip(ts_loc, fl_loc)}
            for t, v in zip(ts_new, fl_new):
                mappa[t] = v
            ts_new = sorted(mappa.keys())
            fl_new = [mappa[t] for t in ts_new]
            print(f"X-ray: unito con locale → {len(ts_new)} campioni totali")
        salva_xray_locale(data_utc, ts_new, fl_new)
        return ts_new, fl_new

    # Fallback: ricostruzione da flare DONKI
    print(f"X-ray: SWPC non disponibile, ricostruzione da flare DONKI...")
    ts_donki, fl_donki = _scarica_xray_donki_flare(data_utc)
    if ts_donki:
        salva_xray_locale(data_utc, ts_donki, fl_donki)
        return ts_donki, fl_donki

    # Ultima spiaggia: quello che c'è nel file locale
    if ts_loc:
        print(f"X-ray: usato file locale come fallback ({len(ts_loc)} campioni)")
        return ts_loc, fl_loc

    print(">>> Nessun dato X-ray disponibile.")
    return [], []

# =============================================================
# FLARE CACHE
# =============================================================
def path_json_flare(data_utc):
    os.makedirs(CARTELLA_FLARES, exist_ok=True)
    data_str = data_utc.strftime("%Y-%m-%d")
    return os.path.join(CARTELLA_FLARES, f"FLARES_{data_str}.json")

def leggi_flare_cache(data_utc):
    oggi_utc = datetime.now(timezone.utc).date()
    if data_utc.date() == oggi_utc:
        return None

    path = path_json_flare(data_utc)
    if not os.path.exists(path):
        return None

    try:
        with open(path, 'r', encoding='utf-8') as f:
            dati = json.load(f)
        flares = []
        for e in dati:
            try:
                e['inizio'] = datetime.strptime(e['inizio'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                e['picco']  = datetime.strptime(e['picco'],  "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                e['fine']   = datetime.strptime(e['fine'],   "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                flares.append(e)
            except Exception:
                continue
        print(f"Flare da cache locale: {len(flares)} eventi ({path})")
        return flares
    except Exception as e:
        print(f"Errore lettura cache flare: {e}")
        return None

def salva_flare_cache(data_utc, flares):
    oggi_utc = datetime.now(timezone.utc).date()
    if data_utc.date() == oggi_utc:
        return

    path = path_json_flare(data_utc)
    try:
        dati = []
        for f in flares:
            if f.get('manuale'):
                continue
            dati.append({
                'inizio': f['inizio'].strftime("%Y-%m-%dT%H:%M:%SZ"),
                'picco':  f['picco'].strftime("%Y-%m-%dT%H:%M:%SZ"),
                'fine':   f['fine'].strftime("%Y-%m-%dT%H:%M:%SZ"),
                'classe': f['classe'],
                'tipo':   f['tipo'],
                'limb':   f.get('limb', False),
            })
        with open(path, 'w', encoding='utf-8') as fout:
            json.dump(dati, fout, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Errore salvataggio cache flare: {e}")

# =============================================================
# DOWNLOAD FLARE
# =============================================================
def scarica_flare(data_utc):
    data_str   = data_utc.strftime("%Y%m%d")
    anno       = data_utc.strftime("%Y")
    oggi_utc   = datetime.now(timezone.utc).date()
    delta_giorni = (oggi_utc - data_utc.date()).days

    url_eventi_swpc = f"https://www.swpc.noaa.gov/archive/{anno}/{data_str}events.txt"

    if delta_giorni <= 30:
        urls = [
            ("json_noaa",  "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json"),
            ("json_noaa",  "https://services.swpc.noaa.gov/json/goes/secondary/xray-flares-7-day.json"),
            ("txt_swpc",   url_eventi_swpc),
        ]
    else:
        data_inizio = data_utc.strftime("%Y-%m-%d")
        data_fine   = data_utc.strftime("%Y-%m-%d")
        urls = [
            ("json_donki", f"{DONKI_BASE}/FLR?startDate={data_inizio}&endDate={data_fine}"),
            ("txt_swpc",   url_eventi_swpc),
        ]

    risultati = []
    for tipo, url in urls:
        try:
            r = requests.get(url, timeout=(5, 8))
            r.raise_for_status()
            risultati.append((r.text, tipo))
            if tipo in ("json_noaa", "json_donki"):
                try:
                    r2 = requests.get(url_eventi_swpc, timeout=(5, 8))
                    r2.raise_for_status()
                    risultati.append((r2.text, "txt_swpc"))
                except Exception:
                    pass
                break
        except Exception:
            pass
    return risultati

def parse_flare_json_noaa(testo, data_utc):
    flares = []
    try:
        dati = json.loads(testo)
        for evento in dati:
            try:
                classe = evento.get('max_class', '')
                if not classe or classe[0] not in 'XMCBA':
                    continue
                if classe[0] in ('A', 'B'):
                    continue
                inizio = datetime.strptime(evento['begin_time'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                picco  = datetime.strptime(evento['max_time'],   "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                fine   = datetime.strptime(evento['end_time'],   "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                if inizio.date() != data_utc.date():
                    continue
                flares.append({'inizio': inizio, 'picco': picco, 'fine': fine, 'classe': classe, 'tipo': classe[0], 'limb': False})
            except Exception:
                continue
    except Exception:
        pass
    return flares

def parse_flare_json_donki(testo, data_utc):
    flares = []
    try:
        dati = json.loads(testo)
        for evento in dati:
            try:
                classe = evento.get('classType', '')
                if not classe or classe[0] not in 'XMCBA':
                    continue
                if classe[0] in ('A', 'B'):
                    continue
                inizio   = datetime.strptime(evento['beginTime'], "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
                picco    = datetime.strptime(evento['peakTime'],  "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
                fine_str = evento.get('endTime')
                fine     = datetime.strptime(fine_str, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc) if fine_str else picco + timedelta(minutes=10)
                if inizio.date() != data_utc.date():
                    continue
                flares.append({'inizio': inizio, 'picco': picco, 'fine': fine, 'classe': classe, 'tipo': classe[0], 'limb': False})
            except Exception:
                continue
    except Exception:
        pass
    return flares

def parse_flare_txt_swpc(testo, data_utc):
    flares   = []
    data_str = data_utc.strftime("%Y-%m-%d")

    def hhmm_to_dt(hhmm):
        return datetime.strptime(f"{data_str} {hhmm[:2]}:{hhmm[2:4]}:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    for riga in testo.splitlines():
        if 'XRA' not in riga or riga.startswith('#'):
            continue
        try:
            parti  = riga.split()
            ora_inizio = parti[2]
            ora_picco  = parti[3]
            ora_fine   = parti[4]

            classe = None
            for p in parti:
                if re.match(r'^[XMCBA]\d+\.\d+$', p) or re.match(r'^[XMCBA]\d+$', p):
                    classe = p
                    break
            if classe is None or classe[0] in ('A', 'B'):
                continue

            is_limb = False
            if len(parti) > 1 and 'P' in parti[1]:
                is_limb = True
            for p in parti:
                m = re.match(r'^[NS]\d+([EW])(\d+)$', p)
                if m:
                    lon = int(m.group(2))
                    if lon > 90:
                        is_limb = True
                    break

            flares.append({'inizio': hhmm_to_dt(ora_inizio), 'picco': hhmm_to_dt(ora_picco),
                           'fine': hhmm_to_dt(ora_fine), 'classe': classe,
                           'tipo': classe[0], 'limb': is_limb})
        except Exception:
            continue
    return flares

def parse_flare_multi(risultati, data_utc):
    flares_json = []
    flares_swpc = []

    for testo, tipo in risultati:
        if tipo == 'json_noaa':
            flares_json.extend(parse_flare_json_noaa(testo, data_utc))
        elif tipo == 'json_donki':
            flares_json.extend(parse_flare_json_donki(testo, data_utc))
        elif tipo == 'txt_swpc':
            flares_swpc.extend(parse_flare_txt_swpc(testo, data_utc))

    flares_extra = []
    for fs in flares_swpc:
        gia_presente = any(abs((fs['picco'] - fj['picco']).total_seconds()) < 300 for fj in flares_json)
        if not gia_presente:
            flares_extra.append(fs)
        elif fs['limb']:
            for fj in flares_json:
                if abs((fs['picco'] - fj['picco']).total_seconds()) < 300:
                    fj['limb'] = True
                    break

    flares_manuali = [f for f in carica_flare_manuali() if f['picco'].date() == data_utc.date()]
    tutti = flares_json + flares_extra + flares_manuali
    tutti.sort(key=lambda f: f['picco'])
    return tutti

# =============================================================
# LETTURA CSV VLF
# =============================================================
def leggi_csv_sid(filepath):
    timestamps, valori = [], []
    with open(filepath, 'r') as f:
        for riga in f:
            riga = riga.strip()
            if riga.startswith('#') or not riga:
                continue
            parti = riga.split(',')
            # Accetta sia 2 che 4 campi (o più)
            if len(parti) < 2:
                continue
            try:
                dt = datetime.strptime(parti[0].strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                val = float(parti[1].strip())
                timestamps.append(dt)
                valori.append(val)
            except Exception:
                continue
    return timestamps, valori

def leggi_csv_sid_con_fase(filepath):
    """
    Come leggi_csv_sid(), ma restituisce anche la fase (3 valori).
    Usata solo dal grafico principale, per non rompere sid_ai.py.
    """
    timestamps, valori, fasi = [], [], []
    with open(filepath, 'r') as f:
        for riga in f:
            riga = riga.strip()
            if riga.startswith('#') or not riga:
                continue
            parti = riga.split(',')
            if len(parti) < 2:
                continue
            try:
                dt = datetime.strptime(parti[0].strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                val = float(parti[1].strip())
                fase = None
                if len(parti) >= 3:
                    try:
                        fase = float(parti[2].strip())
                    except (ValueError, IndexError):
                        fase = None
                timestamps.append(dt)
                valori.append(val)
                fasi.append(fase)
            except Exception:
                continue
    return timestamps, valori, fasi

# =============================================================
# ALBA / TRAMONTO (punto medio del path)
# =============================================================
def calcola_alba_tramonto(data_utc):
    osservatore = ephem.Observer()
    osservatore.lat = '41.1'
    osservatore.lon = '11.1'
    osservatore.elevation = 100
    osservatore.date = data_utc.strftime('%Y/%m/%d 00:00:00')
    osservatore.pressure = 0
    sole = ephem.Sun()
    try:
        alba = osservatore.next_rising(sole).datetime().replace(tzinfo=timezone.utc)
        tramonto = osservatore.next_setting(sole).datetime().replace(tzinfo=timezone.utc)
    except Exception as e:
        print(f"Errore calcolo alba/tramonto: {e}")
        alba, tramonto = None, None
    return alba, tramonto

# =============================================================
# TABELLA SID RILEVATI
# =============================================================
def mostra_tabella_sid(sid_list, data_utc, flares=None):
    import tkinter as tk
    from tkinter import ttk

    if not sid_list:
        root = tk.Tk()
        root.title(f"Nessun SID rilevato - {data_utc.strftime('%Y-%m-%d')}")
        root.attributes('-topmost', True)
        root.geometry("450x200")
        root.configure(bg='#0d0d1a')
        root.resizable(False, False)

        tk.Label(root, text=f"🔍 Nessun SID rilevato per il {data_utc.strftime('%Y-%m-%d')}",
                 bg='#0d0d1a', fg='#FFD700', font=('Consolas', 12, 'bold')).pack(pady=30)
        tk.Label(root, text="Premi H per addestrare l'AI su più giorni",
                 bg='#0d0d1a', fg='#888888', font=('Consolas', 10)).pack()
        tk.Button(root, text="Chiudi", command=root.destroy,
                  bg='#00FF99', fg='black', font=('Consolas', 10, 'bold'),
                  relief='flat', padx=25, pady=6).pack(pady=25)
        root.mainloop()
        return

    sid_con_flare = []
    sid_senza_flare = []

    if flares:
        for sid in sid_list:
            picco = sid['picco']
            flare_associato = None
            for f in flares:
                if f['picco'].tzinfo is None:
                    f['picco'] = f['picco'].replace(tzinfo=timezone.utc)
                diff = abs((picco - f['picco']).total_seconds())
                if diff < 900:
                    flare_associato = f['classe']
                    break
            if flare_associato:
                sid['flare'] = flare_associato
                sid_con_flare.append(sid)
            else:
                sid['flare'] = '—'
                sid_senza_flare.append(sid)
    else:
        for sid in sid_list:
            sid['flare'] = '—'
        sid_con_flare = sid_list

    root = tk.Tk()
    root.title(f"SID rilevati - {data_utc.strftime('%Y-%m-%d')}")
    root.attributes('-topmost', True)
    root.resizable(True, True)
    root.geometry("1050x500")
    root.configure(bg='#f5f5f5')

    style = ttk.Style()
    style.theme_use('default')
    style.configure("Treeview", background="#ffffff", foreground="black",
                    rowheight=25, fieldbackground="#ffffff")
    style.configure("Treeview.Heading", background="#d0d0d0",
                    foreground="black", font=('Consolas', 10, 'bold'))
    style.map('Treeview', background=[('selected', '#347083')])

    frame = ttk.Frame(root)
    frame.pack(fill='both', expand=True, padx=10, pady=10)

    scrollbar = ttk.Scrollbar(frame)
    scrollbar.pack(side='right', fill='y')

    columns = ('Picco UTC', 'Ampiezza (dB)', 'Durata (s)', 'Area (dB·s)',
               'Probabilità AI', 'Flare NOAA', 'Inizio UTC', 'Fine UTC')
    tree = ttk.Treeview(frame, columns=columns, show='headings',
                        yscrollcommand=scrollbar.set, height=20)
    scrollbar.config(command=tree.yview)

    col_widths = {
        'Picco UTC': 110, 'Ampiezza (dB)': 110, 'Durata (s)': 90,
        'Area (dB·s)': 110, 'Probabilità AI': 120,
        'Flare NOAA': 110, 'Inizio UTC': 110, 'Fine UTC': 110
    }
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=col_widths.get(col, 110), anchor='center')

    tree.tag_configure('con_flare', background='#c8e6c9', foreground='#1a3a1a')
    tree.tag_configure('senza_flare', background='#f5e6d3', foreground='#4a2a1a')

    for sid in sid_con_flare:
        prob = sid.get('probabilita', None)
        prob_str = f"{prob:.3f}" if prob is not None else '—'
        tree.insert('', 'end', values=(
            sid['picco'].strftime('%H:%M:%S'),
            f"{sid['ampiezza']:.3f}",
            f"{sid['durata']:.0f}",
            f"{sid['area']:.2f}",
            prob_str,
            sid['flare'],
            sid['inizio'].strftime('%H:%M:%S'),
            sid['fine'].strftime('%H:%M:%S')
        ), tags=('con_flare',))

    for sid in sid_senza_flare:
        prob = sid.get('probabilita', None)
        prob_str = f"{prob:.3f}" if prob is not None else '—'
        tree.insert('', 'end', values=(
            sid['picco'].strftime('%H:%M:%S'),
            f"{sid['ampiezza']:.3f}",
            f"{sid['durata']:.0f}",
            f"{sid['area']:.2f}",
            prob_str,
            sid['flare'],
            sid['inizio'].strftime('%H:%M:%S'),
            sid['fine'].strftime('%H:%M:%S')
        ), tags=('senza_flare',))

    tree.pack(fill='both', expand=True)

    info_label = tk.Label(root,
        text=f"Totale SID: {len(sid_list)}  |  Associati a flare: {len(sid_con_flare)}  |  Non associati: {len(sid_senza_flare)}",
        font=('Consolas', 11, 'bold'), bg='#f5f5f5', fg='#1a3a1a')
    info_label.pack(pady=(6, 4))

    btn_close = tk.Button(root, text="Chiudi", command=root.destroy,
                          bg='#4CAF50', fg='white', font=('Consolas', 10, 'bold'),
                          relief='flat', padx=25, pady=6)
    btn_close.pack(pady=(0, 10))

    root.mainloop()

# =============================================================
# TABELLA FLARE ADDESTRATI (tasto L)
# =============================================================
def mostra_flare_addestrati(cartella_risposte=CARTELLA_RISPOSTE):
    import tkinter as tk
    from tkinter import ttk
    import glob
    import json
    import os
    from datetime import datetime

    print(f"📂 Ricerca risposte in: {cartella_risposte}")
    pattern = os.path.join(cartella_risposte, "risposte_flare_*.json")
    files = glob.glob(pattern)
    files.sort()

    print(f"   Trovati {len(files)} file di risposte")
    for f in files:
        print(f"      - {os.path.basename(f)}")

    if not files:
        modello_path = os.path.join(cartella_risposte, "modello_sid.pkl")
        if os.path.exists(modello_path):
            msg = "Il modello AI esiste ma i file delle risposte non sono stati trovati.\n\n"
            msg += "Potresti aver addestrato con una versione precedente che non salvava le risposte.\n"
            msg += "Premi H per rifare l'addestramento su un giorno con flare."
        else:
            msg = "Nessun file di addestramento trovato.\n\nPremi H per addestrare l'AI."

        root = tk.Tk()
        root.title("Nessun addestramento")
        root.attributes('-topmost', True)
        root.geometry("500x220")
        root.configure(bg='#0d0d1a')
        tk.Label(root, text=msg, bg='#0d0d1a', fg='#FFD700',
                 font=('Consolas', 10), justify='left').pack(pady=20, padx=20)
        tk.Button(root, text="Chiudi", command=root.destroy,
                  bg='#00FF99', fg='black', font=('Consolas', 10, 'bold'),
                  relief='flat', padx=20, pady=5).pack(pady=10)
        root.mainloop()
        return

    flare_set = set()
    flare_list = []

    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                dati = json.load(f)
            data_str = os.path.basename(file_path).replace("risposte_flare_", "").replace(".json", "")
            for r in dati.get('risposte', []):
                picco_iso = r.get('picco')
                classe = r.get('classe', '?')
                etichetta = r.get('etichetta', 0)
                if not picco_iso:
                    continue
                try:
                    if isinstance(picco_iso, str):
                        picco_iso = picco_iso.replace('Z', '+00:00')
                        picco_dt = datetime.fromisoformat(picco_iso)
                    else:
                        picco_dt = picco_iso
                except Exception:
                    continue
                etichetta_str = 'SID' if etichetta == 1 else 'NON SID'
                key = (data_str, picco_dt.strftime('%H:%M:%S'), classe, etichetta_str)
                if key not in flare_set:
                    flare_set.add(key)
                    flare_list.append({
                        'data': data_str,
                        'ora': picco_dt.strftime('%H:%M:%S'),
                        'classe': classe,
                        'etichetta': etichetta_str
                    })
        except Exception as e:
            print(f"Errore leggendo {file_path}: {e}")

    if not flare_list:
        root = tk.Tk()
        root.title("Nessun flare")
        root.attributes('-topmost', True)
        root.geometry("400x150")
        root.configure(bg='#0d0d1a')
        tk.Label(root, text="Nessun flare valido trovato nei file di risposte.",
                 bg='#0d0d1a', fg='white', font=('Consolas', 12)).pack(pady=40)
        tk.Button(root, text="Chiudi", command=root.destroy,
                  bg='#00FF99', fg='black', font=('Consolas', 10),
                  relief='flat', padx=20, pady=5).pack()
        root.mainloop()
        return

    root = tk.Tk()
    root.title(f"Flare addestrati ({len(flare_list)})")
    root.attributes('-topmost', True)
    root.geometry("750x450")
    root.configure(bg='#0d0d1a')

    frame = ttk.Frame(root)
    frame.pack(fill='both', expand=True, padx=10, pady=10)
    scrollbar = ttk.Scrollbar(frame)
    scrollbar.pack(side='right', fill='y')

    columns = ('Data', 'Ora', 'Classe', 'Etichetta')
    tree = ttk.Treeview(frame, columns=columns, show='headings',
                        yscrollcommand=scrollbar.set, height=15)
    scrollbar.config(command=tree.yview)

    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=170, anchor='center')

    tree.tag_configure('SID', background='#c8e6c9', foreground='#1a3a1a')
    tree.tag_configure('NON SID', background='#f5d0c9', foreground='#4a1a1a')

    for f in flare_list:
        tree.insert('', 'end', values=(f['data'], f['ora'], f['classe'], f['etichetta']),
                    tags=(f['etichetta'],))

    tree.pack(fill='both', expand=True)

    def copy_selected(event=None):
        selected = tree.selection()
        if not selected:
            return
        item = selected[0]
        values = tree.item(item, 'values')
        copy_str = "\t".join(values)
        root.clipboard_clear()
        root.clipboard_append(copy_str)
        print("📋 Copiato: " + copy_str)

    root.bind('<Control-c>', copy_selected)
    root.bind('<Control-C>', copy_selected)

    n_sid = sum(1 for f in flare_list if f['etichetta'] == 'SID')
    n_no = sum(1 for f in flare_list if f['etichetta'] == 'NON SID')

    info_frame = tk.Frame(root, bg='#0d0d1a')
    info_frame.pack(fill='x', pady=5)
    info = tk.Label(info_frame,
        text=f"Totale: {len(flare_list)}  |  SID veri: {n_sid}  |  Falsi: {n_no}",
        bg='#0d0d1a', fg='#00FF99', font=('Consolas', 11, 'bold'))
    info.pack(side='left', padx=10)
    hint = tk.Label(info_frame, text="(Ctrl+C per copiare riga selezionata)",
                    bg='#0d0d1a', fg='#888888', font=('Consolas', 9))
    hint.pack(side='right', padx=10)

    btn = tk.Button(root, text="Chiudi", command=root.destroy,
                    bg='#00FF99', fg='black', font=('Consolas', 10, 'bold'),
                    relief='flat', padx=20, pady=5)
    btn.pack(pady=10)
    root.mainloop()

# ============================================================
# SEZIONE GRB — funzioni (solo parti essenziali)
# ============================================================
def _solar_elevation_path(dt_utc):
    doy  = dt_utc.timetuple().tm_yday
    B    = math.radians(360 / 365 * (doy - 81))
    decl = math.radians(23.45 * math.sin(B))
    eot  = 9.87*math.sin(2*B) - 7.53*math.cos(B) - 1.5*math.sin(B)
    lstm = 15 * round(_PATH_MID_LON / 15)
    tc   = 4 * (_PATH_MID_LON - lstm) + eot
    lst  = dt_utc.hour + dt_utc.minute/60 + dt_utc.second/3600 + tc/60
    hra  = math.radians(15 * (lst - 12))
    return math.degrees(math.asin(
        math.sin(math.radians(_PATH_MID_LAT)) * math.sin(decl)
        + math.cos(math.radians(_PATH_MID_LAT)) * math.cos(decl) * math.cos(hra)
    ))

def _grb_altitude(ra_deg, dec_deg, dt_utc):
    y, mo, d = dt_utc.year, dt_utc.month, dt_utc.day
    ut = dt_utc.hour + dt_utc.minute/60 + dt_utc.second/3600
    jd = (367*y - int(7*(y + int((mo+9)/12)) / 4) + int(275*mo/9) + d + 1721013.5 + ut/24)
    T    = (jd - 2451545.0) / 36525.0
    gmst = (280.46061837 + 360.98564736629 * (jd - 2451545.0) + T*T * (0.000387933 - T/38710000.0)) % 360
    lmst = (gmst + _PATH_MID_LON) % 360
    ha   = (lmst - ra_deg) % 360
    if ha > 180:
        ha -= 360
    return math.degrees(math.asin(
        math.sin(math.radians(_PATH_MID_LAT)) * math.sin(math.radians(dec_deg))
        + math.cos(math.radians(_PATH_MID_LAT)) * math.cos(math.radians(dec_deg)) * math.cos(math.radians(ha))
    ))

def _parse_tap_text(text):
    rows = []
    header = None
    for line in text.splitlines():
        line = line.strip()
        if not line or 'Number of rows' in line or 'Number of columns' in line:
            continue
        if '|' not in line:
            continue
        parts = [part.strip() for part in line.split('|')]
        if header is None:
            header = [part.lower() for part in parts]
            continue
        if header and len(parts) == len(header):
            rows.append(dict(zip(header, parts)))
    return rows

def _datetime_to_mjd(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    elif dt.tzinfo != timezone.utc:
        dt = dt.astimezone(timezone.utc)
    ts = dt.timestamp()
    jd = ts / 86400.0 + 2440587.5
    return jd - 2400000.5

def _mjd_to_datetime(mjd):
    jd = mjd + 2400000.5
    ts = (jd - 2440587.5) * 86400.0
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=ts)

def fetch_grb_for_date(date_obj):
    date_str  = date_obj.strftime('%Y-%m-%d')
    date_next = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
    cache_f   = os.path.join(_GRB_CACHE_DIR, f'grb_{date_str}.json')

    if os.path.exists(cache_f):
        with open(cache_f, 'r') as fh:
            raw = json.load(fh)
        for g in raw:
            ts = g.get('trigger_time_str', '')
            for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S',
                        '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
                try:
                    g['trigger_time'] = datetime.strptime(ts, fmt)
                    break
                except ValueError:
                    pass
        return raw

    mjd_start = _datetime_to_mjd(datetime.strptime(date_str + 'T00:00:00', '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc))
    mjd_end = _datetime_to_mjd(datetime.strptime(date_next + 'T00:00:00', '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc))
    query = (
        "SELECT name, trigger_time, t90, fluence, ra, dec "
        "FROM fermigbrst "
        f"WHERE trigger_time >= {mjd_start:.8f} "
        f"AND trigger_time < {mjd_end:.8f}"
    )
    try:
        resp = requests.get("https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync",
                            params={'REQUEST': 'doQuery', 'LANG': 'ADQL', 'QUERY': query, 'FORMAT': 'text'}, timeout=20)
        resp.raise_for_status()
        text = resp.text
        rows = _parse_tap_text(text)
    except Exception as exc:
        print(f"[GRB] Errore query HEASARC: {exc}")
        return []

    result = []
    for r in rows:
        try:
            t_raw = (r.get('trigger_time') or r.get('time') or '').strip()
            if not t_raw:
                continue
            trig = None
            for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S',
                        '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
                try:
                    trig = datetime.strptime(t_raw, fmt).replace(tzinfo=timezone.utc)
                    break
                except ValueError:
                    pass
            if trig is None:
                try:
                    mjd = float(t_raw)
                    trig = _mjd_to_datetime(mjd)
                except ValueError:
                    continue

            flu_raw = next((r[k] for k in ('fluence', 'fluence_1024', 'fluence_band', 'total_fluence')
                            if r.get(k) and r[k].strip() not in ('', 'null', 'NULL')), None)
            if flu_raw is None:
                continue
            fluence = float(flu_raw)
            if fluence <= 0:
                continue

            result.append({
                'name':             (r.get('name') or 'GRB?').strip(),
                'trigger_time':     trig,
                'trigger_time_str': trig.strftime('%Y-%m-%dT%H:%M:%S'),
                't90':              float(r.get('t90') or 0),
                'fluence':          fluence,
                'ra':               float(r.get('ra') or r.get('ra_obj') or 0),
                'dec':              float(r.get('dec') or r.get('dec_obj') or 0),
            })
        except (ValueError, TypeError):
            continue

    if result:
        cache_data = [{k: v for k, v in g.items() if k != 'trigger_time'} for g in result]
        with open(cache_f, 'w') as fh:
            json.dump(cache_data, fh, indent=2)

    return result

def fetch_grb_for_year(year):
    oggi_dt = datetime.now(timezone.utc)
    anno_corrente = oggi_dt.year
    oggi = oggi_dt.date()
    inizio = datetime(year, 1, 1, tzinfo=timezone.utc)
    fine = datetime(year + 1, 1, 1, tzinfo=timezone.utc) if year < anno_corrente else oggi_dt + timedelta(seconds=1)
    data_inizio = inizio.strftime('%Y-%m-%dT%H:%M:%S')
    data_fine = fine.strftime('%Y-%m-%dT%H:%M:%S')
    cache_f = os.path.join(_GRB_CACHE_DIR, f'grb_year_{year}.json')

    if os.path.exists(cache_f):
        if year == anno_corrente:
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(cache_f)).date()
                if oggi > mtime:
                    print(f"[GRB] Cache {os.path.basename(cache_f)} datata {mtime}, rigenero...")
                    os.remove(cache_f)
                else:
                    with open(cache_f, 'r') as fh:
                        raw = json.load(fh)
                    if raw:
                        for g in raw:
                            ts = g.get('trigger_time_str', '')
                            for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S',
                                        '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
                                try:
                                    g['trigger_time'] = datetime.strptime(ts, fmt)
                                    break
                                except ValueError:
                                    pass
                        return raw
                    print(f"[GRB] Cache annuale vuota per il {year}, rifetching da HEASARC...")
            except:
                pass
        else:
            with open(cache_f, 'r') as fh:
                raw = json.load(fh)
            if raw:
                for g in raw:
                    ts = g.get('trigger_time_str', '')
                    for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S',
                                '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
                        try:
                            g['trigger_time'] = datetime.strptime(ts, fmt)
                            break
                        except ValueError:
                            pass
                return raw
            print(f"[GRB] Cache annuale vuota per il {year}, rifetching da HEASARC...")

    mjd_start = _datetime_to_mjd(inizio)
    mjd_end = _datetime_to_mjd(fine)
    query = (
        "SELECT name, trigger_time, t90, fluence, ra, dec "
        "FROM fermigbrst "
        f"WHERE trigger_time >= {mjd_start:.8f} "
        f"AND trigger_time < {mjd_end:.8f}"
    )
    try:
        resp = requests.get("https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync",
                            params={'REQUEST': 'doQuery', 'LANG': 'ADQL', 'QUERY': query, 'FORMAT': 'text'}, timeout=30)
        resp.raise_for_status()
        text = resp.text
        rows = _parse_tap_text(text)
    except Exception as exc:
        print(f"[GRB] Errore query annuale HEASARC: {exc}")
        return []

    result = []
    for r in rows:
        try:
            t_raw = (r.get('trigger_time') or r.get('time') or '').strip()
            if not t_raw:
                continue
            trig = None
            for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S',
                        '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
                try:
                    trig = datetime.strptime(t_raw, fmt).replace(tzinfo=timezone.utc)
                    break
                except ValueError:
                    pass
            if trig is None:
                try:
                    mjd = float(t_raw)
                    trig = _mjd_to_datetime(mjd)
                except ValueError:
                    continue

            flu_raw = next((r[k] for k in ('fluence', 'fluence_1024', 'fluence_band', 'total_fluence')
                            if r.get(k) and r[k].strip() not in ('', 'null', 'NULL')), None)
            if flu_raw is None:
                continue
            fluence = float(flu_raw)
            if fluence <= 0:
                continue

            result.append({
                'name':             (r.get('name') or 'GRB?').strip(),
                'trigger_time':     trig,
                'trigger_time_str': trig.strftime('%Y-%m-%dT%H:%M:%S'),
                't90':              float(r.get('t90') or 0),
                'fluence':          fluence,
                'ra':               float(r.get('ra') or r.get('ra_obj') or 0),
                'dec':              float(r.get('dec') or r.get('dec_obj') or 0),
            })
        except (ValueError, TypeError):
            continue

    if result:
        cache_data = [{k: v for k, v in g.items() if k != 'trigger_time'} for g in result]
        with open(cache_f, 'w') as fh:
            json.dump(cache_data, fh, indent=2)

    return result

def _classifica_grb(grb):
    fluence = grb['fluence']
    trig    = grb['trigger_time']
    try:
        alt = _grb_altitude(grb['ra'], grb['dec'], trig)
    except Exception:
        alt = 0.0
    if alt < _GRB_ELEV_MIN:
        return None, None
    if fluence >= _FLUENZA_ECCEZIONALE:
        return 'ECCEZIONALE', 'red'
    return None, None

def _filtra_grb_rilevabili(grb_list):
    ril = []
    for g in grb_list:
        cat, col = _classifica_grb(g)
        if cat:
            g2 = dict(g)
            g2['categoria'] = cat
            g2['colore']    = col
            ril.append(g2)
    return ril

def mostra_elenco_grb_per_anno(anno):
    grb_anno = fetch_grb_for_year(anno)
    grb_visibili = []
    for g in grb_anno:
        cat, col = _classifica_grb(g)
        if cat:
            g2 = dict(g)
            g2['categoria'] = cat
            g2['colore'] = col
            grb_visibili.append(g2)
    grb_visibili.sort(key=lambda g: g['trigger_time'])

    if not grb_visibili:
        print(f"[GRB] Nessun GRB potenzialmente visibile nel {anno}.")
        return

    linee = [f"GRB potenzialmente visibili nel {anno}: {len(grb_visibili)} eventi"]
    linee.append(
        f"{'Nome':<15s} | {'Trigger UTC':<19s} | {'Categoria':<12s} | "
        f"{'T90 (s)':>6s} | {'Fluence (erg/cm²)':>10s} | {'RA':>6s} | {'DEC':>6s}"
    )
    linee.append("-" * 111)
    for g in grb_visibili:
        linee.append(
            f"{g['name'][:15]:<15s} | {g['trigger_time'].strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{g['categoria'][:12]:<12s} | {g['t90']:6.1f} | {g['fluence']:10.2e} | "
            f"{g['ra']:6.1f} | {g['dec']:6.1f}"
        )

    try:
        root = tk.Tk()
        root.title(f"GRB visibili {anno}")
        root.attributes('-topmost', True)
        root.resizable(True, True)
        text = tk.Text(root, wrap='none', bg='#121212', fg='white',
                       insertbackground='white', font=('Consolas', 10))
        text.pack(fill='both', expand=True)
        text.insert('1.0', '\n'.join(linee))
        text.config(state='disabled')
        yscroll = tk.Scrollbar(root, orient='vertical', command=text.yview)
        yscroll.pack(side='right', fill='y')
        text.config(yscrollcommand=yscroll.set)
        xscroll = tk.Scrollbar(root, orient='horizontal', command=text.xview)
        xscroll.pack(side='bottom', fill='x')
        text.config(xscrollcommand=xscroll.set)
        root.mainloop()
    except Exception as e:
        print(f"[GRB] Errore visualizzazione elenco: {e}")

def mostra_elenco_tutti_grb_per_anno(anno):
    grb_anno = fetch_grb_for_year(anno)
    if not grb_anno:
        print(f"[GRB] Nessun GRB trovato per il {anno}.")
        return
    grb_anno.sort(key=lambda g: g['trigger_time'])
    linee = [f"Tutti i GRB nel {anno}: {len(grb_anno)} eventi"]
    linee.append("\nNome | Trigger UTC | T90 (s) | Fluence (erg/cm²) | RA | DEC")
    linee.append("-" * 96)
    for g in grb_anno:
        linee.append(
            f"{g['name'][:15]:15s} | {g['trigger_time'].strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{g['t90']:6.1f} | {g['fluence']:10.2e} | {g['ra']:6.1f} | {g['dec']:6.1f}"
        )
    try:
        root = tk.Tk()
        root.title(f"Tutti i GRB {anno}")
        root.attributes('-topmost', True)
        root.resizable(True, True)
        text = tk.Text(root, wrap='none', bg='#121212', fg='white',
                       insertbackground='white', font=('Consolas', 10))
        text.pack(fill='both', expand=True)
        text.insert('1.0', '\n'.join(linee))
        text.config(state='disabled')
        yscroll = tk.Scrollbar(root, orient='vertical', command=text.yview)
        yscroll.pack(side='right', fill='y')
        text.config(yscrollcommand=yscroll.set)
        xscroll = tk.Scrollbar(root, orient='horizontal', command=text.xview)
        xscroll.pack(side='bottom', fill='x')
        text.config(xscrollcommand=xscroll.set)
        root.mainloop()
    except Exception as e:
        print(f"[GRB] Errore visualizzazione elenco: {e}")

# ============================================================
# SEZIONE GEOMAGNETISMO (semplificata: solo funzione principale per il pannello)
# ============================================================
def scarica_dati_geomagnetici(data_utc):
    cache_path = os.path.join(CARTELLA_GEOMAG, f"geomag_{data_utc.strftime('%Y-%m-%d')}.json")

    if os.path.exists(cache_path):
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
            eta = (datetime.now() - mtime).total_seconds() / 3600
            if eta < 2:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            else:
                print(f"   ⏰ Cache più vecchia di {eta:.1f} ore, ricarico da DONKI...")
                os.remove(cache_path)
        except:
            pass

    info = {
        'kp_max': None, 'kp_avg': None, 'tempesta': None,
        'proton_flux_max': None, 'proton_event': False,
        'solar_wind_speed_max': None, 'solar_wind_speed_avg': None,
        'cme_analyses': [], 'gst_events': [],
        'impatto': False, 'livello': 'normale', 'note': []
    }

    # Controlla cache annuale
    anno = data_utc.year
    cache_annuale = os.path.join(CARTELLA_GEOMAG, f"geomag_year_{anno}.json")
    if os.path.exists(cache_annuale):
        try:
            with open(cache_annuale, 'r') as f:
                events_raw = json.load(f)
            events = _deserializza_eventi_geomag(events_raw)
            cme_speeds = {}
            for ev in events:
                if ev.get('type') == 'CME':
                    cme_id = ev.get('id', '')
                    velocita = ev.get('velocita_km_s', 0)
                    if cme_id and velocita > 0:
                        cme_speeds[cme_id] = velocita
            for ev in events:
                if ev.get('type') != 'GST':
                    continue
                ev_date = ev.get('date')
                if not isinstance(ev_date, datetime):
                    continue
                if ev_date.date() == data_utc.date():
                    info['kp_max'] = ev.get('kp_max')
                    info['tempesta'] = ev.get('tempesta')
                    info['gst_events'].append({
                        'startTime': ev_date,
                        'kp_max': ev.get('kp_max'),
                        'kp_values': [],
                        'linked_cmes': ev.get('linked_cmes', [])
                    })
                    cme_ids = ev.get('linked_cmes', [])
                    for cme_id in cme_ids:
                        velocita_cme = cme_speeds.get(cme_id, 0)
                        if velocita_cme == 0 and info['kp_max'] is not None and info['kp_max'] >= 7:
                            velocita_cme = 800
                        info['cme_analyses'].append({
                            'time': ev_date,
                            'speed_km_s': velocita_cme,
                            'type': '',
                            'id': cme_id
                        })
                    break
        except Exception as e:
            print(f"   ⚠️ Errore lettura cache annuale: {e}")

    # Se oggi o manca Kp, scarica da DONKI
    oggi = datetime.now(timezone.utc).date()
    if data_utc.date() == oggi or info['kp_max'] is None:
        print(f"   📡 Scarico dati in tempo reale da DONKI...")
        try:
            start = (data_utc - timedelta(days=1)).strftime("%Y-%m-%d")
            end = (data_utc + timedelta(days=1)).strftime("%Y-%m-%d")
            url = f"{DONKI_BASE}/GST?startDate={start}&endDate={end}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                gst_list = resp.json()
                if not isinstance(gst_list, list):
                    print(f"   ⚠️ DONKI GST: risposta inattesa ({type(gst_list).__name__})")
                    gst_list = []
                for gst in gst_list:
                    try:
                        gst_start = gst.get('startTime', '')
                        if not gst_start:
                            continue
                        gst_dt = datetime.strptime(gst_start, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
                        if gst_dt.date() != data_utc.date():
                            continue
                        kp_indices = gst.get('allKpIndex', [])
                        kp_vals = []
                        for kp_entry in kp_indices:
                            try:
                                kp_val = float(kp_entry.get('kpIndex', 0))
                                kp_vals.append(kp_val)
                            except:
                                pass
                        kp_max = max(kp_vals) if kp_vals else None
                        if kp_max is not None:
                            info['kp_max'] = kp_max
                            for soglia, nome in sorted(SOGLIE_KP_TEMPESTA.items(), reverse=True):
                                if kp_max >= soglia:
                                    info['tempesta'] = nome
                                    break
                            linked_cmes = []
                            for cm in gst.get('linkedEvents', []):
                                act_id = cm.get('activityID', '')
                                if act_id:
                                    linked_cmes.append(act_id)
                            for cme_id in linked_cmes:
                                velocita_cme = 0
                                try:
                                    # Il nuovo endpoint DONKI ignora activityId: usiamo una finestra
                                    # di date stretta attorno alla CME e filtriamo in Python.
                                    # cme_id ha formato "YYYY-MM-DDTHH:MM:SS-CME-NNN".
                                    cme_date = datetime.strptime(cme_id[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                                    start_cme = (cme_date - timedelta(days=1)).strftime('%Y-%m-%d')
                                    end_cme   = (cme_date + timedelta(days=1)).strftime('%Y-%m-%d')
                                    url_cme = f"{DONKI_BASE}/CMEAnalysis?startDate={start_cme}&endDate={end_cme}"
                                    resp_cme = requests.get(url_cme, timeout=10)
                                    if resp_cme.status_code == 200:
                                        cme_data = resp_cme.json()
                                        if isinstance(cme_data, list):
                                            # Match esatto sull'ID CME, non sul primo elemento
                                            matching = [
                                                c for c in cme_data
                                                if c.get('associatedCMEID') == cme_id
                                                or c.get('activityID') == cme_id
                                            ]
                                            if matching:
                                                velocita_cme = matching[0].get('speed', 0)
                                except Exception:
                                    pass
                                if velocita_cme == 0 and kp_max >= 7:
                                    velocita_cme = 800
                                info['cme_analyses'].append({
                                    'time': gst_dt,
                                    'speed_km_s': velocita_cme,
                                    'type': '',
                                    'id': cme_id
                                })
                            print(f"   ✅ Tempesta rilevata: Kp={kp_max}, {info['tempesta']}")
                        else:
                            print(f"   ℹ️ Nessuna tempesta rilevata in tempo reale")
                    except Exception as ex:
                        print(f"  Errore parsing GST: {ex}")
                        continue
        except Exception as e:
            print(f"   ⚠️ Errore download DONKI: {e}")

    # Classifica impatto
    impatto, livello, note = _classifica_impatto_combinato(info)
    info['impatto'] = impatto
    info['livello'] = livello
    info['note'] = note

    try:
        with open(cache_path, 'w') as f:
            json.dump(info, f, indent=2, default=str)
        print(f"   💾 Cache aggiornata: {os.path.basename(cache_path)}")
    except Exception as e:
        print(f"   ⚠️ Errore salvataggio cache: {e}")

    return info

def _classifica_impatto_combinato(info):
    impatto = False
    livello = "normale"
    note = []
    punteggio = 0

    kp = info.get('kp_max')
    if kp is not None:
        if kp >= 7:   punteggio += 5; note.append(f"Tempesta geomagnetica Kp={kp:.1f} (G3/G4)")
        elif kp >= 6: punteggio += 4; note.append(f"Tempesta geomagnetica Kp={kp:.1f} (G2)")
        elif kp >= 5: punteggio += 3; note.append(f"Tempesta geomagnetica Kp={kp:.1f} (G1)")
        elif kp >= 4: punteggio += 2; note.append(f"Kp={kp:.1f} (attivo)")

    for cme in info.get('cme_analyses', []):
        vel = cme.get('speed_km_s', 0)
        if vel >= 1000: punteggio += 4; note.append(f"CME {vel:.0f} km/s in arrivo (molto veloce)")
        elif vel >= 600: punteggio += 3; note.append(f"CME {vel:.0f} km/s in arrivo (veloce)")
        elif vel >= 400: punteggio += 2; note.append(f"CME {vel:.0f} km/s in arrivo")
        else: punteggio += 1; note.append(f"CME in arrivo ({vel:.0f} km/s)")

    proton = info.get('proton_flux_max')
    if proton is not None:
        if proton > 100: punteggio += 3; note.append(f"Protoni >10MeV: {proton:.0f} pfu (evento forte)")
        elif proton > 10: punteggio += 2; note.append(f"Protoni >10MeV: {proton:.0f} pfu (evento)")
        elif proton > 1: punteggio += 1; note.append(f"Protoni >10MeV: {proton:.1f} pfu (elevato)")

    sw = info.get('solar_wind_speed_max')
    if sw is not None:
        if sw >= 800: punteggio += 4; note.append(f"Vento solare: {sw:.0f} km/s (molto alto)")
        elif sw >= 600: punteggio += 3; note.append(f"Vento solare: {sw:.0f} km/s (alto)")
        elif sw >= 500: punteggio += 2; note.append(f"Vento solare: {sw:.0f} km/s (elevato)")
        elif sw >= 400: punteggio += 1; note.append(f"Vento solare: {sw:.0f} km/s (moderato)")

    if punteggio >= 8: impatto = True; livello = "forte"
    elif punteggio >= 5: impatto = True; livello = "moderato"
    elif punteggio >= 3: impatto = True; livello = "lieve"
    elif punteggio > 0: impatto = True; livello = "debole"
    else: livello = "normale"; note.append("Nessun indicatore di disturbo")

    return impatto, livello, note

# ============================================================
# SEZIONE ANNUALE EVENTI GEOMAGNETICI (tasto Y)
# ============================================================

def _deserializza_eventi_geomag(events_raw):
    """
    Riconverte i campi datetime serializzati come stringa da json.dump(..., default=str)
    in oggetti datetime timezone-aware. Applicato dopo ogni json.load dalla cache.
    Gestisce anche:
      - kp_diurno : lista di [str_datetime, float] → lista di (datetime, float)
      - sep       : dict con 'inizio' e 'fine' riconvertiti in datetime
    """
    CAMPI_DATA = ('date', 'ssc_utc', 'alba_path', 'tramonto_path')
    for ev in events_raw:
        for campo in CAMPI_DATA:
            val = ev.get(campo)
            if isinstance(val, str):
                try:
                    dt = datetime.fromisoformat(val)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    ev[campo] = dt
                except Exception:
                    ev[campo] = None

        # sep: dict con datetime annidati (inizio, fine)
        sep = ev.get('sep')
        if isinstance(sep, dict):
            for k in ('inizio', 'fine'):
                v = sep.get(k)
                if isinstance(v, str):
                    try:
                        dt = datetime.fromisoformat(v)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        sep[k] = dt
                    except Exception:
                        sep[k] = None

        # kp_diurno: json.dump serializza le tuple come liste → riconverti
        kp_raw = ev.get('kp_diurno')
        if isinstance(kp_raw, list):
            kp_conv = []
            for item in kp_raw:
                try:
                    t_str, v = item[0], item[1]
                    t_dt = datetime.fromisoformat(t_str) if isinstance(t_str, str) else t_str
                    if isinstance(t_dt, datetime) and t_dt.tzinfo is None:
                        t_dt = t_dt.replace(tzinfo=timezone.utc)
                    kp_conv.append((t_dt, float(v)))
                except Exception:
                    pass
            ev['kp_diurno'] = kp_conv
    return events_raw

def _calcola_alba_tramonto_path(data_utc):
    """
    Calcola alba e tramonto sul punto medio del path NSY-GAESID
    (≈41.08°N, 11.06°E), riferimento fisico per l'esistenza dello
    strato D lungo il percorso.
    Restituisce (alba_utc, tramonto_utc) come datetime o (None, None) in caso di errore.
    """
    try:
        obs = ephem.Observer()
        obs.lat       = str(_PATH_MID_LAT)
        obs.lon       = str(_PATH_MID_LON)
        obs.elevation = 100   # quota media path (approssimata)
        obs.pressure  = 0
        obs.date      = data_utc.strftime('%Y/%m/%d 00:00:00')
        sole = ephem.Sun()

        alba = obs.next_rising(sole).datetime().replace(tzinfo=timezone.utc)

        # Avanza l'osservatore all'istante dell'alba, così next_setting
        # restituisce il tramonto successivo all'alba (stesso giorno).
        obs.date = alba.strftime('%Y/%m/%d %H:%M:%S')

        tramonto = obs.next_setting(sole).datetime().replace(tzinfo=timezone.utc)
        return alba, tramonto
    except Exception:
        return None, None


def _stima_durata_tempesta_h(kp_max):
    """
    Stima la durata della main phase + recovery in ore in base alla classe Kp.
    Valori approssimativi basati sulla letteratura SID/ionosferica.
    """
    if kp_max is None:
        return 24
    if kp_max >= 8:   return 48
    elif kp_max >= 7: return 36
    elif kp_max >= 6: return 24
    else:             return 12


def _fetch_kp_gfz_day(data_utc):
    """
    Recupera i valori Kp ogni 3 ore da GFZ Potsdam.
    Se la data è futura (oltre oggi), restituisce [] senza stampare errori.
    Se il server risponde con errore HTTP 500 (es. dati non ancora disponibili per date lontane),
    restituisce [] senza stampare errori per non intasare il log.
    """
    oggi = datetime.now(timezone.utc).date()
    if data_utc.date() > oggi:
        # Silenziosamente ignora le date future
        return []

    start_str = data_utc.strftime('%Y-%m-%dT00:00Z')
    end_str   = data_utc.strftime('%Y-%m-%dT23:59Z')
    url = f"https://kp.gfz-potsdam.de/app/json/?start={start_str}&end={end_str}&index=Kp"
    try:
        resp = requests.get(url, timeout=15, headers={'User-Agent': 'SID-Monitor/1.0'})
        if resp.status_code != 200:
            # HTTP 500 o altri errori: dati non disponibili, errore silenzioso per date passate non troppo lontane
            # ma stampa solo se è un errore anomalo (non 500) su date recenti
            if data_utc.date() <= oggi and resp.status_code != 500:
                print(f"  [GEOMAG] Errore HTTP {resp.status_code} per {data_utc.strftime('%Y-%m-%d')}")
            return []
        dati = resp.json()
        times_raw = dati.get('datetime', [])
        vals_raw  = dati.get('Kp', [])
        risultati = []
        for t_str, kp_v in zip(times_raw, vals_raw):
            try:
                t_dt = datetime.fromisoformat(t_str.replace('Z', '+00:00'))
                risultati.append((t_dt, float(kp_v)))
            except Exception:
                continue
        return sorted(risultati, key=lambda x: x[0])
    except Exception as e:
        if data_utc.date() <= oggi:
            print(f"  [GEOMAG] Errore fetch GFZ Kp: {e}")
        return []


def fetch_geomag_for_year(anno):
    """
    Recupera tutti gli eventi geomagnetici (GST + CME con arrivo) per un intero anno.
    Per ogni GST aggiunge:
      - ssc_utc       : ora SSC (inizio main phase) da allKpIndex[0].observedTime
      - sep           : dict con {pfu, inizio, fine} se esiste un SEP entro ±2gg, altrimenti None
      - finestra_utc  : stringa "HH:MM→HH:MM UTC" della finestra di anomalia stimata
      - alba_path     : datetime alba sul punto medio path NSY-GAESID
      - tramonto_path : datetime tramonto sul punto medio path NSY-GAESID
    """
    cache_f = os.path.join(CARTELLA_GEOMAG, f"geomag_year_{anno}.json")
    anno_corrente = datetime.now(timezone.utc).year
    oggi = datetime.now(timezone.utc).date()

    if os.path.exists(cache_f):
        # Se è l'anno corrente e oggi è un giorno successivo alla data di creazione del file cache,
        # rigenera la cache per includere eventuali nuovi eventi (GST, CME) avvenuti nel frattempo.
        if anno == anno_corrente:
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(cache_f)).date()
                if oggi > mtime:
                    print(f"[GEOMAG] Cache {os.path.basename(cache_f)} datata {mtime}, rigenero...")
                    os.remove(cache_f)  # forza la rigenerazione
                else:
                    try:
                        with open(cache_f, 'r') as fh:
                            events_raw = json.load(fh)
                        if events_raw:
                            return _deserializza_eventi_geomag(events_raw)
                        # cache vuota → rigenera
                    except:
                        pass
            except:
                pass
        else:
            try:
                with open(cache_f, 'r') as fh:
                    events_raw = json.load(fh)
                return _deserializza_eventi_geomag(events_raw)
            except:
                pass

    events = []

    start_date = datetime(anno, 1, 1, tzinfo=timezone.utc)
    end_date   = datetime(anno + 1, 1, 1, tzinfo=timezone.utc)
    delta      = (end_date - start_date).days

    # ----------------------------------------------------------
    # 1. Recupera tutti i SEP dell'anno in un unico dizionario
    #    indicizzato per data (datetime.date) per lookup rapido
    # ----------------------------------------------------------
    sep_per_data = {}   # date → {'pfu': float, 'inizio': datetime, 'fine': datetime}
    for giorno_offset in range(0, delta, 30):
        chunk_start = start_date + timedelta(days=giorno_offset)
        chunk_end   = min(chunk_start + timedelta(days=30), end_date)
        try:
            url = (f"{DONKI_BASE}/SEP"
                   f"?startDate={chunk_start.strftime('%Y-%m-%d')}"
                   f"&endDate={chunk_end.strftime('%Y-%m-%d')}")
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data_sep = resp.json()
                if not isinstance(data_sep, list):
                    print(f"  [GEOMAG] SEP chunk {chunk_start.date()}: risposta inattesa ({type(data_sep).__name__})")
                    continue
                for sep in data_sep:
                    try:
                        # eventTime è l'ora di inizio rilevamento protoni
                        t_str = sep.get('eventTime', '')
                        if not t_str:
                            continue
                        t_inizio = datetime.strptime(t_str, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
                        # picco di flusso: lista instruments → prendiamo il max
                        pfu = 0.0
                        for instr in sep.get('instruments', []):
                            # il campo si chiama 'flux' nella risposta DONKI
                            try:
                                pfu = max(pfu, float(instr.get('flux', 0) or 0))
                            except Exception:
                                pass
                        # fine evento (non sempre presente)
                        t_fine_str = sep.get('endTime', '')
                        t_fine = None
                        if t_fine_str:
                            try:
                                t_fine = datetime.strptime(t_fine_str, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
                            except Exception:
                                pass
                        # Indicizza per ogni giorno compreso nell'evento (±2gg di tolleranza)
                        for delta_d in range(-2, 3):
                            chiave = (t_inizio + timedelta(days=delta_d)).date()
                            if chiave not in sep_per_data or pfu > sep_per_data[chiave].get('pfu', 0):
                                sep_per_data[chiave] = {
                                    'pfu':    pfu,
                                    'inizio': t_inizio,
                                    'fine':   t_fine
                                }
                    except Exception:
                        continue
        except Exception:
            pass

    # ----------------------------------------------------------
    # 2. GST (tempeste geomagnetiche)
    # ----------------------------------------------------------
    for giorno_offset in range(0, delta, 30):
        chunk_start = start_date + timedelta(days=giorno_offset)
        chunk_end   = min(chunk_start + timedelta(days=30), end_date)
        start_str   = chunk_start.strftime("%Y-%m-%d")
        end_str     = chunk_end.strftime("%Y-%m-%d")
        try:
            url = f"{DONKI_BASE}/GST?startDate={start_str}&endDate={end_str}"
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data_gst = resp.json()
                if not isinstance(data_gst, list):
                    print(f"  [GEOMAG] GST chunk {start_str}: risposta inattesa ({type(data_gst).__name__})")
                    continue
                for gst in data_gst:
                    try:
                        gst_start_str = gst.get('startTime', '')
                        if not gst_start_str:
                            continue
                        gst_dt = datetime.strptime(gst_start_str, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)

                        # Kp max e SSC UTC da allKpIndex DONKI
                        # (kp_diurno viene ora recuperato separatamente da GFZ Potsdam,
                        #  con fallback sui dati allKpIndex DONKI se GFZ non risponde)
                        kp_indices  = gst.get('allKpIndex', [])
                        kp_vals     = []
                        ssc_utc     = None
                        kp_diurno   = []   # Inizializza per raccogliere entrambi
                        for kp_entry in kp_indices:
                            t_obs = kp_entry.get('observedTime', '')
                            kp_v  = None
                            try:
                                kp_v = float(kp_entry.get('kpIndex', 0))
                                kp_vals.append(kp_v)
                            except Exception:
                                pass
                            if t_obs and ssc_utc is None:
                                try:
                                    ssc_utc = datetime.strptime(t_obs, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
                                except Exception:
                                    pass
                            # Salva anche la coppia (datetime_utc, kp) per fallback
                            if t_obs and kp_v is not None:
                                try:
                                    t_dt = datetime.strptime(t_obs, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
                                    kp_diurno.append((t_dt, kp_v))
                                except Exception:
                                    pass
                        kp_max = max(kp_vals) if kp_vals else None

                        # Se non trovata SSC da kp_indices, usa startTime come fallback
                        if ssc_utc is None:
                            ssc_utc = gst_dt

                        # Alba/tramonto path — necessari per filtrare kp_serie
                        alba_path, tramonto_path = _calcola_alba_tramonto_path(gst_dt)

                        # Kp giornaliero da GFZ Potsdam (8 valori, copertura completa 00-24 UTC)
                        # Se GFZ non risponde (es. HTTP 500 per date lontane), usa i dati DONKI come fallback
                        oggi = datetime.now(timezone.utc).date()
                        if gst_dt.date() <= oggi:
                            kp_diurno_gfz = _fetch_kp_gfz_day(gst_dt)
                            if kp_diurno_gfz:
                                kp_diurno = kp_diurno_gfz
                            # Se GFZ non ha risposto, kp_diurno mantiene i valori DONKI già estratti
                        else:
                            kp_diurno = []                        

                        # Classifica tempesta
                        tempesta = None
                        if kp_max is not None:
                            for soglia, nome in sorted(SOGLIE_KP_TEMPESTA.items(), reverse=True):
                                if kp_max >= soglia:
                                    tempesta = nome
                                    break

                        # CME collegate
                        linked_cmes = [
                            cm.get('activityID', '')
                            for cm in gst.get('linkedEvents', [])
                            if cm.get('activityID', '')
                        ]

                        # Valutazione impatto
                        punteggio = 0
                        if kp_max is not None:
                            if kp_max >= 7:   punteggio += 5
                            elif kp_max >= 6: punteggio += 4
                            elif kp_max >= 5: punteggio += 3
                            elif kp_max >= 4: punteggio += 2
                        if punteggio >= 8:    impatto = "FORTE"
                        elif punteggio >= 5:  impatto = "MODERATO"
                        elif punteggio >= 3:  impatto = "LIEVE"
                        elif punteggio > 0:   impatto = "DEBOLE"
                        else:                 impatto = "NO"

                        # SEP associato (cerca nel dizionario pre-caricato)
                        sep_info = sep_per_data.get(gst_dt.date())

                        # Finestra anomalia stimata: da SSC_UTC a SSC_UTC + durata_classe
                        durata_h    = _stima_durata_tempesta_h(kp_max)
                        fine_utc    = ssc_utc + timedelta(hours=durata_h)
                        finestra_str = f"{ssc_utc.strftime('%H:%M')}→{fine_utc.strftime('%H:%M')} UTC (+{durata_h}h)"

                        events.append({
                            'type':          'GST',
                            'date':          gst_dt,
                            'kp_max':        kp_max,
                            'tempesta':      tempesta,
                            'linked_cmes':   linked_cmes,
                            'impatto':       impatto,
                            'ssc_utc':       ssc_utc,
                            'sep':           sep_info,   # dict {pfu, inizio, fine} o None
                            'finestra_utc':  finestra_str,
                            'alba_path':     alba_path,
                            'tramonto_path': tramonto_path,
                            'kp_diurno':     kp_diurno,  # lista di (datetime, kp)
                        })
                    except Exception:
                        continue
        except Exception:
            pass

    # ----------------------------------------------------------
    # 3. CME con arrivalTime
    # ----------------------------------------------------------
    for giorno_offset in range(0, delta, 30):
        chunk_start = start_date + timedelta(days=giorno_offset)
        chunk_end   = min(chunk_start + timedelta(days=30), end_date)
        start_str   = chunk_start.strftime("%Y-%m-%d")
        end_str     = chunk_end.strftime("%Y-%m-%d")
        try:
            url = f"{DONKI_BASE}/CMEAnalysis?startDate={start_str}&endDate={end_str}"
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data_cme = resp.json()
                if not isinstance(data_cme, list):
                    print(f"  [GEOMAG] CMEAnalysis chunk {start_str}: risposta inattesa ({type(data_cme).__name__})")
                    continue
                for a in data_cme:
                    try:
                        arr = a.get('arrivalTime')
                        if not arr:
                            continue
                        arr_dt   = datetime.strptime(arr, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
                        velocita = a.get('speed', 0)

                        # Kp giornaliero da GFZ Potsdam per il giorno dell'arrivo
                        oggi = datetime.now(timezone.utc).date()
                        if arr_dt.date() <= oggi:
                            kp_diurno = _fetch_kp_gfz_day(arr_dt)
                        else:
                            kp_diurno = []

                        punteggio = 0
                        if velocita >= 1000:   punteggio = 5
                        elif velocita >= 600:  punteggio = 3
                        elif velocita >= 400:  punteggio = 2
                        else:                  punteggio = 1

                        if punteggio >= 4:    impatto = "FORTE"
                        elif punteggio >= 2:  impatto = "MODERATO"
                        elif punteggio >= 1:  impatto = "LIEVE"
                        else:                 impatto = "NO"

                        events.append({
                            'type':          'CME',
                            'date':          arr_dt,
                            'velocita_km_s': velocita,
                            'id':            a.get('activityID', 'CME'),
                            'impatto':       impatto,
                            'kp_diurno':     kp_diurno
                        })
                    except Exception:
                        continue
        except Exception:
            pass

    # Ordina per data e salva in cache (solo se non vuota!)
    events.sort(key=lambda e: e['date'])
    if events:
        try:
            with open(cache_f, 'w') as fh:
                json.dump(events, fh, indent=2, default=str)
        except Exception:
            pass

    return events


def mostra_elenco_geomag_per_anno(anno):
    """
    Mostra una finestra con tutti gli eventi geomagnetici dell'anno.

    Tabella GST — colonne:
      Data | Kp max | Classe | SSC UTC | SEP | CME coll. | Finestra anomalia | Alba/Tram. path | Impatto SID | Kp orari (UTC)
    """
    print(f"\n[GEOMAG] Recupero eventi geomagnetici per il {anno}...")
    events = fetch_geomag_for_year(anno)

    if not events:
        print(f"[GEOMAG] Nessun evento geomagnetico trovato per il {anno}.")
        return

    gst_events = [e for e in events if e['type'] == 'GST']
    cme_events = [e for e in events if e['type'] == 'CME']

    linee = [
        f"Eventi geomagnetici nel {anno}: {len(events)} totali "
        f"({len(gst_events)} tempeste, {len(cme_events)} CME)",
        "",
        "LEGENDA COLONNE TEMPESTE:",
        "  SSC UTC      = ora inizio main phase (riferimento per cercare anomalie nel plot NSY)",
        "  SEP          = Solar Energetic Particles entro ±2gg (ionizzano D-layer anche di notte)",
        "  Fin.anomalia = finestra UTC stimata per anomalie NSY (SSC + durata per classe Kp)",
        "  Alba/Tram.   = alba e tramonto sul punto medio path NSY-GAESID (41.1°N 11.1°E).",
        "                 Lo strato D (e quindi il segnale NSY) esiste solo tra alba e tramonto.",
        "  Kp orari     = valori Kp ogni 3h per l'intera giornata (fonte GFZ Potsdam).",
        "                 Formato: HH:MM=X.X  — se Kp>=4 il segnale potrebbe essere perturbato.",
        "                 Un Kp elevato già nelle prime ore indica che il disturbo era in corso da notte.",
    ]

    if gst_events:
        linee.append("")
        linee.append("--- TEMPESTE GEOMAGNETICHE (GST) ---")
        linee.append(
            f"{'Data':<10s} | {'Kp':>5s} | {'Classe':<15s} | {'SSC UTC':>7s} | "
            f"{'SEP':<14s} | {'CME':>5s} | {'Finestra anomalia':<22s} | "
            f"{'Alba / Tram. path':<17s} | {'Impatto':<16s} | Kp orari (UTC)"
        )
        linee.append("-" * 160)

        for e in gst_events:
            data     = e['date'].strftime('%Y-%m-%d')
            kp       = f"{e['kp_max']:.1f}" if e['kp_max'] is not None else "N/D"
            tempesta = e.get('tempesta') or "nessuna"
            n_cme    = len(e.get('linked_cmes', []))
            cme_str  = f"{n_cme} CME" if n_cme > 0 else "-"
            impatto  = e.get('impatto', 'N/D')

            # SSC UTC
            ssc = e.get('ssc_utc')
            ssc_str = ssc.strftime('%H:%M') if isinstance(ssc, datetime) else "N/D"

            # SEP
            sep = e.get('sep')
            if sep and isinstance(sep, dict):
                pfu = sep.get('pfu', 0)
                sep_str = f"Si ({pfu:.0f} pfu)" if pfu and pfu > 0 else "Si"
            else:
                sep_str = "No"

            # Finestra anomalia
            finestra = e.get('finestra_utc', 'N/D')

            # Alba / tramonto path
            alba     = e.get('alba_path')
            tramonto = e.get('tramonto_path')
            if isinstance(alba, datetime) and isinstance(tramonto, datetime):
                astro_str = f"{alba.strftime('%H:%M')}/{tramonto.strftime('%H:%M')} UTC"
            else:
                astro_str = "N/D"

            # Kp orari (da GFZ)
            kp_diurno = e.get('kp_diurno', [])
            if kp_diurno:
                kp_diurno_str = "  ".join(
                    f"{t.strftime('%H:%M')}={v:.1f}"
                    for t, v in sorted(kp_diurno, key=lambda x: x[0])
                )
            else:
                kp_diurno_str = "N/D"

            col = 'VERDE'
            if impatto == 'FORTE':                   col = 'ROSSO'
            elif impatto == 'MODERATO':              col = 'ARANCIONE'
            elif impatto in ('LIEVE', 'DEBOLE'):     col = 'GIALLO'

            impatto_col = f"{impatto} ({col})"

            linee.append(
                f"{data:<10s} | {kp:>5s} | {tempesta:<15s} | {ssc_str:>7s} | "
                f"{sep_str:<14s} | {cme_str:>5s} | {finestra:<22s} | "
                f"{astro_str:<17s} | {impatto_col:<16s} | {kp_diurno_str}"
            )

    if cme_events:
        linee.append("")
        linee.append("--- CME CON ARRIVO NELL'ANNO ---")
        linee.append(
            f"{'Data arrivo':<16s} | {'Velocità':<10s} | {'ID CME':<22s} | {'Impatto':<16s} | Kp orari (UTC)"
        )
        linee.append("-" * 140)
        for e in cme_events:
            data    = e['date'].strftime('%Y-%m-%d %H:%M') if isinstance(e['date'], datetime) else str(e['date'])
            vel     = f"{e['velocita_km_s']:.0f} km/s" if e.get('velocita_km_s') else "N/D"
            id_cme  = e.get('id', 'CME')[:22]
            impatto = e.get('impatto', 'N/D')

            # Kp orari per CME
            kp_diurno = e.get('kp_diurno', [])
            if kp_diurno:
                kp_diurno_str = "  ".join(
                    f"{t.strftime('%H:%M')}={v:.1f}"
                    for t, v in sorted(kp_diurno, key=lambda x: x[0])
                )
            else:
                kp_diurno_str = "N/D"

            col = 'VERDE'
            if impatto == 'FORTE':                      col = 'ROSSO'
            elif impatto == 'MODERATO':                 col = 'ARANCIONE'
            elif impatto in ('LIEVE', 'DEBOLE'):        col = 'GIALLO'

            impatto_col = f"{impatto} ({col})"
            linee.append(f"{data:<16s} | {vel:<10s} | {id_cme:<22s} | {impatto_col:<16s} | {kp_diurno_str}")

    try:
        root = tk.Tk()
        root.title(f"Eventi geomagnetici {anno}")
        root.attributes('-topmost', True)
        root.resizable(True, True)
        root.geometry("1400x650")

        yscroll = tk.Scrollbar(root, orient='vertical')
        yscroll.pack(side='right', fill='y')

        xscroll = tk.Scrollbar(root, orient='horizontal')
        xscroll.pack(side='bottom', fill='x')

        text = tk.Text(
            root, wrap='none', bg='#121212', fg='white',
            insertbackground='white', font=('Consolas', 10),
            yscrollcommand=yscroll.set,
            xscrollcommand=xscroll.set,
        )
        text.pack(fill='both', expand=True)

        yscroll.config(command=text.yview)
        xscroll.config(command=text.xview)

        text.insert('1.0', '\n'.join(linee))
        text.config(state='disabled')
        root.mainloop()
    except Exception as ex:
        print(f"[GEOMAG] Errore visualizzazione elenco: {ex}")



# =============================================================
# DISEGNO GRAFICO e STATO
# =============================================================
stato = {
    'data':          None,
    'smooth':        1,
    'fig':           None,
    'ax':            None,
    'xray_on':       True,
    'xray_ts':       [],
    'xray_flux':     [],
    'xray_data':     None,
    'scala_lineare': False,
    'timestamps':    [],
    'valori_grezzi': [],
    # --- NUOVI ---
    'fase_on':       False,   # overlay fase ON/OFF
    'fasi':          [],      # lista fasi grezze (gradi) allineate a timestamps
    'fase_derivata':   True,    # True = srotola la fase, False = usa sin(fase)
}

def disegna_grafico():
    data_utc        = stato['data']
    finestra_smooth = stato['smooth']

    csv_path = trova_csv_per_data(data_utc)
    if csv_path is None:
        aggiorna_titolo_mancante(data_utc)
        return

    print(f"\nLettura CSV VLF: {csv_path}")
    timestamps, valori, fasi = leggi_csv_sid_con_fase(csv_path)
    if timestamps:
        print(f"\n🔍 VERIFICA TIMESTAMP (primi 5 campioni):")
        for i in range(min(5, len(timestamps))):
            print(f"   {i+1}. {timestamps[i].strftime('%Y-%m-%d %H:%M:%S')} UTC")
        alba, tramonto = calcola_alba_tramonto(data_utc)
        if alba and tramonto:
            print(f"   Alba: {alba.strftime('%H:%M')} UTC")
            print(f"   Tramonto: {tramonto.strftime('%H:%M')} UTC")
    if not timestamps:
        print("Nessun dato nel file CSV VLF.")
        return

    stato['timestamps'] = timestamps
    stato['valori_grezzi'] = valori
    stato['fasi'] = fasi

    valori_smooth = applica_smoothing(valori, finestra_smooth) if finestra_smooth > 1 else valori

    # --- Conversione scala ---
    if stato['scala_lineare']:
        # Ampiezza lineare RELATIVA al livello di quiete del giorno.
        # Riferimento = 10° percentile (robusto ai SID positivi).
        #   1.0  = livello di quiete
        #   1.41 = SID di +3 dB
        #   2.00 = SID di +6 dB
        #   3.16 = SID di +10 dB
        # Convenzione compatibile con stazioni supersid (Loudet, ecc.)
        ref_db = float(np.percentile(valori_smooth, 10))
        stato['ref_lineare_db'] = ref_db
        valori_plot = [10 ** ((v - ref_db) / 20.0) for v in valori_smooth]
    else:
        valori_plot = valori_smooth
        stato['ref_lineare_db'] = None

    # Carica flare
    flares_cache = leggi_flare_cache(data_utc)
    if flares_cache is not None:
        flares_manuali = [f for f in carica_flare_manuali() if f['picco'].date() == data_utc.date()]
        flares = flares_cache + flares_manuali
        flares.sort(key=lambda f: f['picco'])
        print(f"Flare totali (cache + manuali): {len(flares)}")
    else:
        risultati_flare = scarica_flare(data_utc)
        flares = parse_flare_multi(risultati_flare, data_utc) if risultati_flare else []
        salva_flare_cache(data_utc, flares)

    if stato['xray_data'] != data_utc.date():
        print("Scarico dati X-ray GOES...")
        xray_ts, xray_flux = scarica_xray(data_utc)
        stato['xray_ts']   = xray_ts
        stato['xray_flux'] = xray_flux
        stato['xray_data'] = data_utc.date()
    else:
        xray_ts   = stato['xray_ts']
        xray_flux = stato['xray_flux']

    alba, tramonto = calcola_alba_tramonto(data_utc)

    fig = stato['fig']
    ax  = stato['ax']
    ax.cla()
    for ax2_old in fig.axes[1:]:
        ax2_old.remove()

    inizio_giorno = data_utc.replace(hour=0, minute=0, second=0, tzinfo=timezone.utc)
    fine_giorno   = data_utc.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
    ax.set_xlim(inizio_giorno, fine_giorno)

    # --- Range asse Y: SEMPRE calcolato dai dati (funziona sia dB che lineare) ---
    v_min = min(valori_plot)
    v_max = max(valori_plot)
    if v_max == v_min:
        margine = abs(v_min) * 0.10 if v_min != 0 else 1.0
    else:
        margine = (v_max - v_min) * 0.10
    ax.set_ylim(v_min - margine, v_max + margine)

    if alba and tramonto:
        ax.axvspan(inizio_giorno, alba, color='#404060', alpha=0.45, zorder=1)
        ax.axvspan(tramonto, fine_giorno, color='#404060', alpha=0.45, zorder=1)
        ax.axvline(alba, color='#FFA500', linewidth=1.2, linestyle='--', alpha=0.8, zorder=2)
        ax.axvline(tramonto, color='#FF6347', linewidth=1.2, linestyle='--', alpha=0.8, zorder=2)
        y_lim = ax.get_ylim()
        y_label = y_lim[0] + (y_lim[1] - y_lim[0]) * 0.05
        ax.text(alba, y_label, f"☀ Alba\n{alba.strftime('%H:%M')} UTC",
                color='#FFA500', fontsize=8, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d0d1a', alpha=0.7, edgecolor='#FFA500'))
        ax.text(tramonto, y_label, f"☀ Tramonto\n{tramonto.strftime('%H:%M')} UTC",
                color='#FF6347', fontsize=8, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d0d1a', alpha=0.7, edgecolor='#FF6347'))

    ax.plot(timestamps, valori_plot, color='#00FF99', linewidth=0.8, alpha=0.9, zorder=3, label='NSY 45.9 kHz VLF')
    ax.set_xlabel('Ora UTC', color='white', fontsize=11)

    # --- Etichetta asse Y ---
    if stato['scala_lineare']:
        ref_db = stato.get('ref_lineare_db')
        if ref_db is not None:
            ax.set_ylabel(
                f'Segnale VLF (lineare, rapporto a {ref_db:+.1f} dB)',
                color='#00FF99', fontsize=11
            )
        else:
            ax.set_ylabel('Segnale VLF (lineare, rapporto)',
                          color='#00FF99', fontsize=11)
        ax.ticklabel_format(style='plain', axis='y', useOffset=False)
        # Riga di riferimento a 1.0 (livello di quiete del giorno)
        ax.axhline(1.0, color='#FFD700', linewidth=0.8,
                   linestyle=':', alpha=0.7, zorder=2)
    else:
        ax.set_ylabel('Segnale VLF (dB)', color='#00FF99', fontsize=11)

    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('#00FF99')
    ax.tick_params(axis='y', colors='#00FF99')
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, color='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
    ax.grid(True, alpha=0.15, color='white')

    # --- Overlay fase (terzo asse Y) ---
    ax3 = None
    fasi_plot = None
    if stato['fase_on'] and any(f is not None for f in fasi):
        fasi_ok = [(t, f) for t, f in zip(timestamps, fasi) if f is not None]
        if fasi_ok:
            ts_fase  = [t for t, _ in fasi_ok]
            fase_raw = np.array([f for _, f in fasi_ok], dtype=float)

            # Detrend lineare su finestra mobile (rimuove rampe ripide)
            finestra_detrend = 1800    # 30 minuti
            n_fase = len(fase_raw)
            fase_detrended = np.zeros(n_fase)
            for i in range(n_fase):
                start = max(0, i - finestra_detrend // 2)
                end = min(n_fase, i + finestra_detrend // 2)
                if end - start < 30:
                    fase_detrended[i] = fase_raw[i]
                    continue
                t_local = np.arange(end - start, dtype=float)
                coeffs = np.polyfit(t_local, fase_raw[start:end], 1)
                fase_detrended[i] = fase_raw[i] - np.polyval(coeffs, i - start)

            # 2) Smoothing prima della derivata (fondamentale!)
            #    Usa un filtro Savitzky-Golay: liscia preservando la forma dei fronti.
            #    Finestra ~31 campioni (~30 s a 1 Hz), polinomio di grado 2.
            from scipy.signal import savgol_filter
            if len(fase_detrended) > 31:
                fase_smooth = savgol_filter(fase_detrended, window_length=31, polyorder=2)
            else:
                fase_smooth = fase_detrended

            if stato.get('fase_derivata', False):
                # 3) Derivata della fase LISCIATA (non quella grezza)
                #    Il fattore 60 converte in °/minuto (più leggibile di °/s)
                dt = 1.0  # secondi tra campioni (log a 1 Hz)
                fase_vals = np.gradient(fase_smooth) / dt * 60.0
                ylabel_fase = 'dFase/dt (°/min)'
            else:
                fase_vals = fase_smooth
                ylabel_fase = 'Variazione di fase (°)'

            ax3 = ax.twinx()
            # Sposta ax3 a destra di ax (che a sua volta è a sinistra di ax2)
            ax3.spines['right'].set_position(('axes', 1.08))
            ax3.spines['left'].set_visible(False)
            ax3.spines['right'].set_visible(True)
            ax3.yaxis.set_label_position('right')
            ax3.yaxis.tick_right()
            ax3.tick_params(axis='y', colors='#4169E1', labelsize=8)
            ax3.yaxis.label.set_color('#4169E1')
            for spine in ax3.spines.values():
                spine.set_edgecolor('#4169E1')
            ax3.set_xlim(inizio_giorno, fine_giorno)
            ax3.plot(ts_fase, fase_vals, color='#4169E1', linewidth=0.9,
                     alpha=0.85, zorder=3, label='Fase VLF')
            ax3.set_ylabel(ylabel_fase, color='#4169E1', fontsize=10)

            fasi_plot = fase_vals

    ax2 = None
    if stato['xray_on'] and xray_ts and xray_flux:
        ax2 = ax.twinx()
        ax2.set_xlim(inizio_giorno, fine_giorno)
        flux_min = max(min(xray_flux) * 0.5, 1e-9)
        flux_max = max(xray_flux) * 3
        ax2.set_yscale('log')
        ax2.set_ylim(flux_min, flux_max)
        for nome_classe, (soglia_inf, soglia_sup) in SOGLIE_CLASSE_XRAY.items():
            y0 = max(soglia_inf, flux_min)
            y1 = min(soglia_sup, flux_max)
            if y1 > y0:
                ax2.axhspan(y0, y1, color=COLORI_FASCE_XRAY[nome_classe], alpha=0.04, zorder=1)
                y_mid = np.sqrt(soglia_inf * soglia_sup)
                if flux_min < y_mid < flux_max:
                    ax2.text(fine_giorno, y_mid, f' {nome_classe}',
                             color=COLORI_FASCE_XRAY[nome_classe], fontsize=8, va='center', ha='left',
                             fontweight='bold', clip_on=False)
        for nome_classe, (soglia_inf, _) in SOGLIE_CLASSE_XRAY.items():
            if flux_min < soglia_inf < flux_max:
                ax2.axhline(soglia_inf, color=COLORI_FASCE_XRAY[nome_classe], linewidth=0.5, linestyle=':', alpha=0.5, zorder=2)
        ax2.plot(xray_ts, xray_flux, color='#FFD700', linewidth=1.2, alpha=0.85, zorder=4, label='GOES X-ray (0.1–0.8 nm)')
        ax2.set_ylabel('Flusso X-ray GOES (W/m²)', color='#FFD700', fontsize=10)
        ax2.tick_params(axis='y', colors='#FFD700', labelsize=8)
        ax2.yaxis.label.set_color('#FFD700')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#444444')
        ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.LogFormatterMathtext())

    legenda_classi = set()
    ha_limb = False
    ha_manuale = False

    if _flare_visible:
        for flare in flares:
            is_limb = flare.get('limb', False)
            is_manuale = flare.get('manuale', False)
            colore = COLORE_LIMB if is_limb else COLORI_FLARE.get(flare['tipo'], '#FFFFFF')
            ax.axvspan(flare['inizio'], flare['fine'], alpha=0.25, color=colore, zorder=4, linestyle='--' if is_limb else '-')
            ax.axvline(flare['picco'], color=colore, linewidth=1.5, alpha=0.8, linestyle=':' if is_limb else '--', zorder=5)
            if is_manuale:
                etichetta = f"{flare['classe']}\n✎ BTL"
            elif is_limb:
                etichetta = f"{flare['classe']}\n{SIMBOLO_LIMB}"
            else:
                etichetta = flare['classe']
            ax.text(flare['picco'], np.percentile(valori_plot, 95), etichetta,
                    color=colore, fontsize=7 if is_limb else 8, ha='center', va='bottom', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d0d1a', alpha=0.7, edgecolor=colore))
            if is_manuale:
                ha_manuale = True
            elif is_limb:
                ha_limb = True
            else:
                legenda_classi.add(flare['tipo'])
    else:
        print("Flare nascosti: premi F per visualizzarli")

    smooth_label = f" (smooth {finestra_smooth}s)" if finestra_smooth > 1 else ""
    if stato['scala_lineare']:
        ref_db = stato.get('ref_lineare_db')
        scala_label = f" [LIN, ref {ref_db:+.1f} dB]" if ref_db is not None else " [LIN]"
    else:
        scala_label = ""
    handles = [mpatches.Patch(color='#00FF99', label=f'NSY 45.9 kHz VLF{smooth_label}{scala_label}')]

    if stato['xray_on'] and xray_ts:
        oggi_utc = datetime.now(timezone.utc).date()
        delta = (oggi_utc - data_utc.date()).days
        xray_label = 'GOES X-ray 0.1–0.8 nm (dati reali)' if delta <= 6 else 'GOES X-ray 0.1–0.8 nm (⚠ profilo ricostruito da flare)'
        handles.append(mpatches.Patch(color='#FFD700', label=xray_label))
    elif stato['xray_on'] and not xray_ts:
        handles.append(mpatches.Patch(color='#FFD700', alpha=0.3, label='GOES X-ray — nessun dato'))
    else:
        handles.append(mpatches.Patch(color='#555555', alpha=0.5, label='GOES X-ray — disattivato (tasto X)'))

    if stato['fase_on'] and fasi_plot is not None:
        handles.append(mpatches.Patch(color='#00BFFF', label='Fase VLF'))
    elif stato['fase_on']:
        handles.append(mpatches.Patch(color='#00BFFF', alpha=0.3, label='Fase VLF — nessun dato'))
    else:
        handles.append(mpatches.Patch(color='#555555', alpha=0.5, label='Fase VLF — disattivata (tasto W)'))

    if alba and tramonto:
        handles.append(mpatches.Patch(color='#404060', alpha=0.6, label='Notte'))
        handles.append(mpatches.Patch(color='#FFA500', alpha=0.8, label=f'Alba {alba.strftime("%H:%M")} UTC'))
        handles.append(mpatches.Patch(color='#FF6347', alpha=0.8, label=f'Tramonto {tramonto.strftime("%H:%M")} UTC'))

    for tipo in sorted(legenda_classi, reverse=True):
        handles.append(mpatches.Patch(color=COLORI_FLARE.get(tipo, '#FFFFFF'), alpha=0.6, label=f'Flare classe {tipo}'))

    if ha_limb:
        handles.append(mpatches.Patch(color=COLORE_LIMB, alpha=0.6, label='Flare beyond-the-limb (★ BTL)'))
    if ha_manuale:
        handles.append(mpatches.Patch(color=COLORE_LIMB, alpha=0.6, label='Flare inserito manualmente (✎ BTL)'))

    ax.legend(handles=handles, loc='lower left', facecolor='#1a1a2e', edgecolor='#444444', labelcolor='white', fontsize=9)

    smooth_title = f" — Smooth {finestra_smooth}s" if finestra_smooth > 1 else ""
    xray_stato = "ON" if stato['xray_on'] else "OFF"
    scala_stato = "LIN" if stato['scala_lineare'] else "dB"

    ax.set_title(
        f"{data_utc.strftime('%Y-%m-%d UTC')}\n"
        f"Segnale VLF NSY - Flusso X-ray GOES - Flare Solari - GRB{smooth_title} - New Filter\n"
        f"A = prec.  |  D = succ.  |  I = data  |  M = flare Man  |  F = flare ON  |  G = GRB ON |  Y = Eventi Geomagnetici |  T = Ricerca SID con AI | W = Fase | shit+W = filtro | B = GRB\n"
        f"Shift+B = tutti GRB  | H = AI addestr  |  C = Correlazioni SID |  "
        f"S / V = smooth lista / libero  |  U = scala [{scala_stato}]  |  L = SID usati da AI  |  "
        f"X = X-ray [{xray_stato}]  |  R = Refresh  |  P = Soglia probab  |  Q = esci",
        color='white', fontsize=11, fontweight='bold', pad=18, y=0.99
    )
    fig.subplots_adjust(right=0.85)   # lascia spazio a destra per due assi

    global _grb_all, _grb_ril, _grb_visible
    print("[GRB] Recupero dati GRB da HEASARC Fermi GBM...")
    _grb_all = fetch_grb_for_date(data_utc)
    _grb_ril = _filtra_grb_rilevabili(_grb_all)
    print(f"[GRB] {len(_grb_all)} GRB totali nel giorno, {len(_grb_ril)} potenzialmente rilevabili a GAESID.")

    info_geo = scarica_dati_geomagnetici(data_utc)
    _aggiorna_status_pannello(fig, _grb_ril, info_geo)
    disegna_grb_markers(ax, _grb_ril, _grb_visible)

    fig.canvas.draw_idle()
    
def aggiorna_titolo_mancante(data_utc):
    ax = stato['ax']
    ax.cla()
    ax.set_facecolor('#0d0d1a')
    ax.text(0.5, 0.5, f"Nessun dato disponibile per\n{data_utc.strftime('%Y-%m-%d')}",
            transform=ax.transAxes, color='#FF6347', fontsize=14, ha='center', va='center')
    ax.set_title(
        f"Segnale VLF NSY  —  {data_utc.strftime('%Y-%m-%d')} UTC\n"
        f"Tasti:  A/D = giorno  |  I = data  |  G = GRB  |  C = Correlazioni SID  |  S/V = smooth",
        color='white', fontsize=10, fontweight='bold', pad=10
    )
    global _grb_all, _grb_ril
    _grb_all = fetch_grb_for_date(data_utc)
    _grb_ril = _filtra_grb_rilevabili(_grb_all)
    info_geo = scarica_dati_geomagnetici(data_utc)
    _aggiorna_status_pannello(stato['fig'], _grb_ril, info_geo)
    stato['fig'].canvas.draw_idle()

def disegna_grb_markers(ax, grb_ril, visible):
    global _grb_artists
    for art in _grb_artists:
        try: art.remove()
        except Exception: pass
    _grb_artists.clear()
    if not visible or not grb_ril:
        return
    y_top = ax.get_ylim()[1]
    for g in grb_ril:
        trig = g['trigger_time']
        col  = g['colore']
        flu  = g['fluence']
        vl = ax.axvline(trig, color=col, linewidth=1.5, linestyle='--', alpha=0.9, zorder=5)
        _grb_artists.append(vl)
        label = f"{g['name']}\nFlu: {flu:.2e} erg/cm²\nT90: {g['t90']:.1f} s\n[{g['categoria']}]"
        ann = ax.annotate(label, xy=(trig, y_top), xytext=(4, -4), textcoords='offset points',
                          fontsize=6.5, color=col, va='top',
                          bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=col, alpha=0.88),
                          zorder=6)
        _grb_artists.append(ann)

def _aggiorna_status_pannello(fig, grb_ril, info_geomag=None):
    global _grb_status_text
    if _grb_status_text is not None:
        try: _grb_status_text.remove()
        except: pass

    righe = []
    colore = 'gray'

    if not grb_ril:
        righe.append("◉ Nessun GRB rilevabile")
        colore = '#AAAAAA'
    else:
        n = len(grb_ril)
        s = 'i' if n > 1 else ''
        righe.append(f"◉ {n} GRB potenzialmente rilevabile{s}")
        colore = '#FF0000'

    if info_geomag:
        kp_max = info_geomag.get('kp_max', None)
        tempesta = info_geomag.get('tempesta', None)
        cme_analyses = info_geomag.get('cme_analyses', [])
        note = info_geomag.get('note', [])

        if kp_max is not None and kp_max >= 5:
            righe.append(f"⚠ TEMPESTA {tempesta.upper() if tempesta else 'GEOMAGNETICA'} (Kp={kp_max:.1f})")
            colore = '#FF0000'
            cme_mostrati = 0
            for cme in cme_analyses:
                vel = cme.get('speed_km_s', 0)
                if vel > 0 and cme_mostrati == 0:
                    righe.append(f"  · CME in arrivo ({vel:.0f} km/s)")
                    cme_mostrati += 1
                elif cme_mostrati == 0:
                    righe.append("  · CME in arrivo")
                    cme_mostrati += 1
            for n in note[:2]:
                if "CME" not in n and "Tempesta" not in n and "Kp" not in n:
                    righe.append(f"  · {n}")
        elif info_geomag.get('impatto', False):
            livello = info_geomag.get('livello', 'normale')
            if livello == "forte":
                righe.append("⚠ Impatto FORTE sul segnale SID")
            elif livello == "moderato":
                righe.append("⚠ Impatto MODERATO sul segnale SID")
            elif livello == "lieve":
                righe.append("⚠ Impatto LIEVE sul segnale SID")
            else:
                righe.append("⚠ Disturbo MINORE sul segnale SID")
            colore = '#FF0000'
            for n in note[:2]:
                if "CME" not in n and "Tempesta" not in n:
                    righe.append(f"  · {n}")
        else:
            if not grb_ril:
                colore = '#006600'
            righe.append("✓ Condizioni tranquille")

    testo = "\n".join(righe)
    _grb_status_text = fig.text(0.99, 0.01, testo, ha='right', va='bottom', fontsize=9, color=colore,
                                transform=fig.transFigure,
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=colore, alpha=0.88),
                                zorder=10)

def _toggle_grb(ax_vlf, fig):
    global _grb_visible
    if not _grb_ril:
        print("[GRB] Nessun GRB rilevabile per questa giornata.")
        return
    if not _grb_visible:
        stato['smooth'] = 1
        _grb_visible = True
        disegna_grafico()
    else:
        _grb_visible = False
        disegna_grb_markers(ax_vlf, _grb_ril, _grb_visible)
        fig.canvas.draw_idle()
    print(f"[GRB] Visualizzazione: {'ON ✓' if _grb_visible else 'OFF'}")

# =============================================================
# HANDLER TASTIERA (corretto)
# =============================================================
def on_key(event):
    if event.key is None:
        return
    
    key_raw = event.key
    shift_pressed = False
    if key_raw is not None:
        key_norm = key_raw.lower()
        if key_norm.startswith('shift+'):
            key_norm = key_norm.split('+', 1)[1]
            shift_pressed = True
        elif len(key_raw) == 1 and key_raw.isalpha() and key_raw == key_raw.upper():
            key_norm = key_raw.lower()
            shift_pressed = True
        else:
            key_norm = key_raw.lower()
    else:
        return
    print(f"[DEBUG] key={event.key!r}  key_norm={key_norm!r}  shift={shift_pressed}")

    gui_event = getattr(event, 'guiEvent', None)
    if gui_event is not None and hasattr(gui_event, 'state'):
        shift_pressed = shift_pressed or (gui_event.state & 0x0001 != 0)

    if key_norm == 'r':
        stato['xray_data'] = None
        disegna_grafico()

    elif key_norm == 'g':
        _toggle_grb(stato['ax'], stato['fig'])

    elif key_norm == 's':
        nuovo = apri_dialogo_smooth_lista()
        if nuovo is not None and nuovo != stato['smooth']:
            stato['smooth'] = nuovo
            disegna_grafico()

    elif key_norm == 'v':
        nuovo = apri_dialogo_smooth_manuale()
        if nuovo is not None and nuovo != stato['smooth']:
            stato['smooth'] = nuovo
            disegna_grafico()

    elif key_norm == 'x':
        stato['xray_on'] = not stato['xray_on']
        disegna_grafico()

    elif key_norm == 'f':
        global _flare_visible
        _flare_visible = not _flare_visible
        print(f"Flare solari {'visibili' if _flare_visible else 'nascosti'}")
        disegna_grafico()

    elif key_norm == 'y':
        anno = apri_calendario_anno(datetime.now(timezone.utc).year)
        if anno is not None:
            mostra_elenco_geomag_per_anno(anno)

    elif key_norm == 'b' and shift_pressed:
        anno = apri_calendario_anno(datetime.now(timezone.utc).year)
        if anno is not None:
            mostra_elenco_tutti_grb_per_anno(anno)

    elif key_norm == 'b':
        anno = apri_calendario_anno(datetime.now(timezone.utc).year)
        if anno is not None:
            mostra_elenco_grb_per_anno(anno)

    elif key_norm == 'u':
        stato['scala_lineare'] = not stato['scala_lineare']
        disegna_grafico()

    elif key_norm == 'a':
        nuova_data = stato['data'] - timedelta(days=1)
        if trova_csv_per_data(nuova_data) is None:
            print(f"Nessun file CSV per {nuova_data.strftime('%Y-%m-%d')}, premi I per inserire una data.")
        else:
            stato['data'] = nuova_data
            disegna_grafico()

    elif key_norm == 'c':
        print("\n📊 Analisi correlazioni SID vs flare (tasto C)...")
        try:
            df = carica_dataset_completo()
            if df.empty:
                print("   ⚠️ Nessun dato disponibile. Addestra l'AI prima (tasto H).")
                return
            mostra_correlazioni_sid_flare(df)
        except Exception as e:
            print(f"   ❌ Errore durante l'analisi: {e}")
            import traceback
            traceback.print_exc()

    elif key_norm == 'd':
        nuova_data = stato['data'] + timedelta(days=1)
        if trova_csv_per_data(nuova_data) is None:
            print(f"Nessun file CSV per {nuova_data.strftime('%Y-%m-%d')}, premi I per inserire una data.")
        else:
            stato['data'] = nuova_data
            disegna_grafico()

    elif key_norm == 'i':
        nuova_data = apri_calendario(stato['data'])
        if nuova_data is not None:
            stato['data'] = nuova_data
            disegna_grafico()

    elif key_norm == 'm':
        parent = getattr(event.canvas, 'window', None)
        data_str = stato['data'].strftime("%Y-%m-%d")
        try:
            classe = tksd.askstring("Flare manuale", f"Classe flare (es. C5.3, M1.2, X1.0)\nData: {data_str}", parent=parent)
            if not classe:
                return
            classe = classe.strip().upper()
            ora_picco = tksd.askstring("Flare manuale", "Ora picco UTC (HH:MM):", parent=parent)
            if not ora_picco:
                return
            ora_picco = ora_picco.strip()
            ora_inizio = tksd.askstring("Flare manuale", f"Ora inizio UTC (HH:MM)\n[lascia vuoto per usare {ora_picco}]:", parent=parent)
            ora_inizio = ora_inizio.strip() if ora_inizio and ora_inizio.strip() else ora_picco
            ora_fine = tksd.askstring("Flare manuale", f"Ora fine UTC (HH:MM)\n[lascia vuoto per usare {ora_picco}]:", parent=parent)
            ora_fine = ora_fine.strip() if ora_fine and ora_fine.strip() else ora_picco
            nota = tksd.askstring("Flare manuale", "Nota opzionale:", parent=parent)
            nota = nota.strip() if nota else ""
            salva_flare_manuale(data_str, classe, ora_inizio, ora_picco, ora_fine, nota)
            disegna_grafico()
        except Exception as e:
            print(f"Errore inserimento flare manuale: {e}")

    elif key_norm == 't':
        print("\n🔍 Rilevazione SID in corso (modello AI su flare)...")
        if 'timestamps' not in stato or 'valori_grezzi' not in stato:
            print("⚠️ Nessun dato disponibile.")
            return

        modello_path = os.path.join(CARTELLA_RISPOSTE, "modello_sid.pkl")
        modello = sid_ai.carica_modello(path=modello_path)
        if modello is None:
            print("⚠️ Modello AI non trovato.")
            # Simula pressione H
            class FakeEvent:
                key = 'h'
            on_key(FakeEvent())
            return

        timestamps = stato['timestamps']
        valori = stato['valori_grezzi']
        data_utc = stato['data']

        flares_noaa = sid_ai.leggi_flare_cache(data_utc)
        flares_manuali = carica_flare_manuali()
        flares_manuali = [f for f in flares_manuali if f['picco'].date() == data_utc.date()]

        soglia_corrente = sid_ai.SOGLIA_PROBABILITA_DEFAULT
        print(f"   📊 Soglia probabilità: {soglia_corrente}")

        sid_detected = sid_ai.rileva_sid_con_modello(
            timestamps, valori, data_utc, modello,
            soglia_prob=soglia_corrente,
            flares_noaa=flares_noaa,
            flares_manuali=flares_manuali
        )

        for s in sid_detected:
            if 'flare' not in s or not s['flare']:
                s['flare'] = '—'
        flares_totali = list(flares_noaa) + flares_manuali
        mostra_tabella_sid(sid_detected, stato['data'], flares_totali)

        if sid_detected:
            print(f"   ✅ Trovati {len(sid_detected)} SID.")
        else:
            print("   ❌ Nessun SID rilevato.")

    elif key_norm == 'h':
        print("\n🧠 Addestramento AI interattivo...")
        if 'timestamps' not in stato or 'valori_grezzi' not in stato:
            print("⚠️ Nessun dato disponibile. Carica prima un giorno.")
            return

        timestamps = stato['timestamps']
        valori = stato['valori_grezzi']
        data_utc = stato['data']

        # Usa tksd per i dialog, parent = finestra principale (se disponibile)
        # parent robusto: gestisce sia eventi matplotlib reali sia FakeEvent()
        canvas = getattr(event, 'canvas', None)
        parent = getattr(canvas, 'window', None) if canvas is not None else None
        classe_soglia = tksd.askstring(
            "Soglia flare",
            "Inserisci la classe minima dei flare da considerare\n"
            "Esempi: C1, C2, M1, M5, X1\n"
            "(I flare inferiori a questa soglia verranno ignorati per il training)",
            parent=parent
        )
        if not classe_soglia:
            print("❌ Addestramento annullato.")
            return
        classe_soglia = classe_soglia.strip().upper()
        print(f"📌 Classe soglia: {classe_soglia}")

        flares = sid_ai.leggi_flare_cache(data_utc)
        if not flares:
            print("⚠️ Nessun flare NOAA trovato per questo giorno.")
            risp = tksd.askstring(
                "Nessun flare",
                "Nessun flare NOAA trovato per questo giorno.\nContinuare lo stesso? [S/N]",
                parent=parent
            )
            if not risp or risp.lower() != 's':
                print("❌ Addestramento annullato.")
                return
            flares_noaa = None
        else:
            flares_noaa = flares

        global modello_ai
        modello, risposte = sid_ai.addestra_interattivo(
            timestamps,
            valori,
            data_utc,
            classe_soglia,
            ax=None,
            fig=None,
            finestra_smooth=stato['smooth'],
            cartella_risposte=CARTELLA_RISPOSTE,
            usa_etichetta_sea=True
        )

        if modello is not None:
            modello_ai = modello
            print("✅ Modello AI addestrato e salvato con successo!")
        else:
            print("❌ Addestramento fallito o annullato.")

    elif key_norm == 'p':
        nuova_soglia = sid_ai.modifica_soglia_probabilita()
        print(f"📊 Soglia probabilità aggiornata a {nuova_soglia}")

    elif key_norm == 'l':
        print("\n📋 Caricamento flare di addestramento...")
        mostra_flare_addestrati(CARTELLA_RISPOSTE)

    elif key_norm == 'q':
        plt.close('all')

    elif key_norm == 'w' and shift_pressed:
        # Shift+W: cambia modalità di visualizzazione della fase
        #   detrended  → fase meno la sua mediana mobile (evidenzia i gradini SID)
        #   derivata   → dFase/dt (evidenzia i fronti ripdi dei SID)
        stato['fase_derivata'] = not stato.get('fase_derivata', False)
        modo = "derivata dFase/dt" if stato['fase_derivata'] else "fase detrended"
        print(f"Modalità fase: {modo}")
        if stato['fase_on']:
            disegna_grafico()

    elif key_norm == 'w':
        # W semplice: attiva/disattiva overlay fase
        if not any(f is not None for f in stato.get('fasi', [])):
            print("⚠️ Nessun dato di fase disponibile nel CSV di questo giorno.")
            return
        stato['fase_on'] = not stato['fase_on']
        print(f"Fase VLF: {'ON ✓' if stato['fase_on'] else 'OFF'}")
        disegna_grafico()

# =============================================================
# DIALOG E MAIN
# =============================================================
def apri_calendario(data_corrente):
    root = tk.Tk()
    root.title("GAESID — Scegli data")
    root.attributes('-topmost', True)
    root.resizable(False, False)
    root.configure(bg='#0d0d1a')
    root.geometry("420x380")
    tk.Label(root, text="Seleziona una data con dati disponibili", bg='#0d0d1a', fg='#00FF99',
             font=('Consolas', 10, 'bold')).pack(pady=(10, 4))
    cal = Calendar(root, selectmode='day', year=data_corrente.year, month=data_corrente.month,
                   day=data_corrente.day, date_pattern='yyyy-mm-dd', locale='it_IT',
                   background='#1a1a2e', foreground='white',
                   headersbackground='#2a2a4e', headersforeground='#00FF99',
                   selectbackground='#00FF99', selectforeground='black',
                   normalbackground='#0d0d1a', normalforeground='white',
                   weekendbackground='#1a1a2e', weekendforeground='#aaaacc',
                   othermonthforeground='#444466', othermonthbackground='#0d0d1a',
                   font=('Consolas', 9))
    today_date = datetime.now(timezone.utc).date()
    displayed_date = data_corrente.date()
    cal.calevent_create(today_date, 'Oggi', 'oggi')
    if displayed_date == today_date:
        cal.calevent_create(displayed_date, 'Oggi/Selezionato', ('oggi', 'corrente'))
    else:
        cal.calevent_create(displayed_date, 'Selezionato', 'corrente')
    cal.tag_config('oggi', background='#005500', foreground='white')
    cal.tag_config('corrente', background='#003399', foreground='white')
    cal.pack(padx=8, pady=6, fill='both', expand=True)
    lbl = tk.Label(root, text="", bg='#0d0d1a', fg='#FF6347', font=('Consolas', 9))
    lbl.pack(pady=(0, 4))
    risultato = {'data': None}

    def conferma():
        data_str = cal.get_date()
        try:
            data = datetime.strptime(data_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return
        if trova_csv_per_data(data) is None:
            lbl.config(text=f"Nessun dato per {data_str}.")
            return
        risultato['data'] = data
        root.destroy()

    def annulla():
        root.destroy()

    frame_btn = tk.Frame(root, bg='#0d0d1a')
    frame_btn.pack(pady=(0, 12))
    tk.Button(frame_btn, text="Conferma", command=conferma, bg='#00FF99', fg='black',
              font=('Consolas', 10, 'bold'), relief='flat', padx=12, pady=4).pack(side='left', padx=8)
    tk.Button(frame_btn, text="Annulla", command=annulla, bg='#333355', fg='white',
              font=('Consolas', 10), relief='flat', padx=12, pady=4).pack(side='left', padx=8)
    root.protocol("WM_DELETE_WINDOW", annulla)
    root.wait_window(root)
    return risultato['data']

def apri_calendario_anno(anno_corrente):
    root = tk.Tk()
    root.title("GAESID — Scegli anno")
    root.attributes('-topmost', True)
    root.resizable(False, False)
    root.configure(bg='#0d0d1a')
    root.geometry("320x220")
    tk.Label(root, text="Seleziona l'anno da cercare", bg='#0d0d1a', fg='#00FF99',
             font=('Consolas', 11, 'bold')).pack(pady=(14, 8))
    frame = tk.Frame(root, bg='#0d0d1a')
    frame.pack(pady=(0, 10))
    tk.Label(frame, text="Anno:", bg='#0d0d1a', fg='white', font=('Consolas', 10)).pack(side='left', padx=(0, 8))
    anno_max = datetime.now(timezone.utc).year
    year_spin = tk.Spinbox(frame, from_=2000, to=anno_max, width=6, justify='center',
                           font=('Consolas', 12), bg='#1a1a2e', fg='white',
                           insertbackground='white', relief='flat')
    year_spin.delete(0, 'end')
    year_spin.insert(0, str(min(anno_corrente, anno_max)))
    year_spin.pack(side='left')
    lbl = tk.Label(root, text="", bg='#0d0d1a', fg='#FF6347', font=('Consolas', 9))
    lbl.pack(pady=(0, 6))
    risultato = {'anno': None}

    def conferma():
        try:
            anno = int(year_spin.get())
            risultato['anno'] = anno
            root.destroy()
        except ValueError:
            lbl.config(text="Inserisci un anno valido")

    def annulla():
        root.destroy()

    frame_btn = tk.Frame(root, bg='#0d0d1a')
    frame_btn.pack(pady=(0, 12))
    tk.Button(frame_btn, text="Conferma", command=conferma, bg='#00FF99', fg='black',
              font=('Consolas', 10, 'bold'), relief='flat', padx=12, pady=4).pack(side='left', padx=10)
    tk.Button(frame_btn, text="Annulla", command=annulla, bg='#333355', fg='white',
              font=('Consolas', 10), relief='flat', padx=12, pady=4).pack(side='left', padx=10)
    root.protocol("WM_DELETE_WINDOW", annulla)
    root.wait_window(root)
    return risultato['anno']

def scegli_data_iniziale():
    return apri_calendario(datetime.now(timezone.utc))

VALORI_SMOOTH_STANDARD = [1, 15, 30, 60, 90, 120]

def scegli_smoothing_iniziale():
    root = tk.Tk()
    root.title("GAESID — Smoothing Iniziale")
    root.attributes('-topmost', True)
    root.resizable(False, False)
    root.configure(bg='#0d0d1a')
    tk.Label(root, text="Scegli la finestra di smoothing (secondi)", bg='#0d0d1a', fg='#00FF99',
             font=('Consolas', 11, 'bold')).pack(pady=(14, 6), padx=20)
    risultato = {'valore': 60}
    frame_std = tk.Frame(root, bg='#0d0d1a')
    frame_std.pack(pady=10, padx=20)
    etichette = {1: "Nessuno\n(1 s)", 15: "15 s", 30: "30 s", 60: "60 s", 90: "90 s", 120: "120 s"}
    btn_sel = {'btn': None}

    def seleziona(val, btn):
        if btn_sel['btn']:
            btn_sel['btn'].config(bg='#1e1e3a', fg='white', relief='flat')
        btn.config(bg='#00FF99', fg='black', relief='sunken')
        btn_sel['btn'] = btn
        risultato['valore'] = val

    for val in VALORI_SMOOTH_STANDARD:
        b = tk.Button(frame_std, text=etichette.get(val, str(val)), width=9, height=2,
                      bg='#1e1e3a', fg='white', font=('Consolas', 10), relief='flat', bd=1)
        b.pack(side='left', padx=5)
        b.config(command=lambda v=val, btn=b: seleziona(v, btn))
        if val == 60:
            seleziona(60, b)

    tk.Button(root, text="Conferma", command=root.destroy, bg='#00FF99', fg='black',
              font=('Consolas', 11, 'bold'), padx=20, pady=8).pack(pady=15)
    root.mainloop()
    return risultato['valore']

def apri_dialogo_smooth_lista():
    root = tk.Tk()
    root.title("GAESID — Smoothing")
    root.attributes('-topmost', True)
    root.resizable(False, False)
    root.configure(bg='#0d0d1a')
    corrente = stato['smooth']
    tk.Label(root, text=f"Scegli smoothing\n(Attuale: {corrente} s)", bg='#0d0d1a', fg='#00FF99',
             font=('Consolas', 11, 'bold')).pack(pady=12)
    risultato = {'valore': None}
    btn_attivo = {'btn': None}
    frame = tk.Frame(root, bg='#0d0d1a')
    frame.pack(pady=8, padx=20)
    etichette = {1: "Nessuno\n(1 s)", 15: "15 s", 30: "30 s", 60: "60 s", 90: "90 s", 120: "120 s"}

    def seleziona(val, btn):
        if btn_attivo['btn'] is not None:
            btn_attivo['btn'].config(bg='#1e1e3a', fg='white', relief='flat')
        btn.config(bg='#00FF99', fg='black', relief='sunken')
        btn_attivo['btn'] = btn
        risultato['valore'] = val

    for val in VALORI_SMOOTH_STANDARD:
        bg_i = '#00FF99' if val == corrente else '#1e1e3a'
        fg_i = 'black' if val == corrente else 'white'
        b = tk.Button(frame, text=etichette.get(val, str(val)), width=9, height=2,
                      bg=bg_i, fg=fg_i, font=('Consolas', 10), relief='flat')
        b.pack(side='left', padx=4)
        b.config(command=lambda v=val, btn=b: seleziona(v, btn))

    def annulla():
        risultato['valore'] = None
        root.destroy()

    frame_btn = tk.Frame(root, bg='#0d0d1a')
    frame_btn.pack(pady=(0, 12))
    tk.Button(frame_btn, text="Conferma", command=root.destroy, bg='#00FF99', fg='black',
              font=('Consolas', 10, 'bold'), relief='flat', padx=12, pady=4).pack(side='left', padx=8)
    tk.Button(frame_btn, text="Annulla", command=annulla, bg='#333355', fg='white',
              font=('Consolas', 10), relief='flat', padx=12, pady=4).pack(side='left', padx=8)
    root.wait_window(root)
    return risultato['valore']

def apri_dialogo_smooth_manuale():
    root = tk.Tk()
    root.title("GAESID — Smoothing manuale")
    root.attributes('-topmost', True)
    root.resizable(False, False)
    root.configure(bg='#0d0d1a')
    corrente = stato['smooth']
    tk.Label(root, text="Inserisci il valore (secondi)", bg='#0d0d1a', fg='#00FF99',
             font=('Consolas', 11, 'bold')).pack(pady=(14, 4), padx=20)
    risultato = {'valore': None}
    frame_e = tk.Frame(root, bg='#0d0d1a')
    frame_e.pack(pady=(0, 6), padx=20)
    entry = tk.Entry(frame_e, width=8, bg='#1e1e3a', fg='white', insertbackground='white',
                     font=('Consolas', 14), justify='center', relief='flat')
    entry.insert(0, str(corrente))
    entry.selection_range(0, tk.END)
    entry.pack(side='left')
    entry.focus_set()
    lbl_err = tk.Label(root, text="", bg='#0d0d1a', fg='#FF6347', font=('Consolas', 9))
    lbl_err.pack(pady=(0, 4))

    def conferma():
        try:
            val = int(entry.get().strip())
            if val < 1:
                lbl_err.config(text="Il valore deve essere >= 1")
                return
            risultato['valore'] = val
            root.destroy()
        except ValueError:
            lbl_err.config(text="Inserisci un numero intero valido")

    frame_btn = tk.Frame(root, bg='#0d0d1a')
    frame_btn.pack(pady=(0, 14))
    tk.Button(frame_btn, text="Conferma", command=conferma, bg='#00FF99', fg='black',
              font=('Consolas', 10, 'bold'), relief='flat', padx=14, pady=5).pack(side='left', padx=6)
    tk.Button(frame_btn, text="Annulla", command=root.destroy, bg='#333355', fg='white',
              font=('Consolas', 10), relief='flat', padx=10, pady=5).pack(side='left', padx=6)
    root.bind('<Return>', lambda e: conferma())
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.wait_window(root)
    return risultato['valore']

# =============================================================
# ANALISI CORRELAZIONI SID vs FLARE (tasto C)
# =============================================================
def _calcola_sza(dt_utc, lat=_PATH_MID_LAT, lon=_PATH_MID_LON):
    """
    Calcola la SZA (Solar Zenith Angle) in gradi al tempo dt_utc
    per il punto medio del path NSY-GAESID.
    0° = Sole allo zenit, 90° = orizzonte, >90° = sotto l'orizzonte.
    """
    try:
        obs = ephem.Observer()
        obs.lat = str(lat)
        obs.lon = str(lon)
        obs.elevation = 100
        obs.pressure = 0
        obs.date = dt_utc.strftime('%Y/%m/%d %H:%M:%S')
        sole = ephem.Sun()
        sole.compute(obs)
        alt = math.degrees(sole.alt)   # altitudine in gradi
        return 90.0 - alt               # SZA = 90 - altitudine
    except Exception:
        return None


def _log_flux_xray_al_picco(picco_dt, data_utc, xray_ts, xray_flux):
    """
    Interpola il log10 del flusso X-ray al tempo del picco del flare.
    Restituisce log10(flux) oppure None se non disponibile.
    """
    if not xray_ts or not xray_flux:
        return None
    try:
        t_target = picco_dt.timestamp()
        ts_float = np.array([t.timestamp() for t in xray_ts])
        fl_array = np.array(xray_flux)
        # Se il picco è fuori dal range disponibile, ritorna None
        if t_target < ts_float[0] or t_target > ts_float[-1]:
            return None
        flux = float(np.interp(t_target, ts_float, fl_array))
        if flux <= 0:
            return None
        return float(np.log10(flux))
    except Exception:
        return None


def carica_dataset_completo(cartella_risposte=CARTELLA_RISPOSTE,
                             cartella_xray=CARTELLA_XRAY):
    """
    Costruisce un DataFrame con tutte le risposte AI salvate, arricchite con:
      - log10(flux X-ray) al picco del flare
      - SZA al picco del flare
      - feature dell'AI già salvate (ampiezza, durata, area, ora,
        pend_salita, pend_discesa, simmetria)
      - classe del flare (lettera: C, M, X)
      - etichetta (1=SID, 0=NON SID)

    Restituisce un pandas.DataFrame. Se non trova nulla, restituisce un DataFrame vuoto.
    """
    import glob

    pattern = os.path.join(cartella_risposte, "risposte_flare_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[CORR] Nessun file risposte_flare_*.json in {cartella_risposte}")
        return pd.DataFrame()

    righe = []
    for fp in files:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                dati = json.load(f)
        except Exception as e:
            print(f"[CORR] Errore lettura {os.path.basename(fp)}: {e}")
            continue

        # La data è nel nome file: risposte_flare_YYYY-MM-DD.json
        try:
            data_str = os.path.basename(fp).replace("risposte_flare_", "").replace(".json", "")
            data_utc = datetime.strptime(data_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            continue

        # Carica X-ray del giorno (una volta sola per file)
        xray_ts, xray_flux = leggi_xray_locale(data_utc)

        for r in dati.get('risposte', []):
            try:
                classe = r.get('classe', '')
                picco_iso = r.get('picco')
                etichetta = r.get('etichetta')
                features = r.get('features')

                if picco_iso is None or etichetta is None or features is None:
                    continue
                if not classe:
                    continue

                picco_dt = datetime.fromisoformat(picco_iso)
                if picco_dt.tzinfo is None:
                    picco_dt = picco_dt.replace(tzinfo=timezone.utc)

                # Features: [ampiezza, durata, area, ora, pend_salita, pend_discesa, simmetria]
                if not isinstance(features, list) or len(features) < 7:
                    continue

                # SZA
                sza = _calcola_sza(picco_dt)

                # log10(flux X-ray) al picco
                log_flux = _log_flux_xray_al_picco(picco_dt, data_utc, xray_ts, xray_flux)

                righe.append({
                    'data': data_str,
                    'picco': picco_dt,
                    'classe': classe,
                    'tipo': classe[0].upper() if classe else '?',
                    'etichetta': int(etichetta),
                    'ampiezza': float(features[0]),
                    'durata': float(features[1]),
                    'area': float(features[2]),
                    'ora': float(features[3]),
                    'pend_salita': float(features[4]),
                    'pend_discesa': float(features[5]),
                    'simmetria': float(features[6]),
                    'sza': sza,
                    'log_flux_xray': log_flux,
                })
            except Exception as e:
                print(f"[CORR] Riga scartata ({e})")
                continue

    df = pd.DataFrame(righe)
    print(f"[CORR] Dataset: {len(df)} flare totali da {len(files)} file")
    if not df.empty:
        n_sid = int((df['etichetta'] == 1).sum())
        n_no  = int((df['etichetta'] == 0).sum())
        print(f"[CORR] SID veri: {n_sid} | NON SID: {n_no}")
        n_sza = int(df['sza'].notna().sum())
        n_xr  = int(df['log_flux_xray'].notna().sum())
        print(f"[CORR] Con SZA valida: {n_sza} | con X-ray valido: {n_xr}")
    return df

def mostra_correlazioni_sid_flare(df=None):
    """
    Mostra 4 scatter plot con regressione lineare:
      1) ampiezza SID vs log10(flux X-ray)
      2) RESIDUI (ampiezza - fit pannello 1) vs SZA
      3) simmetria vs SZA
      4) pendenza di discesa vs SZA (proxy di tau)

    Usa come colore la classe del flare (C, M, X).
    Mostra R² e pendenza nel titolo di ciascun pannello.

    Stampa anche:
      - distribuzione dei punti per fasce di SZA (Test A)
      - coefficienti del fit usato per i residui (Test B)
      - p-value della regressione residui vs SZA (Test di significatività)
    """
    if df is None:
        df = carica_dataset_completo()

    if df.empty:
        print("[CORR] Dataset vuoto: impossibile generare i grafici.")
        return

    # Consideriamo solo i SID veri (etichetta=1) per le regressioni
    df_sid = df[df['etichetta'] == 1].copy()
    if df_sid.empty:
        print("[CORR] Nessun SID positivo nel dataset.")
        return

    # Colore per classe
    colori = {'C': '#FFD700', 'M': '#FF8C00', 'X': '#FF0000'}
    def colore_riga(t):
        return colori.get(t, '#00BFFF')

    fig, assi = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0d0d1a')
    for ax in assi.flat:
        ax.set_facecolor('#0d0d1a')
        ax.tick_params(colors='white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#444444')
        ax.grid(True, alpha=0.15, color='white')

    def scatter_con_reg(ax, x, y, df_source, xlabel, ylabel, titolo):
        """
        Scatter + regressione lineare con R² e pendenza nel titolo.
        df_source: DataFrame da cui provengono x e y (per recuperare 'tipo').
        x, y: pandas.Series con stesso indice di df_source.
        """
        # Maschera costruita sull'indice di x
        mask = x.notna() & y.notna()
        n_validi = int(mask.sum())
        if n_validi < 3:
            ax.text(0.5, 0.5, f"Dati insufficienti (n={n_validi})",
                    transform=ax.transAxes, color='#FF6347',
                    ha='center', va='center', fontsize=12)
            ax.set_title(titolo, color='white', fontsize=11)
            ax.set_xlabel(xlabel, color='white')
            ax.set_ylabel(ylabel, color='white')
            return

        # Prendi solo le righe valide, allineate tra x, y e df_source
        xv = x[mask].values
        yv = y[mask].values
        indici_validi = x[mask].index
        df_filtrato = df_source.loc[indici_validi]

        # Colorazione per classe
        if 'tipo' in df_filtrato.columns:
            tipi = df_filtrato['tipo'].values
        else:
            tipi = ['?'] * len(xv)
        cols = [colore_riga(t) for t in tipi]
        ax.scatter(xv, yv, c=cols, s=35, alpha=0.8, edgecolors='white', linewidths=0.4)

        # Regressione
        try:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(xv, yv)
            xx = np.linspace(xv.min(), xv.max(), 100)
            ax.plot(xx, slope * xx + intercept, color='#00FF99',
                    linewidth=1.5, alpha=0.9)
            titolo_full = f"{titolo}\nR²={r_value**2:.3f}  pend={slope:.3f}  n={n_validi}"
        except Exception:
            titolo_full = f"{titolo}\n(regressione non disponibile)  n={n_validi}"

        ax.set_title(titolo_full, color='white', fontsize=10)
        ax.set_xlabel(xlabel, color='white', fontsize=9)
        ax.set_ylabel(ylabel, color='#00FF99', fontsize=9)

    # --- Pannello 1: ampiezza vs log_flux_xray ---
    scatter_con_reg(
        assi[0, 0],
        df_sid['log_flux_xray'],
        df_sid['ampiezza'],
        df_sid,
        'log10(flux X-ray) [W/m²]',
        'Ampiezza SID (dB)',
        'Ampiezza SID vs Flusso X-ray'
    )

    # =========================================================
    # >>> TEST A: distribuzione dei punti per fascia di SZA <<<
    # =========================================================
    n_sza_bassi = int((df_sid['sza'] < 50).sum())
    n_sza_medi  = int(((df_sid['sza'] >= 50) & (df_sid['sza'] < 70)).sum())
    n_sza_alti  = int((df_sid['sza'] >= 70).sum())
    n_tot       = int(df_sid['sza'].notna().sum())
    print(f"[CORR] Distribuzione SZA (n totale = {n_tot}):")
    print(f"[CORR]   <50°  : {n_sza_bassi} punti")
    print(f"[CORR]   50-70°: {n_sza_medi} punti")
    print(f"[CORR]   >70°  : {n_sza_alti} punti")
    if n_sza_alti < 20:
        print(f"[CORR]   ⚠️ Meno di 20 punti oltre 70°: eventuali effetti SZA in quella fascia")
        print(f"[CORR]      saranno statisticamente fragili.")
    # =========================================================

    # =========================================================
    # >>> TEST B: RESIDUI della regressione ampiezza~log_flux <<<
    # =========================================================
    # Se il modello vero è ampiezza = a + b * log10(flux), il modo
    # corretto di "controllare per" il flusso NON è dividere per il
    # flusso lineare, ma sottrarre la parte di ampiezza spiegata dal
    # flusso (residui). I residui sono la parte di ampiezza non spiegata
    # dal flusso X-ray: se la SZA conta, il residuo deve mostrare una
    # tendenza con SZA.
    # =========================================================
    df_sid_p2 = df_sid.copy()
    mask_fit = df_sid_p2['log_flux_xray'].notna() & df_sid_p2['ampiezza'].notna()
    n_fit = int(mask_fit.sum())

    if n_fit >= 3:
        from scipy.stats import linregress
        x_fit = df_sid_p2.loc[mask_fit, 'log_flux_xray'].values
        y_fit = df_sid_p2.loc[mask_fit, 'ampiezza'].values
        slope_fit, intercept_fit, r_val_fit, _, _ = linregress(x_fit, y_fit)

        # Calcolo dei residui (solo per i punti con log_flux valido)
        df_sid_p2['residuo'] = np.nan
        df_sid_p2.loc[mask_fit, 'residuo'] = (
            df_sid_p2.loc[mask_fit, 'ampiezza']
            - (intercept_fit + slope_fit * df_sid_p2.loc[mask_fit, 'log_flux_xray'])
        )

        print(f"[CORR] Fit per residui: ampiezza = {intercept_fit:.3f} "
              f"+ {slope_fit:.3f} * log_flux  (R²={r_val_fit**2:.3f}, n={n_fit})")

        # --- Test di significatività formale: residui vs SZA ---
        mask_res = df_sid_p2['residuo'].notna() & df_sid_p2['sza'].notna()
        if mask_res.sum() >= 3:
            slope_r, intercept_r, r_val_r, p_value_r, _ = linregress(
                df_sid_p2.loc[mask_res, 'sza'].values,
                df_sid_p2.loc[mask_res, 'residuo'].values
            )
            print(f"[CORR] Residui vs SZA: pendenza={slope_r:.4f}, "
                  f"R²={r_val_r**2:.4f}, p-value={p_value_r:.4f}")
            if p_value_r < 0.05:
                print(f"[CORR] ✅ Effetto SZA statisticamente significativo (p < 0.05)")
            else:
                print(f"[CORR] ⚠️ Effetto SZA non significativo (p = {p_value_r:.4f})")

        scatter_con_reg(
            assi[0, 1],
            df_sid_p2['sza'],
            df_sid_p2['residuo'],
            df_sid_p2,
            'SZA al picco (°)',
            'Residuo ampiezza (dB) — parte non spiegata dal flusso',
            'Residui SID vs SZA (controllo su flusso X-ray)'
        )
    else:
        print(f"[CORR] Pannello 2 saltato: troppi pochi punti validi per il fit (n={n_fit}).")
        assi[0, 1].text(0.5, 0.5, f"Dati insufficienti (n={n_fit})",
                        transform=assi[0, 1].transAxes, color='#FF6347',
                        ha='center', va='center', fontsize=12)
        assi[0, 1].set_title('Residui SID vs SZA', color='white', fontsize=11)
    # =========================================================

    # --- Pannello 3: simmetria vs SZA ---
    scatter_con_reg(
        assi[1, 0],
        df_sid['sza'],
        df_sid['simmetria'],
        df_sid,
        'SZA al picco (°)',
        'Simmetria (pend_salita / pend_discesa)',
        'Simmetria SID vs SZA'
    )

    # --- Pannello 4: pendenza discesa vs SZA (proxy di tau) ---
    scatter_con_reg(
        assi[1, 1],
        df_sid['sza'],
        df_sid['pend_discesa'],
        df_sid,
        'SZA al picco (°)',
        'Pendenza di discesa (dB/campione)',
        'Pendenza di discesa vs SZA (proxy di τ)'
    )

    # Legenda colori
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colori['C'],
               markersize=8, label='Classe C'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colori['M'],
               markersize=8, label='Classe M'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colori['X'],
               markersize=8, label='Classe X'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               facecolor='#1a1a2e', edgecolor='#444444',
               labelcolor='white', fontsize=10)

    fig.suptitle(f"Correlazioni SID vs Flare — {len(df_sid)} SID su {len(df)} flare etichettati",
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.show()
     
# =============================================================
# MAIN
# =============================================================
print("=" * 60)
print("  VLF SID Monitor — NSY 45.9 kHz — Stazione GAESID")
print("  con integrazione GRB + Geomagnetismo e protoni >10MeV")
print("=" * 60)
print()
print("  COMANDI DA TASTIERA (clicca sul grafico prima):")
print("  A  -> Giorno precedente")
print("  D  -> Giorno successivo")
print("  G  -> Attiva/Disattiva overlay dei GRB")
print("  I  -> Calendario per scegliere la data")
print("  M  -> Inserisci flare manuale (beyond-the-limb)")
print("  F  -> Mostra/Nascondi i flare solari")
print("  Y  -> Anno eventi geomagnetici (tempeste, CME con arrivo)")
print("  P  -> Modifica Soglia Probabilità AI")
print("  R  -> Refresh Grafico, Flares e X-ray")
print("  S  -> Cambia smoothing (lista valori standard)")
print("  U  -> Cambia scala asse Y (dB <-> Lineare)")
print("  V  -> Cambia smoothing (valore libero manuale)")
print("  X  -> Attiva/Disattiva overlay flusso X-ray GOES")
print("  B  -> Anno GRB (solo GRB potenzialmente visibili)")
print("  H  -> Addestramento AI interattivo (sul giorno corrente)")
print("  T  -> Rileva SID con modello AI (premi H prima per addestrare)")
print("  C  -> Analisi correlazioni SID vs flare (scatter plot)")
print("  P  -> Modifica soglia probabilità (default 0.5)")
print("  Shift+B -> Anno GRB (tutti i GRB)")
print("  L  -> Mostra tabella flare addestrati (con etichette)")
print("  W  -> Attiva/Disattiva overlay fase VLF (terza colonna CSV)")
print("  Shift+W -> Cambia modalità fase (detrended <-> derivata)")
print()

data_scelta = scegli_data_iniziale()
if data_scelta is None:
    data_scelta = datetime.now(timezone.utc)

finestra_smooth = scegli_smoothing_iniziale()

try:
    _root_tk = tk.Tk()
    _root_tk.withdraw()
    _root_tk.update_idletasks()
    _SW = _root_tk.winfo_screenwidth()
    _SH = _root_tk.winfo_screenheight()
    _root_tk.destroy()
except Exception:
    _SW, _SH = 1920, 1080

_WH = int(_SH * 0.5)

fig, ax = plt.subplots(figsize=(_SW / 100, _WH / 100))
fig.patch.set_facecolor('#0d0d1a')
ax.set_facecolor('#0d0d1a')

try:
    if hasattr(fig.canvas.manager, 'toolbar'):
        fig.canvas.manager.toolbar.pack(side='bottom')
        fig.canvas.manager.toolbar.config(relief='sunken')
except Exception:
    pass

for tasto in ['p','q','s','v','c','a','d','h','r','k','l','o','f','x','u','y','w','b']:
    for keymap in list(plt.rcParams.keys()):
        if keymap.startswith('keymap.') and tasto in plt.rcParams[keymap]:
            try:
                plt.rcParams[keymap].remove(tasto)
            except:
                pass

stato['data']   = data_scelta
stato['smooth'] = finestra_smooth
stato['fig']    = fig
stato['ax']     = ax

modello_ai = sid_ai.carica_modello()
if modello_ai is not None:
    print("✅ Modello AI caricato.")
else:
    print("ℹ️ Nessun modello AI trovato. Premi H per addestrare.")

fig.canvas.mpl_connect('key_press_event', on_key)

disegna_grafico()
plt.tight_layout()

_manager = plt.get_current_fig_manager()
if hasattr(_manager, 'window'):
    _win = _manager.window
    try:
        _win.geometry(f"{_SW}x{_WH}+0+{_SH - _WH}")
    except:
        pass
    if hasattr(_win, 'bind'):
        def _salva_geometria(event=None):
            try:
                geo = _win.geometry()
                if not hasattr(_salva_geometria, 'geo'):
                    _salva_geometria.geo = geo
                else:
                    _salva_geometria.geo = geo
            except:
                pass
        def _applica_geometria(event=None):
            try:
                if hasattr(_salva_geometria, 'geo'):
                    _win.geometry(_salva_geometria.geo)
            except:
                pass
        _win.bind('<Configure>', _salva_geometria, add='+')
        _win.bind('<Map>', _applica_geometria, add='+')
        _win.after(50, lambda: _applica_geometria())
    else:
        print("⚠️ Impossibile gestire il ridimensionamento automatico della finestra.")

plt.show()