import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.widgets as mwidgets
import numpy as np
from datetime import datetime, timezone, timedelta
import re
import os
import json
import ephem
from scipy.signal import savgol_filter
import tkinter as tk
from tkcalendar import Calendar
from tkinter import simpledialog

# --- Import aggiuntivi per modulo GRB ---
import xml.etree.ElementTree as ET
import math


# =============================================================
# CONFIGURAZIONE
# =============================================================

# Directory dove cercare i file CSV — lo script cerca in entrambe,
# dando priorità alla cartella giornaliera se il file esiste in entrambe.
CARTELLA_GIORNALIERA = r"D:\Gaetano\SID\dati\Sdruno"
CARTELLA_STORICO     = r"D:\Gaetano\SID\dati\Sdruno\Storico_NSY"

# Cartella dove salvare i file CSV del flusso X-ray GOES
CARTELLA_XRAY   = r"D:\Gaetano\SID\dati\Sdruno\Storico_NSY\XRAY"

# Cartella dove salvare i file JSON dei flare in cache locale
CARTELLA_FLARES = r"D:\Gaetano\SID\dati\Sdruno\Storico_NSY\Flares"

# File JSON per i flare inseriti manualmente
FILE_FLARE_MANUALI = os.path.join(CARTELLA_STORICO, "flare_manuali.json")

# Colori per le classi di flare
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
# SEZIONE GRB — costanti e stato globale
# ============================================================

# Geometria path NSY (Niscemi) → GAESID (Torino)
_NSY_LAT,    _NSY_LON    =  37.08,  14.44
_GAESID_LAT, _GAESID_LON =  45.07,   7.68
_PATH_MID_LAT = (_NSY_LAT + _GAESID_LAT) / 2   # ≈ 41.1 °N
_PATH_MID_LON = (_NSY_LON + _GAESID_LON) / 2   # ≈ 11.1 °E

# Soglie di fluenza Fermi GBM (erg/cm², banda 10–1000 keV)
# Per riferimento: GRB221009A ("BOAT") ≈ 0.21 erg/cm²
# Tutti i GRB con fluence > 1e-4 erg/cm² sono considerati possibili rilevabili dalla stazione SID.
_FLUENZA_ECCEZIONALE = 1.0e-4
_FLUENZA_FORTE       = 1.0e-5
_FLUENZA_DEBOLE      = 1.0e-6

# Elevazione minima della sorgente GRB sull'orizzonte del path:
# i raggi gamma penetrano leggermente il limbo terrestre (~10°)
_GRB_ELEV_MIN = -10.0

# Elevazione solare sotto cui il path è considerato "notturno"
# (sotto il crepuscolo civile)
_ELEV_NOTTE = -6.0

# Cache locale per non interrogare HEASARC più volte per la stessa data
_GRB_CACHE_DIR = os.path.join(CARTELLA_STORICO, 'grb_cache')
os.makedirs(_GRB_CACHE_DIR, exist_ok=True)

# Stato del modulo GRB (variabili globali)
_grb_all         = []    # tutti i GRB grezzi recuperati per il giorno
_grb_ril         = []    # GRB classificati come potenzialmente rilevabili
_grb_visible     = False # toggle tasto G
_flare_visible   = True  # toggle tasto F, flare solari visibili di default
_grb_artists     = []    # artisti matplotlib correnti (linee + etichette)
_grb_status_text = None  # oggetto Text matplotlib in basso a destra

# ============================================================
#  COMANDI DA TASTIERA
# ============================================================
#  A  →  Giorno precedente
#  D  →  Giorno successivo
#  G  →  Attiva/Disattiva marker GRB
#  I  →  Inserisci nuova data (calendario)
#  M  →  Inserisci flare manuale (beyond-the-limb)
#  P  →  Stampa/salva il grafico corrente come PNG
#  R  →  Aggiorna Grafico e Flares
#  S  →  Cambia smoothing (lista valori standard)
#  U  →  Cambia scala asse Y (dB ↔ Lineare)
#  V  →  Cambia smoothing (valore libero manuale)
#  X  →  Attiva/Disattiva overlay flusso X-ray GOES
#  Q  →  Esci dal programma
# ============================================================


# =============================================================
# SMOOTHING
# =============================================================

def applica_smoothing(valori, finestra):
    if finestra <= 1:
        return valori
    if finestra % 2 == 0:
        finestra += 1
    if finestra < 5:
        finestra = 5
    try:
        return savgol_filter(valori, window_length=finestra, polyorder=3).tolist()
    except Exception:
        kernel   = np.ones(finestra) / finestra
        smoothed = np.convolve(valori, kernel, mode='same')
        meta = finestra // 2
        if len(smoothed) > meta * 2:
            smoothed[:meta]  = smoothed[meta]
            smoothed[-meta:] = smoothed[-meta - 1]
        return smoothed.tolist()


# =============================================================
# ALBA / TRAMONTO
# =============================================================

def calcola_alba_tramonto(data_utc):
    osservatore           = ephem.Observer()
    osservatore.lat       = '45.07'
    osservatore.lon       = '7.68'
    osservatore.elevation = 240
    osservatore.date      = data_utc.strftime('%Y/%m/%d 00:00:00')
    osservatore.pressure  = 0
    sole = ephem.Sun()
    try:
        alba     = osservatore.next_rising(sole).datetime().replace(tzinfo=timezone.utc)
        tramonto = osservatore.next_setting(sole).datetime().replace(tzinfo=timezone.utc)
    except Exception as e:
        print(f"Errore calcolo alba/tramonto: {e}")
        alba, tramonto = None, None
    return alba, tramonto


# =============================================================
# FLARE MANUALI (beyond-the-limb o non in catalogo GOES)
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
    print(f"File: {FILE_FLARE_MANUALI}")

def trova_csv_per_data(data_utc):
    data_str  = data_utc.strftime("%Y-%m-%d")
    nome_file = f"GAESID_NSY_{data_str}.csv"
    for cartella in [CARTELLA_GIORNALIERA, CARTELLA_STORICO]:
        path = os.path.join(cartella, nome_file)
        if os.path.exists(path):
            print(f"File CSV trovato: {path}")
            return path
    print(f"File CSV non trovato in nessuna directory per {data_str}")
    return None

# =============================================================
# DOWNLOAD E CACHE FLUSSO X-RAY GOES
# =============================================================

def path_csv_xray(data_utc):
    """Restituisce il path del file CSV locale per il flusso X-ray."""
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
                    if val > 0:   # ignora valori non validi
                        timestamps.append(dt)
                        flux.append(val)
                except Exception:
                    continue
        print(f"X-ray locale: {len(timestamps)} campioni letti da {path}")
    except Exception as e:
        print(f"Errore lettura CSV X-ray locale: {e}")
    return timestamps, flux


def salva_xray_locale(data_utc, timestamps, flux):
    path = path_csv_xray(data_utc)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# GOES X-ray flux (0.1-0.8 nm, canale B) — scaricato da NOAA/SWPC\n")
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
    CLASSE_BASE = {
        'C': 1e-6,
        'M': 1e-5,
        'X': 1e-4,
    }
    BACKGROUND = 5e-8   # livello B5 tipico attività solare moderata

    inizio_giorno = datetime.combine(
        data_utc.date(), datetime.min.time()
    ).replace(tzinfo=timezone.utc)

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

            durata_rise  = max((t_picco  - t_inizio).total_seconds(), 60.0)
            durata_decay = max((t_fine   - t_picco ).total_seconds(), 60.0)

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


def scarica_xray(data_utc):
    oggi_utc = datetime.now(timezone.utc).date()
    delta    = (oggi_utc - data_utc.date()).days

    # 1. Cache locale
    if delta > 0:
        ts, fl = leggi_xray_locale(data_utc)
        if ts:
            return ts, fl

    # 2. JSON real-time (ultimi ~6 giorni)
    if delta <= 6:
        for sat in ("primary", "secondary"):
            url = f"https://services.swpc.noaa.gov/json/goes/{sat}/xrays-7-day.json"
            print(f"Download X-ray da: {url}")
            try:
                r = requests.get(url, timeout=(8, 20))
                r.raise_for_status()
                timestamps, flux = _parse_xray_json_rt(r.text, data_utc)
                if timestamps:
                    print(f"X-ray scaricato: {len(timestamps)} campioni (json_rt/{sat})")
                    salva_xray_locale(data_utc, timestamps, flux)
                    return timestamps, flux
                else:
                    print(f"  Nessun campione per {data_utc.strftime('%Y-%m-%d')} da {sat}")
            except requests.exceptions.HTTPError as e:
                print(f"  Errore HTTP {e.response.status_code} ({sat}), passo al prossimo...")
            except Exception as e:
                print(f"  Errore ({sat}): {e}")

    # 3. Dati storici: DONKI → ricostruzione curva
    print(f"X-ray storico: ricostruzione da flare DONKI per {data_utc.strftime('%Y-%m-%d')}...")
    data_str = data_utc.strftime("%Y-%m-%d")
    url_donki = f"https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/FLR?startDate={data_str}&endDate={data_str}"
    try:
        r = requests.get(url_donki, timeout=(8, 15))
        r.raise_for_status()
        flares_donki = json.loads(r.text)
        flares_giorno = [
            f for f in flares_donki
            if f.get('peakTime', '').startswith(data_str) or
               f.get('beginTime', '').startswith(data_str)
        ]
        print(f"  {len(flares_giorno)} flare DONKI trovati per la ricostruzione X-ray")
        timestamps, flux = _ricostruisci_xray_da_flare(flares_giorno, data_utc)
        if timestamps:
            print(f"  Curva X-ray ricostruita: {len(timestamps)} campioni (profilo schematico)")
            salva_xray_locale(data_utc, timestamps, flux)
            return timestamps, flux
    except Exception as e:
        print(f"  Errore DONKI per ricostruzione X-ray: {e}")

    # Fallback finale
    ts, fl = leggi_xray_locale(data_utc)
    if ts:
        print("X-ray: usato file locale come fallback.")
        return ts, fl

    print(">>> Nessun dato X-ray disponibile.")
    return [], []

# =============================================================
# CACHE LOCALE FLARE
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
# DOWNLOAD DATI FLARE
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
            ("json_donki", f"https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/FLR?startDate={data_inizio}&endDate={data_fine}"),
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

            flares.append({'inizio': hhmm_to_dt(ora_inizio), 'picco': hhmm_to_dt(ora_picco), 'fine': hhmm_to_dt(ora_fine), 'classe': classe, 'tipo': classe[0], 'limb': is_limb})
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
            if len(parti) != 2:
                continue
            try:
                dt  = datetime.strptime(parti[0].strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                val = float(parti[1].strip())
                timestamps.append(dt)
                valori.append(val)
            except Exception:
                continue
    return timestamps, valori


# ============================================================
# SEZIONE GRB — funzioni
# ============================================================

def _solar_elevation_path(dt_utc):
    """
    Elevazione solare in gradi al punto medio del path NSY-GAESID.
    Formula approssimata con equazione del tempo, precisione ~1°.
    """
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
    """
    Altitudine (°) di una sorgente GRB sull'orizzonte del punto medio del path.
    """
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

def _parse_votable(xml_bytes):
    root   = ET.fromstring(xml_bytes)
    fields = [elem.get('name', '').lower() for elem in root.iter() if elem.tag.endswith('FIELD')]
    rows = []
    for tr in root.iter():
        if tr.tag.endswith('TR'):
            cells = [td.text for td in tr if td.tag.endswith('TD')]
            if len(cells) == len(fields):
                rows.append(dict(zip(fields, cells)))
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


def fetch_grb_for_date(date_obj):
    date_str  = date_obj.strftime('%Y-%m-%d')
    date_next = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
    cache_f   = os.path.join(_GRB_CACHE_DIR, f'grb_{date_str}.json')

    if os.path.exists(cache_f):
        with open(cache_f, 'r') as fh:
            raw = json.load(fh)
        for g in raw:
            ts = g.get('trigger_time_str', '')
            for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
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
            for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
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
            if flu_raw is None: continue
            fluence = float(flu_raw)
            if fluence <= 0: continue

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

    cache_data = [{k: v for k, v in g.items() if k != 'trigger_time'} for g in result]
    with open(cache_f, 'w') as fh:
        json.dump(cache_data, fh, indent=2)

    return result

def fetch_grb_for_year(year):
    oggi = datetime.now(timezone.utc)
    inizio = datetime(year, 1, 1, tzinfo=timezone.utc)
    fine = datetime(year + 1, 1, 1, tzinfo=timezone.utc) if year < oggi.year else oggi + timedelta(seconds=1)
    data_inizio = inizio.strftime('%Y-%m-%dT%H:%M:%S')
    data_fine = fine.strftime('%Y-%m-%dT%H:%M:%S')
    cache_f = os.path.join(_GRB_CACHE_DIR, f'grb_year_{year}.json')

    if os.path.exists(cache_f):
        with open(cache_f, 'r') as fh:
            raw = json.load(fh)
        if raw:
            for g in raw:
                ts = g.get('trigger_time_str', '')
                for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
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
            for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
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

    cache_data = [{k: v for k, v in g.items() if k != 'trigger_time'} for g in result]
    with open(cache_f, 'w') as fh:
        json.dump(cache_data, fh, indent=2)

    return result


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
        text = tk.Text(root, wrap='none', bg='#121212', fg='white', insertbackground='white', font=('Consolas', 10))
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
    linee.append("\nNome | Trigger UTC | Categoria | T90 (s) | Fluence (erg/cm²) | RA | DEC")
    linee.append("-" * 102)
    for g in grb_visibili:
        linee.append(
            f"{g['name'][:15]:15s} | {g['trigger_time'].strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{g['categoria'][:10]:10s} | {g['t90']:6.1f} | {g['fluence']:10.2e} | {g['ra']:6.1f} | {g['dec']:6.1f}"
        )

    try:
        root = tk.Tk()
        root.title(f"GRB visibili {anno}")
        root.attributes('-topmost', True)
        root.resizable(True, True)
        text = tk.Text(root, wrap='none', bg='#121212', fg='white', insertbackground='white', font=('Consolas', 10))
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


def _classifica_grb(grb):
    fluence = grb['fluence']
    trig    = grb['trigger_time']

    try:
        alt = _grb_altitude(grb['ra'], grb['dec'], trig)
    except Exception:
        alt = 0.0
    if alt < _GRB_ELEV_MIN:
        return None, None

    try:
        is_night = _solar_elevation_path(trig) < _ELEV_NOTTE
    except Exception:
        is_night = False

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

def _aggiorna_status_grb(fig, grb_ril):
    global _grb_status_text
    if _grb_status_text is not None:
        try: _grb_status_text.remove()
        except Exception: pass
        _grb_status_text = None

    if not grb_ril:
        txt = "◉ Nessun GRB rilevabile  "
        col = 'gray'
    else:
        n = len(grb_ril)
        cats = [g['categoria'] for g in grb_ril]
        col = ('red' if 'ECCEZIONALE' in cats else 'darkorange' if 'FORTE' in cats else 'goldenrod')
        s = 'i' if n > 1 else ''
        txt = f"◉ {n} GRB potenzialmente rilevabile{s}  [G]  "

    _grb_status_text = fig.text(0.995, 0.005, txt, ha='right', va='bottom', fontsize=8, color=col,
                                transform=fig.transFigure,
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=col, alpha=0.88),
                                zorder=10)

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
# STATO GLOBALE
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
}


# =============================================================
# DISEGNO GRAFICO
# =============================================================

def disegna_grafico():
    data_utc        = stato['data']
    finestra_smooth = stato['smooth']

    csv_path = trova_csv_per_data(data_utc)
    if csv_path is None:
        aggiorna_titolo_mancante(data_utc)
        return

    print(f"\nLettura CSV VLF: {csv_path}")
    timestamps, valori = leggi_csv_sid(csv_path)
    if not timestamps:
        print("Nessun dato nel file CSV VLF.")
        return

    valori_smooth = applica_smoothing(valori, finestra_smooth) if finestra_smooth > 1 else valori

    if stato['scala_lineare']:
        valori_plot = [10 ** (v / 20.0) * 1e5 for v in valori_smooth]
    else:
        valori_plot = valori_smooth

    flares_cache = leggi_flare_cache(data_utc)
    if flares_cache is not None:
        flares_manuali = [f for f in carica_flare_manuali() if f['picco'].date() == data_utc.date()]
        flares = flares_cache + flares_manuali
        flares.sort(key=lambda f: f['picco'])
        print(f"Flare totali (cache + manuali): {len(flares)}")
    else:
        risultati_flare = scarica_flare(data_utc)
        flares          = parse_flare_multi(risultati_flare, data_utc) if risultati_flare else []
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

    inizio_giorno = data_utc.replace(hour=0,  minute=0,  second=0,  tzinfo=timezone.utc)
    fine_giorno   = data_utc.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
    ax.set_xlim(inizio_giorno, fine_giorno)

    v_min   = min(valori_plot)
    v_max   = max(valori_plot)
    if v_max == v_min:
        margine = abs(v_min) * 0.10 if v_min != 0 else 1.0
    else:
        margine = (v_max - v_min) * 0.10
    ax.set_ylim(v_min - margine, v_max + margine)

    if alba and tramonto:
        ax.axvspan(inizio_giorno, alba,   color='#404060', alpha=0.45, zorder=1)
        ax.axvspan(tramonto, fine_giorno, color='#404060', alpha=0.45, zorder=1)
        ax.axvline(alba,     color='#FFA500', linewidth=1.2, linestyle='--', alpha=0.8, zorder=2)
        ax.axvline(tramonto, color='#FF6347', linewidth=1.2, linestyle='--', alpha=0.8, zorder=2)
        y_label = v_min + (v_max - v_min) * 0.05
        ax.text(alba,     y_label, f"☀ Alba\n{alba.strftime('%H:%M')} UTC",
                color='#FFA500', fontsize=8, ha='left',  va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d0d1a', alpha=0.7, edgecolor='#FFA500'))
        ax.text(tramonto, y_label, f"☀ Tramonto\n{tramonto.strftime('%H:%M')} UTC",
                color='#FF6347', fontsize=8, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d0d1a', alpha=0.7, edgecolor='#FF6347'))

    ax.plot(timestamps, valori_plot, color='#00FF99', linewidth=0.8, alpha=0.9, zorder=3, label='NSY 45.9 kHz VLF')
    ax.set_xlabel('Ora UTC', color='white', fontsize=11)
    if stato['scala_lineare']:
        ax.set_ylabel('Segnale VLF (lineare, u.a. ×10^5)', color='#00FF99', fontsize=11)
        ax.ticklabel_format(style='plain', axis='y', useOffset=False)
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
                    ax2.text(fine_giorno, y_mid, f' {nome_classe}', color=COLORI_FASCE_XRAY[nome_classe], fontsize=8, va='center', ha='left', fontweight='bold', clip_on=False)
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
    ha_limb    = False
    ha_manuale = False

    if _flare_visible:
        for flare in flares:
            is_limb    = flare.get('limb', False)
            is_manuale = flare.get('manuale', False)
            colore     = COLORE_LIMB if is_limb else COLORI_FLARE.get(flare['tipo'], '#FFFFFF')

            ax.axvspan(flare['inizio'], flare['fine'], alpha=0.25, color=colore, zorder=4, linestyle='--' if is_limb else '-')
            ax.axvline(flare['picco'], color=colore, linewidth=1.5, alpha=0.8, linestyle=':' if is_limb else '--', zorder=5)

            if is_manuale: etichetta = f"{flare['classe']}\n✎ BTL"
            elif is_limb: etichetta = f"{flare['classe']}\n{SIMBOLO_LIMB}"
            else: etichetta = flare['classe']

            ax.text(flare['picco'], np.percentile(valori_plot, 95), etichetta, color=colore, fontsize=7 if is_limb else 8, ha='center', va='bottom', fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d0d1a', alpha=0.7, edgecolor=colore))

            if is_manuale: ha_manuale = True
            elif is_limb: ha_limb = True
            else: legenda_classi.add(flare['tipo'])
    else:
        print("Flare nascosti: premi F per visualizzarli")

    smooth_label = f" (smooth {finestra_smooth}s)" if finestra_smooth > 1 else ""
    scala_label  = " [LIN]" if stato['scala_lineare'] else ""
    handles = [mpatches.Patch(color='#00FF99', label=f'NSY 45.9 kHz VLF{smooth_label}{scala_label}')]

    if stato['xray_on'] and xray_ts:
        oggi_utc = datetime.now(timezone.utc).date()
        delta    = (oggi_utc - data_utc.date()).days
        xray_label = 'GOES X-ray 0.1–0.8 nm (dati reali)' if delta <= 6 else 'GOES X-ray 0.1–0.8 nm (⚠ profilo ricostruito da flare)'
        handles.append(mpatches.Patch(color='#FFD700', label=xray_label))
    elif stato['xray_on'] and not xray_ts:
        handles.append(mpatches.Patch(color='#FFD700', alpha=0.3, label='GOES X-ray — nessun dato'))
    else:
        handles.append(mpatches.Patch(color='#555555', alpha=0.5, label='GOES X-ray — disattivato (tasto X)'))

    if alba and tramonto:
        handles.append(mpatches.Patch(color='#404060', alpha=0.6, label='Notte'))
        handles.append(mpatches.Patch(color='#FFA500', alpha=0.8, label=f'Alba {alba.strftime("%H:%M")} UTC'))
        handles.append(mpatches.Patch(color='#FF6347', alpha=0.8, label=f'Tramonto {tramonto.strftime("%H:%M")} UTC'))

    for tipo in sorted(legenda_classi, reverse=True):
        handles.append(mpatches.Patch(color=COLORI_FLARE.get(tipo, '#FFFFFF'), alpha=0.6, label=f'Flare classe {tipo}'))

    if ha_limb: handles.append(mpatches.Patch(color=COLORE_LIMB, alpha=0.6, label='Flare beyond-the-limb (★ BTL)'))
    if ha_manuale: handles.append(mpatches.Patch(color=COLORE_LIMB, alpha=0.6, label='Flare inserito manualmente (✎ BTL)'))

    ax.legend(handles=handles, loc='lower left', facecolor='#1a1a2e', edgecolor='#444444', labelcolor='white', fontsize=9)

    smooth_title = f" — Smooth {finestra_smooth}s" if finestra_smooth > 1 else ""
    xray_stato   = "ON" if stato['xray_on'] else "OFF"
    scala_stato  = "LIN" if stato['scala_lineare'] else "dB"
    ax.text(0.5, 1.14, data_utc.strftime('%Y-%m-%d UTC'), transform=ax.transAxes,
            ha='center', va='bottom', color='white', fontsize=13, fontweight='bold')
    ax.set_title(
        f"Segnale VLF NSY - Flusso X-ray GOES - Flare Solari - GRB{smooth_title}\n"
        f"Tasti:  A = prec.  |  D = succ.  |  I = data  |  M = flare Man  |  F = flare ON |  G = GRB ON |  "
        f"B = anno GRB  |  Shift+B = tutti i GRB  |  S = smooth lista  |  V = smooth libero  | "
        f"U = scala [{scala_stato}]  |  X = X-ray [{xray_stato}]  |  R = Aggiorna  |  P = PNG  |  Q = esci",
        color='white', fontsize=9, fontweight='bold', pad=18
    )

    # --- Modulo GRB (Integrazione Blocco 5) ---
    global _grb_all, _grb_ril, _grb_visible
    print("[GRB] Recupero dati GRB da HEASARC Fermi GBM...")
    _grb_all = fetch_grb_for_date(data_utc)
    _grb_ril = _filtra_grb_rilevabili(_grb_all)
    print(f"[GRB] {len(_grb_all)} GRB totali nel giorno, {len(_grb_ril)} potenzialmente rilevabili a GAESID.")
    _aggiorna_status_grb(fig, _grb_ril)
    disegna_grb_markers(ax, _grb_ril, _grb_visible)

    fig.canvas.draw_idle()


def aggiorna_titolo_mancante(data_utc):
    ax = stato['ax']
    ax.cla()
    ax.set_facecolor('#0d0d1a')
    ax.text(0.5, 0.5, f"Nessun dato disponibile per\n{data_utc.strftime('%Y-%m-%d')}", transform=ax.transAxes, color='#FF6347', fontsize=14, ha='center', va='center')
    ax.set_title(
        f"Segnale VLF NSY  —  {data_utc.strftime('%Y-%m-%d')} UTC\n"
        f"Tasti:  A/D = giorno  |  I = data  |  G = GRB  |  S/V = smooth",
        color='white', fontsize=10, fontweight='bold', pad=10
    )
    
    # Anche qui aggiorniamo lo status dei GRB per far capire se ci sono per questa data senza file VLF
    global _grb_all, _grb_ril
    _grb_all = fetch_grb_for_date(data_utc)
    _grb_ril = _filtra_grb_rilevabili(_grb_all)
    _aggiorna_status_grb(stato['fig'], _grb_ril)

    stato['fig'].canvas.draw_idle()


def salva_png():
    data_utc        = stato['data']
    finestra_smooth = stato['smooth']
    smooth_suffix   = f"_smooth{finestra_smooth}s" if finestra_smooth > 1 else ""
    xray_suffix     = "_xray" if stato['xray_on'] else ""
    scala_suffix    = "_lin"  if stato['scala_lineare'] else ""
    output_png = os.path.join(CARTELLA_STORICO, f"VLF_flare_{data_utc.strftime('%Y-%m-%d')}{smooth_suffix}{xray_suffix}{scala_suffix}.png")
    stato['fig'].savefig(output_png, dpi=150, bbox_inches='tight', facecolor=stato['fig'].get_facecolor())
    print(f"Grafico salvato: {output_png}")


# =============================================================
# HANDLER TASTIERA
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

    gui_event = getattr(event, 'guiEvent', None)
    if gui_event is not None and hasattr(gui_event, 'state'):
        shift_pressed = shift_pressed or (gui_event.state & 0x0001 != 0)

    if key_norm == 'r':
        stato['xray_data'] = None
        disegna_grafico()

    elif key_norm == 'g':
        # Handler GRB dal Blocco 4
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
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        data_str = stato['data'].strftime("%Y-%m-%d")
        try:
            classe = simpledialog.askstring("Flare manuale", f"Classe flare (es. C5.3, M1.2, X1.0)\nData: {data_str}", parent=root)
            if not classe: return root.destroy()
            classe = classe.strip().upper()
            ora_picco = simpledialog.askstring("Flare manuale", "Ora picco UTC (HH:MM):", parent=root)
            if not ora_picco: return root.destroy()
            ora_picco = ora_picco.strip()
            ora_inizio = simpledialog.askstring("Flare manuale", f"Ora inizio UTC (HH:MM)\n[lascia vuoto per usare {ora_picco}]:", parent=root)
            ora_inizio = ora_inizio.strip() if ora_inizio and ora_inizio.strip() else ora_picco
            ora_fine = simpledialog.askstring("Flare manuale", f"Ora fine UTC (HH:MM)\n[lascia vuoto per usare {ora_picco}]:", parent=root)
            ora_fine = ora_fine.strip() if ora_fine and ora_fine.strip() else ora_picco
            nota = simpledialog.askstring("Flare manuale", "Nota opzionale:", parent=root)
            nota = nota.strip() if nota else ""
            root.destroy()
            salva_flare_manuale(data_str, classe, ora_inizio, ora_picco, ora_fine, nota)
            disegna_grafico()
        except Exception as e:
            try: root.destroy()
            except Exception: pass

    elif key_norm == 'p':
        salva_png()

    elif key_norm == 'q':
        plt.close('all')


# =============================================================
# AVVIO
# =============================================================

print("=" * 60)
print("  VLF SID Monitor — NSY 45.9 kHz — Stazione GAESID")
print("=" * 60)
print()
print("  COMANDI DA TASTIERA (clicca sul grafico prima):")
print("  A  -> Giorno precedente")
print("  D  -> Giorno successivo")
print("  G  -> Attiva/Disattiva overlay dei GRB (Novità)")
print("  I  -> Calendario per scegliere la data")
print("  M  -> Inserisci flare manuale (beyond-the-limb)")
print("  F  -> Mostra/Nascondi i flare solari")
print("  P  -> Salva grafico come PNG")
print("  R  -> Refresh Grafico, Flares e X-ray")
print("  S  -> Cambia smoothing (lista valori standard)")
print("  U  -> Cambia scala asse Y (dB <-> Lineare)")
print("  V  -> Cambia smoothing (valore libero manuale)")
print("  X  -> Attiva/Disattiva overlay flusso X-ray GOES")
print("  B  -> Anno GRB (solo GRB potenzialmente visibili)")
print("  Shift+B -> Anno GRB (tutti i GRB)")
print()

def apri_calendario(data_corrente):
    root = tk.Tk()
    root.title("GAESID — Scegli data")
    root.attributes('-topmost', True)
    root.resizable(False, False)
    root.configure(bg='#0d0d1a')
    width, height = 420, 380
    root.geometry(f'{width}x{height}')
    root.minsize(width, height)
    root.maxsize(width, height)
    tk.Label(root, text="Seleziona una data con dati disponibili", bg='#0d0d1a', fg='#00FF99', font=('Consolas', 10, 'bold')).pack(pady=(10, 4))
    cal = Calendar(root, selectmode='day', year=data_corrente.year, month=data_corrente.month, day=data_corrente.day, date_pattern='yyyy-mm-dd', locale='it_IT', background='#1a1a2e', foreground='white', headersbackground='#2a2a4e', headersforeground='#00FF99', selectbackground='#00FF99', selectforeground='black', normalbackground='#0d0d1a', normalforeground='white', weekendbackground='#1a1a2e', weekendforeground='#aaaacc', othermonthforeground='#444466', othermonthbackground='#0d0d1a', font=('Consolas', 9))
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
    tk.Button(frame_btn, text="Conferma", command=conferma, bg='#00FF99', fg='black', font=('Consolas', 10, 'bold'), relief='flat', padx=12, pady=4).pack(side='left', padx=8)
    tk.Button(frame_btn, text="Annulla", command=annulla, bg='#333355', fg='white', font=('Consolas', 10), relief='flat', padx=12, pady=4).pack(side='left', padx=8)
    root.protocol("WM_DELETE_WINDOW", annulla)
    root.wait_window(root)
    return risultato['data']


def apri_calendario_anno(anno_corrente):
    root = tk.Tk()
    root.title("GAESID — Scegli anno")
    root.attributes('-topmost', True)
    root.resizable(False, False)
    root.configure(bg='#0d0d1a')
    width, height = 320, 220
    root.geometry(f'{width}x{height}')
    root.minsize(width, height)
    root.maxsize(width, height)
    tk.Label(root, text="Seleziona l'anno da cercare", bg='#0d0d1a', fg='#00FF99', font=('Consolas', 11, 'bold')).pack(pady=(14, 8))

    frame = tk.Frame(root, bg='#0d0d1a')
    frame.pack(pady=(0, 10))
    tk.Label(frame, text="Anno:", bg='#0d0d1a', fg='white', font=('Consolas', 10)).pack(side='left', padx=(0, 8))
    year_spin = tk.Spinbox(frame, from_=2000, to=datetime.now(timezone.utc).year, width=6, justify='center', font=('Consolas', 12), bg='#1a1a2e', fg='white', insertbackground='white', relief='flat')
    year_spin.delete(0, 'end')
    year_spin.insert(0, str(anno_corrente))
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
    tk.Button(frame_btn, text="Conferma", command=conferma, bg='#00FF99', fg='black', font=('Consolas', 10, 'bold'), relief='flat', padx=12, pady=4).pack(side='left', padx=10)
    tk.Button(frame_btn, text="Annulla", command=annulla, bg='#333355', fg='white', font=('Consolas', 10), relief='flat', padx=12, pady=4).pack(side='left', padx=10)
    root.protocol("WM_DELETE_WINDOW", annulla)
    root.wait_window(root)
    return risultato['anno']


def scegli_data_iniziale():
    oggi = datetime.now(timezone.utc)
    return apri_calendario(oggi)


VALORI_SMOOTH_STANDARD = [1, 15, 30, 60, 90, 120]

def scegli_smoothing_iniziale():
    root = tk.Tk()
    root.title("GAESID — Smoothing Iniziale")
    root.attributes('-topmost', True)
    root.resizable(False, False)
    root.configure(bg='#0d0d1a')
    tk.Label(root, text="Scegli la finestra di smoothing (secondi)", bg='#0d0d1a', fg='#00FF99', font=('Consolas', 11, 'bold')).pack(pady=(14, 6), padx=20)
    risultato = {'valore': 60}
    frame_std = tk.Frame(root, bg='#0d0d1a')
    frame_std.pack(pady=10, padx=20)
    etichette = {1: "Nessuno\n(1 s)", 15: "15 s", 30: "30 s", 60: "60 s", 90: "90 s", 120: "120 s"}
    btn_sel = {'btn': None}

    def seleziona(val, btn):
        if btn_sel['btn']: btn_sel['btn'].config(bg='#1e1e3a', fg='white', relief='flat')
        btn.config(bg='#00FF99', fg='black', relief='sunken')
        btn_sel['btn'] = btn
        risultato['valore'] = val

    for val in VALORI_SMOOTH_STANDARD:
        b = tk.Button(frame_std, text=etichette.get(val, str(val)), width=9, height=2, bg='#1e1e3a', fg='white', font=('Consolas', 10), relief='flat', bd=1)
        b.pack(side='left', padx=5)
        b.config(command=lambda v=val, btn=b: seleziona(v, btn))
        if val == 60: seleziona(60, b)

    tk.Button(root, text="Conferma", command=root.destroy, bg='#00FF99', fg='black', font=('Consolas', 11, 'bold'), padx=20, pady=8).pack(pady=15)
    root.mainloop()
    return risultato['valore']

def apri_dialogo_smooth_lista():
    root = tk.Tk()
    root.title("GAESID — Smoothing")
    root.attributes('-topmost', True)
    root.resizable(False, False)
    root.configure(bg='#0d0d1a')
    corrente = stato['smooth']
    tk.Label(root, text=f"Scegli smoothing\n(Attuale: {corrente} s)", bg='#0d0d1a', fg='#00FF99', font=('Consolas', 11, 'bold')).pack(pady=12)
    risultato = {'valore': None}
    btn_attivo = {'btn': None}
    frame = tk.Frame(root, bg='#0d0d1a')
    frame.pack(pady=8, padx=20)
    etichette = {1: "Nessuno\n(1 s)", 15: "15 s", 30: "30 s", 60: "60 s", 90: "90 s", 120: "120 s"}

    def seleziona(val, btn):
        if btn_attivo['btn'] is not None: btn_attivo['btn'].config(bg='#1e1e3a', fg='white', relief='flat')
        btn.config(bg='#00FF99', fg='black', relief='sunken')
        btn_attivo['btn'] = btn
        risultato['valore'] = val

    for val in VALORI_SMOOTH_STANDARD:
        bg_i = '#00FF99' if val == corrente else '#1e1e3a'
        fg_i = 'black' if val == corrente else 'white'
        b = tk.Button(frame, text=etichette.get(val, str(val)), width=9, height=2, bg=bg_i, fg=fg_i, font=('Consolas', 10), relief='flat')
        b.pack(side='left', padx=4)
        b.config(command=lambda v=val, btn=b: seleziona(v, btn))

    def annulla():
        risultato['valore'] = None
        root.destroy()

    frame_btn = tk.Frame(root, bg='#0d0d1a')
    frame_btn.pack(pady=(0, 12))
    tk.Button(frame_btn, text="Conferma", command=root.destroy, bg='#00FF99', fg='black', font=('Consolas', 10, 'bold'), relief='flat', padx=12, pady=4).pack(side='left', padx=8)
    tk.Button(frame_btn, text="Annulla", command=annulla, bg='#333355', fg='white', font=('Consolas', 10), relief='flat', padx=12, pady=4).pack(side='left', padx=8)
    root.wait_window(root)
    return risultato['valore']

def apri_dialogo_smooth_manuale():
    root = tk.Tk()
    root.title("GAESID — Smoothing manuale")
    root.attributes('-topmost', True)
    root.resizable(False, False)
    root.configure(bg='#0d0d1a')
    corrente = stato['smooth']
    tk.Label(root, text="Inserisci il valore (secondi)", bg='#0d0d1a', fg='#00FF99', font=('Consolas', 11, 'bold')).pack(pady=(14, 4), padx=20)
    risultato = {'valore': None}
    frame_e = tk.Frame(root, bg='#0d0d1a')
    frame_e.pack(pady=(0, 6), padx=20)
    entry = tk.Entry(frame_e, width=8, bg='#1e1e3a', fg='white', insertbackground='white', font=('Consolas', 14), justify='center', relief='flat')
    entry.insert(0, str(corrente))
    entry.selection_range(0, tk.END)
    entry.pack(side='left')
    entry.focus_set()
    lbl_err = tk.Label(root, text="", bg='#0d0d1a', fg='#FF6347', font=('Consolas', 9))
    lbl_err.pack(pady=(0, 4))

    def conferma():
        try:
            val = int(entry.get().strip())
            if val < 1: return lbl_err.config(text="Il valore deve essere >= 1")
            risultato['valore'] = val
            root.destroy()
        except ValueError:
            lbl_err.config(text="Inserisci un numero intero valido")

    frame_btn = tk.Frame(root, bg='#0d0d1a')
    frame_btn.pack(pady=(0, 14))
    tk.Button(frame_btn, text="Conferma", command=conferma, bg='#00FF99', fg='black', font=('Consolas', 10, 'bold'), relief='flat', padx=14, pady=5).pack(side='left', padx=6)
    tk.Button(frame_btn, text="Annulla", command=root.destroy, bg='#333355', fg='white', font=('Consolas', 10), relief='flat', padx=10, pady=5).pack(side='left', padx=6)
    root.bind('<Return>', lambda e: conferma())
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.wait_window(root)
    return risultato['valore']

# =============================================================
# MAIN
# =============================================================

data_scelta = scegli_data_iniziale()
if data_scelta is None:
    data_scelta = datetime.now(timezone.utc)

finestra_smooth = scegli_smoothing_iniziale()

# Ottiene le dimensioni dello schermo per creare figura con rapporto corretto
try:
    _root_tk = tk.Tk()
    _root_tk.withdraw()
    _root_tk.update_idletasks()
    _SW = _root_tk.winfo_screenwidth()
    _SH = _root_tk.winfo_screenheight()
    _root_tk.destroy()
except Exception:
    _SW, _SH = 1920, 1080

_WH = int(_SH * 0.5)  # metà schermo in altezza

# Crea figura: larghezza = schermo, altezza = metà schermo
fig, ax = plt.subplots(figsize=(_SW / 100, _WH / 100))
fig.patch.set_facecolor('#0d0d1a')
ax.set_facecolor('#0d0d1a')

# Nota: la lettera 'g' non va disabilitata da plt.rcParams perché altrimenti il nostro bind verrebbe sovrascritto.
for tasto in ['p','q','s','v','c','a','d','h','r','k','l','o','f','x','u']:
    for keymap in list(plt.rcParams.keys()):
        if keymap.startswith('keymap.') and tasto in plt.rcParams[keymap]:
            try: plt.rcParams[keymap].remove(tasto)
            except: pass

stato['data']   = data_scelta
stato['smooth'] = finestra_smooth
stato['fig']    = fig
stato['ax']     = ax

fig.canvas.mpl_connect('key_press_event', on_key)

disegna_grafico()
plt.tight_layout()

# Posiziona la finestra in basso: larga quanto lo schermo, alta metà
geom_finestra = f"{_SW}x{_WH}+0+{_SH - _WH}"
print(f"Geometria finestra: {geom_finestra}  (schermo: {_SW}x{_SH})")

# Ottieni il manager della figura
_manager = plt.get_current_fig_manager()
if hasattr(_manager, 'window'):
    _win = _manager.window
    # Funzione che applica la geometria quando la finestra viene mappata
    def _imposta_geometria(event=None):
        try:
            _win.geometry(geom_finestra)
            _win.update()
        except Exception:
            pass
    # Prova subito
    _imposta_geometria()
    # E anche quando la finestra viene visualizzata (evento <Map>)
    _win.bind('<Map>', _imposta_geometria, add='+')

plt.show()
