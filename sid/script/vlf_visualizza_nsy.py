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

# =============================================================
#  CONFIGURAZIONE UTENTE — modifica questi valori
# =============================================================
#
#  Requisiti: Python 3.13 o superiore
#  Librerie:  pip install requests matplotlib numpy scipy ephem pytz tkcalendar
#
#  CARTELLA_GIORNALIERA : cartella dove si trovano i file CSV del giorno corrente
#  CARTELLA_STORICO     : cartella dove si trovano i file CSV storici
#  CARTELLA_XRAY        : cartella dove vengono salvati i dati X-ray GOES in cache
#  LAT, LON, ELEVAZIONE : coordinate geografiche del sito di ricezione
#  TRASMETTITORE        : nome della stazione VLF monitorata
#  FREQUENZA_KHZ        : frequenza in kHz della stazione monitorata
#
# =============================================================
# CONFIGURAZIONE

# Directory dove cercare i file CSV — lo script cerca in entrambe,
# dando priorità alla cartella giornaliera se il file esiste in entrambe.
CARTELLA_GIORNALIERA = r"C:\SID\dati"           # <-- modifica
CARTELLA_STORICO     = r"C:\SID\dati\Storico"   # <-- modifica

# Cartella dove salvare i file CSV del flusso X-ray GOES
CARTELLA_XRAY = r"C:\SID\dati\Storico\XRAY"    # <-- modifica

STAZIONE_ID    = "GAESID"         # <-- modifica
TRASMETTITORE  = "NSY"             # <-- modifica
FREQUENZA_KHZ  = 45.9              # <-- modifica

# File JSON per i flare inseriti manualmente (beyond-the-limb o non in catalogo).
# Lo script lo crea automaticamente se non esiste.
FILE_FLARE_MANUALI = os.path.join(CARTELLA_STORICO, "flare_manuali.json")

# Colori per le classi di flare
COLORI_FLARE = {
    'X': '#FF0000',
    'M': '#FF8C00',
    'C': '#FFD700',
    'B': '#00BFFF',
    'A': '#90EE90',
}

# Colore e marcatore per flare beyond-the-limb (da catalogo o manuali)
COLORE_LIMB  = '#FF00FF'   # magenta
SIMBOLO_LIMB = '★ BTL'

# Soglie flusso X-ray per le fasce di classe (W/m²) — canale 0.1-0.8 nm
# Usate per le linee di riferimento sull'asse destro
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
#  COMANDI DA TASTIERA
# ============================================================
#  A  →  Giorno precedente
#  D  →  Giorno successivo
#  I  →  Inserisci nuova data (calendario)
#  M  →  Inserisci flare manuale (beyond-the-limb)
#  P  →  Stampa/salva il grafico corrente come PNG
#  R  →  Aggiorna Grafico e Flares
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
    osservatore.lat       = '45.07'   # <-- modifica
    osservatore.lon       = '7.68'    # <-- modifica
    osservatore.elevation = 240       # <-- modifica
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
    nome_file = f"{STAZIONE_ID}_{TRASMETTITORE}_{data_str}.csv"
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
    """
    Legge il CSV locale del flusso X-ray per la data indicata.
    Formato CSV: timestamp_utc,flux_wm2
    Restituisce (timestamps, flux) o ([], []) se il file non esiste.
    """
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
    """Salva il flusso X-ray scaricato in CSV locale."""
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
    """
    Parsa il JSON real-time NOAA (xrays-7-day.json).
    Campi: time_tag, flux, energy ('0.1-0.8nm' = canale B).
    Restituisce (timestamps, flux).
    """
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
    """
    Converte una stringa DONKI nel formato "YYYY-MM-DDTHH:MMZ"
    (o varianti senza Z, con secondi) in un oggetto datetime UTC.
    Restituisce None se la stringa è vuota o non parsabile.
    """
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
    """
    Ricostruisce una curva X-ray approssimativa a partire dalla lista
    di eventi DONKI grezzi (JSON diretto da DONKI API, NON già parsati).

    Ogni elemento è un dict con campi DONKI originali:
      beginTime, peakTime, endTime  → stringhe "YYYY-MM-DDTHH:MMZ"
      classType                     → es. "M1.5", "C3.2", "X1.0"
      peakIntensity                 → dict con 'value' e 'unit' (W/m²)
                                      oppure assente

    Per ogni flare genera una curva rise lineare + decay esponenziale
    campionata ogni minuto sulla griglia giornaliera.

    Mappa classi → flux W/m² di riferimento (mediano della classe GOES-R):
      C → 5e-6,  M → 5e-5,  X → 5e-4   (scala GOES-R, non GOES legacy)
    """
    # Soglie flux GOES-R per il valore "1.0" di ogni classe
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
            # --- Classe e flux picco ---
            classe = (flare.get('classType') or '').strip().upper()
            if not classe or classe[0] not in CLASSE_BASE:
                continue   # ignora A e B — non producono SID rilevabili

            tipo = classe[0]
            try:
                # Estrae il valore numerico: "M1.5" → 1.5
                numero = float(re.sub(r'[^\d.]', '', classe[1:])) if len(classe) > 1 else 1.0
                if numero <= 0:
                    numero = 1.0
            except ValueError:
                numero = 1.0

            # Prova prima peakIntensity dal JSON (W/m²)
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

            # --- Orari ---
            t_inizio = _donki_str_to_dt(flare.get('beginTime'))
            t_picco  = _donki_str_to_dt(flare.get('peakTime'))
            t_fine   = _donki_str_to_dt(flare.get('endTime'))

            if t_picco is None:
                continue
            if t_inizio is None:
                t_inizio = t_picco - timedelta(minutes=max(int(numero * 5), 5))
            if t_fine is None:
                t_fine = t_picco + timedelta(minutes=max(int(numero * 15), 10))

            print(f"  Flare {classe}: picco {t_picco.strftime('%H:%M')} UTC, "
                  f"flux={flux_picco:.2e} W/m²")
            n_ok += 1

            # --- Scrivi curva sulla griglia ---
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
    """
    Scarica il flusso X-ray GOES (canale B, 0.1–0.8 nm) per la data indicata.

    Strategia a cascata:
    1. Cache locale già presente → usata direttamente (salvo se delta==0).
    2. Dati recenti (<=6 giorni) → JSON real-time NOAA xrays-7-day.json
       (la finestra di 7 giorni è dalla mezzanotte UTC corrente, quindi
        sicuro solo fino a ~6 giorni fa).
    3. Dati storici (>6 giorni) → DONKI API per i flare del giorno,
       ricostruzione di una curva X-ray approssimativa basata su
       beginTime/peakTime/endTime/classType.
       Nota: questo è un profilo schematico rise/decay, NON il dato
       strumentale minuto per minuto, ma è sufficiente per identificare
       la correlazione SID visivamente.
    4. Salva il risultato in CSV locale per uso futuro.

    Restituisce (timestamps, flux) con flux in W/m².
    """
    oggi_utc = datetime.now(timezone.utc).date()
    delta    = (oggi_utc - data_utc.date()).days

    # --- 1. Cache locale ---
    if delta > 0:
        ts, fl = leggi_xray_locale(data_utc)
        if ts:
            return ts, fl

    # --- 2. JSON real-time (ultimi ~6 giorni) ---
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

    # --- 3. Dati storici: DONKI → ricostruzione curva ---
    # Per dati >6 giorni fa, NOAA non espone un JSON minuto-per-minuto via HTTP.
    # Usiamo i flare DONKI per ricostruire una curva schematica rise/decay.
    print(f"X-ray storico: ricostruzione da flare DONKI per {data_utc.strftime('%Y-%m-%d')}...")
    data_str = data_utc.strftime("%Y-%m-%d")
    url_donki = f"https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/FLR?startDate={data_str}&endDate={data_str}"
    print(f"Download flare DONKI per X-ray: {url_donki}")
    try:
        r = requests.get(url_donki, timeout=(8, 15))
        r.raise_for_status()
        flares_donki = json.loads(r.text)
        # Filtra solo flare della data richiesta (il DONKI può restituire flare
        # che iniziano il giorno precedente ma hanno il picco nel giorno richiesto)
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

    # --- Fallback finale: prova il locale anche se delta==0 ---
    ts, fl = leggi_xray_locale(data_utc)
    if ts:
        print("X-ray: usato file locale come fallback.")
        return ts, fl

    print(">>> Nessun dato X-ray disponibile.")
    return [], []


# =============================================================
# DOWNLOAD DATI FLARE
# =============================================================

def scarica_flare(data_utc):
    data_str   = data_utc.strftime("%Y%m%d")
    anno       = data_utc.strftime("%Y")
    oggi_utc   = datetime.now(timezone.utc).date()
    delta_giorni = (oggi_utc - data_utc.date()).days

    url_eventi_swpc = f"https://www.swpc.noaa.gov/archive/{anno}/{data_str}events.txt"

    if delta_giorni <= 7:
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
        print(f"Tentativo download flare da: {url}")
        try:
            r = requests.get(url, timeout=(5, 8))
            r.raise_for_status()
            print(f"Download riuscito ({tipo}): {url}")
            risultati.append((r.text, tipo))
            if tipo in ("json_noaa", "json_donki"):
                try:
                    r2 = requests.get(url_eventi_swpc, timeout=(5, 8))
                    r2.raise_for_status()
                    risultati.append((r2.text, "txt_swpc"))
                    print("File eventi SWPC scaricato come supplemento.")
                except Exception:
                    print("File eventi SWPC non disponibile (supplemento ignorato).")
                break
        except requests.exceptions.ConnectTimeout:
            print("Timeout connessione, passo al prossimo...")
        except requests.exceptions.ReadTimeout:
            print("Timeout lettura, passo al prossimo...")
        except requests.exceptions.HTTPError as e:
            print(f"Errore HTTP {e.response.status_code}, passo al prossimo...")
        except Exception as e:
            print(f"Errore: {e}, passo al prossimo...")

    if not risultati:
        print(">>> Nessuna fonte disponibile. Grafico senza dati flare.")
    return risultati


# =============================================================
# PARSING FLARE
# =============================================================

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
                flares.append({
                    'inizio': inizio, 'picco': picco, 'fine': fine,
                    'classe': classe, 'tipo': classe[0], 'limb': False
                })
            except Exception:
                continue
    except Exception as e:
        print(f"Errore parsing JSON NOAA: {e}")
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
                flares.append({
                    'inizio': inizio, 'picco': picco, 'fine': fine,
                    'classe': classe, 'tipo': classe[0], 'limb': False
                })
            except Exception:
                continue
    except Exception as e:
        print(f"Errore parsing JSON DONKI: {e}")
    return flares


def parse_flare_txt_swpc(testo, data_utc):
    flares   = []
    data_str = data_utc.strftime("%Y-%m-%d")

    def hhmm_to_dt(hhmm):
        return datetime.strptime(
            f"{data_str} {hhmm[:2]}:{hhmm[2:4]}:00",
            "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=timezone.utc)

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

            flares.append({
                'inizio': hhmm_to_dt(ora_inizio),
                'picco':  hhmm_to_dt(ora_picco),
                'fine':   hhmm_to_dt(ora_fine),
                'classe': classe,
                'tipo':   classe[0],
                'limb':   is_limb
            })
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
        gia_presente = any(
            abs((fs['picco'] - fj['picco']).total_seconds()) < 300
            for fj in flares_json
        )
        if not gia_presente:
            flares_extra.append(fs)
        elif fs['limb']:
            for fj in flares_json:
                if abs((fs['picco'] - fj['picco']).total_seconds()) < 300:
                    fj['limb'] = True
                    break

    flares_manuali = [
        f for f in carica_flare_manuali()
        if f['picco'].date() == data_utc.date()
    ]

    tutti = flares_json + flares_extra + flares_manuali
    tutti.sort(key=lambda f: f['picco'])

    n_limb    = sum(1 for f in tutti if f.get('limb'))
    n_manuali = sum(1 for f in tutti if f.get('manuale'))
    print(f"Trovati {len(tutti)} flare totali "
          f"({n_limb} beyond-the-limb, {n_manuali} inseriti manualmente)")
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


# =============================================================
# STATO GLOBALE
# =============================================================

stato = {
    'data':        None,
    'smooth':      1,
    'fig':         None,
    'ax':          None,
    'xray_on':     True,    # overlay X-ray attivo/disattivo (tasto X)
    'xray_ts':     [],      # timestamps X-ray dell'ultima acquisizione
    'xray_flux':   [],      # valori flux X-ray dell'ultima acquisizione
    'xray_data':   None,    # data per cui sono stati caricati i dati X-ray
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

    valori_plot = applica_smoothing(valori, finestra_smooth) if finestra_smooth > 1 else valori

    # --- Flare ---
    risultati_flare = scarica_flare(data_utc)
    flares          = parse_flare_multi(risultati_flare, data_utc) if risultati_flare else []

    # --- X-ray: scarica solo se la data è cambiata o non ancora caricata ---
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

    # Rimuovi eventuale asse destro precedente
    for ax2_old in fig.axes[1:]:
        ax2_old.remove()

    inizio_giorno = data_utc.replace(hour=0,  minute=0,  second=0,  tzinfo=timezone.utc)
    fine_giorno   = data_utc.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
    ax.set_xlim(inizio_giorno, fine_giorno)

    v_min   = min(valori_plot)
    v_max   = max(valori_plot)
    margine = (v_max - v_min) * 0.10
    ax.set_ylim(v_min - margine, v_max + margine)

    # --- Notte / alba / tramonto ---
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

    # --- Curva VLF ---
    ax.plot(timestamps, valori_plot, color='#00FF99', linewidth=0.8, alpha=0.9, zorder=3,
            label='NSY 45.9 kHz VLF')
    ax.set_xlabel('Ora UTC', color='white', fontsize=11)
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

    # =========================================================
    # --- Overlay X-ray GOES (asse destro) ---
    # =========================================================
    ax2 = None
    if stato['xray_on'] and xray_ts and xray_flux:
        ax2 = ax.twinx()
        ax2.set_xlim(inizio_giorno, fine_giorno)

        # Scala logaritmica sull'asse destro — stile classico GOES
        flux_min = max(min(xray_flux) * 0.5, 1e-9)
        flux_max = max(xray_flux) * 3
        ax2.set_yscale('log')
        ax2.set_ylim(flux_min, flux_max)

        # Fasce di classe colorate (sfondo leggero)
        for nome_classe, (soglia_inf, soglia_sup) in SOGLIE_CLASSE_XRAY.items():
            y0 = max(soglia_inf, flux_min)
            y1 = min(soglia_sup, flux_max)
            if y1 > y0:
                ax2.axhspan(y0, y1,
                            color=COLORI_FASCE_XRAY[nome_classe],
                            alpha=0.04, zorder=1)
                # Etichetta di classe sull'asse destro
                y_mid = np.sqrt(soglia_inf * soglia_sup)
                if flux_min < y_mid < flux_max:
                    ax2.text(fine_giorno, y_mid, f' {nome_classe}',
                             color=COLORI_FASCE_XRAY[nome_classe],
                             fontsize=8, va='center', ha='left',
                             fontweight='bold', clip_on=False)

        # Linee di soglia classe (tratteggiate sottili)
        for nome_classe, (soglia_inf, _) in SOGLIE_CLASSE_XRAY.items():
            if flux_min < soglia_inf < flux_max:
                ax2.axhline(soglia_inf,
                            color=COLORI_FASCE_XRAY[nome_classe],
                            linewidth=0.5, linestyle=':', alpha=0.5, zorder=2)

        # Curva X-ray
        ax2.plot(xray_ts, xray_flux,
                 color='#FFD700', linewidth=1.2, alpha=0.85, zorder=4,
                 label='GOES X-ray (0.1–0.8 nm)')

        # Stile asse destro
        ax2.set_ylabel('Flusso X-ray GOES (W/m²)', color='#FFD700', fontsize=10)
        ax2.tick_params(axis='y', colors='#FFD700', labelsize=8)
        ax2.yaxis.label.set_color('#FFD700')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#444444')

        # Formattazione tick asse Y destro in notazione scientifica
        ax2.yaxis.set_major_formatter(
            plt.matplotlib.ticker.LogFormatterMathtext()
        )

    # --- Flare markers ---
    legenda_classi = set()
    ha_limb    = False
    ha_manuale = False

    for flare in flares:
        is_limb    = flare.get('limb', False)
        is_manuale = flare.get('manuale', False)
        colore     = COLORE_LIMB if is_limb else COLORI_FLARE.get(flare['tipo'], '#FFFFFF')

        ax.axvspan(flare['inizio'], flare['fine'],
                   alpha=0.25, color=colore, zorder=4,
                   linestyle='--' if is_limb else '-')
        ax.axvline(flare['picco'], color=colore,
                   linewidth=1.5, alpha=0.8,
                   linestyle=':' if is_limb else '--', zorder=5)

        if is_manuale:
            etichetta = f"{flare['classe']}\n✎ BTL"
        elif is_limb:
            etichetta = f"{flare['classe']}\n{SIMBOLO_LIMB}"
        else:
            etichetta = flare['classe']

        ax.text(flare['picco'], np.percentile(valori_plot, 95),
                etichetta, color=colore, fontsize=7 if is_limb else 8,
                ha='center', va='bottom', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d0d1a',
                          alpha=0.7, edgecolor=colore))

        if is_manuale:
            ha_manuale = True
        elif is_limb:
            ha_limb = True
        else:
            legenda_classi.add(flare['tipo'])

    # --- Legenda ---
    smooth_label = f" (smooth {finestra_smooth}s)" if finestra_smooth > 1 else ""
    handles = [mpatches.Patch(color='#00FF99', label=f'NSY 45.9 kHz VLF{smooth_label}')]

    if stato['xray_on'] and xray_ts:
        oggi_utc = datetime.now(timezone.utc).date()
        delta    = (oggi_utc - data_utc.date()).days
        if delta <= 6:
            xray_label = 'GOES X-ray 0.1–0.8 nm (dati reali)'
        else:
            xray_label = 'GOES X-ray 0.1–0.8 nm (⚠ profilo ricostruito da flare)'
        handles.append(mpatches.Patch(color='#FFD700', label=xray_label))
    elif stato['xray_on'] and not xray_ts:
        handles.append(mpatches.Patch(color='#FFD700', alpha=0.3,
                                      label='GOES X-ray — nessun dato'))
    else:
        handles.append(mpatches.Patch(color='#555555', alpha=0.5,
                                      label='GOES X-ray — disattivato (tasto X)'))

    if alba and tramonto:
        handles.append(mpatches.Patch(color='#404060', alpha=0.6, label='Notte'))
        handles.append(mpatches.Patch(color='#FFA500', alpha=0.8,
                                      label=f'Alba {alba.strftime("%H:%M")} UTC'))
        handles.append(mpatches.Patch(color='#FF6347', alpha=0.8,
                                      label=f'Tramonto {tramonto.strftime("%H:%M")} UTC'))

    for tipo in sorted(legenda_classi, reverse=True):
        colore = COLORI_FLARE.get(tipo, '#FFFFFF')
        handles.append(mpatches.Patch(color=colore, alpha=0.6,
                                      label=f'Flare classe {tipo}'))

    if ha_limb:
        handles.append(mpatches.Patch(color=COLORE_LIMB, alpha=0.6,
                                      label='Flare beyond-the-limb (★ BTL)'))
    if ha_manuale:
        handles.append(mpatches.Patch(color=COLORE_LIMB, alpha=0.6,
                                      label='Flare inserito manualmente (✎ BTL)'))

    ax.legend(handles=handles, loc='lower left', facecolor='#1a1a2e',
              edgecolor='#444444', labelcolor='white', fontsize=9)

    # --- Titolo ---
    smooth_title = f" — Smooth {finestra_smooth}s" if finestra_smooth > 1 else ""
    xray_stato   = "ON" if stato['xray_on'] else "OFF"
    ax.set_title(
        f"Segnale VLF NSY + Flusso X-ray GOES + Flare Solari  —  "
        f"{data_utc.strftime('%Y-%m-%d')} UTC{smooth_title}\n"
        f"Tasti:  A = prec.  |  D = succ.  |  I = data  |  M = flare manuale  |  "
        f"X = X-ray [{xray_stato}]  |  R = Aggiorna  |  P = PNG  |  Q = esci",
        color='white', fontsize=10, fontweight='bold', pad=10
    )

    fig.canvas.draw_idle()
    print(f"Grafico aggiornato: {data_utc.strftime('%Y-%m-%d')}  |  X-ray: {xray_stato}")


def aggiorna_titolo_mancante(data_utc):
    ax = stato['ax']
    ax.cla()
    ax.set_facecolor('#0d0d1a')
    ax.text(0.5, 0.5,
            f"Nessun dato disponibile per\n{data_utc.strftime('%Y-%m-%d')}",
            transform=ax.transAxes, color='#FF6347', fontsize=14,
            ha='center', va='center')
    ax.set_title(
        f"Segnale VLF NSY  —  {data_utc.strftime('%Y-%m-%d')} UTC\n"
        f"Tasti:  A = giorno precedente  |  D = giorno successivo  |  "
        f"I = inserisci data  |  R = Aggiorna  |  P = salva PNG  |  Q = esci",
        color='white', fontsize=10, fontweight='bold', pad=10
    )
    stato['fig'].canvas.draw_idle()


# =============================================================
# SALVATAGGIO PNG
# =============================================================

def salva_png():
    data_utc        = stato['data']
    finestra_smooth = stato['smooth']
    smooth_suffix   = f"_smooth{finestra_smooth}s" if finestra_smooth > 1 else ""
    xray_suffix     = "_xray" if stato['xray_on'] else ""
    output_png = os.path.join(
        CARTELLA_STORICO,
        f"VLF_flare_{data_utc.strftime('%Y-%m-%d')}{smooth_suffix}{xray_suffix}.png"
    )
    stato['fig'].savefig(output_png, dpi=150, bbox_inches='tight',
                         facecolor=stato['fig'].get_facecolor())
    print(f"Grafico salvato: {output_png}")


# =============================================================
# HANDLER TASTIERA
# =============================================================

def on_key(event):
    if event.key is None:
        return
    tasto = event.key.lower()
    print(f"[KEY] Tasto ricevuto: '{event.key}' -> '{tasto}'")

    if tasto == 'r':
        print("Aggiornamento grafico e download flares/X-ray...")
        # Forza re-download X-ray anche per la stessa data
        stato['xray_data'] = None
        disegna_grafico()

    elif tasto == 'x':
        stato['xray_on'] = not stato['xray_on']
        print(f"Overlay X-ray: {'ATTIVATO' if stato['xray_on'] else 'DISATTIVATO'}")
        disegna_grafico()

    elif tasto == 'a':
        nuova_data = stato['data'] - timedelta(days=1)
        if trova_csv_per_data(nuova_data) is None:
            print(f"Nessun file CSV per {nuova_data.strftime('%Y-%m-%d')}, "
                  f"premi I per inserire una data.")
        else:
            stato['data'] = nuova_data
            print(f"<- Giorno precedente: {stato['data'].strftime('%Y-%m-%d')}")
            disegna_grafico()

    elif tasto == 'd':
        nuova_data = stato['data'] + timedelta(days=1)
        if trova_csv_per_data(nuova_data) is None:
            print(f"Nessun file CSV per {nuova_data.strftime('%Y-%m-%d')}, "
                  f"premi I per inserire una data.")
        else:
            stato['data'] = nuova_data
            print(f"-> Giorno successivo: {stato['data'].strftime('%Y-%m-%d')}")
            disegna_grafico()

    elif tasto == 'i':
        import tkinter as tk
        from tkcalendar import Calendar

        def apri_calendario():
            root = tk.Tk()
            root.title("Scegli la data")
            root.attributes('-topmost', True)
            root.resizable(False, False)
            root.configure(bg='#0d0d1a')

            data_corrente = stato['data']

            cal = Calendar(
                root,
                selectmode='day',
                year=data_corrente.year,
                month=data_corrente.month,
                day=data_corrente.day,
                date_pattern='yyyy-mm-dd',
                locale='it_IT',
                background='#1a1a2e',
                foreground='white',
                headersbackground='#2a2a4e',
                headersforeground='#00FF99',
                selectbackground='#00FF99',
                selectforeground='black',
                normalbackground='#0d0d1a',
                normalforeground='white',
                weekendbackground='#1a1a2e',
                weekendforeground='#aaaacc',
                othermonthforeground='#444466',
                othermonthbackground='#0d0d1a',
                font=('Consolas', 10),
            )
            cal.pack(padx=10, pady=10)

            lbl = tk.Label(root, text="Seleziona una data con dati disponibili",
                           bg='#0d0d1a', fg='#aaaacc', font=('Consolas', 9))
            lbl.pack(pady=(0, 5))

            risultato = {'data': None}

            def conferma():
                data_scelta_str = cal.get_date()
                try:
                    nuova_data = datetime.strptime(data_scelta_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except ValueError:
                    lbl.config(text=f"Formato non valido: {data_scelta_str}", fg='#FF6347')
                    return
                if trova_csv_per_data(nuova_data) is None:
                    lbl.config(
                        text=f"Nessun dato per {data_scelta_str} — scegli un'altra data.",
                        fg='#FF6347')
                    return
                risultato['data'] = nuova_data
                root.quit()
                root.destroy()

            def annulla():
                root.quit()
                root.destroy()

            frame_btn = tk.Frame(root, bg='#0d0d1a')
            frame_btn.pack(pady=(0, 10))

            tk.Button(
                frame_btn, text="Conferma", command=conferma,
                bg='#00FF99', fg='black', font=('Consolas', 10, 'bold'),
                relief='flat', padx=12, pady=4
            ).pack(side='left', padx=8)

            tk.Button(
                frame_btn, text="Annulla", command=annulla,
                bg='#333355', fg='white', font=('Consolas', 10),
                relief='flat', padx=12, pady=4
            ).pack(side='left', padx=8)

            root.protocol("WM_DELETE_WINDOW", annulla)
            root.mainloop()
            return risultato['data']

        nuova_data = apri_calendario()
        if nuova_data is not None:
            stato['data'] = nuova_data
            print(f"Data selezionata: {nuova_data.strftime('%Y-%m-%d')}")
            disegna_grafico()
        else:
            print("Selezione data annullata.")

    elif tasto == 'm':
        import tkinter as tk
        from tkinter import simpledialog, messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        data_str = stato['data'].strftime("%Y-%m-%d")

        try:
            classe = simpledialog.askstring(
                "Flare manuale",
                f"Classe flare (es. C5.3, M1.2, X1.0)\nData: {data_str}",
                parent=root)
            if not classe:
                root.destroy()
                return
            classe = classe.strip().upper()

            ora_picco = simpledialog.askstring(
                "Flare manuale",
                "Ora picco UTC (HH:MM):",
                parent=root)
            if not ora_picco:
                root.destroy()
                return
            ora_picco = ora_picco.strip()

            ora_inizio = simpledialog.askstring(
                "Flare manuale",
                f"Ora inizio UTC (HH:MM)\n[lascia vuoto per usare {ora_picco}]:",
                parent=root)
            ora_inizio = ora_inizio.strip() if ora_inizio and ora_inizio.strip() else ora_picco

            ora_fine = simpledialog.askstring(
                "Flare manuale",
                f"Ora fine UTC (HH:MM)\n[lascia vuoto per usare {ora_picco}]:",
                parent=root)
            ora_fine = ora_fine.strip() if ora_fine and ora_fine.strip() else ora_picco

            nota = simpledialog.askstring(
                "Flare manuale",
                "Nota opzionale (es. beyond-the-limb, fonte SpaceWeatherLive):",
                parent=root)
            nota = nota.strip() if nota else ""

            root.destroy()

            salva_flare_manuale(data_str, classe, ora_inizio, ora_picco, ora_fine, nota)
            print("Ridisegno il grafico con il nuovo flare...")
            disegna_grafico()

        except Exception as e:
            print(f"Errore inserimento flare manuale: {e}")
            try:
                root.destroy()
            except Exception:
                pass

    elif tasto == 'p':
        salva_png()

    elif tasto == 'q':
        print("Uscita.")
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
print("  I  -> Calendario per scegliere la data")
print("  M  -> Inserisci flare manuale (beyond-the-limb)")
print("  P  -> Salva grafico come PNG")
print("  R  -> Refresh Grafico, Flares e X-ray")
print("  X  -> Attiva/Disattiva overlay flusso X-ray GOES")
print("  Q  -> Esci")
print()
print(f"  Directory CSV VLF:")
print(f"    Giornaliera : {CARTELLA_GIORNALIERA}")
print(f"    Storico     : {CARTELLA_STORICO}")
print(f"  Directory CSV X-ray:")
print(f"    Cache XRAY  : {CARTELLA_XRAY}")
print()

# --- Selezione data iniziale tramite calendario ---
import tkinter as tk
from tkcalendar import Calendar

def scegli_data_iniziale():
    oggi = datetime.now(timezone.utc)
    root = tk.Tk()
    root.title("GAESID — Scegli data iniziale")
    root.attributes('-topmost', True)
    root.resizable(False, False)
    root.configure(bg='#0d0d1a')

    tk.Label(root, text="Scegli la data da visualizzare",
             bg='#0d0d1a', fg='#00FF99',
             font=('Consolas', 11, 'bold')).pack(pady=(12, 4))

    cal = Calendar(
        root,
        selectmode='day',
        year=oggi.year,
        month=oggi.month,
        day=oggi.day,
        date_pattern='yyyy-mm-dd',
        locale='it_IT',
        background='#1a1a2e',
        foreground='white',
        headersbackground='#2a2a4e',
        headersforeground='#00FF99',
        selectbackground='#00FF99',
        selectforeground='black',
        normalbackground='#0d0d1a',
        normalforeground='white',
        weekendbackground='#1a1a2e',
        weekendforeground='#aaaacc',
        othermonthforeground='#444466',
        othermonthbackground='#0d0d1a',
        font=('Consolas', 10),
    )
    cal.pack(padx=10, pady=6)

    lbl = tk.Label(root, text="", bg='#0d0d1a', fg='#FF6347',
                   font=('Consolas', 9))
    lbl.pack(pady=(0, 4))

    risultato = {'data': None}

    def conferma():
        data_str = cal.get_date()
        try:
            data = datetime.strptime(data_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            lbl.config(text=f"Formato non valido: {data_str}")
            return
        if trova_csv_per_data(data) is None:
            lbl.config(text=f"Nessun dato per {data_str} — scegli un'altra data.")
            return
        risultato['data'] = data
        root.destroy()

    def usa_oggi():
        risultato['data'] = oggi
        root.destroy()

    frame_btn = tk.Frame(root, bg='#0d0d1a')
    frame_btn.pack(pady=(0, 12))

    tk.Button(frame_btn, text="Conferma", command=conferma,
              bg='#00FF99', fg='black', font=('Consolas', 10, 'bold'),
              relief='flat', padx=12, pady=4).pack(side='left', padx=8)

    tk.Button(frame_btn, text="Oggi", command=usa_oggi,
              bg='#333355', fg='white', font=('Consolas', 10),
              relief='flat', padx=12, pady=4).pack(side='left', padx=8)

    root.mainloop()
    return risultato['data']

data_scelta = scegli_data_iniziale()
if data_scelta is None:
    print("Nessuna data selezionata, uso oggi.")
    data_scelta = datetime.now(timezone.utc)

# --- Smoothing ---
while True:
    smooth_input = input("Secondi di smoothing (1 = nessuno, es. 120): ").strip()
    if smooth_input == "":
        finestra_smooth = 1
        break
    try:
        finestra_smooth = int(smooth_input)
        if finestra_smooth < 1:
            print("Inserisci un numero >= 1.\n")
            continue
        break
    except ValueError:
        print("Inserisci un numero intero.\n")

fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor('#0d0d1a')
ax.set_facecolor('#0d0d1a')

# Disabilita i keybinding predefiniti di matplotlib che
# confliggono con i nostri tasti personalizzati.
for tasto_da_rimuovere in ['p', 'q', 's', 'c', 'a', 'd', 'g', 'h', 'r', 'k', 'l', 'o', 'v', 'f', 'x']:
    for keymap in plt.rcParams:
        if keymap.startswith('keymap.') and tasto_da_rimuovere in plt.rcParams[keymap]:
            plt.rcParams[keymap].remove(tasto_da_rimuovere)

stato['data']   = data_scelta
stato['smooth'] = finestra_smooth
stato['fig']    = fig
stato['ax']     = ax

fig.canvas.mpl_connect('key_press_event', on_key)

disegna_grafico()
plt.tight_layout()
plt.show()
