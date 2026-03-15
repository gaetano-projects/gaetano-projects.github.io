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

# =============================================================
#  CONFIGURAZIONE UTENTE — modifica questi valori
# =============================================================
#
#  Requisiti: Python 3.13 o superiore
#  Librerie:  pip install requests matplotlib numpy scipy ephem pytz
#
#  CARTELLA_CSV : cartella dove si trovano i file GAESID_DHO38_YYYY-MM-DD.csv
#  STAZIONE_ID  : identificativo della tua stazione (usato nel nome dei file)
#  TRASMETTITORE: nome della stazione VLF monitorata
#  FREQUENZA_KHZ: frequenza in kHz della stazione monitorata
#  LAT, LON     : coordinate geografiche del sito di ricezione (gradi decimali)
#  ELEVAZIONE   : quota del sito di ricezione in metri
#
CARTELLA_CSV  = r"C:\SID\dati"        # <-- modifica con il tuo percorso
STAZIONE_ID   = "GAESID"               # <-- modifica con il tuo ID stazione
TRASMETTITORE = "DHO38"                # <-- modifica con la stazione monitorata
FREQUENZA_KHZ = 23.4                   # <-- modifica con la frequenza in kHz
LAT           = "45.07"               # <-- modifica con la tua latitudine
LON           = "7.68"                # <-- modifica con la tua longitudine
ELEVAZIONE    = 240                    # <-- modifica con la tua quota in metri
# =============================================================

# Colori per le classi di flare
COLORI_FLARE = {
    'X': '#FF0000',
    'M': '#FF8C00',
    'C': '#FFD700',
    'B': '#00BFFF',
    'A': '#90EE90',
}

# ============================================================
#  COMANDI DA TASTIERA
# ============================================================
#  A  →  Giorno precedente
#  D  →  Giorno successivo
#  I  →  Inserisci nuova data manualmente
#  P  →  Stampa/salva il grafico corrente come PNG
#  Q  →  Esci dal programma
# ============================================================

from scipy.signal import savgol_filter

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
        kernel = np.ones(finestra) / finestra
        smoothed = np.convolve(valori, kernel, mode='same')
        meta = finestra // 2
        if len(smoothed) > meta * 2:
            smoothed[:meta] = smoothed[meta]
            smoothed[-meta:] = smoothed[-meta - 1]
        return smoothed.tolist()

def calcola_alba_tramonto(data_utc):
    osservatore = ephem.Observer()
    osservatore.lat = LAT
    osservatore.lon = LON
    osservatore.elevation = ELEVAZIONE
    osservatore.date = data_utc.strftime('%Y/%m/%d 00:00:00')
    osservatore.pressure = 0
    sole = ephem.Sun()
    try:
        alba     = osservatore.next_rising(sole).datetime().replace(tzinfo=timezone.utc)
        tramonto = osservatore.next_setting(sole).datetime().replace(tzinfo=timezone.utc)
    except Exception as e:
        print(f"Errore calcolo alba/tramonto: {e}")
        alba, tramonto = None, None
    return alba, tramonto

def scarica_flare(data_utc):
    data_str = data_utc.strftime("%Y%m%d")
    anno = data_utc.strftime("%Y")
    oggi_utc = datetime.now(timezone.utc).date()
    delta_giorni = (oggi_utc - data_utc.date()).days

    if delta_giorni <= 7:
        urls = [
            ("json_noaa", "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json"),
            ("json_noaa", "https://services.swpc.noaa.gov/json/goes/secondary/xray-flares-7-day.json"),
        ]
    else:
        data_inizio = data_utc.strftime("%Y-%m-%d")
        data_fine   = data_utc.strftime("%Y-%m-%d")
        urls = [
            ("json_donki", f"https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/FLR?startDate={data_inizio}&endDate={data_fine}"),
            ("txt", f"https://www.swpc.noaa.gov/archive/{anno}/{data_str}events.txt"),
        ]

    for tipo, url in urls:
        print(f"Tentativo download da: {url}")
        try:
            r = requests.get(url, timeout=(5, 8))
            r.raise_for_status()
            print(f"Download riuscito da: {url}")
            return r.text, tipo
        except requests.exceptions.ConnectTimeout:
            print(f"Timeout connessione, passo al prossimo...")
        except requests.exceptions.ReadTimeout:
            print(f"Timeout lettura, passo al prossimo...")
        except requests.exceptions.HTTPError as e:
            print(f"Errore HTTP {e.response.status_code}, passo al prossimo...")
        except Exception as e:
            print(f"Errore: {e}, passo al prossimo...")

    print(">>> Nessuna fonte disponibile. Grafico senza dati flare.")
    return None, None

def parse_flare(testo, data_utc, url_fonte=""):
    flares = []
    data_str = data_utc.strftime("%Y-%m-%d")

    def hhmm_to_dt(hhmm):
        return datetime.strptime(
            f"{data_str} {hhmm[:2]}:{hhmm[2:4]}:00",
            "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=timezone.utc)

    if url_fonte == 'json_noaa':
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
                    flares.append({'inizio': inizio, 'picco': picco, 'fine': fine, 'classe': classe, 'tipo': classe[0]})
                except Exception:
                    continue
        except Exception as e:
            print(f"Errore parsing JSON NOAA: {e}")

    elif url_fonte == 'json_donki':
        try:
            dati = json.loads(testo)
            for evento in dati:
                try:
                    classe = evento.get('classType', '')
                    if not classe or classe[0] not in 'XMCBA':
                        continue
                    if classe[0] in ('A', 'B'):
                        continue
                    inizio = datetime.strptime(evento['beginTime'], "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
                    picco  = datetime.strptime(evento['peakTime'],  "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
                    fine_str = evento.get('endTime')
                    fine = datetime.strptime(fine_str, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc) if fine_str else picco + timedelta(minutes=10)
                    if inizio.date() != data_utc.date():
                        continue
                    flares.append({'inizio': inizio, 'picco': picco, 'fine': fine, 'classe': classe, 'tipo': classe[0]})
                except Exception:
                    continue
        except Exception as e:
            print(f"Errore parsing JSON DONKI: {e}")

    else:
        for riga in testo.splitlines():
            if 'XRA' not in riga or riga.startswith('#'):
                continue
            try:
                parti = riga.split()
                ora_inizio = parti[2]
                ora_picco  = parti[3]
                ora_fine   = parti[4]
                classe = None
                for p in parti:
                    if re.match(r'^[XMCBA]\d+\.\d+$', p) or re.match(r'^[XMCBA]\d+$', p):
                        classe = p
                        break
                if classe is None:
                    continue
                flares.append({'inizio': hhmm_to_dt(ora_inizio), 'picco': hhmm_to_dt(ora_picco), 'fine': hhmm_to_dt(ora_fine), 'classe': classe, 'tipo': classe[0]})
            except Exception:
                continue

    print(f"Trovati {len(flares)} flare")
    return flares

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

def trova_csv_per_data(data_utc):
    data_str  = data_utc.strftime("%Y-%m-%d")
    filepath  = os.path.join(CARTELLA_CSV, f"{STAZIONE_ID}_{TRASMETTITORE}_{data_str}.csv")
    return filepath if os.path.exists(filepath) else None

# ---------------------------------------------------------------
#  Stato globale condiviso tra grafico e handler tastiera
# ---------------------------------------------------------------
stato = {
    'data':    None,
    'smooth':  1,
    'fig':     None,
    'ax':      None,
}

def disegna_grafico():
    """Ridisegna il grafico per la data corrente in stato['data']."""
    data_utc      = stato['data']
    finestra_smooth = stato['smooth']

    csv_path = trova_csv_per_data(data_utc)
    if csv_path is None:
        print(f"Nessun file CSV trovato per {data_utc.strftime('%Y-%m-%d')}")
        aggiorna_titolo_mancante(data_utc)
        return

    print(f"\nLettura CSV: {csv_path}")
    timestamps, valori = leggi_csv_sid(csv_path)
    if not timestamps:
        print("Nessun dato nel file CSV.")
        return

    valori_plot = applica_smoothing(valori, finestra_smooth) if finestra_smooth > 1 else valori

    testo_flare, url_fonte = scarica_flare(data_utc)
    flares = parse_flare(testo_flare, data_utc, url_fonte) if testo_flare else []

    alba, tramonto = calcola_alba_tramonto(data_utc)

    fig = stato['fig']
    ax  = stato['ax']
    ax.cla()

    inizio_giorno = data_utc.replace(hour=0,  minute=0,  second=0,  tzinfo=timezone.utc)
    fine_giorno   = data_utc.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
    ax.set_xlim(inizio_giorno, fine_giorno)

    # Calcola i limiti Y dai dati effettivi con un margine del 5%
    v_min = min(valori_plot)
    v_max = max(valori_plot)
    margine = (v_max - v_min) * 0.10
    ax.set_ylim(v_min - margine, v_max + margine)

    if alba and tramonto:
        ax.axvspan(inizio_giorno, alba,     color='#404060', alpha=0.45, zorder=1)
        ax.axvspan(tramonto, fine_giorno,   color='#404060', alpha=0.45, zorder=1)
        ax.axvline(alba,     color='#FFA500', linewidth=1.2, linestyle='--', alpha=0.8, zorder=2)
        ax.axvline(tramonto, color='#FF6347', linewidth=1.2, linestyle='--', alpha=0.8, zorder=2)
        y_label = min(valori_plot) + (max(valori_plot) - min(valori_plot)) * 0.05
        ax.text(alba,     y_label, f"☀ Alba\n{alba.strftime('%H:%M')} UTC",
                color='#FFA500', fontsize=8, ha='left',  va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d0d1a', alpha=0.7, edgecolor='#FFA500'))
        ax.text(tramonto, y_label, f"☀ Tramonto\n{tramonto.strftime('%H:%M')} UTC",
                color='#FF6347', fontsize=8, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d0d1a', alpha=0.7, edgecolor='#FF6347'))

    ax.plot(timestamps, valori_plot, color='#00FF99', linewidth=0.8, alpha=0.9, zorder=3)
    ax.set_xlabel('Ora UTC', color='white', fontsize=11)
    ax.set_ylabel('Segnale VLF (dB)', color='#00FF99', fontsize=11)
    ax.tick_params(colors='white')
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, color='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
    ax.grid(True, alpha=0.15, color='white')

    legenda_classi = set()
    for flare in flares:
        colore = COLORI_FLARE.get(flare['tipo'], '#FFFFFF')
        ax.axvspan(flare['inizio'], flare['fine'], alpha=0.25, color=colore, zorder=4)
        ax.axvline(flare['picco'], color=colore, linewidth=1.5, alpha=0.8, linestyle='--', zorder=5)
        ax.text(flare['picco'], np.percentile(valori_plot, 95),
                flare['classe'], color=colore, fontsize=8,
                ha='center', va='bottom', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d0d1a', alpha=0.7, edgecolor=colore))
        legenda_classi.add(flare['tipo'])

    smooth_label = f" (smooth {finestra_smooth}s)" if finestra_smooth > 1 else ""
    handles = [mpatches.Patch(color='#00FF99', label=f'{TRASMETTITORE} {FREQUENZA_KHZ} kHz VLF{smooth_label}')]
    if alba and tramonto:
        handles.append(mpatches.Patch(color='#404060', alpha=0.6, label='Notte'))
        handles.append(mpatches.Patch(color='#FFA500', alpha=0.8, label=f'Alba {alba.strftime("%H:%M")} UTC'))
        handles.append(mpatches.Patch(color='#FF6347', alpha=0.8, label=f'Tramonto {tramonto.strftime("%H:%M")} UTC'))
    for tipo in sorted(legenda_classi, reverse=True):
        colore = COLORI_FLARE.get(tipo, '#FFFFFF')
        handles.append(mpatches.Patch(color=colore, alpha=0.6, label=f'Flare classe {tipo}'))
    ax.legend(handles=handles, loc='lower left', facecolor='#1a1a2e',
              edgecolor='#444444', labelcolor='white', fontsize=9)

    smooth_title = f" — Smooth {finestra_smooth}s" if finestra_smooth > 1 else ""
    ax.set_title(
        f"Segnale VLF {TRASMETTITORE} + Flare Solari  —  {data_utc.strftime('%Y-%m-%d')} UTC{smooth_title}\n"
        f"  Tasti:  A = giorno precedente  |  D = giorno successivo  |  I = inserisci data  |  P = salva PNG  |  Q = esci",
        color='white', fontsize=11, fontweight='bold', pad=10
    )

    fig.canvas.draw_idle()
    print(f"Grafico aggiornato: {data_utc.strftime('%Y-%m-%d')}")

def aggiorna_titolo_mancante(data_utc):
    ax = stato['ax']
    ax.cla()
    ax.set_facecolor('#0d0d1a')
    ax.text(0.5, 0.5, f"Nessun dato disponibile per\n{data_utc.strftime('%Y-%m-%d')}",
            transform=ax.transAxes, color='#FF6347', fontsize=14,
            ha='center', va='center')
    ax.set_title(
        f"Segnale VLF {TRASMETTITORE}  —  {data_utc.strftime('%Y-%m-%d')} UTC\n"
        f"  Tasti:  A = giorno precedente  |  D = giorno successivo  |  I = inserisci data  |  P = salva PNG  |  Q = esci",
        color='white', fontsize=11, fontweight='bold', pad=10
    )
    stato['fig'].canvas.draw_idle()

def salva_png():
    data_utc = stato['data']
    finestra_smooth = stato['smooth']
    smooth_suffix = f"_smooth{finestra_smooth}s" if finestra_smooth > 1 else ""
    output_png = os.path.join(
        CARTELLA_CSV,
        f"VLF_{TRASMETTITORE}_{data_utc.strftime('%Y-%m-%d')}{smooth_suffix}.png"
    )
    stato['fig'].savefig(output_png, dpi=150, bbox_inches='tight',
                         facecolor=stato['fig'].get_facecolor())
    print(f"Grafico salvato: {output_png}")

def on_key(event):
    if event.key is None:
        return
    tasto = event.key.lower()

    if tasto == 'a':
        # Giorno precedente
        nuova_data = stato['data'] - timedelta(days=1)
        if trova_csv_per_data(nuova_data) is None:
            print(f">>> Nessun file CSV trovato per {nuova_data.strftime('%Y-%m-%d')}, premi I per inserire una data.")
        else:
            stato['data'] = nuova_data
            print(f"← Giorno precedente: {stato['data'].strftime('%Y-%m-%d')}")
            disegna_grafico()

    elif tasto == 'd':
        # Giorno successivo
        nuova_data = stato['data'] + timedelta(days=1)
        if trova_csv_per_data(nuova_data) is None:
            print(f">>> Nessun file CSV trovato per {nuova_data.strftime('%Y-%m-%d')}, premi I per inserire una data.")
        else:
            stato['data'] = nuova_data
            print(f"→ Giorno successivo: {stato['data'].strftime('%Y-%m-%d')}")
            disegna_grafico()

    elif tasto == 'i':
        # Inserisci nuova data
        plt.pause(0.1)
        while True:
            data_input = input("Inserisci data (YYYY-MM-DD): ").strip()
            try:
                nuova_data = datetime.strptime(data_input, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                print(">>> Formato non valido. Usa YYYY-MM-DD (es. 2026-03-13)")
                continue
            if trova_csv_per_data(nuova_data) is None:
                print(f">>> Nessun file CSV trovato per {nuova_data.strftime('%Y-%m-%d')}, riprova.")
                continue
            stato['data'] = nuova_data
            print(f"Data impostata: {nuova_data.strftime('%Y-%m-%d')}")
            disegna_grafico()
            break

    elif tasto == 'p':
        # Salva PNG
        salva_png()

    elif tasto == 'q':
        # Esci
        print("Uscita dal programma.")
        plt.close('all')

# ---------------------------------------------------------------
#  AVVIO
# ---------------------------------------------------------------
print("=" * 60)
print(f"  VLF SID Monitor — {TRASMETTITORE} {FREQUENZA_KHZ} kHz — Stazione {STAZIONE_ID}")
print("=" * 60)
print()
print("  COMANDI DA TASTIERA (clicca sul grafico prima):")
print("  A  →  Giorno precedente")
print("  D  →  Giorno successivo")
print("  I  →  Inserisci nuova data")
print("  P  →  Salva grafico come PNG")
print("  Q  →  Esci")
print()

# Chiedi data iniziale
while True:
    data_input = input("Inserisci la data (YYYY-MM-DD) o premi Enter per oggi: ").strip()
    if data_input == "":
        data_scelta = datetime.now(timezone.utc)
    else:
        try:
            data_scelta = datetime.strptime(data_input, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print(">>> Formato non valido. Usa YYYY-MM-DD (es. 2026-03-13)\n")
            continue
    if trova_csv_per_data(data_scelta) is None:
        print(f">>> Nessun file CSV trovato per {data_scelta.strftime('%Y-%m-%d')}, riprova.\n")
        continue
    break

# Chiedi smoothing
while True:
    smooth_input = input("Secondi di smoothing (1 = nessuno, es. 30): ").strip()
    if smooth_input == "":
        finestra_smooth = 1
        break
    try:
        finestra_smooth = int(smooth_input)
        if finestra_smooth < 1:
            print(">>> Inserisci un numero >= 1.\n")
            continue
        break
    except ValueError:
        print(">>> Inserisci un numero intero.\n")

# Crea figura
fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor('#0d0d1a')
ax.set_facecolor('#0d0d1a')

stato['data']   = data_scelta
stato['smooth'] = finestra_smooth
stato['fig']    = fig
stato['ax']     = ax

# Collega handler tastiera
fig.canvas.mpl_connect('key_press_event', on_key)

# Primo disegno
disegna_grafico()
plt.tight_layout()
plt.show()
