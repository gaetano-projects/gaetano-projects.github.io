import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime, timezone
import re
import os
import ephem
import time
import json
from scipy.signal import savgol_filter

# =============================================================
#  CONFIGURAZIONE UTENTE — modifica questi valori
# =============================================================
#
#  Requisiti: Python 3.13 o superiore
#  Librerie:  pip install requests matplotlib numpy scipy ephem pytz
#
#  CARTELLA_CSV             : cartella dove si trovano i file giornalieri
#  STAZIONE_ID              : identificativo della tua stazione
#  TRASMETTITORE            : nome della stazione VLF monitorata
#  FREQUENZA_KHZ            : frequenza in kHz della stazione monitorata
#  LAT, LON                 : coordinate geografiche del sito di ricezione
#  ELEVAZIONE               : quota del sito di ricezione in metri
#  INTERVALLO_AGGIORNAMENTO : minuti tra un aggiornamento e il successivo
#
CARTELLA_CSV             = r"C:\SID\dati"    # <-- modifica
STAZIONE_ID              = "GAESID"          # <-- modifica
TRASMETTITORE            = "DHO38"           # <-- modifica
FREQUENZA_KHZ            = 23.4             # <-- modifica
LAT                      = "45.07"          # <-- modifica
LON                      = "7.68"           # <-- modifica
ELEVAZIONE               = 240              # <-- modifica
INTERVALLO_AGGIORNAMENTO = 5                # <-- minuti tra aggiornamenti
# =============================================================

# Colori per le classi di flare
COLORI_FLARE = {
    'X': '#FF0000',
    'M': '#FF8C00',
    'C': '#FFD700',
    'B': '#00BFFF',
    'A': '#90EE90',
}


def applica_smoothing(valori, finestra):
    """Savitzky-Golay: preserva meglio i picchi e le forme degli eventi."""
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


def calcola_alba_tramonto(data_utc):
    osservatore           = ephem.Observer()
    osservatore.lat       = LAT
    osservatore.lon       = LON
    osservatore.elevation = ELEVAZIONE
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


def scarica_flare(data_utc):
    """Scarica flare da NOAA con URL multipli come fallback."""
    data_str   = data_utc.strftime("%Y%m%d")
    anno       = data_utc.strftime("%Y")
    oggi_utc   = datetime.now(timezone.utc).date()
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
            ("txt",        f"https://www.swpc.noaa.gov/archive/{anno}/{data_str}events.txt"),
        ]

    for tipo, url in urls:
        print(f"Tentativo download da: {url}")
        try:
            r = requests.get(url, timeout=(5, 8))
            r.raise_for_status()
            print(f"Download riuscito da: {url}")
            return r.text, tipo
        except requests.exceptions.ConnectTimeout:
            print("Timeout connessione, passo al prossimo...")
        except requests.exceptions.ReadTimeout:
            print("Timeout lettura, passo al prossimo...")
        except requests.exceptions.HTTPError as e:
            print(f"Errore HTTP {e.response.status_code}, passo al prossimo...")
        except Exception as e:
            print(f"Errore: {e}, passo al prossimo...")

    print(">>> Nessuna fonte disponibile. Grafico senza dati flare.")
    return None, None


def parse_flare(testo, data_utc, url_fonte=""):
    """Estrae i flare dal JSON NOAA, JSON DONKI o testo archivio."""
    flares   = []
    data_str = data_utc.strftime("%Y-%m-%d")

    def hhmm_to_dt(hhmm):
        return datetime.strptime(
            f"{data_str} {hhmm[:2]}:{hhmm[2:4]}:00", "%Y-%m-%d %H:%M:%S"
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
                    flares.append({'inizio': inizio, 'picco': picco, 'fine': fine,
                                   'classe': classe, 'tipo': classe[0]})
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
                    inizio   = datetime.strptime(evento['beginTime'], "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
                    picco    = datetime.strptime(evento['peakTime'],  "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
                    fine_str = evento.get('endTime')
                    fine     = datetime.strptime(fine_str, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc) if fine_str else picco
                    if inizio.date() != data_utc.date():
                        continue
                    flares.append({'inizio': inizio, 'picco': picco, 'fine': fine,
                                   'classe': classe, 'tipo': classe[0]})
                except Exception:
                    continue
        except Exception as e:
            print(f"Errore parsing JSON DONKI: {e}")

    else:
        for riga in testo.splitlines():
            if 'XRA' not in riga or riga.startswith('#'):
                continue
            try:
                parti      = riga.split()
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
                flares.append({'inizio': hhmm_to_dt(ora_inizio), 'picco': hhmm_to_dt(ora_picco),
                               'fine': hhmm_to_dt(ora_fine), 'classe': classe, 'tipo': classe[0]})
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
    data_str = data_utc.strftime("%Y-%m-%d")
    filepath = os.path.join(CARTELLA_CSV, f"{STAZIONE_ID}_{TRASMETTITORE}_{data_str}.csv")
    return filepath if os.path.exists(filepath) else None


def disegna_grafico(ax, data_utc, finestra_smooth, alba, tramonto, flares,
                    timestamps, valori_plot):
    ax.clear()
    ax.set_facecolor('#0d0d1a')

    inizio_giorno = data_utc.replace(hour=0,  minute=0,  second=0,  tzinfo=timezone.utc)
    fine_giorno   = data_utc.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
    ax.set_xlim(inizio_giorno, fine_giorno)

    # Limiti Y adattativi
    v_min   = min(valori_plot)
    v_max   = max(valori_plot)
    margine = (v_max - v_min) * 0.10
    ax.set_ylim(v_min - margine, v_max + margine)

    if alba and tramonto:
        ax.axvspan(inizio_giorno, alba,    color='#404060', alpha=0.45, zorder=1)
        ax.axvspan(tramonto, fine_giorno,  color='#404060', alpha=0.45, zorder=1)
        ax.axvline(alba,     color='#FFA500', linewidth=1.2, linestyle='--', alpha=0.8, zorder=2)
        ax.axvline(tramonto, color='#FF6347', linewidth=1.2, linestyle='--', alpha=0.8, zorder=2)
        y_label = v_min + (v_max - v_min) * 0.05
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
    handles = [mpatches.Patch(color='#00FF99',
                              label=f'{TRASMETTITORE} {FREQUENZA_KHZ} kHz VLF{smooth_label}')]
    if alba and tramonto:
        handles.append(mpatches.Patch(color='#404060', alpha=0.6, label='Notte'))
        handles.append(mpatches.Patch(color='#FFA500', alpha=0.8,
                                      label=f'Alba {alba.strftime("%H:%M")} UTC'))
        handles.append(mpatches.Patch(color='#FF6347', alpha=0.8,
                                      label=f'Tramonto {tramonto.strftime("%H:%M")} UTC'))
    for tipo in sorted(legenda_classi, reverse=True):
        colore = COLORI_FLARE.get(tipo, '#FFFFFF')
        handles.append(mpatches.Patch(color=colore, alpha=0.6, label=f'Flare classe {tipo}'))
    ax.legend(handles=handles, loc='lower left', facecolor='#1a1a2e',
              edgecolor='#444444', labelcolor='white', fontsize=9)

    ora_aggiornamento = datetime.now(timezone.utc).strftime('%H:%M:%S')
    smooth_title = f" — Smooth {finestra_smooth}s" if finestra_smooth > 1 else ""
    ax.set_title(
        f"Segnale VLF {TRASMETTITORE} + Flare Solari  —  "
        f"{data_utc.strftime('%Y-%m-%d')} UTC{smooth_title}  "
        f"[Aggiornato: {ora_aggiornamento} UTC]",
        color='white', fontsize=12, fontweight='bold', pad=12
    )


def monitor(data_utc, finestra_smooth, intervallo_minuti):
    """Loop di monitoraggio con aggiornamento automatico."""
    alba, tramonto = calcola_alba_tramonto(data_utc)

    plt.ion()
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor('#0d0d1a')

    while True:
        csv_path = trova_csv_per_data(data_utc)
        if csv_path is None:
            print(f">>> File CSV non trovato per {data_utc.strftime('%Y-%m-%d')}, riprovo...")
        else:
            timestamps, valori = leggi_csv_sid(csv_path)
            if timestamps:
                valori_plot = applica_smoothing(valori, finestra_smooth)
                print(f"\n>>> Aggiornamento {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
                testo_flare, url_fonte = scarica_flare(data_utc)
                flares = parse_flare(testo_flare, data_utc, url_fonte) if testo_flare else []
                print(f">>> {len(flares)} flare, {len(timestamps)} campioni")

                disegna_grafico(ax, data_utc, finestra_smooth, alba, tramonto,
                                flares, timestamps, valori_plot)
                fig.canvas.draw()
                fig.canvas.flush_events()

                # Salva PNG aggiornato
                smooth_suffix = f"_smooth{finestra_smooth}s" if finestra_smooth > 1 else ""
                output_png = os.path.join(
                    CARTELLA_CSV,
                    f"VLF_live_{TRASMETTITORE}_{data_utc.strftime('%Y-%m-%d')}{smooth_suffix}.png"
                )
                fig.savefig(output_png, dpi=150, bbox_inches='tight',
                            facecolor=fig.get_facecolor())
                print(f">>> PNG salvato: {output_png}")

        print(f">>> Prossimo aggiornamento tra {intervallo_minuti} minuti. "
              f"Chiudi la finestra del grafico per uscire.")

        for _ in range(intervallo_minuti * 2):
            if not plt.fignum_exists(fig.number):
                print(">>> Finestra chiusa, monitoraggio terminato.")
                return
            plt.pause(30)


# --- AVVIO ---
print("=" * 55)
print(f"  VLF SID Monitor — Monitoraggio in tempo reale")
print(f"  Stazione: {STAZIONE_ID} — {TRASMETTITORE} {FREQUENZA_KHZ} kHz")
print("=" * 55)
print()

data_scelta = datetime.now(timezone.utc)

while True:
    smooth_input = input("Secondi di smoothing (Enter = 60): ").strip()
    if smooth_input == "":
        finestra_smooth = 60
        break
    try:
        finestra_smooth = int(smooth_input)
        if finestra_smooth < 1:
            print(">>> Inserisci un numero >= 1\n")
            continue
        break
    except ValueError:
        print(">>> Inserisci un numero intero\n")

while True:
    intervallo_input = input(f"Intervallo aggiornamento in minuti (Enter = {INTERVALLO_AGGIORNAMENTO}): ").strip()
    if intervallo_input == "":
        intervallo = INTERVALLO_AGGIORNAMENTO
        break
    try:
        intervallo = int(intervallo_input)
        if intervallo < 1:
            print(">>> Inserisci un numero >= 1\n")
            continue
        break
    except ValueError:
        print(">>> Inserisci un numero intero\n")

print(f"\n>>> Avvio monitoraggio — aggiornamento ogni {intervallo} minuti")
print(">>> Chiudi la finestra del grafico per terminare.\n")

monitor(data_scelta, finestra_smooth, intervallo)
