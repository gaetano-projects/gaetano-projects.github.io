import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
#  Librerie:  pip install requests matplotlib numpy scipy ephem pytz
#
#  CARTELLA_CSV              : cartella dove si trovano i file giornalieri
#  STAZIONE_ID               : identificativo della tua stazione
#  TRASMETTITORE             : nome della stazione VLF monitorata
#  FREQUENZA_KHZ             : frequenza in kHz della stazione monitorata
#  LAT, LON                  : coordinate geografiche del sito di ricezione
#  ELEVAZIONE                : quota del sito di ricezione in metri
#  FINESTRA_BASELINE_SECONDI : finestra media mobile globale (secondi)
#  FINESTRA_LOCALE_SECONDI   : finestra baseline locale per lato (secondi)
#
CARTELLA_CSV              = r"C:\SID\dati"    # <-- modifica
STAZIONE_ID               = "GAESID"          # <-- modifica
TRASMETTITORE             = "DHO38"           # <-- modifica
FREQUENZA_KHZ             = 23.4             # <-- modifica
LAT                       = "45.07"          # <-- modifica
LON                       = "7.68"           # <-- modifica
ELEVAZIONE                = 240              # <-- modifica
FINESTRA_BASELINE_SECONDI = 3600             # 60 min — media mobile (fallback)
FINESTRA_LOCALE_SECONDI   = 1800             # 30 min per lato — baseline locale
# =============================================================

COLORI_FLARE = {
    'X': '#FF0000',
    'M': '#FF8C00',
    'C': '#FFD700',
    'B': '#00BFFF',
    'A': '#90EE90',
}

# Soglia di prossimità per raggruppare flare vicini (secondi)
SOGLIA_RAGGRUPPAMENTO_S = 1800  # 30 min

# Classi di flare per cui applicare il metodo locale
CLASSI_METODO_LOCALE = {'C', 'B'}

# Campioni minimi per ancoraggio affidabile (~10 min)
MINUTI_ANCORA = 10


# ---------------------------------------------------------------------------
# SMOOTHING  (Savitzky-Golay)
# ---------------------------------------------------------------------------
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
        meta     = finestra // 2
        if len(smoothed) > meta * 2:
            smoothed[:meta]  = smoothed[meta]
            smoothed[-meta:] = smoothed[-meta - 1]
        return smoothed.tolist()


# ---------------------------------------------------------------------------
# DETRENDING  —  Metodo 1: Media mobile centrata (globale)
# ---------------------------------------------------------------------------
def calcola_detrend_mobile(timestamps, valori, finestra_secondi=3600):
    """Media mobile simmetrica sull'intera giornata."""
    n         = len(valori)
    valori_np = np.array(valori)
    baseline  = np.full(n, np.nan)
    dt_medio  = (timestamps[-1] - timestamps[0]).total_seconds() / (n - 1) if n > 1 else 1.0
    meta      = max(1, int(round(finestra_secondi / dt_medio / 2)))
    for i in range(n):
        baseline[i] = np.mean(valori_np[max(0, i - meta):min(n, i + meta + 1)])
    return (valori_np - baseline).tolist(), baseline.tolist()


# ---------------------------------------------------------------------------
# DETRENDING  —  Metodo 2: Baseline locale con esclusione flare
# ---------------------------------------------------------------------------
def _raggruppa_flare(flares, soglia_s=SOGLIA_RAGGRUPPAMENTO_S):
    """
    Raggruppa i flare la cui distanza temporale è inferiore a soglia_s.
    Applica solo alle classi C e B (M e X usano baseline globale).
    """
    flares_filtrati = [f for f in flares if f['tipo'] in CLASSI_METODO_LOCALE]
    flares_filtrati.sort(key=lambda f: f['inizio'])
    if not flares_filtrati:
        return []
    gruppi          = []
    gruppo_corrente = [flares_filtrati[0]]
    for flare in flares_filtrati[1:]:
        distanza = (flare['inizio'] - gruppo_corrente[-1]['fine']).total_seconds()
        if distanza <= soglia_s:
            gruppo_corrente.append(flare)
            print(f"  Raggruppamento: {flare['classe']} unito al gruppo "
                  f"(distanza={distanza/60:.0f} min < soglia={soglia_s//60} min)")
        else:
            gruppi.append(gruppo_corrente)
            gruppo_corrente = [flare]
    gruppi.append(gruppo_corrente)
    return gruppi


def calcola_detrend_locale(timestamps, valori, flares,
                            finestra_lato_s=1800, margine_flare_s=300):
    """
    Baseline locale robusta con:
      - Raggruppamento automatico dei flare vicini (< 30 min di distanza)
      - Finestra minima adattiva basata sulla durata del gruppo
      - Fit lineare tra due ancoraggi puliti (fuori dalla zona flare)
      - Fallback automatico alla media mobile se la finestra è insufficiente
      - Applicazione solo alle classi C e B (M e X usano baseline globale)
    """
    n         = len(valori)
    valori_np = np.array(valori, dtype=float)
    t_sec     = np.array([(ts - timestamps[0]).total_seconds() for ts in timestamps])
    dt_medio  = t_sec[-1] / (n - 1) if n > 1 else 1.0
    n_ancora  = max(10, int(round(MINUTI_ANCORA * 60 / dt_medio)))

    # Baseline globale come fallback
    meta_g   = max(1, int(round(3600 / dt_medio / 2)))
    baseline = np.full(n, np.nan)
    for i in range(n):
        baseline[i] = np.mean(valori_np[max(0, i - meta_g):min(n, i + meta_g + 1)])

    if not flares:
        print("  Nessun flare: uso media mobile globale.")
        return (valori_np - baseline).tolist(), baseline.tolist()

    esclusi = [f for f in flares if f['tipo'] not in CLASSI_METODO_LOCALE]
    for f in esclusi:
        print(f"  Flare {f['classe']}: classe {f['tipo']} → SID visibile nel "
              f"segnale grezzo, baseline globale mantenuta.")

    gruppi = _raggruppa_flare(flares, SOGLIA_RAGGRUPPAMENTO_S)
    if not gruppi:
        print("  Nessun flare C/B: uso media mobile globale.")
        return (valori_np - baseline).tolist(), baseline.tolist()

    for gruppo in gruppi:
        classi_str    = '+'.join(f['classe'] for f in gruppo)
        t_ini_gruppo  = (gruppo[0]['inizio'] - timestamps[0]).total_seconds()
        t_fin_gruppo  = (gruppo[-1]['fine']  - timestamps[0]).total_seconds()
        durata_gruppo = t_fin_gruppo - t_ini_gruppo
        t_escl_ini    = t_ini_gruppo - margine_flare_s
        t_escl_fin    = t_fin_gruppo + margine_flare_s
        finestra_min  = max(15 * 60, durata_gruppo * 1.5)
        finestra_usata = finestra_lato_s
        if finestra_lato_s < finestra_min:
            print(f"  Gruppo [{classi_str}]: finestra richiesta {finestra_lato_s//60} min "
                  f"< minimo {finestra_min/60:.0f} min → uso {finestra_min/60:.0f} min")
            finestra_usata = finestra_min

        t_win_ini = t_ini_gruppo - finestra_usata
        t_win_fin = t_fin_gruppo + finestra_usata
        mask_sx   = (t_sec >= t_win_ini) & (t_sec <  t_escl_ini)
        mask_dx   = (t_sec >  t_escl_fin) & (t_sec <= t_win_fin)
        mask_zona = (t_sec >= t_win_ini)  & (t_sec <= t_win_fin)
        idx_sx    = np.where(mask_sx)[0]
        idx_dx    = np.where(mask_dx)[0]
        idx_zona  = np.where(mask_zona)[0]

        if len(idx_sx) < n_ancora or len(idx_dx) < n_ancora:
            print(f"  Gruppo [{classi_str}]: campioni puliti insufficienti "
                  f"(sx={len(idx_sx)}, dx={len(idx_dx)}, minimo={n_ancora}) "
                  f"→ fallback baseline globale.")
            continue

        ancora_sx_t = np.mean(t_sec[idx_sx[-n_ancora:]])
        ancora_sx_v = np.mean(valori_np[idx_sx[-n_ancora:]])
        ancora_dx_t = np.mean(t_sec[idx_dx[:n_ancora]])
        ancora_dx_v = np.mean(valori_np[idx_dx[:n_ancora]])

        if ancora_dx_t == ancora_sx_t:
            print(f"  Gruppo [{classi_str}]: ancoraggi coincidenti → skip.")
            continue

        pendenza   = (ancora_dx_v - ancora_sx_v) / (ancora_dx_t - ancora_sx_t)
        intercetta = ancora_sx_v - pendenza * ancora_sx_t
        baseline[idx_zona] = pendenza * t_sec[idx_zona] + intercetta

        n_flare_str = f"{len(gruppo)} flare" if len(gruppo) > 1 else "1 flare"
        print(f"  Gruppo [{classi_str}] ({n_flare_str}): "
              f"finestra ±{finestra_usata//60:.0f} min, "
              f"durata gruppo {durata_gruppo/60:.0f} min, "
              f"pendenza {pendenza*3600:.3f} dB/h")

    return (valori_np - baseline).tolist(), baseline.tolist()


# ---------------------------------------------------------------------------
# ALBA / TRAMONTO
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# DOWNLOAD E PARSING FLARE
# ---------------------------------------------------------------------------
def scarica_flare(data_utc):
    data_str     = data_utc.strftime("%Y%m%d")
    anno         = data_utc.strftime("%Y")
    oggi_utc     = datetime.now(timezone.utc).date()
    delta_giorni = (oggi_utc - data_utc.date()).days
    if delta_giorni <= 7:
        urls = [
            ("json_noaa", "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json"),
            ("json_noaa", "https://services.swpc.noaa.gov/json/goes/secondary/xray-flares-7-day.json"),
        ]
    else:
        data_inizio = data_utc.strftime("%Y-%m-%d")
        urls = [
            ("json_donki", f"https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/FLR?startDate={data_inizio}&endDate={data_inizio}"),
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
                    if not classe or classe[0] not in 'XMCBA' or classe[0] in ('A', 'B'):
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
                    if not classe or classe[0] not in 'XMCBA' or classe[0] in ('A', 'B'):
                        continue
                    inizio   = datetime.strptime(evento['beginTime'], "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
                    picco    = datetime.strptime(evento['peakTime'],  "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
                    fine_str = evento.get('endTime')
                    fine     = datetime.strptime(fine_str, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc) \
                               if fine_str else picco + timedelta(minutes=10)
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
                parti  = riga.split()
                classe = None
                for p in parti:
                    if re.match(r'^[XMCBA]\d+\.\d+$', p) or re.match(r'^[XMCBA]\d+$', p):
                        classe = p
                        break
                if classe is None:
                    continue
                flares.append({'inizio': hhmm_to_dt(parti[2]), 'picco': hhmm_to_dt(parti[3]),
                               'fine':   hhmm_to_dt(parti[4]), 'classe': classe, 'tipo': classe[0]})
            except Exception:
                continue

    print(f"Trovati {len(flares)} flare")
    return flares


# ---------------------------------------------------------------------------
# LETTURA CSV
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ANNOTAZIONE FLARE
# ---------------------------------------------------------------------------
def _annota_flare(ax, flares, valori_plot, posizione_label='top'):
    legenda_classi = set()
    for flare in flares:
        colore = COLORI_FLARE.get(flare['tipo'], '#FFFFFF')
        ax.axvspan(flare['inizio'], flare['fine'], alpha=0.20, color=colore, zorder=4)
        ax.axvline(flare['picco'], color=colore, linewidth=1.5, alpha=0.8, linestyle='--', zorder=5)
        y_lbl = np.percentile(valori_plot, 95) if posizione_label == 'top' else np.percentile(valori_plot, 5)
        va    = 'bottom' if posizione_label == 'top' else 'top'
        ax.text(flare['picco'], y_lbl, flare['classe'],
                color=colore, fontsize=8, ha='center', va=va, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d0d1a', alpha=0.7, edgecolor=colore))
        legenda_classi.add(flare['tipo'])
    return legenda_classi


# ---------------------------------------------------------------------------
# GRAFICO PRINCIPALE
# ---------------------------------------------------------------------------
def grafico(data_utc=None, finestra_smooth=1, metodo='locale',
            finestra_baseline_s=FINESTRA_BASELINE_SECONDI,
            finestra_locale_s=FINESTRA_LOCALE_SECONDI):

    if data_utc is None:
        data_utc = datetime.now(timezone.utc)

    csv_path = trova_csv_per_data(data_utc)
    if csv_path is None:
        print(f"Nessun file CSV trovato per {data_utc.strftime('%Y-%m-%d')}")
        return

    print(f"Lettura CSV: {csv_path}")
    timestamps, valori = leggi_csv_sid(csv_path)
    if not timestamps:
        print("Nessun dato nel file CSV.")
        return

    valori_smooth = applica_smoothing(valori, finestra_smooth) if finestra_smooth > 1 else list(valori)
    if finestra_smooth > 1:
        print(f"Smoothing: {finestra_smooth}s")

    testo_flare, url_fonte = scarica_flare(data_utc)
    flares = parse_flare(testo_flare, data_utc, url_fonte) if testo_flare else []

    if metodo == 'locale':
        print(f"Detrending: baseline locale ±{finestra_locale_s//60} min con esclusione flare")
        valori_detrend, baseline = calcola_detrend_locale(
            timestamps, valori_smooth, flares, finestra_locale_s)
        label_metodo  = f"Baseline locale ±{finestra_locale_s//60}min"
        suffix_metodo = f"_locale{finestra_locale_s//60}min"
    else:
        print(f"Detrending: media mobile {finestra_baseline_s//60} min")
        valori_detrend, baseline = calcola_detrend_mobile(
            timestamps, valori_smooth, finestra_baseline_s)
        label_metodo  = f"Media mobile {finestra_baseline_s//60}min"
        suffix_metodo = f"_baseline{finestra_baseline_s//60}min"

    alba, tramonto = calcola_alba_tramonto(data_utc)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                   gridspec_kw={'height_ratios': [3, 2]},
                                   sharex=True)
    fig.patch.set_facecolor('#0d0d1a')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0d0d1a')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.15, color='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')

    inizio_giorno = data_utc.replace(hour=0,  minute=0,  second=0,  tzinfo=timezone.utc)
    fine_giorno   = data_utc.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
    ax1.set_xlim(inizio_giorno, fine_giorno)

    # Limiti Y adattativi
    v_min   = min(valori_smooth)
    v_max   = max(valori_smooth)
    margine = (v_max - v_min) * 0.10
    ax1.set_ylim(v_min - margine, v_max + margine)

    def _ombra_notte(ax):
        if alba and tramonto:
            ax.axvspan(inizio_giorno, alba,   color='#404060', alpha=0.45, zorder=1)
            ax.axvspan(tramonto, fine_giorno, color='#404060', alpha=0.45, zorder=1)
            ax.axvline(alba,     color='#FFA500', linewidth=1.0, linestyle='--', alpha=0.7, zorder=2)
            ax.axvline(tramonto, color='#FF6347', linewidth=1.0, linestyle='--', alpha=0.7, zorder=2)

    _ombra_notte(ax1)
    _ombra_notte(ax2)

    # Pannello 1: segnale + baseline
    ax1.plot(timestamps, valori_smooth, color='#00FF99', linewidth=0.8, alpha=0.9, zorder=3)
    ax1.plot(timestamps, baseline,      color='#AAAAAA', linewidth=1.0, alpha=0.6,
             linestyle=':', zorder=3)
    ax1.set_ylabel('Segnale VLF (dB)', color='#00FF99', fontsize=11)
    ax1.tick_params(axis='y', colors='#00FF99')
    legenda_classi = _annota_flare(ax1, flares, valori_smooth, posizione_label='top')
    if alba and tramonto:
        y_label = v_min + (v_max - v_min) * 0.05
        ax1.text(alba,     y_label, f"☀ Alba\n{alba.strftime('%H:%M')} UTC",
                 color='#FFA500', fontsize=8, ha='left',  va='bottom',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d0d1a', alpha=0.7, edgecolor='#FFA500'))
        ax1.text(tramonto, y_label, f"☀ Tramonto\n{tramonto.strftime('%H:%M')} UTC",
                 color='#FF6347', fontsize=8, ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d0d1a', alpha=0.7, edgecolor='#FF6347'))

    # Pannello 2: detrended
    ax2.axhline(0, color='#888888', linewidth=0.8, linestyle='-', alpha=0.6, zorder=2)
    valori_det_np = np.array(valori_detrend)
    ax2.fill_between(timestamps, valori_det_np, 0, where=(valori_det_np >= 0),
                     color='#00FF99', alpha=0.35, zorder=3)
    ax2.fill_between(timestamps, valori_det_np, 0, where=(valori_det_np < 0),
                     color='#FF4444', alpha=0.25, zorder=3)
    ax2.plot(timestamps, valori_detrend, color='#00FF99', linewidth=0.7, alpha=0.85, zorder=4)
    ax2.set_ylabel('Variazione VLF (dB)\n[detrended]', color='#00FF99', fontsize=10)
    ax2.tick_params(axis='y', colors='#00FF99')
    ax2.set_xlabel('Ora UTC', color='white', fontsize=11)
    _annota_flare(ax2, flares, valori_detrend, posizione_label='top')
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, color='white')

    # Legenda pannello 1
    smooth_label = f" (smooth {finestra_smooth}s)" if finestra_smooth > 1 else ""
    handles1 = [
        mpatches.Patch(color='#00FF99',
                       label=f'{TRASMETTITORE} {FREQUENZA_KHZ} kHz VLF{smooth_label}'),
        plt.Line2D([0], [0], color='#AAAAAA', linewidth=1, linestyle=':',
                   label=f'Baseline ({label_metodo})'),
    ]
    if alba and tramonto:
        handles1 += [
            mpatches.Patch(color='#404060', alpha=0.6, label='Notte'),
            mpatches.Patch(color='#FFA500', alpha=0.8,
                           label=f'Alba {alba.strftime("%H:%M")} UTC'),
            mpatches.Patch(color='#FF6347', alpha=0.8,
                           label=f'Tramonto {tramonto.strftime("%H:%M")} UTC'),
        ]
    for tipo in sorted(legenda_classi, reverse=True):
        handles1.append(mpatches.Patch(color=COLORI_FLARE.get(tipo, '#FFF'),
                                       alpha=0.6, label=f'Flare classe {tipo}'))
    ax1.legend(handles=handles1, loc='lower left', facecolor='#1a1a2e',
               edgecolor='#444444', labelcolor='white', fontsize=9)

    # Legenda pannello 2
    handles2 = [
        mpatches.Patch(color='#00FF99', alpha=0.5,
                       label='Incremento segnale (possibile SID)'),
        mpatches.Patch(color='#FF4444', alpha=0.4, label='Decremento segnale'),
        plt.Line2D([0], [0], color='#888888', linewidth=0.8, linestyle='-',
                   label='Zero (baseline)'),
    ]
    ax2.legend(handles=handles2, loc='lower left', facecolor='#1a1a2e',
               edgecolor='#444444', labelcolor='white', fontsize=9)

    smooth_title = f" — Smooth {finestra_smooth}s" if finestra_smooth > 1 else ""
    fig.suptitle(
        f"Segnale VLF {TRASMETTITORE} + Flare Solari  —  "
        f"{data_utc.strftime('%Y-%m-%d')} UTC{smooth_title}\n"
        f"Pannello inferiore: detrended — {label_metodo}",
        color='white', fontsize=12, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    smooth_suffix = f"_smooth{finestra_smooth}s" if finestra_smooth > 1 else ""
    output_png = os.path.join(
        CARTELLA_CSV,
        f"VLF_detrend_{TRASMETTITORE}_{data_utc.strftime('%Y-%m-%d')}{smooth_suffix}{suffix_metodo}.png"
    )
    plt.savefig(output_png, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Grafico salvato: {output_png}")
    plt.show()


# ---------------------------------------------------------------------------
# AVVIO INTERATTIVO
# ---------------------------------------------------------------------------
print("=" * 55)
print(f"  VLF SID Detrending — {TRASMETTITORE} {FREQUENZA_KHZ} kHz")
print(f"  Stazione: {STAZIONE_ID}")
print("=" * 55)
print()

while True:
    data_input = input("Inserisci la data (YYYY-MM-DD) o premi Enter per oggi: ").strip()
    if data_input == "":
        data_scelta = datetime.now(timezone.utc)
    else:
        try:
            data_scelta = datetime.strptime(data_input, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print(">>> Formato data non valido. Usa YYYY-MM-DD\n")
            continue
    csv_path = trova_csv_per_data(data_scelta)
    if csv_path is None:
        print(f">>> Nessun file CSV trovato per {data_scelta.strftime('%Y-%m-%d')}, riprova.\n")
        continue

    while True:
        s = input("Secondi di smoothing (1 = nessuno, es. 20): ").strip()
        if s == "":
            finestra_smooth = 1
            break
        try:
            finestra_smooth = int(s)
            if finestra_smooth >= 1:
                break
            print(">>> Inserisci un numero >= 1.\n")
        except ValueError:
            print(">>> Numero intero richiesto.\n")

    print("\nMetodo di detrending:")
    print("  1 = Baseline locale con esclusione flare  (default, ±30 min)")
    print("  2 = Media mobile centrata (globale)")
    while True:
        m = input("Scegli metodo (1/2, Enter = 1): ").strip()
        if m in ('', '1'):
            metodo = 'locale'
            break
        elif m == '2':
            metodo = 'mobile'
            break
        else:
            print(">>> Inserisci 1 o 2.\n")

    finestra_baseline_s = FINESTRA_BASELINE_SECONDI
    finestra_locale_s   = FINESTRA_LOCALE_SECONDI

    if metodo == 'locale':
        MINUTO_MIN_INPUT = 15
        while True:
            v = input(f"Finestra laterale per lato in minuti "
                      f"(min={MINUTO_MIN_INPUT}, Enter = {FINESTRA_LOCALE_SECONDI//60}): ").strip()
            if v == "":
                finestra_locale_s = FINESTRA_LOCALE_SECONDI
                break
            try:
                val = int(v)
                if val < MINUTO_MIN_INPUT:
                    print(f">>> Minimo assoluto {MINUTO_MIN_INPUT} minuti per lato.\n")
                    continue
                finestra_locale_s = val * 60
                if finestra_locale_s < 1800:
                    print(f"  Attenzione: {val} min è vicino al minimo. "
                          f"Lo script potrebbe allargare automaticamente la finestra.")
                break
            except ValueError:
                print(">>> Numero intero richiesto.\n")

    elif metodo == 'mobile':
        while True:
            v = input(f"Finestra baseline in minuti "
                      f"(Enter = {FINESTRA_BASELINE_SECONDI//60}): ").strip()
            if v == "":
                finestra_baseline_s = FINESTRA_BASELINE_SECONDI
                break
            try:
                finestra_baseline_s = int(v) * 60
                if finestra_baseline_s >= 300:
                    break
                print(">>> Minimo 5 minuti.\n")
            except ValueError:
                print(">>> Numero intero richiesto.\n")

    grafico(data_scelta, finestra_smooth, metodo, finestra_baseline_s, finestra_locale_s)
    break
