#!/usr/bin/env python3
"""
Analisi singolo GRB su segnale VLF — versione 4 (Revisione incrociata Claude).
Correzioni critical fix applicate:
  1) Allineamento template con np.interp (bug originale risolto);
  2) Test falsi positivi: inversione TEMPORALE ([::-1]) e non di segno (bug critico risolto);
  3) Distribuzione nulla "Matched": finestre di controllo estratte nella stessa fascia oraria del GRB;
  4) Soglia p-value riportata a 0.01 (grazie ai 14 giorni di statistica);
  5) Rimozione avviso fisico errato sui GRB notturni.
"""

import os, sys, json, math, warnings
from datetime import datetime, timedelta, timezone
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, savgol_filter
import requests

# ============================================================
# CONFIGURAZIONE
# ============================================================
CARTELLA_GIORNALIERA = r"D:\SID\dati\Sdruno"
CARTELLA_STORICO     = r"D:\SID\dati\Sdruno\Storico_NSY"
PREFISSO_CSV         = "GAESID_NSY_"
OUTPUT_DIR           = os.path.join(CARTELLA_STORICO, "GRB_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parametri fisici
TAU_RICOMB = 40.0
FINESTRA_ANALISI_S = 600          # ±10 minuti
SMOOTH_VLF_S = 5                  # smoothing VLF (secondi)

# CORREZIONE CLAUDE 4: Soglia riportata a 0.01 (ora raggiungibile con 14 giorni)
P_VALUE_THRESHOLD = 0.01
LAG_FISICO_MAX = 60.0

# ============================================================
# FUNZIONI DI SUPPORTO
# ============================================================

def trova_csv_per_data(data_utc):
    data_str = data_utc.strftime("%Y-%m-%d")
    nome_file = f"{PREFISSO_CSV}{data_str}.csv"
    for cartella in [CARTELLA_GIORNALIERA, CARTELLA_STORICO]:
        path = os.path.join(cartella, nome_file)
        if os.path.exists(path):
            return path
    return None

def leggi_csv_vlf(filepath):
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
                dt = datetime.strptime(parti[0].strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                val = float(parti[1].strip())
                timestamps.append(dt)
                valori.append(val)
            except Exception:
                continue
    return timestamps, valori

# --- Ricerca parametri GRB ---

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

def get_grb_by_name(grb_name):
    try:
        anno = 2000 + int(grb_name[3:5])
    except:
        print(f"Errore: impossibile estrarre l'anno dal nome {grb_name}")
        return None

    mjd_start = _datetime_to_mjd(datetime(anno, 1, 1, tzinfo=timezone.utc))
    mjd_end   = _datetime_to_mjd(datetime(anno + 1, 1, 1, tzinfo=timezone.utc))
    query = (
        "SELECT name, trigger_time, t90, fluence, ra, dec "
        "FROM fermigbrst "
        f"WHERE trigger_time >= {mjd_start:.8f} "
        f"AND trigger_time < {mjd_end:.8f}"
    )
    try:
        resp = requests.get("https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync",
                            params={'REQUEST': 'doQuery', 'LANG': 'ADQL', 'QUERY': query, 'FORMAT': 'text'},
                            timeout=30)
        resp.raise_for_status()
        rows = _parse_tap_text(resp.text)
    except Exception as e:
        print(f"Errore nella query HEASARC: {e}")
        return None

    for r in rows:
        if r.get('name', '').strip() == grb_name:
            try:
                t_raw = (r.get('trigger_time') or '').strip()
                trig = None
                for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
                    try:
                        trig = datetime.strptime(t_raw, fmt).replace(tzinfo=timezone.utc)
                        break
                    except ValueError:
                        pass
                if trig is None:
                    try:
                        trig = _mjd_to_datetime(float(t_raw))
                    except:
                        continue
                fluence = float(r.get('fluence') or 0)
                t90 = float(r.get('t90') or 0)
                if fluence <= 0:
                    continue
                return {
                    'trigger_time': trig,
                    'fluence': fluence,
                    't90': t90,
                    'ra': float(r.get('ra') or 0),
                    'dec': float(r.get('dec') or 0),
                }
            except (ValueError, TypeError):
                continue
    return None

# --- Lightcurve e template ---
def fetch_lightcurve_fermi(trigger_name, t0, fluence, t90, lightcurve_file=None):
    import astropy.io.fits as fits
    from io import BytesIO

    def _parse_fermi_txt(txt):
        try:
            data = np.loadtxt(txt.splitlines())
        except Exception:
            data = np.genfromtxt(txt.splitlines(), invalid_raise=False)
            if data.ndim == 1:
                data = data[~np.isnan(data)]
                if data.ndim != 2 or data.shape[0] == 0:
                    return np.array([]), np.array([])
            else:
                data = data[~np.isnan(data).any(axis=1)]
        if data.ndim != 2 or data.shape[0] == 0:
            return np.array([]), np.array([])
        ncols = data.shape[1]
        t_rel = data[:, 0]
        if ncols == 2:
            rate = data[:, 1]
        elif ncols >= 9:
            rate = np.sum(data[:, 1:9], axis=1)
        else:
            rate = data[:, 1]
        return t_rel, rate

    def _parse_fermi_ctime(base_url, numero):
        all_times, all_counts = [], []
        detectors = [f'n{i}' for i in range(10)] + ['b0', 'b1']
        for det in detectors:
            url = f"{base_url}/glg_ctime_{det}_bn{numero}_v00.pha"
            try:
                r = requests.get(url, timeout=15)
                if r.status_code != 200: continue
                with fits.open(BytesIO(r.content), ignore_missing_end=True) as hdul:
                    for hdu in hdul:
                        if hdu.is_image or hdu.data is None: continue
                        cols = [c.name.upper() for c in hdu.data.columns]
                        if 'TIME' in cols and 'COUNTS' in cols:
                            tcol = hdu.data.columns[cols.index('TIME')].name
                            ccol = hdu.data.columns[cols.index('COUNTS')].name
                            if 'QUALITY' in cols:
                                qcol = hdu.data.columns[cols.index('QUALITY')].name
                                good = hdu.data[qcol] == 0
                                t, c = hdu.data[tcol][good], hdu.data[ccol][good]
                            else:
                                t, c = hdu.data[tcol], hdu.data[ccol]
                            all_times.append(t)
                            all_counts.append(c)
                            break
            except Exception: continue
        if not all_times: return None, None
        times, counts = np.concatenate(all_times), np.concatenate(all_counts)
        order = np.argsort(times)
        times, counts = times[order], counts[order]
        unique_times, idx = np.unique(times, return_inverse=True)
        summed = np.array([counts[idx == i].sum() for i in range(len(unique_times))])
        return unique_times, summed

    def _parse_fermi_tte(base_url, numero, t0, t90):
        t_start, t_stop = -10.0, t90 + 60.0
        all_times = []
        detectors = [f'n{i}' for i in range(10)] + ['b0', 'b1']
        for det in detectors:
            url = f"{base_url}/glg_tte_{det}_bn{numero}_v00.fit"
            try:
                r = requests.get(url, timeout=30)
                if r.status_code != 200 or len(r.content) == 0: continue
                with fits.open(BytesIO(r.content), ignore_missing_end=True) as hdul:
                    events = None
                    for hdu in hdul:
                        if hdu.is_image or hdu.data is None: continue
                        if hdu.name == 'EVENTS': events = hdu.data; break
                        cols = [c.name.upper() for c in hdu.data.columns]
                        if 'TIME' in cols: events = hdu.data; break
                    if events is None: continue
                    time_col = events.columns.names[events.columns.names.index('TIME')]
                    mask = (events[time_col] >= t_start) & (events[time_col] <= t_stop)
                    if np.sum(mask) > 0: all_times.append(events[time_col][mask])
            except Exception: continue
        if not all_times: return None, None
        all_times = np.concatenate(all_times)
        bins = np.arange(t_start, t_stop + 1, 1.0)
        counts, _ = np.histogram(all_times, bins=bins)
        return (bins[:-1] + bins[1:]) / 2.0, counts.astype(float)

    def _extract_lc_from_hdul(hdul):
        tempo_cands, rate_cands = ['TIME', 'T', 'START_TIME'], ['COUNTS', 'RATE', 'COUNTS_SUM', 'FLUX']
        for hdu in hdul:
            if hdu.is_image or not hasattr(hdu, 'data') or hdu.data is None: continue
            try: nomi = [c.name.upper() for c in hdu.data.columns]
            except Exception: continue
            col_t = next((hdu.data.columns[nomi.index(c)].name for c in tempo_cands if c in nomi), None)
            col_r = next((hdu.data.columns[nomi.index(c)].name for c in rate_cands if c in nomi), None)
            if col_t and col_r:
                times = np.unique(hdu.data[col_t])
                summed = np.array([np.sum(hdu.data[col_r][hdu.data[col_t] == t]) for t in times])
                order = np.argsort(times)
                return times[order], summed[order]
        return None, None

    def _parse_fermi_bcat(path_or_url):
        try:
            from astropy.io import fits; from io import BytesIO
        except ImportError: return None, None
        try:
            if path_or_url.startswith("http"):
                r = requests.get(path_or_url, timeout=30); r.raise_for_status()
                hdulist = fits.open(BytesIO(r.content), ignore_missing_end=True)
            else: hdulist = fits.open(path_or_url)
            with hdulist as hdul: return _extract_lc_from_hdul(hdul)
        except Exception as e: print(f"   Errore BCAT: {e}"); return None, None

    # ---------- Funzione helper per sottrarre fondo e normalizzare ----------
    def _sottrai_fondo_e_normalizza(t_rel, rate, fluence, tipo, sorgente=""):
        if t_rel is None or len(t_rel) == 0:
            return None, None, None

        # Stima fondo come mediana dei conteggi per t < -5 s (pre‑trigger)
        mask_bg = t_rel < -5
        if np.sum(mask_bg) > 5:
            background = np.median(rate[mask_bg])
        else:
            # Se non ci sono dati pre‑trigger, usa il 5° percentile di tutta la curva
            background = np.percentile(rate, 5)
        
        # Sottrai il fondo e tronca a zero
        rate_excess = np.maximum(rate - background, 0.0)
        area = np.trapezoid(rate_excess, t_rel)
        
        if area > 0:
            rate_norm = rate_excess / area * fluence
            print(f"✅ {tipo} {sorgente}: fondo sottratto (bg={background:.2f}), area netta={area:.2f}")
            return t_rel, rate_norm, tipo
        else:
            print(f"⚠️  {tipo} {sorgente}: area netta zero dopo sottrazione fondo")
            return None, None, None

    # ---------- Inizio acquisizione ----------
    numero, anno = trigger_name[3:], 2000 + int(trigger_name[3:5])
    base = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers"

    # ---- 1) File locale ----
    if lightcurve_file and os.path.exists(lightcurve_file):
        t_rel, rate = _parse_fermi_bcat(lightcurve_file)
        if t_rel is not None and len(t_rel) > 0:
            res = _sottrai_fondo_e_normalizza(t_rel, rate, fluence, "real", "da file FITS")
            if res[0] is not None:
                return res
        try:
            with open(lightcurve_file, 'r') as f: txt = f.read()
            t_rel, rate = _parse_fermi_txt(txt)
            if len(t_rel) > 0:
                res = _sottrai_fondo_e_normalizza(t_rel, rate, fluence, "real", "da file testo")
                if res[0] is not None:
                    return res
        except Exception: pass

    # ---- 2) Scarica file di testo medres ----
    for url in [f"{base}/{anno}/bn{numero}/current/glg_lc_medres34_bn{numero}_v00.txt",
                f"{base}/{anno}/bn{numero}/current/glg_lc_medres34_{numero}_v00.txt",
                f"{base}/{anno}/{numero}/current/glg_lc_medres34_bn{numero}_v00.txt"]:
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                t_rel, rate = _parse_fermi_txt(resp.text)
                if len(t_rel) > 0:
                    res = _sottrai_fondo_e_normalizza(t_rel, rate, fluence, "real", "da testo scaricato")
                    if res[0] is not None:
                        return res
        except Exception: continue

    # ---- 3) CTIME ----
    t_rel, rate = _parse_fermi_ctime(f"{base}/{anno}/bn{numero}/current", numero)
    if t_rel is not None and len(t_rel) > 0:
        res = _sottrai_fondo_e_normalizza(t_rel, rate, fluence, "real", "CTIME")
        if res[0] is not None:
            return res

    # ---- 4) TTE ----
    t_rel, rate = _parse_fermi_tte(f"{base}/{anno}/bn{numero}/current", numero, t0, t90)
    if t_rel is not None and len(t_rel) > 0:
        res = _sottrai_fondo_e_normalizza(t_rel, rate, fluence, "tte", "TTE")
        if res[0] is not None:
            return res

    # ---- 5) BCAT ----
    t_rel, rate = _parse_fermi_bcat(f"{base}/{anno}/bn{numero}/current/glg_bcat_all_bn{numero}_v01.fit")
    if t_rel is not None and len(t_rel) > 0:
        res = _sottrai_fondo_e_normalizza(t_rel, rate, fluence, "real", "BCAT")
        if res[0] is not None:
            return res

    # ---- 6) Template sintetico FRED (se tutto fallisce) ----
    print("⚠️  Lightcurve sintetica (FRED) — template approssimato.")
    t_rise, t_decay = t90 * 0.3, t90 * 0.7
    t = np.linspace(-10, t90 + 60, int((t90 + 70) * 1))
    y = np.where(t < 0, 0.0,
                 np.where(t < t_rise, np.exp(-((t - t_rise) / (t_rise / 3)) ** 2),
                          np.exp(-(t - t_rise) / t_decay)))
    area = np.trapezoid(y, t)
    if area > 0:
        y = y / area * fluence
    return t, y, "fred"

def genera_template(t_lc, lc, tau, dt_out=1.0):
    t_kernel = np.arange(0, 5*tau, dt_out)
    kernel = np.exp(-t_kernel / tau); kernel /= kernel.sum()
    t_grid = np.arange(t_lc[0], t_lc[-1] + dt_out, dt_out)
    lc_interp = np.interp(t_grid, t_lc, lc, left=0.0, right=0.0)
    template = np.convolve(lc_interp, kernel, mode='full') * dt_out
    t_template = np.arange(len(template)) * dt_out + t_grid[0]
    mask = (t_template >= 0) & (t_template <= t_grid[-1] + 2*tau)
    return t_template[mask], template[mask]


def cross_correlazione_normalizzata(segnale, template, dt=1.0):
    s = segnale - np.mean(segnale)
    t = template - np.mean(template)
    if np.std(s) == 0 or np.std(t) == 0: return np.array([0]), np.array([0])
    corr = correlate(s, t, mode='same') / (np.std(s) * np.std(t) * len(t))
    n = len(segnale)
    return np.arange(-n//2, n//2) * dt, corr

def valida_segmento(t_unix, valori, max_gap_s=15.0, min_std=None):
    if len(t_unix) < 2 or len(valori) < 2: return False
    if np.max(np.diff(t_unix)) > max_gap_s: return False
    if min_std is not None and np.std(valori) < min_std: return False
    if np.any(np.isnan(valori)) or np.any(np.isinf(valori)): return False
    return True

def costruisci_distribuzione_nulla(timestamps_full, valori_full, template,
                                   t0, finestra_s=600, n_controlli=40,
                                   escludi_raggio_s=600, smooth_s=5, debug=False,
                                   giorni_extra=None, min_std_segmento=None):
    
    if giorni_extra:
        t_all = [np.array([dt.timestamp() for dt in timestamps_full])]
        v_all = [np.asarray(valori_full, dtype=float)]
        for t_extra, v_extra in giorni_extra:
            t_all.append(np.array([dt.timestamp() for dt in t_extra]))
            v_all.append(np.asarray(v_extra, dtype=float))
        t_unix = np.concatenate(t_all)
        valori = np.concatenate(v_all)
    else:
        t_unix = np.array([dt.timestamp() for dt in timestamps_full])
        valori = np.asarray(valori_full, dtype=float)

    t0_unix = t0.timestamp()
    finestre_centri = []

    # CORREZIONE CLAUDE 3: Finestre "Matched" sull'ora del GRB (±3 ore)
    # invece delle 05:30-19:00 fisse.
    def aggiungi_finestre_giorno(data_dt, ora_evento_utc):
        delta_h = 3
        inizio_h = (ora_evento_utc - delta_h) % 24
        fine_h = (ora_evento_utc + delta_h) % 24
        
        def processa_slot(inizio, fine):
            passo = 2 * finestra_s
            durata = (fine - inizio).total_seconds()
            n_finestre = int(durata // passo)
            if n_finestre == 0: return
            centro_start = inizio.timestamp() + finestra_s
            for i in range(n_finestre):
                centro = centro_start + i * passo
                finestre_centri.append((centro, datetime.fromtimestamp(centro, tz=timezone.utc)))

        if inizio_h < fine_h:
            inizio = datetime.combine(data_dt, datetime.min.time()).replace(hour=inizio_h, tzinfo=timezone.utc)
            fine = datetime.combine(data_dt, datetime.min.time()).replace(hour=fine_h, tzinfo=timezone.utc)
            processa_slot(inizio, fine)
        else:
            inizio1 = datetime.combine(data_dt, datetime.min.time()).replace(hour=inizio_h, tzinfo=timezone.utc)
            fine1 = datetime.combine(data_dt + timedelta(days=1), datetime.min.time()).replace(hour=0, minute=0, second=0, tzinfo=timezone.utc)
            inizio2 = fine1
            fine2 = datetime.combine(data_dt + timedelta(days=1), datetime.min.time()).replace(hour=fine_h, tzinfo=timezone.utc)
            processa_slot(inizio1, fine1)
            processa_slot(inizio2, fine2)

    ora_grb = t0.hour
    aggiungi_finestre_giorno(t0.date(), ora_grb)
    
    if giorni_extra:
        date_extra = set()
        for t_extra, _ in giorni_extra:
            for dt in t_extra: date_extra.add(dt.date())
        for d in date_extra:
            aggiungi_finestre_giorno(d, ora_grb)

    if n_controlli and len(finestre_centri) > n_controlli:
        idx = np.linspace(0, len(finestre_centri)-1, n_controlli, dtype=int)
        finestre_centri = [finestre_centri[i] for i in idx]

    coef_max_list, finestre_utilizzate = [], 0

    for i, (centro, centro_dt) in enumerate(finestre_centri, start=1):
        if abs(centro - t0_unix) < escludi_raggio_s: continue
        mask = (t_unix >= centro - finestra_s) & (t_unix <= centro + finestra_s)
        n_campioni = np.sum(mask)
        if n_campioni < 30: continue

        segmento = valori[mask].astype(float)
        if not valida_segmento(t_unix[mask], segmento, max_gap_s=15.0, min_std=min_std_segmento): continue

        if smooth_s > 1 and len(segmento) > smooth_s:
            ws = min(smooth_s, len(segmento) // 2 * 2 + 1)
            if ws >= 3:
                try: segmento = savgol_filter(segmento, ws, 2)
                except Exception: pass

        lags, corr_coef = cross_correlazione_normalizzata(segmento, template, dt=1.0)
        max_abs = np.max(np.abs(corr_coef))
        if np.isnan(max_abs) or np.isinf(max_abs): continue
        
        coef_max_list.append(max_abs)
        finestre_utilizzate += 1
        if debug:
            print(f"   [{i:02d}] {centro_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC → OK ({n_campioni} campioni, max|r|={max_abs:.4f})")

    if debug:
        print(f"   Finestre di controllo usate: {finestre_utilizzate} su {len(finestre_centri)} disponibili")
    return np.array(coef_max_list)


# CORREZIONE CLAUDE 2: Inversione TEMPORALE, non di segno
def test_template_invertito(v_grid, templ_grid, dt=1.0):
    template_invertito = templ_grid[::-1]  # Inverte l'asse del tempo, non l'ampiezza
    lags, corr = cross_correlazione_normalizzata(v_grid, template_invertito, dt=dt)
    return np.max(np.abs(corr))

def valuta_significativita(coeff_max_evento, coef_max_null):
    n = len(coef_max_null)
    if n == 0: return 1.0
    k = np.sum(coef_max_null >= coeff_max_evento)
    return (k + 1) / (n + 1)

def archivia_per_stacking(grb_name, t0, coeff, p_value, rilevato, template_type):
    archivio = os.path.join(OUTPUT_DIR, "grb_stacking_archive.json")
    record = {"name": grb_name, "trigger_utc": t0.isoformat(), "coeff_max": float(coeff), "p_value": float(p_value), "rilevato": bool(rilevato), "template_type": template_type}
    data = json.load(open(archivio, 'r')) if os.path.exists(archivio) else []
    data.append(record)
    with open(archivio, 'w') as f: json.dump(data, f, indent=2)

def carica_dati_giorno(data_utc):
    csv_path = trova_csv_per_data(data_utc)
    if not csv_path: return None
    t, v = leggi_csv_vlf(csv_path)
    return (t, v) if t else None

# ============================================================
# MAIN
# ============================================================
def main(grb_name):
    print("=" * 70)
    print(f"ANALISI GRB – {grb_name}")
    print("=" * 70)

    print("\n[1] Ricerca parametri GRB...")
    grb_params = get_grb_by_name(grb_name)
    if grb_params is None:
        print(f"ERRORE: GRB {grb_name} non trovato."); return
    t0, fluence, t90 = grb_params['trigger_time'], grb_params['fluence'], grb_params['t90']
    print(f"   Trovato: {grb_name} | Trigger: {t0} | Fluence: {fluence:.2e} | T90: {t90:.1f} s")

    print("\n[2] Caricamento CSV VLF...")
    csv_path = trova_csv_per_data(t0)
    if not csv_path:
        print(f"ERRORE: nessun CSV per {t0.strftime('%Y-%m-%d')}."); return
    timestamps, valori = leggi_csv_vlf(csv_path)
    if not timestamps:
        print("ERRORE: dati VLF vuoti."); return
    timestamps_arr, valori_arr = np.array(timestamps), np.array(valori, dtype=float)
    print(f"   {len(valori)} campioni caricati.")

    giorni_extra = []
    for delta in range(-7, 8):
        if delta == 0: continue
        dati_adj = carica_dati_giorno(t0 + timedelta(days=delta))
        if dati_adj: giorni_extra.append(dati_adj)

    t_inizio_stima = t0.replace(hour=10, minute=0, second=0, tzinfo=timezone.utc)
    mask_stima = (timestamps_arr >= t_inizio_stima) & (timestamps_arr <= t_inizio_stima + timedelta(hours=1))
    min_std_segmento = np.std(valori_arr[mask_stima]) * 0.1 if np.sum(mask_stima) > 60 else None

    print("\n[3] Generazione lightcurve e template...")
    t_lc, lc, template_type = fetch_lightcurve_fermi(grb_name, t0, fluence, t90)

    # --- CONVERSIONE: i tempi della lightcurve reale sono in MET (secondi dal 2001-01-01) ---
    # Per i template sintetici (fred), i tempi sono già relativi a T0
    if template_type != "fred":
        # Definisci l'epoca Fermi (MET)
        epoca_fermi = datetime(2001, 1, 1, tzinfo=timezone.utc)
        t0_met = (t0 - epoca_fermi).total_seconds()
        # Converti i tempi assoluti in secondi da T0
        t_lc_rel = np.array(t_lc) - t0_met
        print(f"   DEBUG: t_lc_rel min={t_lc_rel[0]:.1f}, max={t_lc_rel[-1]:.1f}")
    else:
        t_lc_rel = t_lc  # già relativi

    print(f"DEBUG: t_lc min={t_lc[0]:.1f}, max={t_lc[-1]:.1f}, len={len(t_lc)}")
    print(f"DEBUG: lc max={lc.max():.3e}, area={np.trapezoid(lc, t_lc):.3e}")

    # --- DEBUG: visualizza la lightcurve ---
    print(f"\n[DEBUG] Lightcurve: {len(t_lc)} punti, tipo={template_type}")
    print(f"   t_min={t_lc[0]:.1f}, t_max={t_lc[-1]:.1f}")
    print(f"   lc min={lc.min():.3e}, max={lc.max():.3e}, media={np.mean(lc):.3e}")
    print(f"   area_tot={np.trapezoid(lc, t_lc):.3e}")

    t_template, template = genera_template(t_lc_rel, lc, TAU_RICOMB, dt_out=1.0)

    print(f"   DEBUG: t_template min={t_template[0]:.1f}, max={t_template[-1]:.1f}")
    print(f"   DEBUG: template max={template.max():.3e}")

    print(f"\n[4] Finestra VLF ±{FINESTRA_ANALISI_S//60} min attorno a T0...")
    t_start, t_end = t0 - timedelta(seconds=FINESTRA_ANALISI_S), t0 + timedelta(seconds=FINESTRA_ANALISI_S)
    mask_finestra = (timestamps_arr >= t_start) & (timestamps_arr <= t_end)
    if np.sum(mask_finestra) < 100:
        print("ERRORE: dati insufficienti."); return
    t_finestra, v_finestra = timestamps_arr[mask_finestra], valori_arr[mask_finestra].astype(float)

    if not valida_segmento(np.array([dt.timestamp() for dt in t_finestra]), v_finestra, max_gap_s=15.0, min_std=min_std_segmento):
        print("ERRORE: finestra evento non valida."); return
    
    smooth_s = SMOOTH_VLF_S
    if smooth_s > 1:
        ws = min(smooth_s, len(v_finestra)//2*2+1)
        if ws >= 3: v_finestra = savgol_filter(v_finestra, ws, 2)

    t_rel = np.array([(dt - t0).total_seconds() for dt in t_finestra])
    t_grid = np.arange(t_rel[0], t_rel[-1] + 1.0, 1.0)
    v_grid = np.interp(t_grid, t_rel, v_finestra)
    templ_grid = np.interp(t_grid, t_template, template, left=0.0, right=0.0)

    print("\n[5] Cross-correlazione normalizzata...")
    lags, corr_coef = cross_correlazione_normalizzata(v_grid, templ_grid, dt=1.0)
    idx_max = np.argmax(np.abs(corr_coef))
    coeff_max, lag_opt = corr_coef[idx_max], lags[idx_max]
    print(f"   Massimo coefficiente: {coeff_max:.4f} a lag = {lag_opt:.1f} s")

    coeff_invertito = test_template_invertito(v_grid, templ_grid)
    print(f"   Controllo asimmetria (tempo invertito): {coeff_invertito:.4f}")
    spurio = coeff_invertito >= np.abs(coeff_max) * 0.9

    print("\n[6] Costruzione distribuzione nulla...")
    coef_null = costruisci_distribuzione_nulla(
        timestamps_arr, valori_arr, templ_grid, t0, finestra_s=FINESTRA_ANALISI_S,
        n_controlli=None, escludi_raggio_s=600, smooth_s=smooth_s, debug=True,
        giorni_extra=giorni_extra, min_std_segmento=min_std_segmento
    )

    p_value = 1.0
    if len(coef_null) > 0:
        print(f"   Controlli: {len(coef_null)}, media={np.mean(coef_null):.4f}, std={np.std(coef_null):.4f}")
        p_value = valuta_significativita(np.abs(coeff_max), coef_null)
        print(f"   p-value empirico: {p_value:.4f}")

    significativo = p_value < P_VALUE_THRESHOLD
    lag_assoluto = abs(lag_opt)
    
    if significativo and not spurio and lag_assoluto <= LAG_FISICO_MAX:
        rilevato = True
        messaggio = f"\n✅ RILEVATO: Anomalia statisticamente significativa (p={p_value:.4f}, lag={lag_opt:.1f}s)."
    elif significativo and lag_assoluto > LAG_FISICO_MAX:
        rilevato = False
        messaggio = f"\n⚠️  ANOMALIA ma lag non fisico ({lag_opt:.1f}s)."
    elif significativo and spurio:
        rilevato = False
        messaggio = f"\n⚠️  ANOMALIA ma forma spuria (tempo invertito simile al normale)."
    else:
        rilevato = False
        motivi = []
        if fluence < 1e-6: motivi.append("- Fluence molto bassa")
        if np.abs(coeff_max) < 0.1: motivi.append("- Correlazione molto bassa")
        if len(coef_null) > 0 and np.abs(coeff_max) < np.mean(coef_null) + np.std(coef_null): motivi.append("- Correlazione nel rumore normale")
        messaggio = f"\n❌ NON RILEVATO (p={p_value:.4f}).\nMotivi: {'; '.join(motivi) if motivi else 'Nessuna perturbazione VLF'}"

    print("\n" + "=" * 70 + messaggio + "\n" + "=" * 70)
    archivia_per_stacking(grb_name, t0, coeff_max, p_value, rilevato, template_type)

    # Creazione figura con 2 subplot (VLF+template e cross-correlazione)
    fig, (ax0, ax3) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # ---- Subplot 1: VLF + Template (doppio asse) ----
    # Asse principale: VLF
    ax0.plot(t_rel, v_finestra, 'b-', alpha=0.7, label='VLF')
    ax0.axvline(0, color='r', linestyle='--', label='T0')
    ax0.set_ylabel('VLF (unità)', color='b')
    ax0.tick_params(axis='y', labelcolor='b')
    ax0.grid(True)
    ax0.legend(loc='upper left')

    # Asse secondario: Template
    ax0_twin = ax0.twinx()
    ax0_twin.plot(t_grid, templ_grid, 'g-', linewidth=2, label='Template (scalato)')
    ax0_twin.set_ylabel('Template (arb. units)', color='g')
    ax0_twin.tick_params(axis='y', labelcolor='g')
    ax0_twin.legend(loc='upper right')

    # ---- Subplot 2: Cross-correlazione (invariato) ----
    ax3.plot(lags, corr_coef, 'k-', label='X-Corr')
    if len(coef_null) > 0:
        soglia = np.max(coef_null)
        ax3.axhline(soglia, color='orange', linestyle='--', label=f'Soglia nulli ({soglia:.3f})')
        ax3.axhline(-soglia, color='orange', linestyle='--')
    ax3.scatter(lag_opt, coeff_max, color='red', s=80, zorder=5, label=f'Evento (r={coeff_max:.3f})')
    ax3.set_xlabel('Tempo da T0 (s)')
    ax3.set_ylabel('Coefficiente di correlazione')
    ax3.legend()
    ax3.grid(True)
    ax3.set_title(f"p-value = {p_value:.3f} | Controllo asimmetria = {coeff_invertito:.3f} | {'RILEVATO' if rilevato else 'NON RILEVATO'}")

    # Forza la stessa scala temporale su entrambi i subplot (±10 minuti)
    limite_x = 300  # secondi, per vedere il decadimento
    ax0.set_xlim(-limite_x, limite_x)
    ax3.set_xlim(-limite_x, limite_x)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"GRB_{grb_name}_analysis.png"), dpi=150)
    plt.show()

if __name__ == "__main__":
    grb_name = sys.argv[1].strip() if len(sys.argv) > 1 else input("Inserisci nome GRB: ").strip()
    if grb_name: main(grb_name)