# -*- coding: utf-8 -*-
"""
Modulo per il rilevamento SID con AI supervisionata.
Version: 2.0 - Interattivo con recovery e X-ray
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
import ephem
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pickle
import os
import json
from scipy.signal import medfilt
import matplotlib.patches as mpatches
import tkinter as tk
from tkinter import simpledialog

# Costanti
SOGLIA_AMPIEZZA = 0.5  # fissa
SOGLIA_PROBABILITA_DEFAULT = 0.5
TOLLERANZA_SECONDI = 600  # 10 minuti per associare flare NOAA

MODELLO_PATH = "modello_sid.pkl"  # nome del file (senza percorso)


# ============================================================
# 1. GENERAZIONE CANDIDATI
# ============================================================

def genera_candidati(timestamps, valori, data_utc, soglia_ampiezza=SOGLIA_AMPIEZZA):
    """
    Genera candidati SID dal residuo (segnale - quiete).
    """
    if len(timestamps) < 10:
        return []
    
    df = pd.DataFrame({'timestamp': timestamps, 'valore': valori})
    
    def crea_features(df_in):
        df_out = df_in.copy()
        df_out['hour'] = df_out['timestamp'].dt.hour
        df_out['minute'] = df_out['timestamp'].dt.minute + df_out['timestamp'].dt.second / 60.0
        df_out['day_of_year'] = df_out['timestamp'].dt.dayofyear
        df_out['sin_hour'] = np.sin(2 * np.pi * df_out['hour'] / 24)
        df_out['cos_hour'] = np.cos(2 * np.pi * df_out['hour'] / 24)
        df_out['sin_day'] = np.sin(2 * np.pi * df_out['day_of_year'] / 365)
        df_out['cos_day'] = np.cos(2 * np.pi * df_out['day_of_year'] / 365)
        df_out['time_ratio'] = df_out['hour'] + df_out['minute'] / 60.0
        return df_out
    
    df = crea_features(df)
    features = ['sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'time_ratio']
    X = df[features].values
    y = df['valore'].values
    
    # Modello di quiete
    model_quiete = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    model_quiete.fit(X, y)
    
    df['quiete'] = model_quiete.predict(X)
    df['residuo'] = df['valore'] - df['quiete']
    dt = (timestamps[1] - timestamps[0]).total_seconds()
    
    # Filtro notte
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
    except Exception:
        alba = data_utc.replace(hour=6, minute=0, second=0, microsecond=0)
        tramonto = data_utc.replace(hour=18, minute=0, second=0, microsecond=0)
    
    alba_stabile = alba + timedelta(minutes=30)
    tramonto_stabile = tramonto - timedelta(minutes=30)
    
    # Maschera binaria
    maschera = (df['residuo'].values > soglia_ampiezza).astype(int)
    
    inizio_eventi = []
    fine_eventi = []
    in_ev = False
    for i, val in enumerate(maschera):
        if not in_ev and val == 1:
            inizio_eventi.append(i)
            in_ev = True
        elif in_ev and val == 0:
            fine_eventi.append(i - 1)
            in_ev = False
    if in_ev:
        fine_eventi.append(len(maschera) - 1)
    
    candidati = []
    for i_start, i_end in zip(inizio_eventi, fine_eventi):
        durata = (i_end - i_start) * dt
        if durata < 30 or durata > 7200:
            continue
        
        idx_picco = i_start + np.argmax(df['residuo'].values[i_start:i_end+1])
        picco_time = df['timestamp'].iloc[idx_picco]
        ampiezza = df['residuo'].iloc[idx_picco]
        
        if picco_time < alba_stabile or picco_time > tramonto_stabile:
            continue
        
        baseline = df['quiete'].iloc[idx_picco]
        area = np.trapezoid(df['residuo'].values[i_start:i_end+1], dx=dt)
        
        ora = picco_time.hour + picco_time.minute/60.0
        
        if idx_picco - i_start > 2:
            pend_salita = (df['residuo'].iloc[idx_picco] - df['residuo'].iloc[i_start]) / (idx_picco - i_start)
        else:
            pend_salita = 0
        
        if i_end - idx_picco > 2:
            pend_discesa = (df['residuo'].iloc[idx_picco] - df['residuo'].iloc[i_end]) / (i_end - idx_picco)
        else:
            pend_discesa = 0
        
        simmetria = pend_salita / (pend_discesa + 1e-6)
        
        candidati.append({
            'inizio': df['timestamp'].iloc[i_start],
            'fine': df['timestamp'].iloc[i_end],
            'picco': picco_time,
            'ampiezza': ampiezza,
            'durata': durata,
            'area': area,
            'baseline': baseline,
            'ora': ora,
            'pend_salita': pend_salita,
            'pend_discesa': pend_discesa,
            'simmetria': simmetria,
            'idx_start': i_start,
            'idx_end': i_end,
            'idx_picco': idx_picco,
        })
    
    return candidati

def stima_quiete_locale(df, finestra_minuti=60, dt_secondi=1.0):
    """
    Stima la curva di quiete usando il percentile 10 mobile.
    Robusta ai picchi (SID) perché il percentile basso li ignora.
    Non richiede dati storici.
    """
    valori = df['valore'].values
    n = len(valori)
    finestra_camp = max(10, int(finestra_minuti * 60 / dt_secondi))
    quiete = np.empty(n)
    
    for i in range(n):
        i_start = max(0, i - finestra_camp // 2)
        i_end   = min(n, i + finestra_camp // 2)
        quiete[i] = np.percentile(valori[i_start:i_end], 10)
    
    return quiete

def unisci_candidati_ravvicinati(candidati, gap_max=180):
    """
    Unisce candidati che hanno picchi distanti meno di gap_max secondi.
    Mantiene il candidato con ampiezza maggiore e estende inizio/fine.
    """
    if len(candidati) <= 1:
        return candidati
    
    # Ordina per tempo di picco
    candidati.sort(key=lambda c: c['picco'])
    
    uniti = []
    current = candidati[0]
    
    for next_c in candidati[1:]:
        delta = (next_c['picco'] - current['picco']).total_seconds()
        
        if delta < gap_max:
            # Unisci: tieni il più forte
            if next_c['ampiezza'] > current['ampiezza']:
                current['picco'] = next_c['picco']
                current['ampiezza'] = next_c['ampiezza']
                current['baseline'] = next_c['baseline']
                current['idx_picco'] = next_c['idx_picco']
            
            # Estendi inizio e fine
            if next_c['inizio'] < current['inizio']:
                current['inizio'] = next_c['inizio']
                current['idx_start'] = next_c['idx_start']
            if next_c['fine'] > current['fine']:
                current['fine'] = next_c['fine']
                current['idx_end'] = next_c['idx_end']
            
            # Ricalcola durata e area (somma)
            current['durata'] = (current['fine'] - current['inizio']).total_seconds()
            current['area'] += next_c['area']
        else:
            uniti.append(current)
            current = next_c
    
    uniti.append(current)
    return uniti

# ============================================================
# 2. FUNZIONI PER CARICARE FLARE
# ============================================================

def leggi_flare_cache(data_utc, cartella_flares=None):
    """Carica i flare NOAA per un giorno (cache locale)."""
    if cartella_flares is None:
        # Cerca nella cartella standard
        cartella_flares = r"D:\SID\dati\Sdruno\Storico_NSY\Flares"
    
    data_str = data_utc.strftime('%Y-%m-%d')
    path = os.path.join(cartella_flares, f"FLARES_{data_str}.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            dati = json.load(f)
        flares = []
        for e in dati:
            try:
                inizio = datetime.strptime(e['inizio'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                picco = datetime.strptime(e['picco'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                fine = datetime.strptime(e['fine'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                flares.append({
                    'inizio': inizio,
                    'picco': picco,
                    'fine': fine,
                    'classe': e['classe'],
                    'tipo': e['tipo'],
                    'limb': e.get('limb', False)
                })
            except Exception:
                continue
        return flares
    except Exception:
        return []


def carica_xray_local(data_utc, cartella_xray=None):
    """Carica flusso X-ray da cache locale."""
    if cartella_xray is None:
        cartella_xray = r"D:\SID\dati\Sdruno\Storico_NSY\XRAY"
    
    data_str = data_utc.strftime('%Y-%m-%d')
    path = os.path.join(cartella_xray, f"GOES_XRAY_{data_str}.csv")
    if not os.path.exists(path):
        return [], []
    
    timestamps, flux = [], []
    try:
        with open(path, 'r') as f:
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
                    if val > 0:
                        timestamps.append(dt)
                        flux.append(val)
                except Exception:
                    continue
    except Exception:
        pass
    return timestamps, flux


# ============================================================
# 3. FUNZIONI DI PERSISTENZA
# ============================================================

def salva_modello(modello, path=None):
    """Salva il modello su disco."""
    if path is None:
        path = MODELLO_PATH
    with open(path, 'wb') as f:
        pickle.dump(modello, f)
    print(f"   💾 Modello salvato su {path}")


def carica_modello(path=None):
    """Carica il modello da disco."""
    if path is None:
        path = MODELLO_PATH
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"   ⚠️ Errore caricamento modello: {e}")
        return None


def salva_risposte(data_utc, risposte, cartella=None):
    """Salva le risposte per un giorno (recovery)."""
    if cartella is None:
        cartella = os.path.dirname(os.path.abspath(__file__))
    
    data_str = data_utc.strftime('%Y-%m-%d')
    path = os.path.join(cartella, f"risposte_{data_str}.json")
    try:
        # Carica risposte esistenti
        dati = {'risposte': [], 'completato': False}
        if os.path.exists(path):
            with open(path, 'r') as f:
                dati = json.load(f)
        
        # Aggiorna con le nuove risposte
        dati['risposte'] = []
        for r in risposte:
            c = r['candidato']
            dati['risposte'].append({
                'picco': c['picco'].isoformat(),
                'ampiezza': float(c['ampiezza']),
                'durata': float(c['durata']),
                'area': float(c['area']),
                'ora': float(c['ora']),
                'etichetta': int(r['etichetta']),
                'flare_vicino': r.get('flare_vicino', None)
            })
        
        with open(path, 'w') as f:
            json.dump(dati, f, indent=2)
    except Exception as e:
        print(f"   ⚠️ Errore salvataggio risposte: {e}")


def carica_risposte(data_utc, cartella=None):
    """Carica le risposte salvate per un giorno."""
    if cartella is None:
        cartella = os.path.dirname(os.path.abspath(__file__))
    
    data_str = data_utc.strftime('%Y-%m-%d')
    path = os.path.join(cartella, f"risposte_{data_str}.json")
    if not os.path.exists(path):
        return [], False
    
    try:
        with open(path, 'r') as f:
            dati = json.load(f)
        
        risposte = []
        for r in dati.get('risposte', []):
            risposte.append({
                'picco': datetime.fromisoformat(r['picco']),
                'ampiezza': r['ampiezza'],
                'durata': r['durata'],
                'area': r['area'],
                'ora': r['ora'],
                'etichetta': r['etichetta'],
                'flare_vicino': r.get('flare_vicino', None)
            })
        completato = dati.get('completato', False)
        return risposte, completato
    except Exception:
        return [], False


def segna_completato(data_utc, cartella=None):
    """Segna un giorno come completato."""
    if cartella is None:
        cartella = os.path.dirname(os.path.abspath(__file__))
    
    data_str = data_utc.strftime('%Y-%m-%d')
    path = os.path.join(cartella, f"risposte_{data_str}.json")
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                dati = json.load(f)
            dati['completato'] = True
            with open(path, 'w') as f:
                json.dump(dati, f, indent=2)
        except Exception:
            pass


# ============================================================
# 4. FUNZIONI AUSILIARIE
# ============================================================

def parse_classe(classe_str):
    """
    Converte una classe in valore numerico per il confronto.
    """
    if not classe_str:
        return 0
    
    classe_str = classe_str.upper().strip()
    if not classe_str:
        return 0
    
    tipo = classe_str[0]
    try:
        num = float(classe_str[1:])
    except ValueError:
        num = 1.0
    
    mappa = {'A': 0, 'B': 1, 'C': 2, 'M': 3, 'X': 4}
    base = mappa.get(tipo, 0)
    
    return base + num / 10.0

def estrai_features_flare(timestamps, valori, picco_flare, finestra_secondi=600):
    """
    Estrae le features per un flare dato il suo picco.
    Calcola dt in modo robusto (mediana delle differenze tra timestamp consecutivi).
    """
    import numpy as np

    # 1. Calcola dt in modo robusto (ignora buchi e duplicati)
    if len(timestamps) < 2:
        return None

    diffs = []
    for i in range(min(50, len(timestamps)-1)):
        diff = (timestamps[i+1] - timestamps[i]).total_seconds()
        if diff > 0:  # ignora duplicati
            diffs.append(diff)

    if not diffs:
        return None

    dt = np.median(diffs)
    if dt <= 0 or dt > 10:  # se dt è >10 secondi, qualcosa non va
        print(f"   ⚠️ dt anomalo: {dt:.2f} s, fallback a 1.0")
        dt = 1.0

    # 2. Cerca l'indice del picco
    idx_picco = None
    for i, t in enumerate(timestamps):
        if abs((t - picco_flare).total_seconds()) < 5:
            idx_picco = i
            break

    if idx_picco is None:
        return None

    # 3. Calcola finestra in campioni
    finestra_idx = int(finestra_secondi / dt)
    inizio_idx = max(0, idx_picco - finestra_idx)
    fine_idx = min(len(valori), idx_picco + finestra_idx)

    # 4. Baseline (percentile 10)
    baseline = np.percentile(valori[inizio_idx:idx_picco], 10) if idx_picco > inizio_idx else valori[idx_picco]
    ampiezza = valori[idx_picco] - baseline

    # 5. Durata (half-width)
    soglia = valori[idx_picco] - ampiezza * 0.5
    inizio = idx_picco
    fine = idx_picco
    for j in range(idx_picco, inizio_idx, -1):
        if valori[j] <= soglia:
            inizio = j
            break
    for j in range(idx_picco, fine_idx):
        if valori[j] <= soglia:
            fine = j
            break
    durata = (fine - inizio) * dt
    area = np.trapezoid(valori[inizio:fine+1], dx=dt) if fine > inizio else 0

    # 6. Ora del giorno
    ora = picco_flare.hour + picco_flare.minute/60.0

    # 7. Pendenze
    if idx_picco - inizio > 2:
        pend_salita = (valori[idx_picco] - valori[inizio]) / (idx_picco - inizio)
    else:
        pend_salita = 0
    if fine - idx_picco > 2:
        pend_discesa = (valori[idx_picco] - valori[fine]) / (fine - idx_picco)
    else:
        pend_discesa = 0
    simmetria = pend_salita / (pend_discesa + 1e-6)

    return {
        'ampiezza': ampiezza,
        'durata': durata,
        'area': area,
        'ora': ora,
        'pend_salita': pend_salita,
        'pend_discesa': pend_discesa,
        'simmetria': simmetria,
        'baseline': baseline,
        'idx_picco': idx_picco,
        'inizio': timestamps[inizio],
        'fine': timestamps[fine]
    }

# ============================================================
# 5. ADDESTRAMENTO INTERATTIVO (VERSIONE FINALE)
# ============================================================
def addestra_interattivo(timestamps, valori, data_utc, classe_soglia,
                         flares_noaa=None, ax=None, fig=None,
                         cartella_flares=None, cartella_xray=None,
                         cartella_risposte=None,
                         finestra_smooth=60,
                         finestra_visualizzazione=30):
    """
    Addestramento interattivo basato sui FLARE NOAA.
    Per ogni flare NOAA, chiede all'utente se ha prodotto un SID visibile.
    L'addestramento è CUMULATIVO: carica tutte le risposte salvate in precedenza
    e addestra il modello su tutto il dataset storico.
    """
    import glob
    import json
    import os
    from scipy.signal import savgol_filter
    from sklearn.ensemble import RandomForestClassifier
    import matplotlib.patches as mpatches
    import tkinter as tk
    from tkinter import simpledialog

    # Carica flare
    if flares_noaa is None:
        flares_noaa = leggi_flare_cache(data_utc, cartella_flares)

    # Filtra per classe soglia
    classe_soglia_val = parse_classe(classe_soglia)
    flares_filtrati = [f for f in flares_noaa if parse_classe(f['classe']) >= classe_soglia_val]

    # Filtro notte
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
    except Exception:
        alba = data_utc.replace(hour=6, minute=0, second=0, microsecond=0)
        tramonto = data_utc.replace(hour=18, minute=0, second=0, microsecond=0)

    alba_stabile = alba + timedelta(minutes=30)
    tramonto_stabile = tramonto - timedelta(minutes=30)

    flares_giorno = [f for f in flares_filtrati if alba_stabile <= f['picco'] <= tramonto_stabile]

    if not flares_giorno:
        print(f"⚠️ Nessun flare diurno >= {classe_soglia} per {data_utc.strftime('%Y-%m-%d')}")
        return None, []

    print(f"\n📋 Trovati {len(flares_giorno)} flare NOAA diurni per {data_utc.strftime('%Y-%m-%d')}:")
    for f in flares_giorno:
        print(f"   {f['classe']} alle {f['picco'].strftime('%H:%M')} UTC")

    # --- RECOVERY: carica risposte salvate per questo giorno (se esistono) ---
    risposte_salvate, completato = carica_risposte_flare(data_utc, cartella_risposte)

    if risposte_salvate:
        print(f"   📂 Trovate {len(risposte_salvate)} risposte salvate per questo giorno (completato={completato}).")
        if completato:
            print(f"   ℹ️ Training già completato per questo giorno.")
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            risp = simpledialog.askstring(
                "Training esistente",
                "Training già completato per questo giorno.\nRifare da capo? [S/N]",
                parent=root
            )
            root.destroy()
            if risp and risp.lower() != 's':
                modello_path = os.path.join(cartella_risposte, "modello_sid.pkl") if cartella_risposte else MODELLO_PATH
                modello = carica_modello(modello_path)
                return modello, risposte_salvate
            else:
                print("   🔄 Ricomincio da capo.")
                risposte_salvate = []
        else:
            print(f"   ⚠️ Training non completato. Riprendo dalle {len(risposte_salvate)} risposte salvate.")
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            risp = simpledialog.askstring(
                "Training non completato",
                f"Ci sono {len(risposte_salvate)} risposte salvate ma training non completato.\n"
                "Continuare da dove si era interrotto? [S/N]\n"
                "(S=continua, N=ricomincia da capo)",
                parent=root
            )
            root.destroy()
            if risp and risp.lower() != 's':
                print("   🔄 Ricomincio da capo.")
                risposte_salvate = []
            else:
                print(f"   ➡️ Continuo da {len(risposte_salvate)} risposte.")
    else:
        print("   📂 Nessuna risposta salvata per questo giorno.")

    # Smoothing
    if finestra_smooth > 1:
        finestra_sg = finestra_smooth if finestra_smooth % 2 == 1 else finestra_smooth + 1
        if finestra_sg < 5:
            finestra_sg = 5
        polyordine = min(2, finestra_sg - 1)
        try:
            valori_plot = savgol_filter(valori, window_length=finestra_sg, polyorder=polyordine)
            print(f"   📊 Smoothing applicato: {finestra_smooth}s")
        except Exception:
            valori_plot = valori
    else:
        valori_plot = valori

    # Figura
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))
        fig.patch.set_facecolor('#0d0d1a')
        ax.set_facecolor('#0d0d1a')
        mostra_figura = True
    else:
        mostra_figura = False

    xray_ts, xray_flux = carica_xray_local(data_utc, cartella_xray)

    # Inizializza risposte con quelle salvate
    risposte = risposte_salvate.copy() if risposte_salvate else []
    indice_partenza = len(risposte)
    modello = None
    print(f"   📝 Punto di partenza: candidato {indice_partenza+1}/{len(flares_giorno)}")

    inizio_giorno = data_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    fine_giorno = data_utc.replace(hour=23, minute=59, second=59, microsecond=0)
    ax.set_xlim(inizio_giorno, fine_giorno)

    try:
        timestamps_float = np.array([t.timestamp() for t in timestamps])

        # --- CREAZIONE FINESTRA TOTEM (Una volta sola, fuori dal ciclo) ---
        root = tk.Tk()
        root.title("Conferma SID")
        root.attributes('-topmost', True)
        root.geometry("500x200")
        root.configure(bg='#0d0d1a')

        messaggio_var = tk.StringVar(value="Caricamento in corso...")
        tk.Label(root, textvariable=messaggio_var, bg='#0d0d1a', fg='white',
                 font=('Consolas', 10), justify='left').pack(pady=15, padx=15)

        risultato = {'valore': None}
        wait_var = tk.StringVar(value='')

        def set_risposta(val):
            risultato['valore'] = val
            wait_var.set('procedi')

        def set_sid(): set_risposta('s')
        def set_no(): set_risposta('n')
        def set_skip(): set_risposta('skip')
        def set_quit(): set_risposta('quit')
        def on_closing(): set_risposta('skip')

        root.protocol("WM_DELETE_WINDOW", on_closing)

        frame_btn = tk.Frame(root, bg='#0d0d1a')
        frame_btn.pack(pady=10)
        for txt, cmd, col in [
            ("SÌ (è SID)", set_sid, '#00FF99'),
            ("NO", set_no, '#FF4444'),
            ("SKIP", set_skip, '#444466'),
            ("QUIT", set_quit, '#663333')
        ]:
            tk.Button(frame_btn, text=txt, command=cmd,
                      bg=col, fg='white' if col != '#00FF99' else 'black',
                      font=('Consolas', 10, 'bold') if col in ('#00FF99','#FF4444') else ('Consolas', 10),
                      relief='flat', padx=15, pady=5).pack(side='left', padx=5)

        for i, flare in enumerate(flares_giorno[indice_partenza:], start=indice_partenza):
            picco_flare = flare['picco']
            ax.cla()

            # Segnale VLF
            ax.plot(timestamps, valori_plot, color='#00FF99', linewidth=0.8, alpha=0.7, label='VLF')

            # Evidenzia il flare corrente
            inizio_flare = picco_flare - timedelta(minutes=finestra_visualizzazione)
            fine_flare = picco_flare + timedelta(minutes=finestra_visualizzazione)
            ax.axvline(picco_flare, color='#FF0000', linewidth=2, linestyle='--', zorder=5)
            ax.axvspan(inizio_flare, fine_flare, alpha=0.15, color='#FFFF00', zorder=3)

            # Tutti i flare
            for f in flares_giorno:
                col = 'red' if f['tipo'] == 'X' else ('orange' if f['tipo'] == 'M' else 'yellow')
                ax.axvline(f['picco'], color=col, linewidth=1.0, alpha=0.5, linestyle='-')
                y_pos = np.percentile(valori_plot, 95) if len(valori_plot) > 0 else 0
                ax.text(f['picco'], y_pos, f['classe'],
                       color=col, fontsize=8, ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d0d1a', alpha=0.6))

            # Cerchio sul picco del flare corrente
            y_picco = np.interp(picco_flare.timestamp(), timestamps_float, valori_plot)
            ax.plot(picco_flare, y_picco, 'ro', markersize=10, zorder=6)

            # X-ray
            if xray_ts and xray_flux:
                ax2 = ax.twinx()
                ax2.set_yscale('log')
                ax2.plot(xray_ts, xray_flux, color='#FFD700', linewidth=1.0, alpha=0.6, label='X-ray')
                ax2.set_ylabel('X-ray flux (W/m²)', color='#FFD700', fontsize=9)
                ax2.tick_params(axis='y', colors='#FFD700', labelsize=8)

            ax.set_xlim(inizio_giorno, fine_giorno)
            ax.set_title(f"Flare {i+1}/{len(flares_giorno)}: {flare['classe']} alle {picco_flare.strftime('%H:%M:%S')} UTC",
                        color='white', fontsize=12, fontweight='bold')
            ax.set_xlabel('Ora UTC', color='white', fontsize=10)
            ax.set_ylabel('Segnale VLF', color='#00FF99', fontsize=10)
            ax.tick_params(colors='white', labelsize=9)
            ax.grid(True, alpha=0.15, color='white')
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=1))

            # Dettagli
            dettagli = f"""
FLARE NOAA:
Classe: {flare['classe']}
Picco: {picco_flare.strftime('%H:%M:%S')} UTC
Inizio: {flare['inizio'].strftime('%H:%M')} UTC
Fine: {flare['fine'].strftime('%H:%M')} UTC
{'⚠️ Beyond-the-limb' if flare.get('limb', False) else ''}
            """
            ax.text(0.02, 0.98, dettagli, transform=ax.transAxes,
                    color='white', fontsize=9, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', alpha=0.85))

            # Legenda
            handles = [
                mpatches.Patch(color='#00FF99', label='VLF'),
                mpatches.Patch(color='#FFFF00', alpha=0.2, label='Finestra flare'),
                mpatches.Patch(color='#FF0000', label='Picco flare')
            ]
            if xray_ts and xray_flux:
                handles.append(mpatches.Patch(color='#FFD700', label='X-ray'))
            ax.legend(handles=handles, loc='upper right', facecolor='#1a1a2e',
                     edgecolor='#444444', labelcolor='white', fontsize=8)

            if mostra_figura:
                plt.tight_layout()
                plt.draw()
                plt.pause(0.1)
            else:
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            # --- CHIEDI CONFERMA ---
            msg = f"Flare {i+1}/{len(flares_giorno)}: {flare['classe']} alle {picco_flare.strftime('%H:%M:%S')} UTC\n\n"
            msg += "Vedi un SID (aumento del segnale) associato a questo flare?"
            
            messaggio_var.set(msg)       # Aggiorna solo il testo nella finestra fissa
            risultato['valore'] = None
            
            # Ciclo di attesa che non blocca Matplotlib
            while risultato['valore'] is None:
                root.update()
                time.sleep(0.05)
            
            risposta = risultato['valore'] or 'skip'

            if risposta == 'quit':
                print("   ⏹️ Addestramento interrotto dall'utente.")
                if risposte:
                    salva_risposte_flare(data_utc, risposte, cartella_risposte)
                raise KeyboardInterrupt
            elif risposta == 'skip':
                print(f"   ⏭️ Flare {flare['classe']} alle {picco_flare.strftime('%H:%M')} → Saltato")
                continue

            etichetta = 1 if risposta == 's' else 0

            # --- CALCOLA FEATURES e SALVA CON FEATURES ---
            features = estrai_features_flare(timestamps, valori, picco_flare)
            if features is None:
                print(f"   ⚠️ Impossibile estrarre features per {flare['classe']} alle {picco_flare.strftime('%H:%M')}, salto.")
                continue

            # Aggiungi alla lista risposte con le features
            risposte.append({
                'classe': flare['classe'],
                'picco': picco_flare,
                'etichetta': etichetta,
                'features': [
                    features['ampiezza'], features['durata'], features['area'],
                    features['ora'], features['pend_salita'], features['pend_discesa'],
                    features['simmetria']
                ]
            })
            print(f"   {'✅' if etichetta else '❌'} Flare {flare['classe']} alle {picco_flare.strftime('%H:%M')} → {'SID VERO' if etichetta else 'NON è SID'}")

            # Salva subito (con features)
            salva_risposte_flare(data_utc, risposte, cartella_risposte)
            print(f"   📝 DEBUG: risposta salvata! Totale risposte: {len(risposte)}")
            if len(risposte) % 5 == 0:
                print(f"   💾 Salvataggio automatico ({len(risposte)} risposte)")

        # --- FINE CICLO ---

        # --- ADDESTRAMENTO CUMULATIVO ---
        print(f"\n   🔍 DEBUG: risposte totali dopo il ciclo = {len(risposte)}")
        for idx, r in enumerate(risposte):
            print(f"      {idx+1}. {r['classe']} alle {r['picco'].strftime('%H:%M')} -> {'SID VERO' if r['etichetta'] == 1 else 'NON SID'}")

        # Costruisci X_train, y_train da TUTTI i file JSON presenti nella cartella
        X_train_all = []
        y_train_all = []

        # 1. Carica tutti i file storici
        pattern = os.path.join(cartella_risposte, "risposte_flare_*.json")
        for file_path in glob.glob(pattern):
            try:
                with open(file_path, 'r') as f:
                    dati = json.load(f)
                    for r in dati.get('risposte', []):
                        if 'features' in r and r['features']:
                            X_train_all.append(r['features'])
                            y_train_all.append(r['etichetta'])
                        else:
                            print(f"   ⚠️ File {os.path.basename(file_path)} non ha features, saltato.")
            except Exception as e:
                print(f"   ⚠️ Errore caricamento {file_path}: {e}")

        # 2. Aggiungi le risposte del giorno corrente (già in 'risposte')
        for r in risposte:
            if 'features' in r and r['features']:
                X_train_all.append(r['features'])
                y_train_all.append(r['etichetta'])
            else:
                # Se mancano, ricalcola (ma non dovrebbe accadere)
                features = estrai_features_flare(timestamps, valori, r['picco'])
                if features:
                    X_train_all.append([
                        features['ampiezza'], features['durata'], features['area'],
                        features['ora'], features['pend_salita'], features['pend_discesa'],
                        features['simmetria']
                    ])
                    y_train_all.append(r['etichetta'])
                else:
                    print(f"   ⚠️ Features mancanti per {r['classe']} alle {r['picco']}, saltato.")

        print(f"   📚 Dataset totale per training: {len(X_train_all)} esempi (storico + nuovo)")

        if len(X_train_all) >= 1:
            classi_presenti = set(y_train_all)
            ha_entrambe = len(classi_presenti) >= 2

            if ha_entrambe:
                modello = RandomForestClassifier(
                    n_estimators=50, max_depth=6, random_state=42,
                    class_weight='balanced'
                )
                modello.fit(X_train_all, y_train_all)
                print(f"\n✅ Addestramento completato con entrambe le classi!")
            else:
                classe_unica = classi_presenti.pop() if classi_presenti else 0
                nome_classe = "SID VERI" if classe_unica == 1 else "NON SID"
                print(f"\n⚠️ Attenzione: tutti i {len(X_train_all)} esempi sono {nome_classe}.")
                print(f"   Il modello sarà addestrato solo su una classe (meno robusto).")
                modello = RandomForestClassifier(
                    n_estimators=50, max_depth=6, random_state=42,
                    class_weight='balanced'
                )
                modello.fit(X_train_all, y_train_all)
                print(f"   ✅ Modello addestrato (solo {nome_classe}).")

            # Salva modello
            modello_path = os.path.join(cartella_risposte, "modello_sid.pkl") if cartella_risposte else MODELLO_PATH
            salva_modello(modello, path=modello_path)
            segna_completato_flare(data_utc, cartella_risposte)

            n_veri = sum(y_train_all)
            n_falsi = len(y_train_all) - n_veri
            print(f"\n📊 Riepilogo addestramento:")
            print(f"   Esempi totali: {len(X_train_all)}")
            print(f"   SID veri: {n_veri}")
            print(f"   SID falsi: {n_falsi}")
            print(f"   Modello salvato su: {modello_path}")

            if not ha_entrambe:
                print(f"\n   💡 SUGGERIMENTO: Il modello è meno robusto perché addestrato solo su {nome_classe}.")
                print(f"      Aggiungi più giorni di training per migliorarlo.")
        else:
            print("   ❌ Nessun esempio valido per l'addestramento.")

    except KeyboardInterrupt:
        print("\n⏹️ Addestramento interrotto.")
        if mostra_figura and fig:
            plt.close(fig)
        return None, risposte
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        if mostra_figura and fig:
            plt.close(fig)
        return None, risposte
    finally:
        if mostra_figura and fig:
            plt.close(fig)
        # Chiude la finestra Tkinter fissa alla fine del processo
        try:
            root.destroy()
        except:
            pass

    return modello, risposte

#=======================================================
# 6. RILEVAZIONE SID CON MODELLO
# ============================================================

def rileva_sid_con_modello(timestamps, valori, data_utc, modello,
                           soglia_prob=SOGLIA_PROBABILITA_DEFAULT,
                           flares_noaa=None, flares_manuali=None,
                           cartella_flares=None):
    """
    Rileva SID usando il modello addestrato.
    Per ogni flare NOAA e manuale (diurno), classifica se è SID.
    """
    if flares_noaa is None:
        flares_noaa = leggi_flare_cache(data_utc, cartella_flares)
    
    flares = list(flares_noaa)
    if flares_manuali:
        flares.extend(flares_manuali)
    
    if not flares:
        print("   ℹ️ Nessun flare trovato per questo giorno.")
        return []
    
    # Filtra notte
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
    except Exception:
        alba = data_utc.replace(hour=6, minute=0, second=0, microsecond=0)
        tramonto = data_utc.replace(hour=18, minute=0, second=0, microsecond=0)
    
    alba_stabile = alba + timedelta(minutes=30)
    tramonto_stabile = tramonto - timedelta(minutes=30)
    
    sid_list = []
    for f in flares:
        picco_flare = f['picco']
        if picco_flare < alba_stabile or picco_flare > tramonto_stabile:
            continue
        
        features = estrai_features_flare(timestamps, valori, picco_flare)
        if features is None:
            continue
        
        X = np.array([[
            features['ampiezza'],
            features['durata'],
            features['area'],
            features['ora'],
            features['pend_salita'],
            features['pend_discesa'],
            features['simmetria']
        ]])
        
        # --- GESTIONE MODELLO CON UNA SOLA CLASSE ---
        if hasattr(modello, 'predict_proba'):
            proba = modello.predict_proba(X)
            # Verifica quante colonne ha la probabilità
            if proba.shape[1] >= 2:
                # Modello con entrambe le classi: prendi probabilità classe 1
                prob = proba[0, 1]
            else:
                # Modello con una sola classe: la probabilità della classe presente è 1.0
                # Se la classe presente è 1 (SID), prob = 1.0; se è 0, prob = 0.0
                # Ma dato che abbiamo addestrato solo su una classe, prendiamo il valore
                # come probabilità (sarà 1.0 o 0.0)
                prob = proba[0, 0] if hasattr(modello, 'classes_') and modello.classes_[0] == 1 else 1.0 - proba[0, 0]
                # Se il modello ha una sola classe e la classe è 1, la probabilità è 1.0
                # Se la classe è 0, la probabilità è 0.0
                # Semplifichiamo: se la classe è 1 (SID), prob = 1.0, altrimenti 0.0
                if hasattr(modello, 'classes_') and len(modello.classes_) == 1:
                    prob = 1.0 if modello.classes_[0] == 1 else 0.0
                else:
                    prob = proba[0, 0]
        else:
            prob = 0.5  # fallback
        
        if prob >= soglia_prob:
            sid_list.append({
                'picco': picco_flare,
                'ampiezza': float(features['ampiezza']),
                'durata': float(features['durata']),
                'area': float(features['area']),
                'inizio': features['inizio'],
                'fine': features['fine'],
                'baseline': float(features['baseline']),
                'probabilita': float(prob),
                'flare': f.get('classe', 'manuale')
            })
    
    print(f"   ✅ SID rilevati su {len(flares)} flare diurni: {len(sid_list)}")
    return sid_list

# ============================================================
# 7. FUNZIONE PER MODIFICARE SOGLIA (utile)
# ============================================================

def modifica_soglia_probabilita():
    """Permette all'utente di modificare la soglia di probabilità usando una finestra Tkinter."""

    global SOGLIA_PROBABILITA_DEFAULT
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    msg = f"Soglia probabilità attuale: {SOGLIA_PROBABILITA_DEFAULT:.2f}\nInserisci nuovo valore (0.1-0.9) o lascia vuoto per mantenere:"
    risp = simpledialog.askstring("Modifica soglia", msg, parent=root)
    root.destroy()
    
    if risp:
        try:
            val = float(risp.strip())
            if 0.0 <= val <= 1.0:
                SOGLIA_PROBABILITA_DEFAULT = val
                print(f"   ✅ Soglia aggiornata a {val}")
                return val
            else:
                print(f"   ⚠️ Valore non valido. Mantengo {SOGLIA_PROBABILITA_DEFAULT}")
        except ValueError:
            print(f"   ⚠️ Valore non valido. Mantengo {SOGLIA_PROBABILITA_DEFAULT}")
    else:
        print(f"   ℹ️ Soglia mantenuta a {SOGLIA_PROBABILITA_DEFAULT}")
    
    return SOGLIA_PROBABILITA_DEFAULT

def chiedi_conferma_tk(messaggio, titolo="Conferma SID"):
    import tkinter as tk
    from tkinter import simpledialog
    
    while True:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        risposta = simpledialog.askstring(
            titolo,
            messaggio + "\n\n" + "-" * 50 + "\n" +
            "Inserisci:\n" +
            "  S  -> Sì, è un SID\n" +
            "  N  -> No, non è un SID\n" +
            "  skip -> Salta questo flare\n" +
            "  quit -> Interrompi l'addestramento\n" +
            "-" * 50,
            parent=root
        )
        root.destroy()
        
        if risposta is None:
            return 'quit'
        
        risposta = risposta.strip().lower()
        
        if risposta in ('s', 'si', 'yes', 'y'):
            return 's'
        elif risposta in ('n', 'no'):
            return 'n'
        elif risposta in ('skip', ''):
            return 'skip'
        elif risposta in ('quit', 'q', 'esci', 'exit'):
            return 'quit'
        else:
            # Mostra errore e ricomincia il ciclo
            root_err = tk.Tk()
            root_err.withdraw()
            root_err.attributes('-topmost', True)
            simpledialog.askstring(
                "Risposta non valida",
                f"⚠️ '{risposta}' non è valido.\n\nUsa: S, N, skip o quit.",
                parent=root_err
            )
            root_err.destroy()
            # Il ciclo continua
            
def segna_completato_flare(data_utc, cartella=None):
    """
    Segna un giorno come completato (formato flare).
    
    Args:
        data_utc (datetime): Data del giorno da segnare come completato
        cartella (str, optional): Cartella dove cercare il file.
            Default: ./training_responses/
    """
    if cartella is None:
        cartella = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_responses")
    
    # Assicura che la cartella esista
    os.makedirs(cartella, exist_ok=True)
    
    data_str = data_utc.strftime('%Y-%m-%d')
    path = os.path.join(cartella, f"risposte_flare_{data_str}.json")
    
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                dati = json.load(f)
            dati['completato'] = True
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(dati, f, indent=2)
            print(f"   ✅ Giorno {data_str} segnato come completato.")
        except Exception as e:
            print(f"   ⚠️ Errore durante il salvataggio: {e}")
    else:
        print(f"   ⚠️ File non trovato: {path}")

# ============================================================
# FUNZIONI PER IL SALVATAGGIO DELLE RISPOSTE (FORMATO FLARE)
# ============================================================
def salva_risposte_flare(data_utc, risposte, cartella=None):
    """
    Salva le risposte per un giorno (formato flare) includendo le features.
    """
    if cartella is None:
        cartella = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_responses")
    os.makedirs(cartella, exist_ok=True)
    
    data_str = data_utc.strftime('%Y-%m-%d')
    path = os.path.join(cartella, f"risposte_flare_{data_str}.json")
    
    # Carica risposte esistenti (per evitare duplicati)
    dati = {'risposte': [], 'completato': False}
    if os.path.exists(path):
        with open(path, 'r') as f:
            dati = json.load(f)
    
    # Dizionario per deduplicare: chiave = (classe, picco_iso)
    risposte_dict = {}
    for r in dati.get('risposte', []):
        key = (r.get('classe', '?'), r.get('picco', ''))
        risposte_dict[key] = r  # manteniamo l'intero record
    
    # Aggiungi le nuove risposte (sovrascrivono se stesso picco/classe)
    for r in risposte:
        # r può essere nel formato vecchio (con 'flare') o nuovo (con 'classe', 'picco')
        if 'flare' in r:
            flare = r['flare']
            classe = flare['classe']
            picco = flare['picco']
        else:
            classe = r.get('classe')
            picco = r.get('picco')
        
        if classe is None or picco is None:
            continue
        
        if hasattr(picco, 'isoformat'):
            picco_iso = picco.isoformat()
        else:
            picco_iso = str(picco)
        
        # Estrai le features se presenti (sono già state calcolate al momento dell'etichettatura)
        features = r.get('features')  # lista di 7 float, o None se non presente
        
        # Se non ci sono features, le calcoliamo ora (ma avremo bisogno di timestamps/valori, 
        # che potrebbero non essere disponibili in questo contesto). 
        # Per sicurezza, se mancano, le calcoliamo con i dati correnti se passati, 
        # ma per semplicità assumiamo che chi chiama questa funzione le abbia già calcolate.
        # Per evitare problemi, se features è None, salviamo una lista di zeri (ma è meglio evitare).
        if features is None:
            features = [0.0]*7  # fallback, ma non dovrebbe accadere se usiamo la nuova logica
        
        key = (classe, picco_iso)
        risposte_dict[key] = {
            'classe': classe,
            'picco': picco_iso,
            'etichetta': int(r.get('etichetta', 0)),
            'features': features
        }
    
    # Ricostruisci la lista
    nuove_risposte = list(risposte_dict.values())
    dati['risposte'] = nuove_risposte
    # Non modificare il flag 'completato' qui (verrà gestito separatamente)
    
    with open(path, 'w') as f:
        json.dump(dati, f, indent=2)
    
    print(f"   💾 Salvate {len(nuove_risposte)} risposte uniche in {path}")
    return len(nuove_risposte)

    
def carica_risposte_flare(data_utc, cartella=None):
    """Carica le risposte salvate per un giorno (formato flare)."""
    if cartella is None:
        cartella = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_responses")
    
    data_str = data_utc.strftime('%Y-%m-%d')
    path = os.path.join(cartella, f"risposte_flare_{data_str}.json")
    if not os.path.exists(path):
        return [], False
    
    try:
        with open(path, 'r') as f:
            dati = json.load(f)
        risposte = []
        for r in dati.get('risposte', []):
            risposte.append({
                'classe': r['classe'],
                'picco': datetime.fromisoformat(r['picco']),
                'etichetta': r['etichetta'],
                'features': r.get('features', None)
            })
        completato = dati.get('completato', False)
        return risposte, completato
    except Exception:
        return [], False
    
def get_modello_path(cartella):
    """Restituisce il percorso completo del modello in una cartella."""
    return os.path.join(cartella, "modello_sid.pkl")