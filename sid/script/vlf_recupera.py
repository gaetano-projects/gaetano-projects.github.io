import os
import glob
from datetime import datetime, timezone
import pytz

# =============================================================
#  CONFIGURAZIONE UTENTE — modifica questi valori
# =============================================================
#
#  Requisiti: Python 3.13 o superiore
#  Librerie:  pip install pytz
#
#  CARTELLA_SDRUNO       : cartella dove si trovano i file SDRuno_PWRSNR*.csv
#  CARTELLA_DESTINAZIONE : cartella dove salvare i file giornalieri recuperati
#  STAZIONE_ID           : identificativo della tua stazione
#  TRASMETTITORE         : nome della stazione VLF monitorata
#  FREQUENZA_HZ          : frequenza in Hz della stazione monitorata
#  SITO                  : nome del sito di ricezione
#  LONGITUDINE           : longitudine del sito (gradi decimali)
#  LATITUDINE            : latitudine del sito (gradi decimali)
#  FUSO_LOCALE           : fuso orario locale (es. "Europe/Rome")
#
CARTELLA_SDRUNO       = r"C:\SID\dati"       # <-- modifica
CARTELLA_DESTINAZIONE = r"C:\SID\dati"       # <-- modifica
STAZIONE_ID           = "GAESID"             # <-- modifica
TRASMETTITORE         = "DHO38"              # <-- modifica
FREQUENZA_HZ          = 23400                # <-- modifica
SITO                  = "Torino"             # <-- modifica
LONGITUDINE           = "7.68"              # <-- modifica
LATITUDINE            = "45.07"             # <-- modifica
FUSO_LOCALE           = pytz.timezone("Europe/Rome")  # <-- modifica
# =============================================================


# ---------------------------------------------------------------
#  UTILITÀ
# ---------------------------------------------------------------

def nome_file_giorno(data_utc):
    return os.path.join(
        CARTELLA_DESTINAZIONE,
        f"{STAZIONE_ID}_{TRASMETTITORE}_{data_utc.strftime('%Y-%m-%d')}.csv"
    )


def crea_header_compatibile(filepath, data_utc):
    header = (
        f"# Site = {SITO}\n"
        f"# Longitude = {LONGITUDINE}\n"
        f"# Latitude = {LATITUDINE}\n"
        "#\n"
        f"# UTC_StartTime = {data_utc.strftime('%Y-%m-%d')} 00:00:00\n"
        "# LogInterval = 1\n"
        "# LogType = raw\n"
        f"# MonitorID = {STAZIONE_ID}\n"
        f"# StationID = {TRASMETTITORE}\n"
        f"# Frequency = {FREQUENZA_HZ}\n"
    )
    with open(filepath, "w") as f:
        f.write(header)
    print(f">>> File creato: {filepath}")


def locale_to_utc(dt_str):
    dt_naive  = datetime.strptime(dt_str, "%d/%m/%Y %H:%M:%S")
    dt_locale = FUSO_LOCALE.localize(dt_naive)
    return dt_locale.astimezone(timezone.utc)


# ---------------------------------------------------------------
#  RICERCA FILE SDRUNO
# ---------------------------------------------------------------

def trova_file_sdruno():
    """
    Trova tutti i file SDRuno_PWRSNR*.csv nella cartella,
    ordinati per data di creazione (dal più vecchio al più recente).
    Restituisce lista di tuple (path, data_creazione).
    """
    pattern = os.path.join(CARTELLA_SDRUNO, "SDRuno_PWRSNR*.csv")
    files   = glob.glob(pattern)
    if not files:
        return []

    risultati = []
    for f in files:
        try:
            ctime = os.path.getctime(f)
            dt_creazione = datetime.fromtimestamp(ctime, tz=timezone.utc)
            risultati.append((f, dt_creazione))
        except Exception as e:
            print(f">>> Impossibile leggere data di {f}: {e}")

    # Ordina dal più vecchio al più recente
    risultati.sort(key=lambda x: x[1])
    return risultati


def seleziona_file_candidati(data_utc, files_ordinati):
    """
    Dato l'elenco dei file ordinati per data di creazione,
    seleziona il/i file candidati per la data cercata.

    Strategia:
    - Il file che contiene i dati di 'data_utc' è quello creato
      PRIMA di data_utc+1giorno il cui SUCCESSIVO è creato
      DOPO data_utc (o non esiste).
    - In caso di ambiguità restituisce fino a 2 file contigui
      per sicurezza (il file potrebbe essere stato creato a cavallo
      della mezzanotte).
    - Se la ricerca per data di creazione non trova nulla,
      restituisce tutti i file (fallback completo).
    """
    if not files_ordinati:
        return []

    target      = data_utc.replace(hour=0,  minute=0,  second=0,  microsecond=0)
    target_fine = data_utc.replace(hour=23, minute=59, second=59, microsecond=0)

    candidati = []
    n = len(files_ordinati)

    for i, (path, ctime) in enumerate(files_ordinati):
        ctime_successivo = files_ordinati[i + 1][1] if i + 1 < n else None

        if ctime <= target_fine:
            if ctime_successivo is None or ctime_successivo >= target:
                candidati.append(path)
                # Aggiungi anche il precedente per sicurezza (cavallo mezzanotte)
                if i > 0 and files_ordinati[i-1][0] not in candidati:
                    candidati.insert(0, files_ordinati[i-1][0])

    if candidati:
        return candidati

    # Fallback: restituisce tutti i file
    print(">>> Ricerca per data di creazione non conclusiva, scansione completa.")
    return [f for f, _ in files_ordinati]


# ---------------------------------------------------------------
#  RECUPERO DATI
# ---------------------------------------------------------------

def recupera_da_file(filepath, data_da_recuperare, f_out):
    """
    Legge 'filepath' e scrive su f_out le righe relative a data_da_recuperare.
    Restituisce (contatore, errori).
    """
    contatore = 0
    errori    = 0
    try:
        with open(filepath, "r", encoding='utf-8', errors='ignore') as f_in:
            for riga in f_in:
                riga = riga.strip()
                if not riga:
                    continue
                parti = riga.split(',')
                if len(parti) < 3:
                    continue
                if parti[0].strip() == 'Date Stamp':
                    continue
                try:
                    dt_utc = locale_to_utc(parti[0].strip())
                    if dt_utc.date() != data_da_recuperare:
                        continue
                    ts_iso = dt_utc.strftime("%Y-%m-%d %H:%M:%S")
                    pwr    = float(parti[2].strip()) + 150
                    f_out.write(f"{ts_iso},{pwr:.2f}\n")
                    contatore += 1
                    if contatore % 3600 == 0:
                        print(f"    ...{contatore} campioni scritti ({ts_iso})")
                except Exception:
                    errori += 1
    except Exception as e:
        print(f">>> Errore apertura {filepath}: {e}")
    return contatore, errori


def recupera_giorno(data_utc):
    """Estrae da uno o più file SDRuno tutti i dati del giorno specificato."""
    file_destinazione  = nome_file_giorno(data_utc)
    data_da_recuperare = data_utc.date()

    if os.path.exists(file_destinazione):
        risposta = input(
            f">>> Il file {os.path.basename(file_destinazione)} esiste già. "
            f"Sovrascrivere? (s/n): "
        ).strip().lower()
        if risposta != 's':
            print(">>> Operazione annullata.")
            return

    # Trova tutti i file SDRuno disponibili
    files_ordinati = trova_file_sdruno()
    if not files_ordinati:
        print(f">>> Nessun file SDRuno_PWRSNR*.csv trovato in:\n    {CARTELLA_SDRUNO}")
        return

    print(f"\n>>> File SDRuno trovati ({len(files_ordinati)}):")
    for path, ctime in files_ordinati:
        print(f"    {os.path.basename(path):30s}  creato: {ctime.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # Seleziona i file candidati per la data cercata
    candidati = seleziona_file_candidati(data_utc, files_ordinati)
    if not candidati:
        print(f">>> Nessun file candidato trovato per il {data_da_recuperare}.")
        return

    print(f"\n>>> File candidati per il {data_da_recuperare}:")
    for c in candidati:
        print(f"    {os.path.basename(c)}")

    # Crea file destinazione e scansiona i candidati
    crea_header_compatibile(file_destinazione, data_utc)
    contatore_totale = 0
    errori_totali    = 0

    with open(file_destinazione, "a") as f_out:
        for filepath in candidati:
            print(f"\n>>> Scansione: {os.path.basename(filepath)} ...")
            cnt, err = recupera_da_file(filepath, data_da_recuperare, f_out)
            contatore_totale += cnt
            errori_totali    += err
            print(f"    {cnt} campioni trovati in questo file.")

    sep = '=' * 55
    if contatore_totale > 0:
        print(f"\n{sep}")
        print(f"  OPERAZIONE COMPLETATA CON SUCCESSO")
        print(sep)
        print(f"  Data recuperata : {data_da_recuperare}")
        print(f"  Campioni scritti: {contatore_totale}")
        print(f"  Righe ignorate  : {errori_totali}")
        print(f"  File salvato    : {os.path.basename(file_destinazione)}")
        print(sep)
    else:
        print(f"\n{sep}")
        print(f"  OPERAZIONE FALLITA")
        print(sep)
        print(f"  Nessun dato trovato per il {data_da_recuperare}")
        print(f"  nei file scansionati.")
        print(sep)
        os.remove(file_destinazione)


# ---------------------------------------------------------------
#  AVVIO
# ---------------------------------------------------------------
print("=" * 55)
print(f"  Script recupero dati SDRuno -> SuperSID")
print(f"  Stazione: {STAZIONE_ID} — {TRASMETTITORE} {FREQUENZA_HZ} Hz")
print("  (ricerca automatica su file SDRuno_PWRSNR*.csv)")
print("=" * 55)
print()

while True:
    data_input = input("Inserisci la data da recuperare (YYYY-MM-DD): ").strip()
    if data_input == "":
        print(">>> Devi inserire una data.\n")
        continue
    try:
        data_scelta = datetime.strptime(data_input, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        break
    except ValueError:
        print(">>> Formato non valido. Usa YYYY-MM-DD (es. 2026-03-03)\n")

recupera_giorno(data_scelta)

print()
input("Premi INVIO per chiudere la finestra...")
