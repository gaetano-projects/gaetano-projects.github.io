import time
import os
from datetime import datetime, timezone
import pytz

# =============================================================
#  CONFIGURAZIONE UTENTE — modifica questi valori
# =============================================================
#
#  Requisiti: Python 3.13 o superiore
#  Librerie:  pip install pytz
#
#  FILE_ORIGINE          : percorso del file CSV generato da SDRuno
#  CARTELLA_DESTINAZIONE : cartella dove salvare i file giornalieri
#  STAZIONE_ID           : identificativo della tua stazione
#  TRASMETTITORE         : nome della stazione VLF monitorata
#  FREQUENZA_HZ          : frequenza in Hz della stazione monitorata
#  SITO                  : nome del sito di ricezione
#  LONGITUDINE           : longitudine del sito (gradi decimali)
#  LATITUDINE            : latitudine del sito (gradi decimali)
#  FUSO_LOCALE           : fuso orario locale (es. "Europe/Rome")
#
FILE_ORIGINE          = r"C:\SID\dati\SDRuno_PWRSNR.csv"  # <-- modifica
CARTELLA_DESTINAZIONE = r"C:\SID\dati"                    # <-- modifica
STAZIONE_ID           = "GAESID"                          # <-- modifica
TRASMETTITORE         = "DHO38"                           # <-- modifica
FREQUENZA_HZ          = 23400                             # <-- modifica
SITO                  = "Torino"                          # <-- modifica
LONGITUDINE           = "7.68"                            # <-- modifica
LATITUDINE            = "45.07"                           # <-- modifica
FUSO_LOCALE           = pytz.timezone("Europe/Rome")      # <-- modifica
# =============================================================


def nome_file_giorno(data_utc):
    """Genera il nome file per il giorno UTC corrente."""
    return os.path.join(
        CARTELLA_DESTINAZIONE,
        f"{STAZIONE_ID}_{TRASMETTITORE}_{data_utc.strftime('%Y-%m-%d')}.csv"
    )


def crea_header_compatibile(filepath, data_utc):
    """Header in formato SID classico per il giorno specificato."""
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
    print(f">>> Nuovo file creato: {filepath}")


def get_utc_oggi():
    """Restituisce la data UTC corrente (solo data, senza orario)."""
    return datetime.now(timezone.utc).date()


def locale_to_utc(dt_str):
    """Converte una stringa di ora locale in datetime UTC.
    Gestisce automaticamente ora solare e legale.
    """
    dt_naive  = datetime.strptime(dt_str, "%d/%m/%Y %H:%M:%S")
    dt_locale = FUSO_LOCALE.localize(dt_naive)
    return dt_locale.astimezone(timezone.utc)


def leggi_ultimo_timestamp_csv(filepath):
    """Legge l'ultimo timestamp già scritto nel file CSV."""
    ultimo = None
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        for riga in f:
            if riga.startswith('#') or not riga.strip():
                continue
            parti = riga.strip().split(',')
            if len(parti) == 2:
                try:
                    ultimo = datetime.strptime(
                        parti[0].strip(), "%Y-%m-%d %H:%M:%S"
                    ).replace(tzinfo=timezone.utc)
                except Exception:
                    pass
    return ultimo


# --- AVVIO ---
print("=" * 55)
print(f"  VLF SID Converter — {TRASMETTITORE} {FREQUENZA_HZ} Hz")
print(f"  Stazione: {STAZIONE_ID} — {SITO}")
print("=" * 55)
print("--- In attesa di dati da SDRuno... ---")

data_corrente_utc = get_utc_oggi()
file_corrente     = nome_file_giorno(datetime.now(timezone.utc))

# Legge l'ultimo timestamp già presente nel CSV per evitare duplicati
ultimo_timestamp_scritto = leggi_ultimo_timestamp_csv(file_corrente)

if ultimo_timestamp_scritto:
    print(f">>> Riprendo da: {ultimo_timestamp_scritto.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    try:
        with open(FILE_ORIGINE, "r", encoding='utf-8', errors='ignore') as f:
            righe_processate = sum(1 for _ in f)
        print(f">>> File sorgente: {righe_processate} righe esistenti, parto da qui.")
    except Exception:
        righe_processate = 0
else:
    # File CSV assente o vuoto: recupera tutti i dati di oggi
    print(">>> Nessun dato precedente, recupero tutti i dati di oggi...")
    if not os.path.exists(file_corrente):
        crea_header_compatibile(file_corrente, datetime.now(timezone.utc))
    righe_processate = 0

while True:
    try:
        # Controlla cambio giorno UTC
        oggi_utc = get_utc_oggi()
        if oggi_utc != data_corrente_utc:
            print(f">>> Mezzanotte UTC! Nuovo giorno: {oggi_utc}")
            data_corrente_utc = oggi_utc
            file_corrente     = nome_file_giorno(datetime.now(timezone.utc))
            crea_header_compatibile(file_corrente, datetime.now(timezone.utc))
            ultimo_timestamp_scritto = None
            try:
                with open(FILE_ORIGINE, "r", encoding='utf-8', errors='ignore') as f:
                    righe_processate = sum(1 for _ in f)
            except Exception:
                righe_processate = 0

        # Legge SOLO le righe nuove aggiunte da SDRuno
        with open(FILE_ORIGINE, "r", encoding='utf-8', errors='ignore') as f_in:
            tutte_righe = f_in.readlines()

        nuove_righe = tutte_righe[righe_processate:]

        for riga in nuove_righe:
            riga = riga.strip()
            righe_processate += 1
            if not riga:
                continue

            parti = riga.split(',')
            if len(parti) < 3:
                continue

            # Salta la riga di intestazione
            if parti[0].strip() == 'Date Stamp':
                continue

            try:
                dt_utc = locale_to_utc(parti[0].strip())

                # Scarta dati di altri giorni UTC
                if dt_utc.date() != data_corrente_utc:
                    continue

                # Scarta dati già scritti (utile dopo un riavvio)
                if ultimo_timestamp_scritto and dt_utc <= ultimo_timestamp_scritto:
                    continue

                ts_iso = dt_utc.strftime("%Y-%m-%d %H:%M:%S")
                pwr    = float(parti[2].strip())

                with open(file_corrente, "a") as f_out:
                    f_out.write(f"{ts_iso},{pwr:.2f}\n")

                ultimo_timestamp_scritto = dt_utc
                print(f"Dato registrato: {ts_iso} | {pwr:.2f}")

            except Exception:
                continue

    except Exception as e:
        log_path = os.path.join(CARTELLA_DESTINAZIONE, "acquisizione_log.txt")
        with open(log_path, "a") as log:
            log.write(
                f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} "
                f"ERRORE: {e}\n"
            )

    time.sleep(1)
