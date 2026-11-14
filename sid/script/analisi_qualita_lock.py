import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os
import tkinter as tk
from datetime import datetime, date

try:
    from tkcalendar import Calendar
    HAS_TKCALENDAR = True
except ImportError:
    HAS_TKCALENDAR = False

# Cartella dati (in linea con il contesto progetto)
import glob

# Cartelle dati (in linea con visualizza_tempesta.py)
DATI_DIR    = r"D:\Gaetano\SID\dati\Sdruno"
STORICO_DIR = r"D:\Gaetano\SID\dati\Sdruno\Storico_NSY"

# ----------------------------------------------------------------------
# RICERCA FILE CSV (Sdruno + Storico_NSY)
# ----------------------------------------------------------------------
def trova_file_csv(d: date):
    """Cerca il CSV per la data richiesta.
    1) D:\\...\\Sdruno   -> nome standard, non ricorsiva
    2) D:\\...\\Storico_NSY -> ricorsiva, nome standard o pattern con la data
    Ritorna (path | None, provenienza_str)."""
    data_str = d.isoformat()
    nome_std = f"GAESID_NSY_{data_str}.csv"

    # 1) Cartella principale, nome standard
    p = os.path.join(DATI_DIR, nome_std)
    if os.path.exists(p):
        return p, "Sdruno"

    # 2) Storico_NSY ricorsiva: prima nome standard, poi qualsiasi
    #    CSV che contenga la data nel nome (per file storici con naming diverso)
    if os.path.isdir(STORICO_DIR):
        std_ricorsivo = glob.glob(
            os.path.join(STORICO_DIR, "**", nome_std), recursive=True)
        if std_ricorsivo:
            return std_ricorsivo[0], "Storico_NSY"

        # Fallback: qualunque *.csv contenente la data nel nome
        candidati = [
            f for f in glob.glob(
                os.path.join(STORICO_DIR, "**", "*.csv"), recursive=True)
            if data_str in os.path.basename(f)
        ]
        if candidati:
            # Se più di uno, scegli il più recente (mtime) e logga gli altri
            candidati.sort(key=os.path.getmtime, reverse=True)
            if len(candidati) > 1:
                print(f"Attenzione: {len(candidati)} CSV trovati per {data_str} "
                      f"in Storico_NSY. Uso il più recente:")
                for c in candidati:
                    print(f"   - {c}")
            return candidati[0], "Storico_NSY (fallback)"

    return None, "non trovato"

# ----------------------------------------------------------------------
# SELEZIONE DATA (calendario Tk standalone)
# ----------------------------------------------------------------------
def scegli_data(data_iniziale=None):
    """Apre un calendario Tk standalone e ritorna un datetime.date.
    Ritorna None se l'utente annulla o chiude la finestra."""
    if not HAS_TKCALENDAR:
        # Fallback testuale se tkcalendar non è installato
        default = (data_iniziale or date.today()).isoformat()
        s = input(f"Inserisci data (YYYY-MM-DD) [{default}]: ").strip()
        if not s:
            return data_iniziale or date.today()
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except ValueError:
            print("Formato non valido.")
            return None

    data_scelta = [None]

    root = tk.Tk()
    root.title("Seleziona data")
    root.resizable(False, False)

    if data_iniziale is not None:
        cal = Calendar(root, selectmode='day',
                       year=data_iniziale.year,
                       month=data_iniziale.month,
                       day=data_iniziale.day,
                       date_pattern='yyyy-mm-dd')
    else:
        cal = Calendar(root, selectmode='day', date_pattern='yyyy-mm-dd')
    cal.pack(padx=10, pady=10)

    def conferma():
        try:
            data_scelta[0] = cal.selection_get()
        except Exception:
            data_scelta[0] = None
        root.destroy()

    def annulla():
        root.destroy()

    frame = tk.Frame(root)
    frame.pack(pady=(0, 10))
    tk.Button(frame, text="OK", command=conferma, width=10).pack(side='left', padx=5)
    tk.Button(frame, text="Annulla", command=annulla, width=10).pack(side='left', padx=5)

    root.bind('<Return>', lambda e: conferma())
    root.bind('<Escape>', lambda e: annulla())

    root.mainloop()
    return data_scelta[0]


# ----------------------------------------------------------------------
# ANALIZZATORE INTERATTIVO
# ----------------------------------------------------------------------
class AnalizzatoreLock:
    def __init__(self, data_iniziale: date):
        self.data_corrente = data_iniziale

        self.fig, self.axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        self.fig.suptitle("Analisi Qualità Lock V2", fontsize=14, fontweight='bold')
        #self.fig.tight_layout()

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Primo caricamento
        self.carica_e_disegna()

        # Aggiunge un'indicazione dei tasti disponibili
        self.fig.text(0.01, 0.005,
                      "Tasti: I = cambia data (calendario) | R = refresh | Q = esci",
                      fontsize=8, color='gray')

        plt.show()

    # ------------------------------------------------------------------
    def _path_per_data(self, d: date) -> str:
        return os.path.join(DATI_DIR, f"GAESID_NSY_{d.isoformat()}.csv")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def carica_e_disegna(self):
        file_path, provenienza = trova_file_csv(self.data_corrente)

        for ax in self.axs:
            ax.clear()

        if file_path is None:
            self.fig.suptitle(
                f"Analisi Qualità Lock V2: {self.data_corrente} — FILE NON TROVATO",
                fontsize=14, fontweight='bold', color='red')
            for ax in self.axs:
                ax.grid(True, alpha=0.3)
            self.fig.canvas.draw_idle()
            print(f"File non trovato per {self.data_corrente} "
                  f"(cercato in Sdruno e Storico_NSY).")
            return

        print(f"Caricamento dati [{provenienza}]: {file_path}")
        df = pd.read_csv(file_path, comment="#")
        df['Timestamp_UTC'] = pd.to_datetime(df['Timestamp_UTC'], utc=True)

        # Verifica colonne V2
        if 'Corr_Mag' not in df.columns or 'Phase_Resid_Deg' not in df.columns:
            self.fig.suptitle(
                f"Analisi Qualità Lock V2: {os.path.basename(file_path)} "
                f"— COLONNE V2 MANCANTI",
                fontsize=14, fontweight='bold', color='red')
            for ax in self.axs:
                ax.grid(True, alpha=0.3)
            self.fig.canvas.draw_idle()
            print("Errore: il file non contiene le colonne della V2 "
                  "(Corr_Mag, Phase_Resid_Deg).")
            return

        # Varianza locale del residuo (finestra 5 min)
        df['Resid_Var'] = df['Phase_Resid_Deg'].rolling(
            window=300, center=True, min_periods=60).var()

        # 1. Potenza
        self.axs[0].plot(df['Timestamp_UTC'], df['Power_dB'],
                         color='green', linewidth=0.5)
        self.axs[0].set_ylabel('Potenza (dB)')
        self.axs[0].grid(True, alpha=0.3)
        self.axs[0].set_title('Potenza del segnale NSY', loc='left', fontsize=10)

        # 2. Corr_Mag
        self.axs[1].plot(df['Timestamp_UTC'], df['Corr_Mag'],
                         color='blue', linewidth=0.7)
        self.axs[1].set_ylabel('Corr_Mag (0-1)')
        self.axs[1].set_ylim(0, 1.05)
        self.axs[1].grid(True, alpha=0.3)
        self.axs[1].set_title('Coerenza del segnale (Corr_Mag)',
                              loc='left', fontsize=10)
        self.axs[1].axhline(0.3, color='red', linestyle='--', alpha=0.6,
                            label='Soglia Lock Marginale')
        self.axs[1].legend(loc='upper right')

        # 3. Phase Resid + Std locale
        self.axs[2].plot(df['Timestamp_UTC'], df['Phase_Resid_Deg'],
                         color='orange', linewidth=0.5, alpha=0.7,
                         label='Residuo grezzo')
        self.axs[2].plot(df['Timestamp_UTC'], np.sqrt(df['Resid_Var']),
                         color='red', linewidth=1.5,
                         label='Std Locale (5 min)')
        self.axs[2].set_ylabel('Fase Residua (°)')
        self.axs[2].grid(True, alpha=0.3)
        self.axs[2].set_title('Errore di fase del PLL (Phase_Resid_Deg)',
                              loc='left', fontsize=10)
        self.axs[2].legend(loc='upper right')

        # 4. Frequenza PLL
        self.axs[3].plot(df['Timestamp_UTC'], df['PLL_Freq_Hz'],
                         color='purple', linewidth=0.5)
        self.axs[3].set_ylabel('Freq (Hz)')
        self.axs[3].grid(True, alpha=0.3)
        self.axs[3].set_title('Frequenza stimata dal PLL',
                              loc='left', fontsize=10)
        self.axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        # Titolo con provenienza
        self.fig.suptitle(
            f"Analisi Qualità Lock V2: {os.path.basename(file_path)}  "
            f"[{provenienza}]",
            fontsize=14, fontweight='bold', color='black')

        self.fig.autofmt_xdate()
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        # Salvataggio PNG (nome che include la data)
        out_png = f"analisi_qualita_lock_{self.data_corrente.isoformat()}.png"
        #self.fig.savefig(out_png, dpi=150)
        print(f"Grafico salvato in: {out_png}")

        # Analisi statistica
        self._stampa_statistiche(df)

    # ------------------------------------------------------------------
    def _stampa_statistiche(self, df):
        print("\n" + "=" * 50)
        print(" VALIDAZIONE TEORICA: Corr_Mag vs Varianza del Residuo")
        print("=" * 50)

        bins = np.linspace(0, 1, 11)
        df['Mag_Bin'] = pd.cut(df['Corr_Mag'], bins=bins)
        stats = df.groupby('Mag_Bin', observed=True)['Resid_Var'].mean().dropna()

        print("Media della Varianza del Residuo per livello di Corr_Mag:")
        for mag_bin, var in stats.items():
            print(f"  Mag {mag_bin}: Varianza = {var:.1f} °² (Std = {np.sqrt(var):.1f}°)")

        print("\nNota: Se la teoria è corretta, la varianza dovrebbe diminuire "
              "all'aumentare di Corr_Mag.")

    # ------------------------------------------------------------------
    def _on_key(self, event):
        if event.key is None:
            return
        k = event.key.lower()
        if k == 'i':
            nuova_data = scegli_data(self.data_corrente)
            if nuova_data is not None:
                self.data_corrente = nuova_data
                self.carica_e_disegna()
        elif k == 'r':
            self.carica_e_disegna()
        elif k == 'q':
            plt.close(self.fig)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            data_iniziale = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
        except ValueError:
            print("Formato data non valido. Usa YYYY-MM-DD.")
            sys.exit(1)
    else:
        data_iniziale = scegli_data()
        if data_iniziale is None:
            print("Nessuna data selezionata. Uscita.")
            sys.exit(0)

    AnalizzatoreLock(data_iniziale) 