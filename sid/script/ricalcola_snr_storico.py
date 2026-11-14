# ============================================================
# ricalcola_snr_storico.py
#
# Audit offline, READ-ONLY sui dati e su analisi_doppler2.py.
# Ricalcola le metriche degli eventi storici usando la v2.1,
# in modo da avere un dataset OMOGENEO per il Blind Test.
#
# Principio: usa le STESSE funzioni di analisi_doppler2.py.
# Non reimplementa nulla. Non modifica CSV né il file di analisi.
#
# Output: D:\Gaetano\SID\pysdrplay\audit_blind_test\
# ============================================================

import os
import io
import contextlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd

# --- Import silenzioso di analisi_doppler2 ---
_buf_import = io.StringIO()
with contextlib.redirect_stdout(_buf_import):
    import analisi_doppler2 as ad2

OUTPUT_DIR = r"D:\Gaetano\SID\pysdrplay\audit_blind_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Eventi da analizzare ---
# (data, HH:MM del picco GOES, classe_goes, nome logico)
EVENTI = [
    # Novembre (giorni con Corr_Mag, 9 colonne)
    ("2026-11-07", "11:57", "C6.2", "C6.2_07nov"),
    ("2026-11-07", "12:43", "M1.2", "M1.2_07nov"),
    ("2026-11-08", "08:43", "C8.0", "C8.0_08nov"),
    ("2026-11-08", "09:47", "C1.5", "C1.5_08nov"),
    ("2026-11-08", "11:27", "C2.7", "C2.7_08nov"),
    ("2026-11-08", "11:44", "C2.8", "C2.8_08nov"),
    ("2026-11-10", "08:34", "C9.3", "C9.3_10nov"),
    ("2026-11-10", "09:31", "C2.6", "C2.6_10nov"),
    ("2026-11-10", "10:02", "C8.6", "C8.6_10nov"),
    ("2026-11-10", "13:06", "C1.9", "C1.9_10nov"),
    # Ottobre: aggiungere qui DOPO il Blocco 2 (regressione)
]

# Valori di riferimento dal contesto §22.2 (v2.1, 13 Nov)
# Solo per il test di non-regressione. Tolleranza ±0.2 dB.
RIFERIMENTI = {
    "C8.6_10nov": {"pot_picco":  6.58, "tol": 0.2},
    "C2.6_10nov": {"pot_picco":  2.61, "tol": 0.2},
    "C9.3_10nov": {"pot_picco": -4.76, "tol": 0.2},
    "C1.9_10nov": {"pot_picco":  1.14, "tol": 0.2},
    "C8.0_08nov": {"pot_picco":  7.90, "tol": 0.2},
    "C1.5_08nov": {"pot_picco":  4.54, "tol": 0.2},
    "C2.7_08nov": {"pot_picco":  0.82, "tol": 0.2},
    "C2.8_08nov": {"pot_picco":  3.28, "tol": 0.2},
}


def setup_paths(data_str):
    """Imposta i global di ad2 per la data corrente."""
    ad2.DATA = data_str
    ad2.PATH_XRAY = os.path.join(ad2.CARTELLA_XRAY,
                                  f"GOES_XRAY_{data_str}.csv")
    ad2.PATH_FLARES = os.path.join(ad2.CARTELLA_FLARES,
                                    f"FLARES_{data_str}.json")


def trova_flare_per_ora(flares, ora_str, tol_sec=180):
    """Trova il flare con picco più vicino a ora_str (HH:MM).
    Confronta solo l'ora del giorno: i flares sono già filtrati per data
    da ad2.scarica_flares()."""
    h, m = map(int, ora_str.split(":"))
    target_sec_day = h * 3600 + m * 60
    best, best_diff = None, None
    for f in flares:
        pk = ad2.parse_flare_datetime(f.get("picco", ""))
        if pk is None:
            continue
        pk_sec_day = pk.hour * 3600 + pk.minute * 60 + pk.second
        diff = abs(pk_sec_day - target_sec_day)
        if best_diff is None or diff < best_diff:
            best_diff, best = diff, f
    if best_diff is not None and best_diff <= tol_sec:
        return best
    return None

def calcola_contaminazione(df, picco_dt, flares):
    """Calcola evento_contaminato esternamente, senza toccare ad2."""
    if picco_dt is None:
        return None
    ts = df["Timestamp"].values
    m_altri = ad2.maschera_flare_vicini(ts, picco_dt, flares)
    t_sec = (ts - np.datetime64(picco_dt.replace(tzinfo=None))) \
            / np.timedelta64(1, "s")
    m_ev = (t_sec >= -600) & (t_sec <= 300)
    return bool(np.any(m_ev & m_altri))


def main():
    righe = []
    log = ["=== AUDIT BLOCCO 1 — Ricalcolo metriche storiche v2.1 ==="]
    cache_giorno = {}

    for data_str, ora_str, classe_goes, nome in EVENTI:
        log.append("")
        log.append(f"--- {nome}  {data_str} {ora_str}  ({classe_goes}) ---")

        # Setup giorno (cachato)
        try:
            if data_str not in cache_giorno:
                setup_paths(data_str)
                csv_path = ad2.trova_csv(data_str)
                if csv_path is None:
                    log.append(f"  [SKIP] CSV non trovato per {data_str}")
                    cache_giorno[data_str] = None
                    continue

                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    df = ad2.leggi_csv_tracker(csv_path)
                    ts_xray, flux_xray = ad2.scarica_xray(data_str)
                    flares = ad2.scarica_flares(
                        data_str, ts_xray=ts_xray, flux_xray=flux_xray)

                cache_giorno[data_str] = {
                    "df": df, "ts_xray": ts_xray, "flux_xray": flux_xray,
                    "flares": flares,
                }
                log.append(f"  CSV: {csv_path}")
                log.append(f"  X-ray: {len(ts_xray)} campioni, "
                            f"flares: {len(flares)}")

            ctx = cache_giorno[data_str]
            if ctx is None:
                continue

            df = ctx["df"]
            flares = ctx["flares"]

            flare = trova_flare_per_ora(flares, ora_str)
            if flare is None:
                log.append(f"  [SKIP] Nessun flare GOES vicino a {ora_str}")
                continue
            log.append(f"  Flare: {flare.get('classe')} @ {flare.get('picco')}")

            # Chiamata silenziosa ad analizza_flare
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    result = ad2.analizza_flare(
                        df, flare, ctx["ts_xray"], ctx["flux_xray"],
                        flares=flares,
                    )
                except Exception as e:
                    log.append(f"  [ERR] analizza_flare: {e}")
                    continue

            if result is None:
                log.append("  [SKIP] analizza_flare = None "
                            "(probabile Reality Check X-ray fallito)")
                continue

            picco_dt = ad2.parse_flare_datetime(flare.get("picco"))
            contaminato = calcola_contaminazione(df, picco_dt, flares)

            righe.append({
                "evento":            nome,
                "data":              data_str,
                "ora":               ora_str,
                "classe_goes":       classe_goes,
                "contaminato":       contaminato,
                "pot_picco_dB":      result.get("delta_pwr_picco"),
                "snr_pot_picco":     result.get("snr_pwr_picco"),
                "pot_media_dB":      result.get("delta_pwr_media"),
                "snr_pot_media":     result.get("snr_pwr_media"),
                "snr_doppler":       result.get("bump_snr_locale"),
                "bump_ampiezza_mhz": result.get("bump_ampiezza_mhz"),
                "std_quieta_mhz":    result.get("bump_std_quieta"),
                "forma_doppler_su":  result.get("bump_forma_su"),
                "delta_phi_deg":     result.get("delta_fase_int_deg"),
                "delta_phi_sigma":   result.get("snr_fase_int"),
                "forma_phi_su":      result.get("forma_su_int"),
                "std_quiete_phi_deg": result.get("std_quiete_gradi"),
                "frazione_invalida": result.get("frazione_invalida"),
                "n_reset":           result.get("n_reset"),
                "categoria_v21":     result.get("categoria"),
                "spiegazione":       result.get("spiegazione"),
                "versione_script":   "v2.1_audit_13nov",
            })

            log.append(f"  Categoria: {result.get('categoria')}")
            log.append(f"  Pot picco: {result.get('delta_pwr_picco'):+.2f} dB"
                        f"  (SNR {result.get('snr_pwr_picco'):+.1f})")
            log.append(f"  Doppler SNR: {result.get('bump_snr_locale'):.2f}"
                        f"  forma: {'su' if result.get('bump_forma_su') else 'giu'}")
            log.append(f"  Fase Int: Δφ/σ = {result.get('snr_fase_int'):.2f}"
                        f"  Δφ = {result.get('delta_fase_int_deg'):+.2f}°")
            log.append(f"  Frazione invalida: "
                        f"{result.get('frazione_invalida')*100:.1f}%")
            log.append(f"  Contaminato: {contaminato}")

        except Exception as e:
            import traceback
            log.append(f"  [ERR] {e}")
            log.append(traceback.format_exc())

    # --- Salva CSV ---
    df_out = pd.DataFrame(righe)
    out_csv = os.path.join(OUTPUT_DIR, "eventi_storici_omogenei.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"[OK] CSV scritto: {out_csv}")

    # --- Salva log ---
    out_log = os.path.join(OUTPUT_DIR, "eventi_storici_omogenei_log.txt")
    with open(out_log, "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    print(f"[OK] Log scritto: {out_log}")

    # --- Test di non-regressione ---
    verifica = ["", "=== TEST DI NON-REGRESSIONE (rif. contesto §22.2) ===", ""]
    verifica.append(f"{'Evento':<15} {'Atteso':>10} {'Ottenuto':>10} "
                     f"{'Tol':>6} {'Esito':>10}")
    verifica.append("-" * 62)
    passati, falliti = 0, 0
    for _, row in df_out.iterrows():
        nome = row["evento"]
        if nome not in RIFERIMENTI:
            continue
        atteso = RIFERIMENTI[nome]["pot_picco"]
        tol = RIFERIMENTI[nome]["tol"]
        ottenuto = row["pot_picco_dB"]
        if pd.isna(ottenuto):
            esito, falliti = "NaN", falliti + 1
        elif abs(ottenuto - atteso) <= tol:
            esito, passati = "OK", passati + 1
        else:
            esito, falliti = "FALLITO", falliti + 1
        verifica.append(f"{nome:<15} {atteso:>+10.2f} {ottenuto:>+10.2f} "
                         f"{tol:>6.2f} {esito:>10}")
    verifica.append("-" * 62)
    verifica.append(f"Passati: {passati}  Falliti: {falliti}")

    out_verifica = os.path.join(OUTPUT_DIR,
                                  "eventi_storici_omogenei_confronto.md")
    with open(out_verifica, "w", encoding="utf-8") as f:
        f.write("# Test di non-regressione — Audit Blocco 1\n\n")
        f.write("Riferimento: contesto_35.txt §22.2 (v2.1, 13 Nov 2026)\n\n")
        f.write("```\n" + "\n".join(verifica) + "\n```\n")
    print(f"[OK] Confronto scritto: {out_verifica}")

    print("\n".join(verifica))
    if falliti > 0:
        print()
        print("[!] ATTENZIONE: alcuni valori non riproducono quelli attesi.")
        print("    NON aggiustare lo script. Documenta le discrepanze e")
        print("    verifica la definizione di sigma_locale o di finestra evento.")


if __name__ == "__main__":
    main()