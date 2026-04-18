# VLF Monitoring Station - Solar Flares Detection 🛰️

Questo progetto monitora in tempo reale le **Sudden Ionospheric Disturbances (SID)** causate dai **Solar Flares** (brillamenti solari), utilizzando segnali radio in banda **VLF** (Very Low Frequency).

🔗 **Sito Live:** [gaetano-projects.github.io/sid/](https://github.io)

## 📋 Descrizione
La stazione riceve segnali da trasmettitori internazionali (attualmente la stazione NSY a 45.9 KHz) e analizza le variazioni di ampiezza del segnale per identificare l'impatto dei raggi X e UV sulla ionosfera terrestre.

### Hardware & Software utilizzato:
*   **SDR:** Utilizzo del software **SDRuno** per l'acquisizione dei segnali.
*   **Antenna:** Antenna Loop sintonizzata sui 45.9 KHz, dal diametro di 85 cm e 50 spire di filo smaltato.
*   **Analisi:** Script personalizzati per la generazione dei grafici e l'upload dei dati.

## 📊 Monitoraggio in Tempo Reale
I dati sono registrati in tempo reale. Il sistema è ottimizzato per rilevare:
*   Brillamenti solari di classe C, M e X.
*   Effetto "Diurnal Curve" (transizione alba/tramonto).
*   Anomalie elettromagnetiche ambientali.

## 🚀 Come Funziona
1.  Il segnale viene captato e processato tramite **SDRuno**.
2.  I dati estratti vengono elaborati per creare grafici di ampiezza/tempo.

---
*Progetto curato da Gaetano.*
