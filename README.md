# Advanced Portfolio Analytics (Decision Support System)

## 🎓 Akademischer Kontext
Dieses Software-Artefakt wurde im Rahmen einer Bachelorarbeit im Bereich Wirtschaftsinformatik entwickelt. Ziel war es, die naive 1/n-Heuristik beim Portfolio-Aufbau durch datengetriebene Präskription zu dekonstruieren und ein nutzerzentriertes Decision Support System gemäß dem Design Science Research (DSR) Framework zu implementieren.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://portfolio-analyser.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Ein intelligentes, webbasiertes Decision Support System zur quantitativen Risikoanalyse und Portfolio-Optimierung. Diese Applikation übersetzt komplexe finanzmathematische Modelle in intuitive, handlungsorientierte Dashboards für Privatanleger.

**👉 [Live-Demo der Applikation ansehen](https://portfolio-analyser.streamlit.app/)**

---

## Kernfunktionen (Features)

* **Zentrale Daten-Engine:** Live-Abruf von historischen Preis- und Fundamentaldaten via Yahoo Finance API inkl. dynamischer Währungsbereinigung (Live FX-Engine).
* **Unsupervised Learning (Clustering):** Hierarchische Clusteranalyse (Ward-Linkage) zur Identifikation verborgener Klumpenrisiken. Die optimale Cluster-Anzahl wird endogen via Silhouette-Score ermittelt.
* **Stochastische Risiko-Simulation:** Monte-Carlo-Simulation (1.000 Zukünfte) auf Basis empirischer Korrelationen (injiziert via Cholesky-Zerlegung) zur Berechnung des 95% Value at Risk (VaR).
* **Expertensystem & UI:** Automatische Übersetzung der komplexen Stochastik-Metriken in natürlichsprachliche Warnboxen und Handlungsempfehlungen.
* **Executive PDF-Report:** One-Click-Generierung eines professionellen, revisionssicheren PDF-Tearsheets mit allen relevanten KPIs, Diagrammen und dem automatisierten Anlage-Fazit.

---

## Tech-Stack & Architektur

* **Frontend / UI:** [Streamlit](https://streamlit.io/)
* **Data Processing:** Pandas, NumPy
* **Machine Learning / Mathematik:** SciPy, Scikit-Learn
* **Visualisierung:** Plotly Express, Matplotlib
* **Schnittstellen:** yfinance, requests
* **Reporting:** FPDF

---

## 🛠️ Lokale Installation (Setup)

Falls du das Projekt lokal auf deinem Rechner ausführen möchtest, folge diesen Schritten:

**1. Repository klonen:**
```bash
git clone https://github.com/DEIN_GITHUB_NAME/DEIN_REPO_NAME.git
cd DEIN_REPO_NAME
```

**2. Virtuelle Umgebung erstellen (empfohlen):**
```bash
python -m venv venv
source venv/bin/activate  # Auf Windows: venv\Scripts\activate
```

**3. Abhängigkeiten installieren:**
```bash
pip install -r requirements.txt
```

**4. Applikation starten:**
```bash
streamlit run app.py
```
*Die App öffnet sich nun automatisch in deinem Standard-Webbrowser unter `http://localhost:8501`.*

---

## 📸 Screenshots

<img width="1915" height="1011" alt="Screenshot 2026-05-04 154103" src="https://github.com/user-attachments/assets/fef32624-ff0d-46f5-9ef8-bdbbc939a8aa" />
<img width="845" height="436" alt="Cluster Diagnose nur Musterportfolio" src="https://github.com/user-attachments/assets/9759afb2-b01a-4023-a1cf-a53f35dbab30" />
<img width="781" height="663" alt="Screenshot 2026-05-06 140515" src="https://github.com/user-attachments/assets/ed9c64e4-a178-4f17-9fd6-dc42fdc7987a" />
<img width="1702" height="997" alt="Screenshot 2026-04-24 103705" src="https://github.com/user-attachments/assets/28618cc6-ec8e-481a-adbf-d7a6500074dd" />
<img width="669" height="944" alt="Screenshot 2026-05-07 104727" src="https://github.com/user-attachments/assets/b8f97a45-b572-4115-9dae-9894cfb4a0df" />

