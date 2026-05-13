# Advanced Portfolio Analytics (Decision Support System)

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

## Lokale Installation (Setup)

Falls du das Projekt lokal auf deinem Rechner ausführen möchtest, folge diesen Schritten:

1. **Repository klonen:**
   ```bash
   git clone [https://github.com/DEIN_GITHUB_NAME/DEIN_REPO_NAME.git](https://github.com/DEIN_GITHUB_NAME/DEIN_REPO_NAME.git)
   cd DEIN_REPO_NAME
