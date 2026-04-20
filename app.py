import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
import requests
import concurrent.futures


# 0. HILFSFUNKTIONEN (FX Engine & Search)
def search_ticker(query):
    """Sucht über die Yahoo Finance API nach dem Ticker anhand eines Firmennamens."""
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        if 'quotes' in data and len(data['quotes']) > 0:
            for quote in data['quotes']:
                if quote.get('quoteType') in ['EQUITY', 'ETF', 'CRYPTOCURRENCY']:
                    return quote['symbol'], quote.get('shortname', quote.get('longname', query))
            return data['quotes'][0]['symbol'], data['quotes'][0].get('shortname', query)
    except Exception:
        pass
    return None, None


@st.cache_data(ttl=3600)
def fetch_currencies(tickers):
    currency_map = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).fast_info
            currency_map[ticker] = info.get('currency', 'USD')
        except Exception:
            if ticker.endswith('.DE') or ticker.endswith('.AS') or ticker.endswith('.PA'):
                currency_map[ticker] = 'EUR'
            else:
                currency_map[ticker] = 'USD'
    return currency_map


@st.cache_data(ttl=3600)
def fetch_live_fx(currency):
    if currency == 'EUR': return 1.0
    try:
        fx_ticker = f"{currency}EUR=X"
        fx_data = yf.download(fx_ticker, period="5d")
        if 'Close' in fx_data:
            close_data = fx_data['Close']
        else:
            close_data = fx_data
        if isinstance(close_data, pd.DataFrame): return float(close_data.iloc[-1, 0])
        return float(close_data.dropna().iloc[-1])
    except:
        return 1.0


@st.cache_data(ttl=3600)
def fetch_historical_fx(currency, period="3y"):
    if currency == 'EUR': return None
    try:
        fx_ticker = f"{currency}EUR=X"
        fx_data = yf.download(fx_ticker, period=period)
        if 'Close' in fx_data: return fx_data['Close']
        return fx_data
    except:
        return None



# 1. SETUP & INTERAKTIVE EINGABEMASKE (Sidebar)
st.set_page_config(page_title="Advanced Portfolio Analytics", layout="wide", initial_sidebar_state="expanded")

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "URTH", "BTC-USD"],
        "Name": ["Apple Inc.", "Microsoft Corporation", "iShares MSCI World ETF", "Bitcoin USD"],
        "Stücke": [15.5, 10.0, 50.0, 0.15],
        "Buy_In_EUR": [129.03, 250.00, 60.00, 23333.33],
        "Kaufwert_EUR": [2000.0, 2500.0, 3000.0, 3500.0]
    })

st.sidebar.title("⚙️ Portfolio-Manager")
st.sidebar.markdown("Füge hier neue Positionen per Suchbegriff hinzu.")

with st.sidebar.form("add_position_form", clear_on_submit=True):
    search_term = st.text_input("Suchbegriff (z.B. Apple, Tesla, SAP)")
    shares = st.number_input("Stückzahl", min_value=0.00001, value=1.0, step=1.0, format="%.5f")
    buy_in = st.number_input("Buy-In (EUR pro Stück)", min_value=0.01, value=100.0, step=10.0, format="%.2f")

    submitted = st.form_submit_button("➕ Position hinzufügen")

    if submitted:
        if search_term.strip() == "":
            st.warning("Bitte gib einen Suchbegriff ein.")
        else:
            with st.spinner("Suche an der Börse..."):
                ticker, official_name = search_ticker(search_term)

                if ticker:
                    kaufwert_gesamt = float(shares) * float(buy_in)
                    new_row = pd.DataFrame({
                        "Ticker": [ticker],
                        "Name": [official_name],
                        "Stücke": [shares],
                        "Buy_In_EUR": [buy_in],
                        "Kaufwert_EUR": [kaufwert_gesamt]
                    })
                    st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
                    st.success(f"✅ Gefunden: **{official_name}** ({ticker}) wurde hinzugefügt!")
                else:
                    st.error(f"❌ '{search_term}' konnte an der Börse nicht gefunden werden.")

st.sidebar.divider()

st.sidebar.subheader("Aktuelle Positionen")
if not st.session_state.portfolio.empty:
    st.sidebar.markdown(
        "**Tipp:** Klicke in eine Zelle, um Zahlen zu ändern. Markiere eine Zeile links, um sie zu löschen (Papierkorb-Symbol).")


    # SPEICHERN & LADEN
    with st.sidebar.expander("💾 Speichern & Laden"):
        st.markdown("Sichere das aktuelle Portfolio lokal ab oder lade einen alten Stand.")

        # EXPORT (Nur wenn es etwas zu speichern gibt)
        if not st.session_state.portfolio.empty:
            csv_export = st.session_state.portfolio.to_csv(index=False, sep=";")
            st.download_button(
                label="📥 Aktuelles Portfolio speichern",
                data=csv_export,
                file_name="mein_portfolio.csv",
                mime="text/csv",
                use_container_width=True
            )

        # IMPORT
        uploaded_file = st.file_uploader("📂 Gespeichertes Portfolio laden", type=["csv"])
        if uploaded_file is not None:
            if st.button("📥 Import bestätigen", use_container_width=True):
                try:
                    imported_df = pd.read_csv(uploaded_file, sep=";")
                    required_cols = ["Ticker", "Name", "Stücke", "Buy_In_EUR", "Kaufwert_EUR"]

                    if all(col in imported_df.columns for col in required_cols):
                        st.session_state.portfolio = imported_df
                        st.rerun()
                    else:
                        st.error("Fehler: Die Datei hat nicht das richtige Format.")
                except Exception as e:
                    st.error(f"Fehler beim Lesen der Datei: {e}")

    # Wir übergeben das dynamische Portfolio an die alte Variable
    portfolio_data = st.session_state.portfolio.copy()

    edited_df = st.sidebar.data_editor(
        st.session_state.portfolio[["Ticker", "Name", "Stücke", "Buy_In_EUR"]],
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
            "Name": st.column_config.TextColumn("Name", disabled=True),
            "Stücke": st.column_config.NumberColumn("Stückzahl", format="%.5f", min_value=0.00001),
            "Buy_In_EUR": st.column_config.NumberColumn("Buy-In (€)", format="%.2f", min_value=0.01)
        },
        hide_index=False,
        num_rows="dynamic",
        use_container_width=True,
        key="portfolio_editor"
    )

    edited_df = edited_df.dropna(subset=["Ticker"])

    if not edited_df.empty:
        edited_df["Kaufwert_EUR"] = edited_df["Stücke"] * edited_df["Buy_In_EUR"]

    st.session_state.portfolio = edited_df.reset_index(drop=True)

    if st.sidebar.button("Portfolio leeren", use_container_width=True):
        st.session_state.portfolio = pd.DataFrame(columns=["Ticker", "Name", "Stücke", "Buy_In_EUR", "Kaufwert_EUR"])
        st.rerun()
else:
    st.sidebar.info("Dein Portfolio ist leer. Füge oben Aktien hinzu oder lade ein gespeichertes Portfolio!")


# 2. ZENTRALE DATEN-ENGINE
st.title("Advanced Portfolio Analytics")

eur_stock_data = pd.DataFrame()
bad_tickers = []
benchmark_ticker = "URTH"

if not portfolio_data.empty:
    with st.spinner("Zentrale Daten-Engine: Lade, prüfe und harmonisiere Marktdaten... ⏳"):
        tickers = portfolio_data["Ticker"].unique().tolist()
        all_tickers_to_fetch = list(set(tickers + [benchmark_ticker]))

        currency_map = fetch_currencies(all_tickers_to_fetch)

        raw_data = yf.download(all_tickers_to_fetch, period="3y")
        if 'Close' in raw_data:
            central_stock_data = raw_data['Close']
        else:
            central_stock_data = raw_data

        if isinstance(central_stock_data, pd.Series):
            central_stock_data = central_stock_data.to_frame(name=all_tickers_to_fetch[0])

        # Verhindert Pandas-Crashes bei doppelten Datumsangaben
        central_stock_data = central_stock_data[~central_stock_data.index.duplicated(keep='last')]

        fx_data_dict = {}
        for cur in set(currency_map.values()):
            if cur != 'EUR':
                fx_data_dict[cur] = fetch_historical_fx(cur, period="3y")

        eur_stock_data = central_stock_data.copy()

        for t in all_tickers_to_fetch:
            if t not in eur_stock_data.columns or eur_stock_data[t].dropna().empty:
                bad_tickers.append(t)
                continue

            asset_cur = currency_map.get(t, 'USD')
            if asset_cur != 'EUR' and asset_cur in fx_data_dict and fx_data_dict[asset_cur] is not None:
                fx_series = fx_data_dict[asset_cur]
                if isinstance(fx_series, pd.DataFrame):
                    fx_series = fx_series.iloc[:, 0]

                fx_series = fx_series[~fx_series.index.duplicated(keep='last')]
                eur_stock_data[t] = eur_stock_data[t] * fx_series.reindex(eur_stock_data.index, method='ffill')

        if bad_tickers:
            bad_tickers_in_portfolio = [t for t in bad_tickers if t != benchmark_ticker]
            if bad_tickers_in_portfolio:
                st.warning(
                    f"⚠️ **Data Health Check:** Folgende Ticker wurden ignoriert (z.B. Delisting): **{', '.join(bad_tickers_in_portfolio)}**")
                portfolio_data = portfolio_data[~portfolio_data["Ticker"].isin(bad_tickers)]
                tickers = portfolio_data["Ticker"].unique().tolist()

else:
    st.info("Bitte füge über die Seitenleiste Positionen hinzu, um die Analyse zu starten.")
    st.stop()

# --- Globale Variablen für alle Tabs initialisieren ---
returns = pd.DataFrame()
if not eur_stock_data.empty and len(tickers) >= 2:
    raw_returns = eur_stock_data[tickers].pct_change()
    returns = raw_returns.dropna(axis=1, how='all').dropna()

# 3. HAUPT-DASHBOARD TABS
tab1, tab2, tab3, tab4 = st.tabs(
    ["Mein Portfolio", "Diversifikation (Clustering)", "Risiko-Simulation (Monte Carlo)",
     "Sektoren & Fundamentaldaten"])

# TAB 1: MEIN PORTFOLIO
with tab1:
    st.header("Aktueller Portfolio-Status (Währungsbereinigt)")

    current_values = []
    current_prices_eur = []

    for ticker, shares in zip(portfolio_data["Ticker"], portfolio_data["Stücke"]):
        if ticker in eur_stock_data.columns:
            valid_prices = eur_stock_data[ticker].dropna()
            if not valid_prices.empty:
                price_in_eur = float(valid_prices.iloc[-1])
                current_prices_eur.append(price_in_eur)
                current_values.append(price_in_eur * float(shares))
            else:
                current_prices_eur.append(0)
                current_values.append(0)
        else:
            current_prices_eur.append(0)
            current_values.append(0)

    portfolio_data["Kurs_aktuell_EUR"] = current_prices_eur
    portfolio_data["Aktueller_Wert_EUR"] = current_values
    portfolio_data["Rendite_EUR"] = portfolio_data["Aktueller_Wert_EUR"] - portfolio_data["Kaufwert_EUR"]
    portfolio_data["Rendite_%"] = ((portfolio_data["Aktueller_Wert_EUR"] / portfolio_data["Kaufwert_EUR"]) - 1) * 100

    total_invested = portfolio_data["Kaufwert_EUR"].sum()
    total_current = portfolio_data["Aktueller_Wert_EUR"].sum()
    total_performance_eur = total_current - total_invested
    total_performance_pct = ((total_current / total_invested) - 1) * 100 if total_invested > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Investiertes Kapital", f"{total_invested:,.2f} €")
    col2.metric("Aktueller Wert (in EUR)", f"{total_current:,.2f} €",
                f"{total_performance_eur:+,.2f} € ({total_performance_pct:+.2f} %)")
    col3.metric("Anzahl Positionen", len(portfolio_data))

    st.divider()

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("Asset Allocation")
        fig_pie = px.pie(portfolio_data, values='Aktueller_Wert_EUR', names='Name', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        st.subheader("Performance pro Position")
        fig_bar = px.bar(
            portfolio_data, x='Name', y='Rendite_%',
            color='Rendite_%',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Positionsübersicht")
    display_df = portfolio_data[[
        "Name", "Stücke", "Buy_In_EUR", "Kurs_aktuell_EUR",
        "Aktueller_Wert_EUR", "Rendite_%", "Rendite_EUR"
    ]].copy()

    display_df.rename(columns={
        "Stücke": "Stückzahl",
        "Buy_In_EUR": "Buy In",
        "Kurs_aktuell_EUR": "Aktueller Kurs",
        "Aktueller_Wert_EUR": "Gesamtwert",
        "Rendite_%": "Performance (%)",
        "Rendite_EUR": "Rendite (€)"
    }, inplace=True)

    abs_max_perf = max(abs(display_df["Performance (%)"].min()), abs(display_df["Performance (%)"].max()), 5.0)
    abs_max_rendite = max(abs(display_df["Rendite (€)"].min()), abs(display_df["Rendite (€)"].max()), 100.0)

    styled_df = display_df.style.format({
        "Stückzahl": "{:,.4f}",
        "Buy In": "{:,.2f} €",
        "Aktueller Kurs": "{:,.2f} €",
        "Gesamtwert": "{:,.2f} €",
        "Performance (%)": "{:+.2f} %",
        "Rendite (€)": "{:+,.2f} €"
    }).background_gradient(
        subset=["Performance (%)"],
        cmap="RdYlGn",
        vmin=-abs_max_perf,
        vmax=abs_max_perf
    ).background_gradient(
        subset=["Rendite (€)"],
        cmap="RdYlGn",
        vmin=-abs_max_rendite,
        vmax=abs_max_rendite
    )

    st.dataframe(styled_df, use_container_width=True, hide_index=True)


    # BENCHMARK VERGLEICH (MSCI WORLD)

    st.divider()
    st.subheader("Historischer Benchmark-Vergleich (1 Jahr Rückblick)")
    st.markdown(
        "Wie hätte sich dein heutiges Portfolio im letzten Jahr im Vergleich zum Gesamtmarkt (**MSCI World**) geschlagen? *(Simulierter Constant-Mix Backtest in EUR)*")

    if total_current > 0 and benchmark_ticker not in bad_tickers and benchmark_ticker in eur_stock_data.columns:
        all_t = list(set(tickers + [benchmark_ticker]))
        hist_eur = eur_stock_data[all_t].ffill().tail(252).dropna()

        if not hist_eur.empty:
            daily_returns = hist_eur.pct_change().dropna()
            weights = portfolio_data["Aktueller_Wert_EUR"] / total_current
            weight_dict = dict(zip(portfolio_data["Ticker"], weights))

            port_daily_ret = pd.Series(0.0, index=daily_returns.index)
            for t in tickers:
                if t in daily_returns.columns:
                    port_daily_ret = port_daily_ret + (daily_returns[t] * weight_dict.get(t, 0))

            port_cum = (1 + port_daily_ret).cumprod() * 100
            bench_cum = (1 + daily_returns[benchmark_ticker]).cumprod() * 100

            port_final_pct = port_cum.iloc[-1] - 100
            bench_final_pct = bench_cum.iloc[-1] - 100

            col_b1, col_b2, col_b3 = st.columns(3)
            col_b1.metric("Dein Portfolio (1 Jahr)", f"{port_final_pct:+.2f} %")
            col_b2.metric("MSCI World (1 Jahr)", f"{bench_final_pct:+.2f} %")

            delta_val = port_final_pct - bench_final_pct
            col_b3.metric("Out-/Underperformance", f"{delta_val:+.2f} %", delta=f"{delta_val:+.2f} % vs Markt",
                          delta_color="normal")
            st.write("")

            chart_df = pd.DataFrame({"Dein Portfolio": port_cum, "MSCI World (Benchmark)": bench_cum})
            fig_bench = px.line(chart_df, labels={"value": "Wertentwicklung (Start = 100 €)", "Date": "Datum",
                                                  "variable": "Strategie"})
            fig_bench.update_traces(line=dict(width=2.5))
            st.plotly_chart(fig_bench, use_container_width=True)
        else:
            st.warning("Nicht genug historische Daten für den Benchmark-Vergleich vorhanden.")
    else:
        st.info("Füge dem Portfolio eine Aktie hinzu, um den Benchmark-Vergleich zu sehen.")


    # STRESSTEST: HISTORISCHE SZENARIO-ANALYSE
    st.divider()
    st.subheader("Stresstest: Historische Szenario-Analyse")
    st.markdown("Was würde passieren, wenn sich eine reale Marktkrise heute exakt so wiederholen würde?")

    # Szenario-Auswahl
    scenario = st.selectbox("Wähle ein historisches Stress-Szenario:", [
        "Corona-Crash (19. Feb 2020 - 23. Mär 2020)",
        "Ukraine-Krieg Ausbruch (16. Feb 2022 - 08. Mär 2022)",
        "Zinswende & Bärenmarkt (03. Jan 2022 - 12. Okt 2022)"
    ], key="stress_scenario_select")

    # Zeiträume definieren
    if "Corona" in scenario:
        start_d, end_d = "2020-02-19", "2020-03-24"
    elif "Ukraine" in scenario:
        start_d, end_d = "2022-02-16", "2022-03-09"
    else:
        start_d, end_d = "2022-01-03", "2022-10-13"

    # Berechnungs-Button
    if st.button("Stresstest durchführen", key="run_stress_test", use_container_width=True):
        # Snapshot des aktuellen Portfolios aus dem Session State
        current_portfolio = st.session_state.portfolio.copy()

        if current_portfolio.empty:
            st.warning("Dein Portfolio ist aktuell leer. Füge Positionen in der Sidebar hinzu.")
        else:
            with st.spinner(f"Analysiere historische Daten für {len(current_portfolio)} Positionen..."):
                try:
                    # --- DATEN-VALIDIERUNG & VORBEREITUNG ---

                    if "Aktueller_Wert_EUR" not in current_portfolio.columns:
                        # Fallback: Falls Haupt-Engine noch nicht lief, nehmen wir Buy-In als Basis
                        price_col = "Kurs_aktuell_EUR" if "Kurs_aktuell_EUR" in current_portfolio.columns else "Buy_In_EUR"
                        current_portfolio["Aktueller_Wert_EUR"] = current_portfolio["Stücke"] * current_portfolio[
                            price_col]

                    temp_total_current = current_portfolio["Aktueller_Wert_EUR"].sum()
                    safe_tickers = current_portfolio["Ticker"].tolist()

                    # --- HISTORISCHE DATEN LADEN ---
                    crisis_data = yf.download(safe_tickers, start=start_d, end=end_d, progress=False)
                    c_close = crisis_data['Close'] if 'Close' in crisis_data else crisis_data

                    # Handle Single-Ticker Portfolios (Series to DataFrame)
                    if isinstance(c_close, pd.Series):
                        c_close = c_close.to_frame(name=safe_tickers[0])

                    c_close = c_close.dropna(axis=1, how='all')

                    if c_close.empty:
                        st.error("Keine historischen Marktdaten für diesen Zeitraum gefunden.")
                    else:
                        # Berechnung der prozentualen Änderung (Letzter Tag vs. Erster Tag der Krise)
                        first_prices = c_close.iloc[0]
                        last_prices = c_close.iloc[-1]
                        drops = (last_prices - first_prices) / first_prices

                        simulated_total_impact_eur = 0
                        impact_list = []

                        for t in safe_tickers:
                            if t in drops.index and not pd.isna(drops[t]):
                                # Wert der Position im heutigen Depot
                                current_val = \
                                current_portfolio.loc[current_portfolio["Ticker"] == t, "Aktueller_Wert_EUR"].values[0]
                                asset_name = current_portfolio.loc[current_portfolio["Ticker"] == t, "Name"].values[0]

                                # Euro-Impact berechnen
                                asset_impact_eur = current_val * drops[t]
                                simulated_total_impact_eur += asset_impact_eur

                                impact_list.append({
                                    "Name": asset_name,
                                    "Impact_EUR": asset_impact_eur,
                                    "Performance": drops[t] * 100
                                })

                        if not impact_list:
                            st.info(
                                "Hinweis: Deine gewählten Assets existierten im Krisenzeitraum noch nicht an der Börse.")
                        else:
                            # --- ERGEBNIS-DARSTELLUNG ---
                            loss_pct = (
                            simulated_total_impact_eur / temp_total_current) * 100 if temp_total_current > 0 else 0

                            # Große Metrik
                            st.metric(
                                label=f"Simulierter Depot-Effekt ({scenario.split(' (')[0]})",
                                value=f"{simulated_total_impact_eur:+,.2f} €",
                                delta=f"{loss_pct:+,.2f} % des Gesamtdepots",
                                delta_color="normal"
                            )

                            # Dynamische Einordnung & Sortierung
                            if loss_pct <= -15:
                                st.error(f"**Massiver Einbruch:** Dein Depot verliert deutlich.")
                                analysis_title = "**Haupt-Verlustbringer:**"
                                # Sortieren nach schlimmstem Verlust
                                impact_df = pd.DataFrame(impact_list).sort_values(by="Impact_EUR", ascending=True)
                            elif loss_pct < 0:
                                st.warning(f"**Negatives Szenario:** Dein Depot korrigiert.")
                                analysis_title = "**Haupt-Verlustbringer (Größter Euro-Impact):**"

                                impact_df = pd.DataFrame(impact_list).sort_values(by="Impact_EUR", ascending=True)
                            else:
                                st.success(f"**Resilientes Szenario:** Dein Depot bleibt stabil und legt sogar zu!")
                                analysis_title = "**Performance-Treiber in dieser Phase:**"

                                impact_df = pd.DataFrame(impact_list).sort_values(by="Impact_EUR", ascending=False)

                            st.write("---")
                            st.markdown(analysis_title)

                            # Top 3 Karten anzeigen
                            cols = st.columns(min(3, len(impact_df)))
                            for i, col in enumerate(cols):
                                row = impact_df.iloc[i]
                                with col:
                                    st.caption(row["Name"])
                                    # Dynamische Farbe für den Euro-Betrag (Grün/Rot)
                                    text_color = "#2ecc71" if row["Impact_EUR"] >= 0 else "#e74c3c"
                                    st.markdown(
                                        f"<h3 style='color:{text_color}; margin:0;'>{row['Impact_EUR']:+,.2f} €</h3>",
                                        unsafe_allow_html=True)
                                    st.caption(f"{row['Performance']:+.2f} % Kursänderung")

                            # Detaillierte Tabelle für alle Positionen
                            with st.expander("Vollständige Analyse aller Positionen anzeigen"):
                                st.dataframe(
                                    impact_df.rename(columns={
                                        "Impact_EUR": "Impact auf Depot (€)",
                                        "Performance": "Kursbewegung Asset (%)"
                                    }).style.format({
                                        "Impact auf Depot (€)": "{:+,.2f} €",
                                        "Kursbewegung Asset (%)": "{:+.2f} %"
                                    }),
                                    use_container_width=True,
                                    hide_index=True
                                )

                except Exception as e:
                    st.error(f"Ein technischer Fehler ist aufgetreten: {e}")


# TAB 2: DIVERSIFIKATION (Clustering)
with tab2:
    st.header("Diversifikations- und Cluster-Analyse")
    st.markdown("Analyse der Verhaltensähnlichkeit der Assets (inklusive Währungseffekten).")

    if len(tickers) < 2:
        st.warning(
            "Für eine Cluster-Analyse (Diversifikation) müssen mindestens 2 verschiedene Assets im Portfolio sein.")
    else:
        if returns.empty or len(returns.columns) < 2:
            st.error("Nach der Datenbereinigung sind nicht genügend historische Daten übrig.")
        else:
            corr_matrix = returns.corr()
            col_heat, col_text = st.columns([2, 1])

            with col_heat:
                st.subheader("Korrelations-Heatmap")
                fig_corr = px.imshow(
                    corr_matrix, text_auto=".2f",
                    color_continuous_scale='RdYlGn_r', zmin=-1, zmax=1, aspect="auto"
                )
                st.plotly_chart(fig_corr, use_container_width=True)

            with col_text:
                st.info(
                    "**Wie lese ich diese Matrix?**\n\n"
                    "• **Dunkelrot (+1.00):** Die Assets bewegen sich exakt gleich. Fällt das eine, fällt das andere (Schlecht für die Diversifikation).\n\n"
                    "• **Gelb/Weiß (um 0.00):** Die Assets haben wenig miteinander zu tun.\n\n"
                    "• **Dunkelgrün (-1.00):** Die Assets bewegen sich entgegengesetzt. Ein perfekter Schutzkappen-Effekt (Hedge).\n\n"
                    "**(Hinweis: Währungseffekte sind in diesen Zahlen bereits realitätsnah eingepreist. Die Matrix beinhaltet daher reale Wechselkursrisiken (Uncovered Interest Rate Parity).**")

            st.divider()

            st.subheader("Machine Learning: Cluster-Dendrogramm")
            col_dendro, col_dendro_text = st.columns([2, 1])

            with col_dendro:
                dist_matrix = 1 - corr_matrix
                dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
                dist_matrix = np.clip(dist_matrix, 0, 2).fillna(0)
                dist_values = dist_matrix.to_numpy(copy=True)
                np.fill_diagonal(dist_values, 0)
                condensed_dist = ssd.squareform(dist_values)
                Z = sch.linkage(condensed_dist, method='ward')

                fig_dendro, ax = plt.subplots(figsize=(10, 5))
                sch.dendrogram(Z, labels=corr_matrix.columns, ax=ax, leaf_rotation=90)
                plt.ylabel("Distanz (Unähnlichkeit)")
                plt.tight_layout()
                st.pyplot(fig_dendro)

            with col_dendro_text:
                st.success(
                    "**Wie lese ich diesen Baum?**\n\n"
                    "Stell dir vor, dies ist ein Stammbaum des Risikos deines Portfolios:\n\n"
                    "• Je **weiter unten** sich zwei Linien verbinden, desto ähnlicher schwingen diese Aktien im Börsenalltag.\n\n"
                    "• Oben am Stamm teilt das KI-Modell dein Depot in die großen Haupt-Risikoblöcke auf. \n\n"
                    "**Dein Ziel:** Ein krisenfestes Portfolio sollte Blätter an vielen verschiedenen, weit voneinander entfernten Ästen besitzen!")

            st.divider()
            st.subheader("Portfolio-Diagnose & Handlungsempfehlungen")
            st.markdown(
                "Das Machine-Learning-Modell hat den Baum oben mathematisch zerschnitten und **Risiko-Gruppen (Klumpenrisiken)** identifiziert. Hier ist die detaillierte Diagnose deines Portfolios:")

            max_clusters = min(4, len(corr_matrix.columns))
            cluster_labels = sch.fcluster(Z, t=max_clusters, criterion='maxclust')

            name_dict = dict(zip(portfolio_data["Ticker"], portfolio_data["Name"]))
            value_dict = dict(zip(portfolio_data["Ticker"], portfolio_data["Aktueller_Wert_EUR"]))

            cluster_df = pd.DataFrame({
                "Ticker": corr_matrix.columns,
                "Name": [name_dict.get(t, t) for t in corr_matrix.columns],
                "Cluster_ID": cluster_labels
            })

            cluster_df["Wert_EUR"] = cluster_df["Ticker"].map(value_dict).fillna(0)
            cluster_cols = st.columns(max_clusters)

            for i in range(1, max_clusters + 1):
                assets_in_cluster = cluster_df[cluster_df["Cluster_ID"] == i]
                with cluster_cols[i - 1]:
                    if not assets_in_cluster.empty:
                        cluster_val = assets_in_cluster["Wert_EUR"].sum()
                        cluster_pct = (cluster_val / total_current) * 100 if total_current > 0 else 0

                        st.markdown(f"### Gruppe {i}")
                        st.metric("Gewichtung im Depot", f"{cluster_pct:.1f} %", f"{cluster_val:,.0f} €",
                                  delta_color="off")
                        asset_names = assets_in_cluster["Name"].tolist()
                        st.write(f"**Enthält:** {', '.join(asset_names)}")

                        if cluster_pct >= 50:
                            st.error(
                                "**⚠️ Kritisches Klumpenrisiko**\n\nDein Portfolio ist extrem abhängig von dieser Verhaltensgruppe. Wenn diese Assets durch einen Branchen-Schock fallen, reißt es einen Großteil deines Depots mit.\n\n**Empfehlung:** Erwäge ein *Rebalancing* (Verkauf von Anteilen) oder lenke künftige Sparraten gezielt in andere Gruppen.")
                        elif cluster_pct >= 25:
                            st.warning(
                                "**🟡 Erhöhte Gewichtung**\n\nDies ist eine starke Säule deines Depots, die sich sehr ähnlich verhält.\n\n**Empfehlung:** Beobachte diese Gruppe. Weitere Zukäufe sollten eher in kleinere, unkorrelierte Gruppen fließen, um die Balance zu wahren.")
                        else:
                            st.success(
                                "**🟢 Gut diversifiziert**\n\nDiese Gruppe dient als gesunde Beimischung und Ausgleich zu deinen Hauptpositionen.\n\n**Empfehlung:** Hervorragend zur Stabilisierung. Bei Markt-Rücksetzern könntest du hier antizyklisch nachkaufen.")
                    else:
                        st.markdown(f"### Gruppe {i}")
                        st.write("(Leer)")

# ------------------------------------------
# TAB 3: RISIKO-SIMULATION
# ------------------------------------------
with tab3:
    st.header("Monte-Carlo-Simulation & Value at Risk (VaR)")
    st.markdown("Simulation von 1.000 möglichen Zukünften basierend auf historischer Volatilität (Währungsbereinigt).")

    if total_current <= 0 or pd.isna(total_current):
        st.error("⚠️ Abbruch: Der Gesamtportfolio-Wert ist 0 € oder ungültig.")
    elif returns.empty:
        st.error("⚠️ Abbruch: Historische Renditen fehlen (siehe Tab 2).")
    else:
        try:
            portfolio_data["Gewicht"] = portfolio_data["Aktueller_Wert_EUR"] / total_current
            clean_returns = returns.copy()

            weight_dict = dict(zip(portfolio_data["Ticker"], portfolio_data["Gewicht"]))
            aligned_weights_list = [weight_dict.get(ticker, 0) for ticker in clean_returns.columns]
            aligned_weights = np.array(aligned_weights_list)

            if aligned_weights.sum() > 0:
                aligned_weights = aligned_weights / aligned_weights.sum()

            mean_returns = clean_returns.mean()
            cov_matrix = clean_returns.cov()

            col_params, col_results = st.columns([1, 2])

            with col_params:
                st.subheader("Simulations-Parameter")
                days = st.slider("Simulierter Zeitraum (Tage)", 30, 252, 252)
                iterations = st.selectbox("Anzahl der Simulationen", [100, 500, 1000, 5000], index=2)
                st.info(f"Startkapital: **{total_current:,.2f} €**")

            with st.spinner("Simuliere tausende Zukünfte... ⏳"):
                try:
                    L = np.linalg.cholesky(cov_matrix)
                except np.linalg.LinAlgError:
                    cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6
                    try:
                        L = np.linalg.cholesky(cov_matrix)
                    except np.linalg.LinAlgError:
                        st.error(
                            "Abbruch: Die Simulation konnte aufgrund perfekter oder fehlerhafter Korrelationen nicht berechnet werden.")
                        st.stop()

                portfolio_sims = np.zeros((days, iterations))

                # Konservativer Ansatz: mean_returns auf 0 setzen für reine Risiko-Simulation
                zero_drift = np.zeros_like(mean_returns.values)

                for i in range(iterations):
                    Z = np.random.normal(size=(days, len(aligned_weights)))
                    daily_returns = zero_drift + np.dot(Z, L.T)
                    portfolio_returns = np.dot(daily_returns, aligned_weights)
                    portfolio_sims[:, i] = total_current * np.cumprod(1 + portfolio_returns)

                final_values = portfolio_sims[-1, :]

                if np.isnan(final_values).any():
                    st.error("Mathematischer Fehler: Die Simulation hat ungültige Werte (NaN) generiert.")
                else:
                    var_95_value = np.percentile(final_values, 5)
                    var_amount = total_current - var_95_value
                    median_value = np.median(final_values)

                    var_percent = (var_amount / total_current) * 100

                    with col_results:
                        st.subheader("Ergebnis der Risikoanalyse")
                        st.metric("Value at Risk (95% Konfidenz)", f"- {var_amount:,.2f} €",
                                  help="Mit 95% Wahrscheinlichkeit verlierst du im gewählten Zeitraum NICHT mehr als diesen Betrag.")
                        st.metric("Erwarteter Depotwert (Median)", f"{median_value:,.2f} €",
                                  f"{(median_value / total_current - 1) * 100:+.2f} %")

                    st.divider()
                    st.subheader("Deine persönliche Risiko-Diagnose")

                    if var_percent < 10:
                        st.success(
                            f"**Risikoprofil: Defensiv / Konservativ (VaR: {var_percent:.1f} %)**\n\nDein Portfolio schwankt relativ wenig. Selbst in einem extrem schlechten Jahr (95% Konfidenz) verlierst du voraussichtlich weniger als 10 % deines Kapitals. **Bewertung:** Ideal für den Vermögenserhalt oder wenn du das Geld in naher Zukunft brauchst.")
                    elif var_percent < 25:
                        st.warning(
                            f"**Risikoprofil: Ausgewogen / Wachstumsorientiert (VaR: {var_percent:.1f} %)**\n\nDies ist ein typisches, gesundes Aktien-Portfolio. Du nimmst moderate Schwankungen in Kauf (bis zu 25 % Verlustrisiko im Stress-Szenario), um langfristig eine ordentliche Rendite zu erwirtschaften. **Bewertung:** Gut für den langfristigen Vermögensaufbau geeignet.")
                    else:
                        st.error(
                            f"**Risikoprofil: Hoch riskant / Aggressiv (VaR: {var_percent:.1f} %)**\n\nSchnall dich an! Dein Portfolio ist hochvolatil oder stark konzentriert (Klumpenrisiko). Du könntest in einem schlechten Jahr über ein Viertel deines Geldes verlieren. **Bewertung:** Das ist nur für sehr langfristige Anlagehorizonte (10+ Jahre), extrem starke Nerven oder als reines 'Spielgeld' geeignet.")

                    with st.expander("Wie kommen diese Zahlen zustande? (Blick unter die Haube)"):
                        st.markdown("""
                        Diese Analyse ist kein Bauchgefühl, sondern basiert auf der **Monte-Carlo-Simulation**, einem Standardverfahren bei Großbanken. So rechnet der Algorithmus:

                        1. **Der Rückspiegel:** Das System analysiert das tägliche Auf und Ab (Volatilität) deiner Aktien der letzten 3 Jahre.
                        2. **Gezinkte Würfel (Cholesky-Zerlegung):** Das System würfelt zufällige Marktbewegungen für die Zukunft. Aber: Die Würfel sind aneinander gekoppelt. Wenn in der Simulation die Apple-Aktie crasht, 'weiß' der Algorithmus durch historische Daten, dass Microsoft wahrscheinlich auch fällt. Das macht die Simulation extrem realistisch.
                        3. **1.000 Paralleluniversen:** Der Algorithmus rechnet dieses Würfelspiel für das nächste Jahr nicht nur einmal, sondern 1.000 Mal durch. Jede blaue Linie im linken Chart unten ist eine dieser simulierten Zukünfte.
                        4. **Der Value at Risk (VaR):** Wir nehmen alle 1.000 Ergebnisse und sortieren sie von extremem Verlust bis zu extremem Gewinn. Der *Value at Risk (95%)* ist einfach das Ergebnis, das genau an der 5%-Marke von unten steht. Das bedeutet: In 950 von 1.000 Zukünften hast du mehr Geld als diesen Betrag.
                        """)

                    st.divider()
                    st.subheader("Visualisierung der simulierten Zukünfte")
                    st.markdown(
                        "**Erklärung zum VaR:** Das Modell zeichnet hier die 1.000 simulierten Paralleluniversen deines Portfolios auf:")

                    col_plot1, col_plot2 = st.columns(2)
                    with col_plot1:
                        st.markdown("**1. Die Preispfade**")
                        st.caption(
                            "Jede blaue Linie repräsentiert eine mögliche Entwicklung deines Depots über die gewählten Tage. Je breiter der Trichter auseinandergeht, desto höher ist die Unsicherheit. Die schwarze gestrichelte Linie ist dein Startkapital.")

                        fig_sim, ax_sim = plt.subplots(figsize=(10, 6))
                        ax_sim.plot(portfolio_sims, color='#1f77b4', alpha=0.05)  # Modernes Blau
                        ax_sim.axhline(y=total_current, color='black', linestyle='--')
                        ax_sim.set_ylabel("Portfolio Wert (€)")
                        ax_sim.set_xlabel("Tage in der Zukunft")
                        st.pyplot(fig_sim)

                    with col_plot2:
                        st.markdown("**2. Die Wahrscheinlichkeitsverteilung**")
                        st.caption(
                            "Dieses Histogramm zeigt, wo die 1.000 Pfade am Ende gelandet sind. Der höchste Balken ist das wahrscheinlichste Szenario. Die rote gestrichelte Linie markiert deinen Value at Risk (die Grenze zu den schlimmsten 5 %).")

                        fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
                        ax_hist.hist(final_values, bins=50, color='#85C1E9', edgecolor='white')  # Corporate Light Blue
                        ax_hist.axvline(var_95_value, color='#E74C3C', linestyle='dashed', linewidth=2,
                                        label="VaR Grenze")
                        ax_hist.axvline(total_current, color='#2C3E50', linestyle='solid', linewidth=2,
                                        label="Startkapital")
                        ax_hist.set_ylabel("Häufigkeit")
                        ax_hist.set_xlabel("Portfolio Endwert (€)")
                        ax_hist.legend()
                        st.pyplot(fig_hist)


        except Exception as e:
            st.error(f"Fehler bei der Berechnung in Tab 3. Details: {e}")


# TAB 4: SEKTOREN & FUNDAMENTALDATEN
with tab4:
    st.header("Sektor-Allokation & Fundamentaldaten")
    st.markdown("Analyse der Branchenverteilung und fundamentalen Bewertungskennzahlen deines Portfolios.")

    if total_current <= 0:
        st.info("Bitte füge Positionen hinzu, um die Fundamentaldaten zu sehen.")
    else:
        with st.spinner("Lade Fundamentaldaten von Yahoo Finance... ⏳"):

            @st.cache_data(ttl=3600)
            def fetch_fundamentals(ticker_list):
                def get_info(t):
                    try:
                        ticker_obj = yf.Ticker(t)
                        info = ticker_obj.info

                        # Berechnung der Dividende
                        div_rate = info.get("dividendRate")
                        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")

                        if div_rate and price and price > 0:
                            div_pct = (div_rate / price) * 100
                        else:
                            raw_div = info.get("dividendYield") or 0.0
                            div_pct = raw_div if raw_div > 0.5 else raw_div * 100

                        return {
                            "Ticker": t,
                            "Sektor": info.get("sector", "Krypto / ETF / Sonstige"),
                            "Industrie": info.get("industry", "Krypto / ETF / Sonstige"),
                            "KGV (PE)": info.get("trailingPE", None),
                            "Div. Rendite (%)": div_pct
                        }
                    except Exception:
                        return {
                            "Ticker": t,
                            "Sektor": "Unbekannt",
                            "Industrie": "Unbekannt",
                            "KGV (PE)": None,
                            "Div. Rendite (%)": 0.0
                        }

                fund_data = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    results = list(executor.map(get_info, ticker_list))
                    fund_data = results

                return pd.DataFrame(fund_data)

            # --- WICHTIG: Daten-Harmonisierung vor dem Merge ---
            fund_df = fetch_fundamentals(tickers)
            
            # Ticker-Typen angleichen, um leere Merges zu verhindern
            portfolio_data["Ticker"] = portfolio_data["Ticker"].astype(str).str.strip()
            fund_df["Ticker"] = fund_df["Ticker"].astype(str).str.strip()

            merged_portfolio = pd.merge(portfolio_data, fund_df, on="Ticker", how="left")
            
            # Standardwerte für fehlende Daten setzen
            merged_portfolio["Sektor"] = merged_portfolio["Sektor"].fillna("Unbekannt")
            merged_portfolio["Div. Rendite (%)"] = merged_portfolio["Div. Rendite (%)"].fillna(0.0)

            st.divider()

            col_tree, col_kpi = st.columns([2, 1])

            with col_tree:
                st.subheader("Sektor-Gewichtung (Treemap)")
                fig_tree = px.treemap(
                    merged_portfolio,
                    path=[px.Constant("Mein Portfolio"), 'Sektor', 'Name'],
                    values='Aktueller_Wert_EUR',
                    color='Rendite_%',
                    color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=0
                )
                fig_tree.update_traces(root_color="lightgrey")
                fig_tree.update_layout(margin=dict(t=20, l=10, r=10, b=10))
                st.plotly_chart(fig_tree, use_container_width=True)

            with col_kpi:
                st.subheader("Portfolio-Charakteristik")
                st.info(
                    "**Treemap-Erklärung:**\nDie Größe der Kacheln spiegelt den monetären Wert der Position wider. Die Farbe zeigt die Performance.")

                # Berechnung der durchschnittlichen Dividendenrendite (Gewichtet)
                merged_portfolio['Gewicht'] = merged_portfolio['Aktueller_Wert_EUR'] / total_current
                avg_div = (merged_portfolio['Div. Rendite (%)'] * merged_portfolio['Gewicht']).sum()

                st.metric("Ø Dividendenrendite (Gewichtet)", f"{avg_div:.2f} %")

            st.divider()
            st.subheader("Fundamental-Übersicht")

            display_fund_df = merged_portfolio[
                ["Name", "Sektor", "Industrie", "KGV (PE)", "Div. Rendite (%)", "Aktueller_Wert_EUR"]].copy()

            # Robuste Formatierung mit Lambda-Funktionen
            styled_fund_df = display_fund_df.style.format({
                "KGV (PE)": lambda x: f"{x:.2f}" if (pd.notnull(x) and isinstance(x, (int, float))) else "N/A",
                "Div. Rendite (%)": lambda x: f"{x:.2f} %" if pd.notnull(x) else "0.00 %",
                "Aktueller_Wert_EUR": "{:,.2f} €"
            }, na_rep="N/A").background_gradient(subset=["Div. Rendite (%)"], cmap="Greens")

            st.dataframe(styled_fund_df, use_container_width=True, hide_index=True)


# GLOBARE SIDEBAR-AKTIONEN (Executive PDF Report)
st.sidebar.divider()
st.sidebar.subheader("📄 PDF Report Export")

if st.sidebar.button("Executive Report generieren", use_container_width=True):
    if total_current > 0:
        with st.spinner("Generiere hochauflösendes PDF-Tearsheet..."):
            try:
                from fpdf import FPDF
                import datetime

                # --- 1. DATEN-SYNC MIT DEM DASHBOARD ---

                pdf_var_amount = var_amount if 'var_amount' in locals() else 0.0
                pdf_var_percent = var_percent if 'var_percent' in locals() else 0.0

                if pdf_var_percent < 10 and pdf_var_percent > 0:
                    risk_profile = "Defensiv / Konservativ"
                elif pdf_var_percent < 25 and pdf_var_percent > 0:
                    risk_profile = "Ausgewogen / Wachstumsorientiert"
                elif pdf_var_percent >= 25:
                    risk_profile = "Aggressiv / Hoch riskant"
                else:
                    risk_profile = "Nicht berechenbar"

                # B) Stresstest (Corona) berechnen
                stress_loss = 0.0
                stress_pct = 0.0
                try:
                    c_data = yf.download(tickers, start="2020-02-19", end="2020-03-24")
                    c_close = c_data['Close'] if 'Close' in c_data else c_data
                    if isinstance(c_close, pd.Series): c_close = c_close.to_frame(name=tickers[0])
                    c_close = c_close.dropna(axis=1, how='all')
                    if not c_close.empty:
                        drops = (c_close.iloc[-1] - c_close.iloc[0]) / c_close.iloc[0]
                        for t in drops.index:
                            if not pd.isna(drops[t]):
                                val = portfolio_data[portfolio_data["Ticker"] == t]["Aktueller_Wert_EUR"].values[0]
                                stress_loss += val * drops[t]
                        stress_pct = (stress_loss / total_current) * 100
                except:
                    pass

                # C) Benchmark (MSCI World 1 Jahr) synchronisieren
                pdf_port_1y, pdf_bench_1y, pdf_outperf = 0.0, 0.0, 0.0
                try:
                    if benchmark_ticker in eur_stock_data.columns:
                        all_t = list(set(tickers + [benchmark_ticker]))
                        hist_bench = eur_stock_data[all_t].ffill().tail(252).dropna()
                        if not hist_bench.empty:
                            daily_bench_ret = hist_bench.pct_change().dropna()
                            b_weights = np.array(
                                [portfolio_data.loc[portfolio_data["Ticker"] == t, "Aktueller_Wert_EUR"].values[0] for t
                                 in tickers if t in daily_bench_ret.columns]) / total_current
                            p_daily = daily_bench_ret[[t for t in tickers if t in daily_bench_ret.columns]].dot(
                                b_weights)

                            p_cum = (1 + p_daily).cumprod() * 100
                            b_cum = (1 + daily_bench_ret[benchmark_ticker]).cumprod() * 100

                            pdf_port_1y = p_cum.iloc[-1] - 100
                            pdf_bench_1y = b_cum.iloc[-1] - 100
                            pdf_outperf = pdf_port_1y - pdf_bench_1y
                except:
                    pass


                # --- 2. PDF LAYOUT & DESIGN KLASSE ---
                class PDF(FPDF):
                    def header(self):
                        # Corporate Header (Dunkelblau)
                        self.set_fill_color(20, 35, 60)
                        self.rect(0, 0, 210, 38, 'F')
                        self.set_y(12)

                        # Titel
                        self.set_font('Arial', 'B', 24)
                        self.set_text_color(255, 255, 255)
                        self.cell(110, 10, 'PORTFOLIO TEARSHEET', 0, 0, 'L')

                        # Timestamp rechts oben
                        self.set_font('Arial', '', 10)
                        self.set_text_color(200, 200, 200)
                        current_date = datetime.datetime.now().strftime("%d. %b %Y")
                        self.cell(80, 10, f'Erstellt am: {current_date}', 0, 1, 'R')

                        # Untertitel
                        self.set_font('Arial', '', 11)
                        self.set_text_color(170, 185, 200)
                        self.cell(0, 5, 'Quantitative Analyse, Risk Assessment & Benchmark', 0, 1, 'L')
                        self.set_text_color(0, 0, 0)  # Reset color
                        self.ln(15)

                    def footer(self):
                        self.set_y(-15)
                        self.set_font('Arial', 'I', 8)
                        self.set_text_color(150, 150, 150)
                        self.set_draw_color(200, 200, 200)
                        self.line(10, 282, 200, 282)
                        self.cell(0, 10, f'Advanced Portfolio Analytics DSS | Vertraulich | Seite {self.page_no()}', 0,
                                  0, 'C')

                    def section_title(self, title):
                        self.set_font('Arial', 'B', 14)
                        self.set_text_color(20, 35, 60)
                        self.cell(0, 10, title, 0, 1, 'L')
                        # Schicke Unterstreichung
                        self.set_draw_color(20, 35, 60)
                        self.set_line_width(0.6)
                        self.line(self.get_x(), self.get_y(), self.get_x() + 190, self.get_y())
                        self.set_line_width(0.2)
                        self.ln(4)

                    def set_val_color(self, val):
                        if val > 0:
                            self.set_text_color(34, 139, 34)  # Forest Green
                        elif val < 0:
                            self.set_text_color(220, 20, 60)  # Crimson Red
                        else:
                            self.set_text_color(0, 0, 0)  # Black


                # --- 3. PDF GENERIERUNG ---
                pdf = PDF()
                pdf.add_page()

                # --- Sektion 1: Key Performance Indicators (Grid Layout) ---
                pdf.section_title('1. Core Metrics & Benchmark (1 Jahr Historie)')

                # Zeile 1: Absolute Werte
                pdf.set_font('Arial', 'B', 9)
                pdf.set_text_color(120, 120, 120)
                pdf.cell(63, 5, 'INVESTIERTES KAPITAL', 0, 0)
                pdf.cell(63, 5, 'AKTUELLER MARKTWERT', 0, 0)
                pdf.cell(63, 5, 'GESAMTRENDITE (ALL-TIME)', 0, 1)

                pdf.set_font('Arial', 'B', 15)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(63, 8, f'{total_invested:,.2f} EUR', 0, 0)
                pdf.cell(63, 8, f'{total_current:,.2f} EUR', 0, 0)

                pdf.set_val_color(total_performance_eur)
                pdf.cell(63, 8, f'{total_performance_eur:+,.2f} EUR ({total_performance_pct:+.2f}%)', 0, 1)
                pdf.ln(4)

                # Zeile 2: Benchmark (1 Jahr)
                pdf.set_font('Arial', 'B', 9)
                pdf.set_text_color(120, 120, 120)
                pdf.cell(63, 5, 'PORTFOLIO PERFORMANCE (1Y)', 0, 0)
                pdf.cell(63, 5, 'MSCI WORLD BENCHMARK (1Y)', 0, 0)
                pdf.cell(63, 5, 'OUTPERFORMANCE VS MARKT', 0, 1)

                pdf.set_font('Arial', 'B', 13)
                pdf.set_val_color(pdf_port_1y)
                pdf.cell(63, 8, f'{pdf_port_1y:+.2f} %', 0, 0)
                pdf.set_val_color(pdf_bench_1y)
                pdf.cell(63, 8, f'{pdf_bench_1y:+.2f} %', 0, 0)

                pdf.set_val_color(pdf_outperf)
                pdf.cell(63, 8, f'{pdf_outperf:+.2f} % Punkte', 0, 1)
                pdf.set_text_color(0, 0, 0)  # Reset color
                pdf.ln(8)

                # --- Sektion 2: Risk Assessment ---
                pdf.section_title('2. Risiko & Stress-Analyse')
                pdf.set_font('Arial', '', 10)

                # Professionelle Info-Boxen
                pdf.set_fill_color(245, 247, 250)
                pdf.set_draw_color(220, 225, 230)

                pdf.cell(90, 9, ' Analysiertes Risikoprofil:', 1, 0, 'L', 1)
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(100, 9, f' {risk_profile}', 1, 1, 'L', 1)

                pdf.set_font('Arial', '', 10)
                pdf.cell(90, 9, ' Monte-Carlo VaR (95% / Simulation):', 1, 0, 'L', 1)
                pdf.set_font('Arial', 'B', 10)
                pdf.set_val_color(-pdf_var_amount)  # Farbe für Verlust
                pdf.cell(100, 9, f' - {pdf_var_amount:,.2f} EUR ({pdf_var_percent:.2f} %)', 1, 1, 'L', 1)
                pdf.set_text_color(0, 0, 0)

                pdf.set_font('Arial', '', 10)
                pdf.cell(90, 9, ' Historischer Stresstest (Corona 2020):', 1, 0, 'L', 1)
                pdf.set_font('Arial', 'B', 10)
                pdf.set_val_color(stress_loss)
                pdf.cell(100, 9, f' {stress_loss:+,.2f} EUR ({stress_pct:+.2f} %)', 1, 1, 'L', 1)
                pdf.set_text_color(0, 0, 0)
                pdf.ln(10)

                # --- Sektion 3: Logisch geschlossene Asset Tabelle ---
                pdf.section_title('3. Asset Allokation & Rendite-Uebersicht')
                pdf.set_font('Arial', 'B', 9)

                # Neuer Tabellen-Header (Dunkelblau für Kontrast)
                pdf.set_fill_color(30, 45, 75)
                pdf.set_text_color(255, 255, 255)
                pdf.cell(60, 8, ' ASSET NAME', 0, 0, 'L', 1)
                pdf.cell(25, 8, ' GEWICHT', 0, 0, 'C', 1)
                pdf.cell(35, 8, ' INVESTIERT', 0, 0, 'C', 1)
                pdf.cell(35, 8, ' MARKTWERT', 0, 0, 'C', 1)
                pdf.cell(35, 8, ' RENDITE (EUR)', 0, 1, 'C', 1)

                pdf.set_text_color(0, 0, 0)
                pdf.set_font('Arial', '', 10)
                sorted_port = portfolio_data.sort_values(by="Aktueller_Wert_EUR", ascending=False)

                fill = False  # Für das Zebra-Muster
                for idx, row in sorted_port.iterrows():
                    pdf.set_fill_color(245, 245, 245)  # Sehr helles Grau für Zebra

                    name_clean = str(row["Name"])
                    name_clean = (name_clean[:28] + '..') if len(name_clean) > 28 else name_clean
                    weight = (row["Aktueller_Wert_EUR"] / total_current) * 100

                    # Die Spalten-Logik (Investiert -> Marktwert -> Gewinn)
                    pdf.cell(60, 8, f' {name_clean}', 0, 0, 'L', fill)
                    pdf.cell(25, 8, f'{weight:.1f} %', 0, 0, 'C', fill)
                    pdf.cell(35, 8, f'{row["Kaufwert_EUR"]:,.0f} EUR', 0, 0, 'C', fill)
                    pdf.cell(35, 8, f'{row["Aktueller_Wert_EUR"]:,.0f} EUR', 0, 0, 'C', fill)

                    pdf.set_font('Arial', 'B', 10)
                    pdf.set_val_color(row["Rendite_EUR"])
                    pdf.cell(35, 8, f'{row["Rendite_EUR"]:+,.0f} EUR', 0, 1, 'C', fill)

                    pdf.set_text_color(0, 0, 0)
                    pdf.set_font('Arial', '', 10)
                    fill = not fill  # Farbe für nächste Zeile umschalten

                pdf.ln(10)

                # --- Sektion 4: Executive Conclusion ---
                pdf.section_title('4. Executive Conclusion')
                pdf.set_font('Arial', '', 10)
                pdf.set_fill_color(240, 248, 255)  # Helles Blau für das Fazit
                pdf.set_draw_color(176, 196, 222)

                # KI generiertes dynamisches Fazit
                fazit = f"Das Portfolio weist ein Anlagevolumen von {total_invested:,.0f} EUR auf und wird aktuell mit {total_current:,.0f} EUR bewertet. "
                if pdf_outperf > 0:
                    fazit += f"Auf 1-Jahres-Sicht konnte der Referenzmarkt (MSCI World) um {pdf_outperf:.2f} Prozentpunkte geschlagen werden (Alpha-Generierung). "
                else:
                    fazit += f"Auf 1-Jahres-Sicht underperformt das Portfolio den Referenzmarkt (MSCI World) um {abs(pdf_outperf):.2f} Prozentpunkte. "

                fazit += f"Das ermittelte Risikoprofil ('{risk_profile}') spiegelt einen maximalen erwarteten Verlust von {abs(pdf_var_amount):,.0f} EUR im berechneten Konfidenzintervall wider. Im Falle eines extremen historischen Marktschocks (vergleichbar mit Maerz 2020) waere ein kurzfristiger Drawdown von ca. {abs(stress_pct):.1f} % zu erwarten."

                pdf.multi_cell(0, 6, fazit, 1, 'L', 1)
                pdf.ln(5)

                # Disclaimer
                pdf.set_font('Arial', 'I', 7)
                pdf.set_text_color(150, 150, 150)
                pdf.multi_cell(0, 3,
                               "Methodik: Die Kennzahlen basieren auf historischen Marktdaten (Yahoo Finance) und stochastischen Modellen (Monte-Carlo). Keine Anlageberatung. Waehrungsrisiken sind in den historischen EUR-Renditen eingepreist.")

                # Output
                pdf_out = pdf.output(dest='S')
                pdf_bytes = pdf_out.encode('latin1') if isinstance(pdf_out, str) else bytes(pdf_out)

                st.sidebar.success("✅ Executive Report (PDF) erstellt!")
                st.sidebar.download_button(
                    label="📥 Report jetzt herunterladen",
                    data=pdf_bytes,
                    file_name="Executive_Portfolio_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            except Exception as e:
                st.sidebar.error(f"Fehler beim Generieren: {e}")
    else:
        st.sidebar.info("Bitte füge dem Portfolio Aktien hinzu, um einen Report zu erstellen.")
