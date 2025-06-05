import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import os
import sys
import logging

# Logging konfigurieren (statt st.write)
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Dynamischer Pfad zu src/
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # Vortraegen/
SRC_DIR = os.path.join(BASE_DIR, "src")
logger.info(f"Arbeitsverzeichnis: {os.getcwd()}")
logger.info(f"BASE_DIR: {BASE_DIR}")
logger.info(f"SRC_DIR: {SRC_DIR}")
logger.info(f"sys.path: {sys.path}")

# Arbeitsverzeichnis auf BASE_DIR setzen
os.chdir(BASE_DIR)
logger.info(f"Neues Arbeitsverzeichnis: {os.getcwd()}")

# Sicherstellen, dass SRC_DIR existiert
if not os.path.exists(SRC_DIR):
    st.error(f"Das Verzeichnis 'src/' existiert nicht: {SRC_DIR}")
    st.stop()

# SRC_DIR nur hinzufügen, wenn nicht bereits in sys.path
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Überprüfe alle erforderlichen Dateien
required_files = {
    "co2_stahl_2024_2035_prognose.py": os.path.join(SRC_DIR, "co2_stahl_2024_2035_prognose.py"),
    "stahlproduktion_prognose.py": os.path.join(SRC_DIR, "stahlproduktion_prognose.py"),
    "stahl_ml_prognose.py": os.path.join(SRC_DIR, "stahl_ml_prognose.py"),
    "prognose_gruener_wasserstoff.py": os.path.join(SRC_DIR, "prognose_gruener_wasserstoff.py"),
    "prognose_recyclingquote.py": os.path.join(SRC_DIR, "prognose_recyclingquote.py"),
    "prognose_globaler_stahlschrott.py": os.path.join(SRC_DIR, "prognose_globaler_stahlschrott.py"),
    "prognose_politische_massnahmen.py": os.path.join(SRC_DIR, "prognose_politische_massnahmen.py"),
    "__init__.py": os.path.join(SRC_DIR, "__init__.py")
}

for file_name, file_path in required_files.items():
    if not os.path.exists(file_path):
        st.error(f"Die Datei '{file_name}' wurde nicht in 'src/' gefunden: {file_path}")
        st.stop()

# Imports
try:
    from co2_stahl_2024_2035_prognose import get_co2_prognose
    from stahlproduktion_prognose import get_stahlproduktion_prognose
    from stahl_ml_prognose import get_stahl_ml_prognose
    from prognose_gruener_wasserstoff import get_wasserstoff_prognose
    from prognose_recyclingquote import get_recycling_prognose
    from prognose_globaler_stahlschrott import get_stahlschrott_prognose
    from prognose_politische_massnahmen import get_massnahmen_prognose
except ModuleNotFoundError as e:
    st.error(f"Fehler beim Importieren der Skripte: {e}. Überprüfe:\n1. Existenz der Dateien in 'src/'.\n2. '__init__.py' in 'src/'.\n3. Korrekte Dateinamen (keine versteckten Zeichen).\n4. Abhängigkeiten in requirements.txt.")
    st.stop()
except ImportError as e:
    st.error(f"Importfehler: {e}. Überprüfe die Skripte in 'src/' auf Syntaxfehler oder fehlende Abhängigkeiten.")
    st.stop()

# CSS einbinden
STATIC_DIR = os.path.join(BASE_DIR, "static")
CSS_PATH = os.path.join(STATIC_DIR, "style.css")
if not os.path.exists(STATIC_DIR):
    st.error(f"Das Verzeichnis 'static/' existiert nicht: {STATIC_DIR}")
    st.stop()
if not os.path.exists(CSS_PATH):
    st.error(f"Die Datei 'style.css' wurde nicht gefunden: {CSS_PATH}")
    st.stop()
try:
    with open(CSS_PATH, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.error(f"Fehler beim Laden von 'style.css': {CSS_PATH}")
    st.stop()

# Titel
st.markdown("<h2 style='color:#adee45; text-align:center;'>Produktionsprognosen: Stahl & Scrap-Stahl <br>Mily Koyikkara<br> Alexander Hörber<br> Said Mohamed Ahmed</h2>", unsafe_allow_html=True)

# Sidebar für Interaktivität
st.sidebar.header("Einstellungen")
analysis_type = st.sidebar.selectbox("Analyse auswählen", [
    "CO2-Prognose Stahl",
    "Stahlproduktion weltweit",
    "ML-basierte Stahlprognose",
    "Prognose mit grünem Wasserstoff",
    "Recyclingquote nach Region",
    "Globaler Stahlschrott",
    "Politische Maßnahmen nach Region"
])

# Tab-Setup
tabs = st.tabs(["Historische Daten", "Prognose", "Scrap-Stahl"])

# CO2-Prognose
if analysis_type == "CO2-Prognose Stahl":
    try:
        result = get_co2_prognose()
        mae = result["mae"]
        fig = result["fig"]
        future_df = result["future_df"]
        historical_df = result["historical_df"]
        model = result["model"]
        poly = result["poly"]
    except Exception as e:
        st.error(f"Fehler beim Laden der CO2-Prognose: {e}")
        st.stop()

    # Slider für CO2-Prognose
    eisenschrott_quote = st.sidebar.slider("Eisenschrott-Quote (%)", 20, 50, (30, 45), key="eisenschrott")
    co2_preis = st.sidebar.slider("CO2-Preis (€/t)", 50, 200, (90, 150), key="co2_preis")

    # Dynamische Prognose basierend auf Slider
    future_updated = future_df.copy()
    future_updated["Eisenschrott_Quote"] = np.linspace(eisenschrott_quote[0], eisenschrott_quote[1], 12)
    future_updated["CO2_Preis"] = np.linspace(co2_preis[0], co2_preis[1], 12)
    # Konvertiere DataFrame in NumPy-Array, um Feature-Namen-Warnungen zu vermeiden
    X_future = future_updated[["Jahr", "Produktion_Mio_t", "Strompreis", "Eisenschrott_Quote", "CO2_Preis"]].to_numpy()
    future_updated["CO2_pro_t_Pred"] = model.predict(poly.transform(X_future))
    future_updated["Gesamt_CO2_Mio_t"] = future_updated["CO2_pro_t_Pred"] * future_updated["Produktion_Mio_t"] / 1000

    # Plot
    updated_fig = px.line(historical_df, x="Jahr", y="CO2_pro_t", title="CO2-Emissionen pro Tonne Rohstahl")
    updated_fig.add_scatter(x=future_updated["Jahr"], y=future_updated["CO2_pro_t_Pred"], mode="lines", name="Prognose", line=dict(dash="dash", color="#005B9A"))
    updated_fig.update_layout(
        yaxis_title="kg CO₂ pro t Rohstahl",
        showlegend=True,
        template="plotly_white",
        font=dict(family="Segoe UI", size=12),
        title_x=0.5,
        plot_bgcolor="#F5F6F5",
        paper_bgcolor="#F5F6F5"
    )

    with tabs[0]:
        st.subheader("Historische CO2-Daten")
        st.dataframe(historical_df)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader("CO2-Prognose 2024–2035")
        st.metric("MAE", f"{mae} kg CO₂/t")
        st.dataframe(future_updated)
        st.plotly_chart(updated_fig, use_container_width=True)
        st.download_button(
            label="Prognose herunterladen",
            data=future_updated.to_csv(index=False),
            file_name="co2_prognose_2024-2035.csv",
            mime="text/csv",
            key="download_co2"
        )

# Stahlproduktion weltweit
elif analysis_type == "Stahlproduktion weltweit":
    try:
        result = get_stahlproduktion_prognose()
        mae = result["mae"]
        mse = result["mse"]
        rmse = result["rmse"]
        fig = result["fig"]
        future_df = result["future_df"]
        historical_df = result["historical_df"]
    except Exception as e:
        st.error(f"Fehler beim Laden der Stahlproduktionsprognose: {e}")
        st.stop()

    with tabs[0]:
        st.subheader("Historische Produktionsdaten")
        st.dataframe(historical_df)
        st.pyplot(fig)

    with tabs[1]:
        st.subheader("Produktionsprognose 2024–2035")
        st.metric("MAE", f"{mae:.2f} Mio. Tonnen")
        st.metric("MSE", f"{mse:.2f} Mio. Tonnen²")
        st.metric("RMSE", f"{rmse:.2f} Mio. Tonnen")
        st.dataframe(future_df)
        st.pyplot(fig)
        st.download_button(
            label="Prognose herunterladen",
            data=future_df.to_csv(index=False),
            file_name="stahlproduktion_prognose_2024-2035.csv",
            mime="text/csv",
            key="download_stahlproduktion"
        )

# ML-basierte Stahlprognose
elif analysis_type == "ML-basierte Stahlprognose":
    try:
        result = get_stahl_ml_prognose()
        mse = result["mse"]
        rmse = result["rmse"]
        fig = result["fig"]
        future_df = result["future_df"]
        historical_df = result["historical_df"]
    except Exception as e:
        st.error(f"Fehler beim Laden der ML-basierten Stahlprognose: {e}")
        st.stop()

    with tabs[0]:
        st.subheader("Historische Daten")
        st.dataframe(historical_df)
        st.pyplot(fig)

    with tabs[1]:
        st.subheader("ML-basierte Prognose 2024–2035")
        st.metric("MSE", f"{mse:.2f} Mio. Tonnen²")
        st.metric("RMSE", f"{rmse:.2f} Mio. Tonnen")
        st.dataframe(future_df)
        st.pyplot(fig)
        st.download_button(
            label="Prognose herunterladen",
            data=future_df.to_csv(index=False),
            file_name="stahl_ml_prognose_2024-2035.csv",
            mime="text/csv",
            key="download_stahl_ml"
        )

# Prognose mit grünem Wasserstoff
elif analysis_type == "Prognose mit grünem Wasserstoff":
    try:
        result = get_wasserstoff_prognose()
        fig = result["fig"]
        prognose_df = result["prognose_df"]
        historical_df = result["historical_df"]
    except Exception as e:
        st.error(f"Fehler beim Laden der Prognose mit grünem Wasserstoff: {e}")
        st.stop()

    with tabs[0]:
        st.subheader("Historische Daten (Maßnahmen)")
        st.dataframe(historical_df)
        st.pyplot(fig)

    with tabs[1]:
        st.subheader("Prognose mit grünem Wasserstoff 2024–2035")
        st.dataframe(prognose_df)
        st.pyplot(fig)
        st.download_button(
            label="Prognose herunterladen",
            data=prognose_df.to_csv(index=False),
            file_name="prognose_gruener_wasserstoff_2024-2035.csv",
            mime="text/csv",
            key="download_wasserstoff"
        )

# Recyclingquote nach Region
elif analysis_type == "Recyclingquote nach Region":
    try:
        result = get_recycling_prognose()
        fig = result["fig"]
        prognose_df = result["prognose_df"]
        historical_df = result["historical_df"]
    except Exception as e:
        st.error(f"Fehler beim Laden der Recyclingquote-Prognose: {e}")
        st.stop()

    with tabs[0]:
        st.subheader("Historische Recyclingquoten")
        st.dataframe(historical_df)
        st.pyplot(fig)

    with tabs[1]:
        st.subheader("Prognose der Recyclingquote 2024–2035")
        st.dataframe(prognose_df)
        st.pyplot(fig)
        st.download_button(
            label="Prognose herunterladen",
            data=prognose_df.to_csv(index=False),
            file_name="prognose_recyclingquote_2024-2035.csv",
            mime="text/csv",
            key="download_recyclingquote"
        )

# Globaler Stahlschrott
elif analysis_type == "Globaler Stahlschrott":
    try:
        result = get_stahlschrott_prognose()
        fig = result["fig"]
        prognose_df = result["prognose_df"]
        historical_df = result["historical_df"]
    except Exception as e:
        st.error(f"Fehler beim Laden der Stahlschrott-Prognose: {e}")
        st.stop()

    # Schieberegler für Interaktivität
    nachfrage_wachstum = st.sidebar.slider("Wachstumsrate der Nachfrage (%)", -20, 20, (0, 10), key="nachfrage_wachstum")
    nachfrageueberhang_reduktion = st.sidebar.slider("Reduktion des Nachfrageüberhangs (%)", 0, 50, (0, 20), key="nachfrageueberhang_reduktion")

    # Dynamische Anpassung der Prognose basierend auf Slider
    prognose_updated = prognose_df.copy()
    nachfrage_wachstum_faktor = np.linspace(1 + nachfrage_wachstum[0]/100, 1 + nachfrage_wachstum[1]/100, len(prognose_updated))
    prognose_updated["Nachfrage"] = prognose_updated["Nachfrage"] * nachfrage_wachstum_faktor
    nachfrageueberhang_reduktion_faktor = np.linspace(1 - nachfrageueberhang_reduktion[0]/100, 1 - nachfrageueberhang_reduktion[1]/100, len(prognose_updated))
    prognose_updated["Nachfrageüberhang"] = prognose_updated["Nachfrageüberhang"] * nachfrageueberhang_reduktion_faktor

    # Aktualisierter Plot
    fig_updated, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(historical_df["Jahr"], historical_df["Nachfrage"], color="blue", label="Historische Nachfrage")
    ax.plot(prognose_updated["Jahr"], prognose_updated["Nachfrage"], color="red", linestyle="dashed", label="Prognose Nachfrage")
    ax.scatter(historical_df["Jahr"], historical_df["Nachfrageüberhang"], color="green", label="Historischer Nachfrageüberhang")
    ax.plot(prognose_updated["Jahr"], prognose_updated["Nachfrageüberhang"], color="orange", linestyle="dashed", label="Prognose Nachfrageüberhang")
    ax.set_xlabel("Jahr")
    ax.set_ylabel("Mio. t")
    ax.set_title("Prognose der globalen Stahlschrott-Nachfrage & Nachfrageüberhang")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    with tabs[0]:
        st.subheader("Historische Stahlschrott-Daten")
        st.dataframe(historical_df)
        st.pyplot(fig)

    with tabs[1]:
        st.subheader("Prognose der Stahlschrott-Nachfrage 2024–2035")
        st.dataframe(prognose_updated)
        st.pyplot(fig_updated)
        st.download_button(
            label="Prognose herunterladen",
            data=prognose_updated.to_csv(index=False),
            file_name="prognose_globaler_stahlschrott_2024-2035.csv",
            mime="text/csv",
            key="download_stahlschrott"
        )

# Politische Maßnahmen nach Region
elif analysis_type == "Politische Maßnahmen nach Region":
    try:
        result = get_massnahmen_prognose()
        fig = result["fig"]
        prognose_df = result["prognose_df"]
        historical_df = result["historical_df"]
    except Exception as e:
        st.error(f"Fehler beim Laden der Prognose für politische Maßnahmen: {e}")
        st.stop()

    with tabs[0]:
        st.subheader("Historische Daten (Politische Maßnahmen)")
        st.dataframe(historical_df)
        st.pyplot(fig)

    with tabs[1]:
        st.subheader("Prognose der politischen Maßnahmen 2024–2035")
        st.dataframe(prognose_df)
        st.pyplot(fig)
        st.download_button(
            label="Prognose herunterladen",
            data=prognose_df.to_csv(index=False),
            file_name="prognose_politische_massnahmen_2024-2035.csv",
            mime="text/csv",
            key="download_massnahmen"
        )

# Scrap-Stahl (Globaler Stahlschrott)
with tabs[2]:
    if analysis_type == "Globaler Stahlschrott":
        st.subheader("Globale Stahlschrott-Prognose")
        try:
            result = get_stahlschrott_prognose()
            fig = result["fig"]
            prognose_df = result["prognose_df"]
            historical_df = result["historical_df"]
        except Exception as e:
            st.error(f"Fehler beim Laden der Stahlschrott-Prognose: {e}")
            st.stop()
        
        st.dataframe(historical_df)
        st.pyplot(fig)
        st.dataframe(prognose_df)
        st.download_button(
            label="Prognose herunterladen",
            data=prognose_df.to_csv(index=False),
            file_name="prognose_globaler_stahlschrott_2024-2035.csv",
            mime="text/csv",
            key="download_stahlschrott_tab"
        )
    else:
        st.write("Bitte wählen Sie 'Globaler Stahlschrott' in der Sidebar, um die Prognose anzuzeigen.")