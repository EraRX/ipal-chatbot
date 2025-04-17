import pandas as pd
import streamlit as st
import traceback
import io
import os
import base64

st.set_page_config(page_title="Helpdesk Zoekfunctie", layout="wide")

# 📷 Logo en titel in gekleurde balk
encoded_logo = base64.b64encode(open("logo.png", "rb").read()).decode()
st.markdown(
    f"""
    <div style='background-color: rgb(42, 68, 173); padding: 10px 20px; display: flex; align-items: center; border-radius: 0 0 10px 10px;'>
        <img src='data:image/png;base64,{encoded_logo}' style='height: 100px; margin-right: 25px;'>
        <h1 style='color: white; font-size: 2.5rem;'>🔍 Helpdesk Zoekfunctie</h1>
    </div>
    <style>
        html, body, .stApp {{
            background-color: #FFD3AC;
        }}
        .stSelectbox label, .stTextInput label, .stRadio label {{
            color: rgb(42, 68, 173) !important;
            font-weight: bold;
        }}
        .stDownloadButton button, .stButton button {{
            background-color: rgb(42, 68, 173);
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }}
        .stDownloadButton button:hover, .stButton button:hover {{
            background-color: rgb(30, 50, 130);
        }}
        div[data-baseweb="select"] > div {{
            border-radius: 10px !important;
            box-shadow: 1px 1px 4px rgba(0,0,0,0.1);
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# 🔐 Wachtwoord
if "wachtwoord_ok" not in st.session_state:
    wachtwoord = st.text_input("Wachtwoord:", type="password")
    if wachtwoord != "ipal2024":
        st.warning("Voer het juiste wachtwoord in om toegang te krijgen.")
        st.stop()
    else:
        st.session_state.wachtwoord_ok = True
        st.rerun()

# 📄 Excel inladen
try:
    df = pd.read_excel("faq.xlsx")
except Exception as e:
    st.error("Fout bij inlezen Excelbestand.")
    st.code(traceback.format_exc())
    st.stop()

# 🔍 Zoektekst samenstellen voor vrije zoekfunctie (inclusief antwoord)
df["zoektekst"] = (
    df["Systeem"].astype(str) + " " +
    df["Subthema"].astype(str) + " " +
    df["Categorie"].astype(str) + " " +
    df["Omschrijving melding"].astype(str) + " " +
    df["Toelichting melding"].astype(str) + " " +
    df["Soort melding"].astype(str) + " " +
    df["Antwoord of oplossing"].astype(str)
)

# 🔀 Keuze tussen Vrij of Gefilterd zoeken
st.markdown("---")
keuze = st.radio("🔎 Kies zoekmethode:", ["🎯 Gefilterde zoekopdracht", "🔍 Vrij zoeken"], horizontal=True)

# ✅ Gefilterde zoekopdracht
if keuze == "🎯 Gefilterde zoekopdracht":
    st.subheader("🎯 Gefilterde zoekopdracht")

    if "zoek_reset" not in st.session_state:
        st.session_state["zoek_reset"] = ""

    filter_df = df.copy()

    systeem = st.selectbox("Systeem", sorted(filter_df["Systeem"].dropna().unique()), key="systeem")
    filter_df = filter_df[filter_df["Systeem"] == systeem]
    st.session_state["zoek_reset"] = systeem

    subthema = st.selectbox("Subthema", sorted(filter_df["Subthema"].dropna().unique()), key="subthema")
    filter_df = filter_df[filter_df["Subthema"] == subthema]
    st.session_state["zoek_reset"] += subthema

    categorie = st.selectbox("Categorie", sorted(filter_df["Categorie"].dropna().unique()), key="categorie")
    filter_df = filter_df[filter_df["Categorie"] == categorie]
    st.session_state["zoek_reset"] += categorie

    omschrijving = st.selectbox("Omschrijving melding", sorted(filter_df["Omschrijving melding"].dropna().unique()), key="omschrijving")
    filter_df = filter_df[filter_df["Omschrijving melding"] == omschrijving]
    st.session_state["zoek_reset"] += omschrijving

    toelichting = st.selectbox("Toelichting melding", sorted(filter_df["Toelichting melding"].dropna().unique()), key="toelichting")
    filter_df = filter_df[filter_df["Toelichting melding"] == toelichting]
    st.session_state["zoek_reset"] += toelichting

    soort = st.selectbox("Soort melding", sorted(filter_df["Soort melding"].dropna().unique()), key="soort")
    filter_df = filter_df[filter_df["Soort melding"] == soort]
    st.session_state["zoek_reset"] += str(soort) if soort else ""

    st.subheader("📋 Resultaat op basis van filters")
    if filter_df.empty:
        st.warning("Geen resultaat gevonden op basis van de filters.")
    else:
        for _, rij in filter_df.iterrows():
            antwoord = rij["Antwoord of oplossing"]
            if pd.notna(antwoord) and antwoord.strip():
                st.markdown("**💬 Antwoord of oplossing:**")
                st.markdown(f"<div style='color: rgb(42, 68, 173); font-weight: bold;'>{antwoord}</div>", unsafe_allow_html=True)
                st.markdown("<br><hr>", unsafe_allow_html=True)
            else:
                st.info("ℹ️ Geen antwoord of oplossing beschikbaar voor deze melding.")
            st.markdown(f"<div style='color: rgb(42, 68, 173); font-weight: bold;'>🗂️ Subthema: {rij['Subthema']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color: rgb(42, 68, 173); font-weight: bold;'>📌 Categorie: {rij['Categorie']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color: rgb(42, 68, 173); font-weight: bold;'>📝 Omschrijving melding: {rij['Omschrijving melding']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color: rgb(42, 68, 173); font-weight: bold;'>ℹ️ Toelichting melding: {rij['Toelichting melding']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color: rgb(42, 68, 173); font-weight: bold;'>🏷️ Soort melding: {rij['Soort melding']}</div>", unsafe_allow_html=True)
            st.markdown("<br><hr>", unsafe_allow_html=True)
            st.markdown("---")

# 🔍 Vrij zoeken
if keuze == "🔍 Vrij zoeken":
    st.subheader("🔎 Vrij zoeken in alle velden (inclusief antwoord)")

    zoekterm = st.text_input("Zoek in Systeem, Subthema, Categorie, Omschrijving, Toelichting, Soort melding of Antwoord:")

    zoek_resultaten = pd.DataFrame()
    if zoekterm:
        zoek_resultaten = df[df["zoektekst"].str.contains(zoekterm, case=False, na=False)]
        st.subheader(f"📄 {len(zoek_resultaten)} resultaat/resultaten gevonden:")

        if zoek_resultaten.empty:
            st.warning("Geen resultaten gevonden.")
        else:
            for _, rij in zoek_resultaten.iterrows():
                st.markdown(f"**📁 Systeem:** {rij['Systeem']}")
                st.markdown(f"**🗂️ Subthema:** {rij['Subthema']}")
                st.markdown(f"**📌 Categorie:** {rij['Categorie']}")
                st.markdown(f"**📝 Omschrijving melding:** {rij['Omschrijving melding']}")
                st.markdown(f"**ℹ️ Toelichting melding:** {rij['Toelichting melding']}")
                st.markdown(f"**🏷️ Soort melding:** {rij['Soort melding']}")
                antwoord = rij["Antwoord of oplossing"]
                if pd.notna(antwoord) and antwoord.strip():
                    st.markdown("**💬 Antwoord of oplossing:**")
                    st.markdown(f"<div style='color: rgb(42, 68, 173); font-weight: bold;'>{antwoord}</div>", unsafe_allow_html=True)
                    st.markdown("<br><hr>", unsafe_allow_html=True)
                else:
                    st.info("ℹ️ Geen antwoord of oplossing beschikbaar voor deze melding.")
                st.markdown("---")

        # 📥 Downloadknop voor zoekresultaten
        buffer = io.BytesIO()
        zoek_resultaten.drop(columns=["zoektekst"], errors="ignore").to_excel(buffer, index=False)
        st.download_button(
            label="📥 Download resultaten als Excel",
            data=buffer.getvalue(),
            file_name="zoekresultaten.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
