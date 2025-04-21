import streamlit as st
import pandas as pd
import base64
import io
import traceback

st.set_page_config(page_title="Helpdesk Zoekfunctie", layout="wide")

# 📷 Logo's laden
ipal_logo = base64.b64encode(open("logo.png", "rb").read()).decode()
docbase_logo = base64.b64encode(open("logo-docbase-icon.png", "rb").read()).decode()
exact_logo = base64.b64encode(open("Exact.png", "rb").read()).decode()

# 🎨 CSS + Header layout
st.markdown(f"""
    <style>
        html, body, .stApp {{
            background-color: #FFD3AC;
            font-family: "Segoe UI", sans-serif;
        }}
        .header-container {{
            background-color: #2A44AD;
            border-radius: 0 0 20px 20px;
            padding: 20px;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .header-container img {{
            margin: 0 20px;
        }}
        .header-title {{
            flex-basis: 100%;
            text-align: center;
            color: white;
            font-size: 2.2rem;
            margin-top: 10px;
        }}
        .stSelectbox > div {{
            border-radius: 10px !important;
            border: 1px solid #2A44AD;
            box-shadow: 1px 1px 6px rgba(0,0,0,0.15);
            padding: 4px;
        }}
        .stSelectbox label, .stTextInput label, .stRadio label {{
            color: #2A44AD !important;
            font-weight: bold;
        }}
        .stDownloadButton button, .stButton button {{
            background-color: #2A44AD;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }}
        .stDownloadButton button:hover, .stButton button:hover {{
            background-color: #1E3282;
        }}
    </style>
    <div class='header-container'>
        <img src='data:image/png;base64,{ipal_logo}' style='height: 60px;'>
        <img src='data:image/png;base64,{docbase_logo}' style='height: 50px;'>
        <img src='data:image/png;base64,{exact_logo}' style='height: 40px;'>
        <div class='header-title'>🔍 Helpdesk Zoekfunctie</div>
    </div>
""", unsafe_allow_html=True)

# 🔐 Wachtwoord
if "wachtwoord_ok" not in st.session_state:
    st.session_state.wachtwoord_ok = False

if not st.session_state.wachtwoord_ok:
    st.title("🔐 Helpdesk Toegang")
    wachtwoord = st.text_input("Voer wachtwoord in om verder te gaan:", type="password")
    if wachtwoord == "ipal2024":
        st.session_state.wachtwoord_ok = True
        st.experimental_rerun()
    elif wachtwoord:
        st.error("Ongeldig wachtwoord.")
    st.stop()

# 📄 Excel laden
try:
    df = pd.read_excel("faq.xlsx")
except Exception as e:
    st.error("Fout bij laden van het Excelbestand.")
    st.code(traceback.format_exc())
    st.stop()

# 🔍 Zoektekst kolom
kolommen = ["Systeem", "Subthema", "Categorie", "Omschrijving melding", "Toelichting melding", "Soort melding", "Antwoord of oplossing"]
df["zoektekst"] = df[kolommen].astype(str).agg(" ".join, axis=1)

# 🔘 Zoekmethode
st.markdown("---")
keuze = st.radio("🔎 Kies zoekmethode:", ["🎯 Gefilterde zoekopdracht", "🔍 Vrij zoeken"], horizontal=True)

if keuze == "🎯 Gefilterde zoekopdracht":
    st.subheader("🎯 Gefilterde zoekopdracht")
    filter_df = df.copy()

    for veld in ["Systeem", "Subthema", "Categorie", "Omschrijving melding", "Toelichting melding", "Soort melding"]:
        opties = sorted(filter_df[veld].dropna().unique())
        keuze = st.selectbox(veld, opties, key=veld)
        filter_df = filter_df[filter_df[veld] == keuze]

    st.subheader("📋 Resultaat op basis van filters")
    if filter_df.empty:
        st.warning("Geen resultaten gevonden.")
    else:
        for _, rij in filter_df.iterrows():
            st.markdown("**💬 Antwoord of oplossing:**")
            st.markdown(f"<div style='color: #2A44AD; font-weight: bold;'>{rij['Antwoord of oplossing']}</div>", unsafe_allow_html=True)
            st.markdown(f"🗂️ **Subthema:** {rij['Subthema']}")
            st.markdown(f"📌 **Categorie:** {rij['Categorie']}")
            st.markdown(f"📝 **Omschrijving melding:** {rij['Omschrijving melding']}")
            st.markdown(f"ℹ️ **Toelichting melding:** {rij['Toelichting melding']}")
            st.markdown(f"🏷️ **Soort melding:** {rij['Soort melding']}")
            st.markdown("<br><hr>", unsafe_allow_html=True)

elif keuze == "🔍 Vrij zoeken":
    st.subheader("🔍 Vrij zoeken in alle velden (inclusief antwoord)")
    zoekterm = st.text_input("Zoek in alle velden:")
    if zoekterm:
        result = df[df["zoektekst"].str.contains(zoekterm, case=False, na=False)]
        st.subheader(f"📄 {len(result)} resultaat/resultaten gevonden:")
        if result.empty:
            st.warning("Geen resultaten gevonden.")
        else:
            for _, rij in result.iterrows():
                st.markdown("**💬 Antwoord of oplossing:**")
                st.markdown(f"<div style='color: #2A44AD; font-weight: bold;'>{rij['Antwoord of oplossing']}</div>", unsafe_allow_html=True)
                st.markdown(f"📁 **Systeem:** {rij['Systeem']}")
                st.markdown(f"🗂️ **Subthema:** {rij['Subthema']}")
                st.markdown(f"📌 **Categorie:** {rij['Categorie']}")
                st.markdown(f"📝 **Omschrijving melding:** {rij['Omschrijving melding']}")
                st.markdown(f"ℹ️ **Toelichting melding:** {rij['Toelichting melding']}")
                st.markdown(f"🏷️ **Soort melding:** {rij['Soort melding']}")
                st.markdown("<br><hr>", unsafe_allow_html=True)

        buffer = io.BytesIO()
        result.drop(columns=["zoektekst"], errors="ignore").to_excel(buffer, index=False)
        st.download_button(
            label="📥 Download resultaten als Excel",
            data=buffer.getvalue(),
            file_name="zoekresultaten.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
