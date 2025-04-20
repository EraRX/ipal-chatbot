# main.py  ────────────────────────────────────────────────────────────────────
import pandas as pd
import streamlit as st
import traceback, io, base64

# ── Pagina‑config
st.set_page_config(page_title="Helpdesk Zoekfunctie", layout="wide")

# ── Logo‑assets lezen
ipal_logo   = base64.b64encode(open("logo.png",               "rb").read()).decode()
doc_logo    = base64.b64encode(open("logo-docbase-icon.png",  "rb").read()).decode()
exact_logo  = base64.b64encode(open("Exact.png",              "rb").read()).decode()

# ── Globale CSS  (soft‑UI stijl)
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{{--accent:#2A44AD;--accent2:#5A6AF4;--bg:#FFD3AC;--card:#FFFFFF;--text:#0F274A;}}
body,.stApp{{background:var(--bg);font-family:"Inter",sans-serif;color:var(--text)}}

.header{{background:linear-gradient(135deg,var(--accent)0%,var(--accent2)100%);
        padding:12px 24px;border-radius:0 0 14px 14px;display:flex;flex-wrap:wrap;
        align-items:center;justify-content:center;gap:24px;box-shadow:0 4px 12px rgba(0,0,0,.15)}}
.header img{{height:60px}}
.header h1{{flex-basis:100%;margin:6px 0 0;text-align:center;color:#fff;font-size:2rem;font-weight:600}}
@media(min-width:760px){{.header h1{{flex-basis:auto;margin-left:18px}}}}

.stSelectbox>div{{border:none!important;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.1);
                   padding:8px 12px;background:#fff}}
.stSelectbox label,.stTextInput label,.stRadio label{{font-weight:600;color:var(--accent)}}
.stDownloadButton button,.stButton button{{background:var(--accent);color:#fff;border-radius:8px;font-weight:600}}
.stDownloadButton button:hover,.stButton button:hover{{background:#1B2E8A}}

.card{{background:var(--card);border-radius:16px;box-shadow:0 4px 14px rgba(0,0,0,.08);
       padding:22px;margin:22px 0}}
</style>
""", unsafe_allow_html=True)

# ── Header HTML
st.markdown(f"""
<div class='header'>
  <img src='data:image/png;base64,{ipal_logo}'>
  <img src='data:image/png;base64,{doc_logo}'>
  <img src='data:image/png;base64,{exact_logo}'>
  <h1>🔍 Helpdesk&nbsp;Zoekfunctie</h1>
</div>
""", unsafe_allow_html=True)

# ── Wachtwoord
if "ok" not in st.session_state:
    st.session_state.ok = False

if not st.session_state.ok:
    st.title("🔐 Helpdesk Toegang")
    pw = st.text_input("Voer wachtwoord in:", type="password")
    if pw == "ipal2024":
        st.session_state.ok = True
        st.success("Toegang verleend.")
    elif pw:
        st.error("Ongeldig wachtwoord.")
    if not st.session_state.ok:
        st.stop()

# ── Data inladen
try:
    df = pd.read_excel("faq.xlsx")
except Exception:
    st.error("Excelbestand niet gevonden of onleesbaar.")
    st.code(traceback.format_exc())
    st.stop()

# Zoekkolom
df["zoektekst"] = df.fillna("").astype(str).agg(" ".join, axis=1)

# ── Zoekmethode
st.markdown("---")
mode = st.radio("🔎 Kies zoekmethode:", ["🎯 Gefilterde zoekopdracht", "🔍 Vrij zoeken"], horizontal=True)

if mode == "🎯 Gefilterde zoekopdracht":
    st.subheader("🎯 Gefilterde zoekopdracht")
    f = df.copy()

    s  = st.selectbox("Systeem",     sorted(f["Systeem"].dropna().unique()))
    f  = f[f["Systeem"]==s]

    st1 = st.selectbox("Subthema",   sorted(f["Subthema"].dropna().unique()))
    f   = f[f["Subthema"]==st1]

    cat = st.selectbox("Categorie",  sorted(f["Categorie"].dropna().unique()))
    f   = f[f["Categorie"]==cat]

    oms = st.selectbox("Omschrijving melding", sorted(f["Omschrijving melding"].dropna().unique()))
    f   = f[f["Omschrijving melding"]==oms]

    tol = st.selectbox("Toelichting melding",  sorted(f["Toelichting melding"].dropna().unique()))
    f   = f[f["Toelichting melding"]==tol]

    typ = st.selectbox("Soort melding",        sorted(f["Soort melding"].dropna().unique()))
    res = f[f["Soort melding"]==typ]

else:
    st.subheader("🔍 Vrij zoeken in alle velden")
    term = st.text_input("Zoekterm:")
    res  = df[df["zoektekst"].str.contains(term, case=False, na=False)] if term else pd.DataFrame()

# ── Resultaten tonen
if res.empty:
    st.info("Geen resultaten gevonden.")
else:
    st.subheader(f"📄 {len(res)} resultaat/resultaten")
    for _, r in res.iterrows():
        st.markdown(f"""
<div class='card'>
  <strong>💬 Antwoord:</strong><br>
  <div style='color:var(--accent);font-weight:600;'>{r['Antwoord of oplossing'] or '–'}</div><hr>
  <ul style='list-style:none;padding:0;margin:0'>
    <li>📁 <b>Systeem:</b> {r['Systeem']}</li>
    <li>🗂️ <b>Subthema:</b> {r['Subthema']}</li>
    <li>📌 <b>Categorie:</b> {r['Categorie']}</li>
    <li>📝 <b>Omschrijving:</b> {r['Omschrijving melding']}</li>
    <li>ℹ️ <b>Toelichting:</b> {r['Toelichting melding']}</li>
    <li>🏷️ <b>Soort:</b> {r['Soort melding']}</li>
  </ul>
</div>
""", unsafe_allow_html=True)

# ── Downloadknop (alleen bij vrije zoek en resultaten)
if mode == "🔍 Vrij zoeken" and not res.empty:
    buff = io.BytesIO(); res.drop(columns=["zoektekst"], errors="ignore").to_excel(buff, index=False)
    st.download_button("📥 Download resultaten", data=buff.getvalue(),
                       file_name="zoekresultaten.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
# ─────────────────────────────────────────────────────────────
