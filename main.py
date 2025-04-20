# main.py – volledige werkende versie
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import traceback, io, base64

# ────────────────────  Basis‑config
st.set_page_config(page_title="Helpdesk Zoekfunctie", layout="wide")

# ────────────────────  Assets laden
ipal_logo  = base64.b64encode(open("logo.png",              "rb").read()).decode()
doc_logo   = base64.b64encode(open("logo-docbase-icon.png", "rb").read()).decode()
exact_logo = base64.b64encode(open("Exact.png",             "rb").read()).decode()

# ────────────────────  Globale CSS  (soft‑UI)
st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{
  --accent:#2A44AD;--accent2:#5A6AF4;--bg:#FFD3AC;--card:#FFFFFF;--text:#0F274A;
}
body,.stApp{background:var(--bg);font-family:'Inter',sans-serif;color:var(--text)}

/* Header */
.topbar{
  background:linear-gradient(135deg,var(--accent)0%,var(--accent2)100%);
  padding:14px 24px;border-radius:0 0 16px 16px;
  display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:24px;
  box-shadow:0 4px 12px rgba(0,0,0,.15)
}
.topbar img{height:60px}
.topbar h1{
  flex-basis:100%;text-align:center;color:#fff;font-size:2rem;font-weight:600;margin:6px 0 0
}
@media(min-width:768px){
  .topbar h1{flex-basis:auto;margin-left:18px}
}

/* Widgets */
.stSelectbox>div{
  border:none!important;border-radius:12px;padding:8px 12px;
  box-shadow:0 2px 8px rgba(0,0,0,.1);background:#fff
}
.stSelectbox label,.stTextInput label,.stRadio label{
  font-weight:600;color:var(--accent)
}
.stDownloadButton button,.stButton button{
  background:var(--accent);color:#fff;font-weight:600;border-radius:8px
}
.stDownloadButton button:hover,.stButton button:hover{
  background:#1b2e8a
}

/* Result card */
.card{
  background:var(--card);border-radius:16px;
  box-shadow:0 4px 14px rgba(0,0,0,.08);padding:22px;margin:22px 0
}
</style>
""",
    unsafe_allow_html=True)

# ────────────────────  Header HTML
st.markdown(
    f"""
<div class='topbar'>
  <img src='data:image/png;base64,{ipal_logo}'>
  <img src='data:image/png;base64,{doc_logo}'>
  <img src='data:image/png;base64,{exact_logo}'>
  <h1>🔍 Helpdesk&nbsp;Zoekfunctie</h1>
</div>
""",
    unsafe_allow_html=True)

# ────────────────────  Login
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.title("🔐 Helpdesk Toegang")
    pw = st.text_input("Wachtwoord:", type="password")
    if pw == "ipal2024":
        st.session_state.auth_ok = True
        st.success("Toegang verleend.")
    elif pw:
        st.error("Onjuist wachtwoord.")
    if not st.session_state.auth_ok:
        st.stop()

# ────────────────────  Data inladen
try:
    df = pd.read_excel("faq.xlsx")
except Exception:
    st.error("Fout bij inlezen Excelbestand.")
    st.code(traceback.format_exc())
    st.stop()

# Extra kolom voor vrije zoek
df["zoektxt"] = df.fillna("").astype(str).agg(" ".join, axis=1)

# ────────────────────  UI
st.markdown("---")
mode = st.radio("🔎 Zoekmethode", ["🎯 Gefilterde zoekopdracht", "🔍 Vrij zoeken"], horizontal=True)

# Gefilterde zoekopdracht
if mode == "🎯 Gefilterde zoekopdracht":
    st.subheader("Gefilterde zoekopdracht")
    f = df.copy()

    s   = st.selectbox("Systeem",     sorted(f["Systeem"].dropna().unique()))
    f   = f[f["Systeem"] == s]

    sub = st.selectbox("Subthema",    sorted(f["Subthema"].dropna().unique()))
    f   = f[f["Subthema"] == sub]

    cat = st.selectbox("Categorie",   sorted(f["Categorie"].dropna().unique()))
    f   = f[f["Categorie"] == cat]

    oms = st.selectbox("Omschrijving", sorted(f["Omschrijving melding"].dropna().unique()))
    f   = f[f["Omschrijving melding"] == oms]

    tol = st.selectbox("Toelichting", sorted(f["Toelichting melding"].dropna().unique()))
    f   = f[f["Toelichting melding"] == tol]

    typ = st.selectbox("Soort melding", sorted(f["Soort melding"].dropna().unique()))
    res = f[f["Soort melding"] == typ]

# Vrij zoeken
else:
    st.subheader("Vrij zoeken in alle velden")
    term = st.text_input("Zoekterm")
    res  = df[df["zoektxt"].str.contains(term, case=False, na=False)] if term else pd.DataFrame()

# ────────────────────  Resultaten tonen
if res.empty:
    st.info("Geen resultaten gevonden.")
else:
    st.subheader(f"📄 {len(res)} resultaat/resultaten")
    for _, r in res.iterrows():
        card_html = f"""
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
</div>"""
        est_height = 250 + (len(str(r["Antwoord of oplossing"])) // 80) * 18
        components.html(card_html, height=est_height, scrolling=False)

# ────────────────────  Download
if mode == "🔍 Vrij zoeken" and not res.empty:
    buf = io.BytesIO()
    res.drop(columns=["zoektxt"], errors="ignore").to_excel(buf, index=False)
    st.download_button(
        "📥 Download resultaten (Excel)",
        data=buf.getvalue(),
        file_name="zoekresultaten.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
