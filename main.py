# main.py — volledige werkende versie (soft‑UI, geen Markdown‑crashes)
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import traceback, io, base64

# ────────────────────  Pagina‑config  ───────────────────────
st.set_page_config(page_title="Helpdesk Zoekfunctie", layout="wide")

# ────────────────────  Assets laden  ────────────────────────
ipal_logo  = base64.b64encode(open("logo.png",              "rb").read()).decode()
doc_logo   = base64.b64encode(open("logo-docbase-icon.png", "rb").read()).decode()
exact_logo = base64.b64encode(open("Exact.png",             "rb").read()).decode()

# ────────────────────  Globale CSS  ─────────────────────────
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <style>
    :root{
      --accent:#2A44AD;--accent2:#5A6AF4;
      --bg:#FFD3AC;--card:#FFFFFF;--text:#0F274A;
    }
    body,.stApp{background:var(--bg);font-family:'Inter',sans-serif;color:var(--text)}

    .topbar{background:linear-gradient(135deg,var(--accent)0%,var(--accent2)100%);
            padding:14px 24px;border-radius:0 0 16px 16px;
            display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:24px;
            box-shadow:0 4px 12px rgba(0,0,0,.15)}
    .topbar img{height:60px}
    .topbar h1{flex-basis:100%;text-align:center;color:#fff;font-size:2rem;font-weight:600;margin:6px 0 0}
    @media(min-width:768px){.topbar h1{flex-basis:auto;margin-left:18px}}

    .stSelectbox>div{border:none!important;border-radius:12px;padding:8px 12px;
                     box-shadow:0 2px 8px rgba(0,0,0,.1);background:#fff}
    .stSelectbox label,.stTextInput label,.stRadio label{font-weight:600;color:var(--accent)}
    .stDownloadButton button,.stButton button{background:var(--accent);color:#fff;font-weight:600;border-radius:8px}
    .stDownloadButton button:hover,.stButton button:hover{background:#1b2e8a}
    .card{background:var(--card);border-radius:16px;box-shadow:0 4px 14px rgba(0,0,0,.08);padding:22px;margin:22px 0}
    </style>
    """,
    unsafe_allow_html=True)

# ────────────────────  Header HTML  ─────────────────────────
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

# ────────────────────  Login  ───────────────────────────────
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.title("🔐 Helpdesk Toegang")
    pw = st.text_input("Wachtwoord:", type="password")
    if pw == "ipal2024":
        st.session_state.auth_ok = True
        st.success("Toegang verleend.")
        st.experimental_rerun()  # automatische doorgang naar zoekinterface
    elif pw:
        st.error("Onjuist wachtwoord.")
    st.stop()  # login vereist

# ────────────────────  Data inladen  ────────────────────────
try:
    df = pd.read_excel("faq.xlsx")
except Exception:
    st.error("Fout bij inlezen Excelbestand.")
    st.code(traceback.format_exc())
    st.stop()

# extra kolom voor vrije zoek
df["zoektxt"] = df.fillna("").astype(str).agg(" ".join, axis=1)

# ────────────────────  Zoekmodus  ───────────────────────────
st.markdown("---")
mode = st.radio("🔎 Zoekmethode", ["🎯 Gefilterde", "🔍 Vrij"], horizontal=True)

if mode == "🎯 Gefilterde":
    st.subheader("Gefilterde zoekopdracht")
    f = df.copy()
    s   = st.selectbox("Systeem",     sorted(f["Systeem"].dropna().unique()))
    f   = f[f["Systeem"]==s]
    sub = st.selectbox("Subthema",    sorted(f["Subthema"].dropna().unique()))
    f   = f[f["Subthema"]==sub]
    cat = st.selectbox("Categorie",   sorted(f["Categorie"].dropna().unique()))
    f   = f[f["Categorie"]==cat]
    oms = st.selectbox("Omschrijving", sorted(f["Omschrijving melding"].dropna().unique()))
    f   = f[f["Omschrijving melding"]==oms]
    tol = st.selectbox("Toelichting", sorted(f["Toelichting melding"].dropna().unique()))
    f   = f[f["Toelichting melding"]==tol]
    typ = st.selectbox("Soort", sorted(f["Soort melding"].dropna().unique()))
    res = f[f["Soort melding"]==typ]
else:
    st.subheader("Vrij zoeken in alle velden")
    term = st.text_input("Zoekterm")
    res  = df[df["zoektxt"].str.contains(term, case=False, na=False)] if term else pd.DataFrame()

# ────────────────────  Card‑renderer  ───────────────────────
def render_card(row):
    html = f"""
    <div class='card'>
      <strong>💬 Antwoord:</strong><br>
      <div style='color:var(--accent);font-weight:600;'>{row['Antwoord of oplossing'] or '–'}</div><hr>
      <ul style='list-style:none;padding:0;margin:0'>
        <li>📁 <b>Systeem:</b> {row['Systeem']}</li>
        <li>🗂️ <b>Subthema:</b> {row['Subthema']}</li>
        <li>📌 <b>Categorie:</b> {row['Categorie']}</li>
        <li>📝 <b>Omschrijving:</b> {row['Omschrijving melding']}</li>
        <li>ℹ️ <b>Toelichting:</b> {row['Toelichting melding']}</li>
        <li>🏷️ <b>Soort:</b> {row['Soort melding']}</li>
      </ul>
    </div>"""
    h = 270 + (len(str(row["Antwoord of oplossing"])) // 80) * 18
    components.html(html, height=h, scrolling=False)

# ────────────────────  Resultaten  ──────────────────────────
if res.empty:
    st.info("Geen resultaten gevonden.")
else:
    st.subheader(f"📄 {len(res)} resultaat/resultaten")
    res.apply(render_card, axis=1)

# ────────────────────  Download  ────────────────────────────
if mode == "🔍 Vrij" and not res.empty:
    buf = io.BytesIO()
    res.drop(columns=["zoektxt"], errors="ignore").to_excel(buf, index=False)
    st.download_button(
        "📥 Download resultaten (Excel)",
        data=buf.getvalue(),
        file_name="zoekresultaten.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
