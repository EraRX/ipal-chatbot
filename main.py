# main.py — responsive gradient‑header + iPhone safe‑area support
import pandas as pd, streamlit as st, io, base64, traceback
import streamlit.components.v1 as components

st.set_page_config(page_title="Helpdesk Zoekfunctie", layout="wide")

# ───────────── Assets
ipal  = base64.b64encode(open("logo.png",              "rb").read()).decode()
doc   = base64.b64encode(open("logo-docbase-icon.png", "rb").read()).decode()
exact = base64.b64encode(open("Exact.png",             "rb").read()).decode()

# ───────────── Header & globale CSS via components
components.html(f"""
<meta name="viewport" content="width=device-width, initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{{--accent:#2A44AD;--accent2:#5A6AF4;--bg:#FFD3AC;--text:#0F274A;--card:#FFF}}
html,body,.stApp{{margin:0;padding:0;background:var(--bg);font-family:'Inter',sans-serif;color:var(--text)}}

/* HEADER */
.topbar{{
  position:relative;width:100vw;left:50%;transform:translateX(-50%);
  background:linear-gradient(135deg,var(--accent)0%,var(--accent2)100%);
  padding:calc(14px + env(safe-area-inset-top)) 24px 14px;
  display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:22px;
  border-radius:0 0 20px 20px;box-shadow:0 4px 12px rgba(0,0,0,.15)}
.topbar img{{height:58px}}
.topbar h1{{flex-basis:100%;text-align:center;color:#fff;margin:6px 0 0;font-size:2rem;font-weight:600}}
@media(min-width:768px){{.topbar h1{{flex-basis:auto;margin-left:18px}}}}

/* USER‑WIDGETS */
.stSelectbox>div{{border:none!important;border-radius:12px;padding:8px 12px;box-shadow:0 2px 8px rgba(0,0,0,.1);background:#fff}}
.stSelectbox label,.stTextInput label,.stRadio label{{font-weight:600;color:var(--accent)}}
.stDownloadButton button,.stButton button{{background:var(--accent);color:#fff;font-weight:600;border-radius:8px}}
.stDownloadButton button:hover,.stButton button:hover{{background:#1b2e8a}}

/* RESULT CARD */
.card{{background:var(--card);border-radius:16px;box-shadow:0 4px 14px rgba(0,0,0,.08);padding:22px;margin:20px 0}}
</style>
<div class="topbar">
  <img src="data:image/png;base64,{ipal}">
  <img src="data:image/png;base64,{doc}">
  <img src="data:image/png;base64,{exact}">
  <h1>🔍 Helpdesk&nbsp;Zoekfunctie</h1>
</div>
""", height=140, scrolling=False)

# ───────────── Login
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    pw = st.text_input("🔐 Wachtwoord", type="password")
    if pw == "ipal2024":
        st.session_state.auth = True
        st.success("Toegang verleend – start met zoeken ↓")
    elif pw:
        st.error("Onjuist wachtwoord.")
    st.stop()

# ───────────── Data
try:
    df = pd.read_excel("faq.xlsx")
except Exception:
    st.error("Excelbestand ontbreekt of is beschadigd.")
    st.code(traceback.format_exc())
    st.stop()

df["zoek"] = df.fillna("").astype(str).agg(" ".join, axis=1)

# ───────────── Zoek‑UI
st.subheader("Kies zoekmethode")
mode = st.radio("", ["🎯 Gefilterd", "🔍 Vrij zoeken"], horizontal=True, label_visibility="collapsed")

if mode == "🎯 Gefilterd":
    f = df.copy()
    for col,label in [("Systeem","Systeem"),("Subthema","Subthema"),
                      ("Categorie","Categorie"),("Omschrijving melding","Omschrijving"),
                      ("Toelichting melding","Toelichting")]:
        val = st.selectbox(label, sorted(f[col].dropna().unique()))
        f   = f[f[col]==val]
    soort = st.selectbox("Soort melding", sorted(f["Soort melding"].dropna().unique()))
    res   = f[f["Soort melding"]==soort]
else:
    term = st.text_input("🔍 Zoekterm in alle velden")
    res  = df[df["zoek"].str.contains(term, case=False, na=False)] if term else pd.DataFrame()

# ───────────── Kaart‑renderer
def kaart(r):
    html = f"""
<div class='card'>
  <b>💬 Antwoord</b><br><br>{r['Antwoord of oplossing'] or '–'}<hr>
  <b>Systeem:</b> {r['Systeem']}<br>
  <b>Subthema:</b> {r['Subthema']}<br>
  <b>Categorie:</b> {r['Categorie']}<br>
  <b>Omschrijving:</b> {r['Omschrijving melding']}<br>
  <b>Toelichting:</b> {r['Toelichting melding']}<br>
  <b>Soort:</b> {r['Soort melding']}
</div>"""
    h = 300 + (len(str(r["Antwoord of oplossing"]))//90)*18
    components.html(html, height=h, scrolling=False)

# ───────────── Resultaten
if res.empty:
    st.info("Geen resultaten gevonden.")
else:
    st.write(f"### {len(res)} resultaat/resultaten")
    res.apply(kaart, axis=1)

# ───────────── Download
if mode == "🔍 Vrij zoeken" and not res.empty:
    buf = io.BytesIO(); res.drop(columns=["zoek"]).to_excel(buf, index=False)
    st.download_button("📥 Download Excel", buf.getvalue(),
                       "zoekresultaten.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
