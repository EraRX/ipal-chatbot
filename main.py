# main.py — afgeschermd, zonder Markdown‑crashes
import pandas as pd, streamlit as st, base64, io, traceback
import streamlit.components.v1 as components

st.set_page_config(page_title="Helpdesk Zoekfunctie", layout="wide")

# ───────────────────  LOGO'S  ───────────────────────────────
logo_ipal  = base64.b64encode(open("logo.png", "rb").read()).decode()
logo_doc   = base64.b64encode(open("logo-docbase-icon.png", "rb").read()).decode()
logo_exact = base64.b64encode(open("Exact.png", "rb").read()).decode()

# ───────────────────  HEADER + CSS  (via components) ────────
components.html(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{{--accent:#2A44AD;--accent2:#5A6AF4;--bg:#FFD3AC;--card:#FFF;--text:#0F274A}}
body,.stApp{{margin:0;background:var(--bg);font-family:'Inter',sans-serif;color:var(--text)}}
.top{{background:linear-gradient(135deg,var(--accent)0%,var(--accent2)100%);
      padding:14px 24px;border-radius:0 0 16px 16px;display:flex;flex-wrap:wrap;
      align-items:center;justify-content:center;gap:22px;box-shadow:0 4px 12px rgba(0,0,0,.12)}}
.top img{{height:60px}}
.top h1{{flex-basis:100%;text-align:center;color:#fff;margin:6px 0 0;font-size:2rem;font-weight:600}}
@media(min-width:768px){{.top h1{{flex-basis:auto;margin-left:18px}}}}
.card{{background:var(--card);border-radius:14px;box-shadow:0 4px 12px rgba(0,0,0,.08);padding:22px;margin:20px 0}}
select, input, .stApp button {{font-size:15px}}
</style>
<div class="top">
  <img src="data:image/png;base64,{logo_ipal}">
  <img src="data:image/png;base64,{logo_doc}">
  <img src="data:image/png;base64,{logo_exact}">
  <h1>🔍 Helpdesk Zoekfunctie</h1>
</div>
""", height=120, scrolling=False)

# ───────────────────  LOGIN BLOK  ───────────────────────────
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title("🔐  Voer wachtwoord in")
    pw = st.text_input("Wachtwoord", type="password")
    if pw == "ipal2024":
        st.session_state.auth = True
        st.success("Toegang verleend. De pagina wordt geladen …")
        st.stop()          # stopt dit script‑run; volgende run toont FAQ
    elif pw:
        st.error("Onjuist wachtwoord.")
    st.stop()              # wachtwoord vereist; verder niets tonen

# ───────────────────  DATA INLADEN  ─────────────────────────
try:
    df = pd.read_excel("faq.xlsx")
except Exception:
    st.error("Excelbestand ontbreekt of is beschadigd.")
    st.code(traceback.format_exc())
    st.stop()

df["zoek"] = df.fillna("").astype(str).agg(" ".join, axis=1)

# ───────────────────  ZOEK‑UI  ──────────────────────────────
st.subheader("Selecteer zoekmethode")
modus = st.radio("", ["🎯 Gefilterd", "🔍 Vrij zoeken"], horizontal=True, label_visibility="collapsed")

if modus == "🎯 Gefilterd":
    f = df.copy()
    f = f[f["Systeem"] == st.selectbox("Systeem", sorted(f["Systeem"].dropna().unique()))]
    f = f[f["Subthema"] == st.selectbox("Subthema", sorted(f["Subthema"].dropna().unique()))]
    f = f[f["Categorie"] == st.selectbox("Categorie", sorted(f["Categorie"].dropna().unique()))]
    f = f[f["Omschrijving melding"] == st.selectbox("Omschrijving", sorted(f["Omschrijving melding"].dropna().unique()))]
    f = f[f["Toelichting melding"] == st.selectbox("Toelichting", sorted(f["Toelichting melding"].dropna().unique()))]
    res = f[f["Soort melding"] == st.selectbox("Soort melding", sorted(f["Soort melding"].dropna().unique()))]
else:
    term = st.text_input("🔍  Zoekterm in alle velden")
    res = df[df["zoek"].str.contains(term, case=False, na=False)] if term else pd.DataFrame()

# ───────────────────  RESULTATEN  ───────────────────────────
if res.empty:
    st.info("Geen resultaten gevonden.")
else:
    st.write(f"### {len(res)} resultaat/resultaten")
    for _, r in res.iterrows():
        st.markdown(f"<div class='card'><b>💬 Antwoord</b><br><br>{r['Antwoord of oplossing'] or '–'}</div>",
                    unsafe_allow_html=True)
        st.text(f"Systeem: {r['Systeem']} | Subthema: {r['Subthema']}")
        st.text(f"Categorie: {r['Categorie']} | Soort: {r['Soort melding']}")
        st.text(f"Omschrijving: {r['Omschrijving melding']}")
        st.text(f"Toelichting: {r['Toelichting melding']}")
        st.markdown("---")

# ───────────────────  DOWNLOAD  ─────────────────────────────
if modus == "🔍 Vrij zoeken" and not res.empty:
    buf = io.BytesIO()
    res.drop(columns=["zoek"]).to_excel(buf, index=False)
    st.download_button("📥  Download Excel", buf.getvalue(),
                       "resultaten.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
