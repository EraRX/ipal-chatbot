# main.py — stabiele versie zónder experimental_rerun
import pandas as pd, streamlit as st, io, base64, traceback
import streamlit.components.v1 as components

st.set_page_config(page_title="Helpdesk Zoekfunctie", layout="wide")

# ───────────────────  Assets (logo’s) ───────────────────────
ipal  = base64.b64encode(open("logo.png",              "rb").read()).decode()
doc   = base64.b64encode(open("logo-docbase-icon.png", "rb").read()).decode()
exact = base64.b64encode(open("Exact.png",             "rb").read()).decode()

# ───────────────────  Header + CSS  ─────────────────────────
components.html(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{{--acc:#2A44AD;--acc2:#5A6AF4;--bg:#FFD3AC;--card:#FFF;--txt:#0F274A}}
body,.stApp{{margin:0;background:var(--bg);font-family:'Inter',sans-serif;color:var(--txt)}}
.top{{background:linear-gradient(135deg,var(--acc)0%,var(--acc2)100%);
      padding:14px 24px;border-radius:0 0 16px 16px;display:flex;flex-wrap:wrap;
      align-items:center;justify-content:center;gap:22px;box-shadow:0 4px 12px rgba(0,0,0,.12)}}
.top img{{height:60px}}
.top h1{{flex-basis:100%;text-align:center;color:#fff;font-size:2rem;font-weight:600;margin:6px 0 0}}
@media(min-width:768px){{.top h1{{flex-basis:auto;margin-left:18px}}}}
.card{{background:var(--card);border-radius:14px;box-shadow:0 4px 12px rgba(0,0,0,.08);padding:22px;margin:20px 0}}
</style>
<div class="top">
  <img src="data:image/png;base64,{ipal}">
  <img src="data:image/png;base64,{doc}">
  <img src="data:image/png;base64,{exact}">
  <h1>🔍 Helpdesk Zoekfunctie</h1>
</div>
""", height=120, scrolling=False)

# ───────────────────  Login  ────────────────────────────────
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    pw = st.text_input("🔐 Voer wachtwoord in", type="password")
    if pw == "ipal2024":
        st.session_state.auth = True
        st.success("Toegang verleend. Interface geladen ↓")
    elif pw:
        st.error("Onjuist wachtwoord.")
        st.stop()      # verkeerd wachtwoord → rest blokkeren
    else:
        st.stop()      # nog niets ingevuld → rest blokkeren

# ───────────────────  Data inladen  ─────────────────────────
try:
    df = pd.read_excel("faq.xlsx")
except Exception:
    st.error("Excelbestand ontbreekt of is beschadigd.")
    st.code(traceback.format_exc())
    st.stop()

df["zoek"] = df.fillna("").astype(str).agg(" ".join, axis=1)

# ───────────────────  Zoek‑interface  ───────────────────────
st.subheader("Kies zoekmethode")
mode = st.radio("", ["🎯 Gefilterd", "🔍 Vrij zoeken"], horizontal=True, label_visibility="collapsed")

if mode == "🎯 Gefilterd":
    f = df.copy()
    f = f[f["Systeem"]             == st.selectbox("Systeem",      sorted(f["Systeem"].dropna().unique()))]
    f = f[f["Subthema"]            == st.selectbox("Subthema",     sorted(f["Subthema"].dropna().unique()))]
    f = f[f["Categorie"]           == st.selectbox("Categorie",    sorted(f["Categorie"].dropna().unique()))]
    f = f[f["Omschrijving melding"]== st.selectbox("Omschrijving", sorted(f["Omschrijving melding"].dropna().unique()))]
    f = f[f["Toelichting melding"] == st.selectbox("Toelichting",  sorted(f["Toelichting melding"].dropna().unique()))]
    res = f[f["Soort melding"]     == st.selectbox("Soort melding",sorted(f["Soort melding"].dropna().unique()))]
else:
    term = st.text_input("🔍 Zoekterm in alle velden")
    res  = df[df["zoek"].str.contains(term, case=False, na=False)] if term else pd.DataFrame()

# ───────────────────  Resultaten  ───────────────────────────
def show_card(r):
    components.html(f"""
    <div class='card'>
      <b>💬 Antwoord</b><br><br>{r['Antwoord of oplossing'] or '–'}<hr>
      <b>Systeem:</b> {r['Systeem']}<br>
      <b>Subthema:</b> {r['Subthema']}<br>
      <b>Categorie:</b> {r['Categorie']}<br>
      <b>Omschrijving:</b> {r['Omschrijving melding']}<br>
      <b>Toelichting:</b> {r['Toelichting melding']}<br>
      <b>Soort:</b> {r['Soort melding']}
    </div>""", height=300, scrolling=False)

if res.empty:
    st.info("Geen resultaten gevonden.")
else:
    st.write(f"### {len(res)} resultaat/resultaten")
    res.apply(show_card, axis=1)

# ───────────────────  Download‑knop  ────────────────────────
if mode == "🔍 Vrij zoeken" and not res.empty:
    buf = io.BytesIO(); res.drop(columns=["zoek"]).to_excel(buf, index=False)
    st.download_button("📥 Download Excel", buf.getvalue(),
                       "resultaten.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
