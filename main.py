# main.py — eenvoudige, crash‑vrije versie
import pandas as pd
import streamlit as st
import io, base64, traceback

st.set_page_config(page_title="Helpdesk Zoekfunctie", layout="wide")

# ── Logo's inline laden
ipal = base64.b64encode(open("logo.png",              "rb").read()).decode()
doc  = base64.b64encode(open("logo-docbase-icon.png", "rb").read()).decode()
exac = base64.b64encode(open("Exact.png",             "rb").read()).decode()

# ── Header (alle CSS via components.html zodat Markdown niet parseert)
import streamlit.components.v1 as components
components.html(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
body,html,.stApp{{margin:0;font-family:'Inter',sans-serif;background:#FFD3AC}}
.bar{{display:flex;flex-wrap:wrap;justify-content:center;align-items:center;
      background:#2A44AD;padding:12px 20px;border-radius:0 0 16px 16px}}
.bar img{{height:60px;margin:0 14px}}
.bar h1{{color:#fff;font-size:2rem;font-weight:600;margin:6px 0 0;text-align:center;flex-basis:100%}}
@media(min-width:768px){{.bar h1{{flex-basis:auto}}}}
.card{{background:#fff;border-radius:14px;padding:18px;margin:18px 0;box-shadow:0 4px 10px rgba(0,0,0,.08)}}
</style>
<div class='bar'>
  <img src='data:image/png;base64,{ipal}'><img src='data:image/png;base64,{doc}'><img src='data:image/png;base64,{exac}'><h1>🔍 Helpdesk&nbsp;Zoekfunctie</h1>
</div>
""", height=120, scrolling=False)

# ── Login
if "ok" not in st.session_state:
    st.session_state.ok = False
if not st.session_state.ok:
    pw = st.text_input("🔐 Wachtwoord", type="password")
    if pw == "ipal2024":
        st.session_state.ok = True
        st.experimental_rerun()
    elif pw: st.error("Onjuist wachtwoord"); st.stop()

# ── Data
try:
    df = pd.read_excel("faq.xlsx")
except Exception:
    st.error("Fout bij inlezen Excel."); st.code(traceback.format_exc()); st.stop()
df["zoek"] = df.fillna("").astype(str).agg(" ".join, axis=1)

# ── Modus
mode = st.radio("Zoekmethode", ["🎯 Gefilterd", "🔍 Vrij"], horizontal=True)

if mode == "🎯 Gefilterd":
    f = df.copy()
    s   = st.selectbox("Systeem",      sorted(f["Systeem"].dropna().unique())); f = f[f["Systeem"]==s]
    sub = st.selectbox("Subthema",     sorted(f["Subthema"].dropna().unique())); f = f[f["Subthema"]==sub]
    cat = st.selectbox("Categorie",    sorted(f["Categorie"].dropna().unique())); f = f[f["Categorie"]==cat]
    oms = st.selectbox("Omschrijving", sorted(f["Omschrijving melding"].dropna().unique())); f = f[f["Omschrijving melding"]==oms]
    tol = st.selectbox("Toelichting",  sorted(f["Toelichting melding"].dropna().unique())); f = f[f["Toelichting melding"]==tol]
    typ = st.selectbox("Soort",        sorted(f["Soort melding"].dropna().unique())); res = f[f["Soort melding"]==typ]
else:
    term = st.text_input("Zoekterm in alle velden"); res = df[df["zoek"].str.contains(term, case=False, na=False)] if term else pd.DataFrame()

# ── Resultaten
if res.empty:
    st.info("Geen resultaten gevonden.")
else:
    st.write(f"### {len(res)} resultaat/resultaten")
    for _, r in res.iterrows():
        with st.container():
            st.markdown(f"<div class='card'>**💬 Antwoord**\n\n{r['Antwoord of oplossing'] or '–'}</div>", unsafe_allow_html=True)
            st.markdown(
f"""
• **Systeem:** {r['Systeem']}  
• **Subthema:** {r['Subthema']}  
• **Categorie:** {r['Categorie']}  
• **Omschrijving:** {r['Omschrijving melding']}  
• **Toelichting:** {r['Toelichting melding']}  
• **Soort:** {r['Soort melding']}
""")

# ── Download
if mode == "🔍 Vrij" and not res.empty:
    buf = io.BytesIO(); res.drop(columns=["zoek"]).to_excel(buf, index=False)
    st.download_button("Download Excel", buf.getvalue(), "resultaten.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
