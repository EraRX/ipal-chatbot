# IPAL Chatbox — main.py (fix: cascade 1→6 + geen geneste expander)
from __future__ import annotations
import os, re, io, json, hashlib, logging
from typing import Tuple, Optional, List
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ---------------- UI/config ----------------
st.set_page_config(page_title="IPAL Chatbox", page_icon="💬", layout="centered")
st.markdown("""
<style>
  .block-container { padding-top: 1rem; }
  html, body, [class*="css"] { font-size: 18px; }
  button[kind="primary"] { font-size: 18px !important; padding:.55em 1.0em; }
  .card { border:1px solid #e5e7eb;border-radius:16px;background:#fff;
          padding:1rem 1.1rem;margin:.5rem 0;box-shadow:0 3px 10px rgba(0,0,0,.04); }
  .badge { display:inline-block;background:#eef2ff;border:1px solid #e5e7eb;
           border-radius:999px;padding:.2rem .55rem;font-size:.78rem;margin-right:.4rem }
</style>
""", unsafe_allow_html=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------- Cascade/const ----------------
CASCADE = [
    "ID",
    "Systeem",                # 1
    "Subthema",               # 2
    "Categorie",              # 3
    "Omschrijving melding",   # 4
    "Toelichting melding",    # 5
    "Antwoord of oplossing",  # 6  (NIET opschonen)
    "Afbeelding",
]
EIGHT_CATS = [
    "Inloggen & Beveiliging",
    "Navigatie & Administraties",
    "Koppelingen & Synchronisatie",
    "Ledenbeheer & Toezeggingen",
    "Financiële inrichting",
    "Inkoop & Verkoop",
    "Bank & Betalingen & Incasso",
    "Rapportages & Jaarafsluiting",
]
CAT_RULES = [
    ("Inloggen & Beveiliging", r"inloggen|wachtwoord|2fa|tweestaps|beveiliging|rechten|autorisatie"),
    ("Navigatie & Administraties", r"hoofdmenu|navigeren|administraties|schakelen|navigatie"),
    ("Koppelingen & Synchronisatie", r"koppeling|synchronisatie|rel-?id|sila|docbase|integratie"),
    ("Ledenbeheer & Toezeggingen", r"ledenbeheer|[^a-z]lid[^a-z]|toezegging|kerkbijdrage"),
    ("Financiële inrichting", r"grootboek|dagboek|btw(?!-aang)|kostenplaats|kostendrager|begroting|inrichting|financieel"),
    ("Inkoop & Verkoop", r"inkoop|verkoop|creditnota|digitale brievenbus|scan ?& ?herken|inkooporder|factuur"),
    ("Bank & Betalingen & Incasso", r"\bbank(?!et)\b|bankmut|sepa|betaling|incasso|bic"),
    ("Rapportages & Jaarafsluiting", r"rapportage|rapporten|jaarafsluiting|jaaroverzicht|btw-?aang"),
]
SEARCH_COLS = ["Systeem","Subthema","Categorie","Omschrijving melding","Toelichting melding"]
AI_INFO = (
    "AI-Antwoord Info:\n"
    "1. Dit antwoord komt uit de IPAL chatbox. Controleer bij twijfel altijd de officiële bron.\n"
    "2. Hulp nodig met DocBase of Exact? Kijk eerst in de FAQ of maak een ticket in DocBase.\n"
    "- Veelgestelde vragen DocBase nieuw 2024\n"
    "- Veelgestelde vragen Exact Online"
)

# ---------------- Helpers ----------------
def clean_soft(s: str) -> str:
    if s is None: return ""
    s = str(s).replace("\u00A0"," ")
    repl = {
        "\u0091":"'", "\u0092":"'", "\u0093":'"', "\u0094":'"',
        "\u0096":"-", "\u0097":"-", "\u0085":"...",
        "\u2018":"'", "\u2019":"'", "\u201C":'"', "\u201D":'"',
        "\u2013":"-", "\u2014":"-", "\u2212":"-", "\u00AD":"", "\u2026":"...",
    }
    for k,v in repl.items(): s = s.replace(k,v)
    return re.sub(r"\s+"," ", s).strip()

def read_csv_smart(path: str) -> Tuple[pd.DataFrame, dict]:
    tries = [
        dict(encoding="utf-8", sep=";"),
        dict(encoding="utf-8-sig", sep=";"),
        dict(encoding="utf-8", sep=","),
        dict(encoding="utf-8-sig", sep=","),
        dict(encoding="cp1252", sep=";"),
        dict(encoding="latin1", sep=";"),
        dict(engine="python", encoding="utf-8", sep=None),
    ]
    last = None
    for kw in tries:
        try:
            return pd.read_csv(path, **kw), kw
        except Exception as e:
            last = e
    raise last

def eight_name(s: str) -> str:
    base = re.sub(r"^\s*\d+(?:\.\d+)*\s*", "", (s or "").strip())
    base = re.sub(r"\s*\(.*?\)\s*$", "", base)
    for canon in EIGHT_CATS:
        if canon.lower() in base.lower() or base.lower() in canon.lower():
            return canon
    return ""

# ---------------- CSV laden ----------------
@st.cache_data(show_spinner=False)
def load_faq(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"FAQ-bestand '{path}' niet gevonden.")
        return pd.DataFrame(columns=CASCADE).set_index(["Systeem","Subthema","Categorie"], drop=False)

    df, meta = read_csv_smart(path)
    df.columns = [str(c).strip().replace("\ufeff","") for c in df.columns]

    # kolommen afdwingen + volgorde
    for c in CASCADE:
        if c not in df.columns: df[c] = ""

    # Soort melding weg
    if "Soort melding" in df.columns:
        df.drop(columns=["Soort melding"], inplace=True)

    # opschonen (NIET op Antwoord)
    for c in ["ID","Systeem","Subthema","Categorie","Omschrijving melding","Toelichting melding","Afbeelding"]:
        df[c] = df[c].astype(str).map(clean_soft)

    # Systeem normaliseren
    def norm_sys(s: str) -> str:
        t = (s or "").strip().lower()
        if t.startswith("exact"): return "Exact"
        if t.startswith("docbase") or t == "docbase": return "DocBase"
        if t.startswith("sila"): return "SILA"
        if t.startswith("algemeen"): return "Algemeen"
        return s or ""
    df["Systeem"] = df["Systeem"].map(norm_sys)

    # Exact: mappen naar 8 vaste categorieën
    def map_exact_cat(row):
        if row["Systeem"] != "Exact":
            return row.get("Categorie","")
        cur = eight_name(row.get("Categorie",""))
        if cur: return cur
        sub = eight_name(row.get("Subthema",""))
        if sub: return sub
        blob = " ".join([row.get("Categorie",""), row.get("Subthema",""),
                         row.get("Omschrijving melding",""), row.get("Toelichting melding","")]).lower()
        for label, pat in CAT_RULES:
            if re.search(pat, blob): return label
        return "Navigatie & Administraties"
    df["Categorie"] = df.apply(map_exact_cat, axis=1)

    # Subthema fallback voor Exact
    df["Subthema"] = df.apply(lambda r: ("Exact Online" if r["Systeem"]=="Exact" and not r.get("Subthema","").strip() else r.get("Subthema","")), axis=1)

    # combined voor zoeken
    df["combined"] = df[SEARCH_COLS].fillna("").agg(" ".join, axis=1)

    # kolomvolgorde afdwingen
    ordered = [c for c in CASCADE if c in df.columns]
    extras = [c for c in df.columns if c not in ordered + ["combined"]]
    df = df[ordered + extras + ["combined"]]

    # index zetten, kolommen behouden
    return df.set_index(["Systeem","Subthema","Categorie"], drop=False)

CSV_PATH = os.getenv("FAQ_CSV", "/mnt/data/faq (1).csv") if os.path.exists("/mnt/data/faq (1).csv") else os.getenv("FAQ_CSV", "faq.csv")
faq_df = load_faq(CSV_PATH)

# ---------------- Zoeken/ranking ----------------
STOPWORDS_NL = {"de","het","een","en","of","maar","want","dus","als","dan","dat","die","dit","deze",
"ik","jij","hij","zij","wij","jullie","u","ze","je","mijn","jouw","zijn","haar","ons","hun",
"van","voor","naar","met","bij","op","in","aan","om","tot","uit","over","onder","boven","zonder",
"ook","nog","al","wel","niet","nooit","altijd","hier","daar","ergens","niets","iets","alles",
"is","was","wordt","zijn","heeft","heb","hebben","doe","doet","doen","kan","kunnen","moet","moeten"}

def _tok(s: str) -> list[str]:
    return [w for w in re.findall(r"[0-9A-Za-zÀ-ÿ_]+", (s or "").lower()) if len(w)>2 and w not in STOPWORDS_NL]

def token_score(q: str, t: str) -> int:
    return len(set(_tok(q)) & set(_tok(t)))

def rank_rows(df: pd.DataFrame, q: str) -> pd.DataFrame:
    ql = (q or "").strip().lower()
    if not ql: return df.assign(_score=0)
    out = df.copy()
    out["_score"] = out["combined"].apply(lambda t: token_score(ql, t))
    out = out.sort_values("_score", ascending=False)
    return out[out["_score"]>0]

# ---------------- Copy button (FIX: geen expander binnen expander) ----------
def copy_button(text: str, key_suffix: str):
    payload = text or ""
    js_text = json.dumps(payload)
    components.html(
        """
<div style="margin-top:8px;">
  <button id="COPY_BTN_ID" style="padding:6px 10px;font-size:16px;">Kopieer antwoord</button>
  <span id="COPY_STATE_ID" style="margin-left:8px;font-size:14px;"></span>
</div>
<script>
(function(){
  const btn = document.getElementById('COPY_BTN_ID');
  const state = document.getElementById('COPY_STATE_ID');
  if (btn) {
    btn.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(JS_TEXT);
            state.textContent='Gekopieerd!'; setTimeout(()=>state.textContent='',1500);
      } catch(e){ state.textContent='Niet gelukt — zet "Toon kopieertekst" aan.'; setTimeout(()=>state.textContent='',3000); }
    });
  }
})();
</script>
        """.replace("COPY_BTN_ID", f"copybtn-{key_suffix}")
           .replace("COPY_STATE_ID", f"copystate-{key_suffix}")
           .replace("JS_TEXT", js_text),
        height=60
    )
    # i.p.v. expander: checkbox voorkomt nested-expander fout
    if st.checkbox("Toon kopieertekst", key=f"show_copy_{key_suffix}"):
        st.text_area("Tekst", payload, height=150, key=f"copy_fallback_{key_suffix}")

# ---------------- PDF ----------------
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, ListFlowable, ListItem
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import cm
from reportlab.lib import colors

def _strip_md(s: str) -> str:
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"#+\s*([^\n]+)", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)
    return s

def make_pdf(question: str, answer: str) -> bytes:
    q = clean_soft(question or "")
    a = clean_soft(_strip_md(answer or ""))
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leading=16, spaceAfter=10, alignment=TA_LEFT)
    head = ParagraphStyle("Head", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=colors.HexColor("#333"))
    story = []
    if os.path.exists("logopdf.png"):
        try:
            banner = Image("logopdf.png"); banner._restrictSize(A4[0]-4*cm, 10000); banner.hAlign="LEFT"; story += [banner, Spacer(1,8)]
        except Exception: pass
    story += [Paragraph(f"Vraag: {q}", head), Spacer(1,8), Paragraph("Antwoord:", head)]
    for line in (a.split("\n") if a else []):
        if line.strip(): story.append(Paragraph(line.strip(), body))
    story += [Spacer(1,8), Paragraph(_strip_md(AI_INFO).replace("\n","<br/>"), body)]
    doc.build(story); buf.seek(0); return buf.getvalue()

# ---------------- UI helpers ----------------
def render_result(row: pd.Series):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<span class='badge'>{row['Systeem']} › {row['Categorie']}</span>", unsafe_allow_html=True)
    st.markdown(f"**{row['Omschrijving melding']}**")
    if str(row.get("Toelichting melding","")).strip():
        st.caption(row["Toelichting melding"])
    with st.expander("Antwoord tonen", expanded=False):
        # Belangrijk: het ruwe antwoord NIET opschonen
        ans = str(row.get("Antwoord of oplossing","") or "")
        st.markdown(ans)
        st.markdown("---")
        st.markdown("### In het kort")
        # eenvoudige samenvatting (niet-AI)
        short = []
        for line in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+"," ", ans.strip())):
            if any(k in line.lower() for k in ["klik", "open", "ga naar", "selecteer", "vul", "opslaan", "menu"]):
                short.append(line)
            if len(short) >= 5: break
        if not short:
            short = [l for l in re.split(r"(?<=[.!?])\s+", ans) if l.strip()][:3]
        for b in short:
            st.markdown(f"- {b}")
        # downloads & copy
        pdf = make_pdf(row.get("Omschrijving melding","Vraag"), ans + "\n\n" + AI_INFO)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📄 Download PDF", data=pdf, file_name="antwoord.pdf", mime="application/pdf")
        with col2:
            copy_button(ans + "\n\n" + AI_INFO, hashlib.md5(ans.encode("utf-8")).hexdigest()[:8])
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- App ----------------
def main():
    st.header("Welkom bij IPAL Chatbox")

    # BOVENSTE KNOPPEN — blijven altijd zichtbaar
    if "mode" not in st.session_state: st.session_state.mode = None
    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button("Exact", use_container_width=True):   st.session_state.mode = "Exact"
    if c2.button("DocBase", use_container_width=True): st.session_state.mode = "DocBase"
    if c3.button("Zoeken", use_container_width=True):  st.session_state.mode = "Zoeken"
    if c4.button("Internet", use_container_width=True):st.session_state.mode = "Internet"
    if c5.button("🔄 Reset", use_container_width=True):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.session_state.mode = None
        try: st.cache_data.clear()
        except Exception: pass
        st.rerun()

    mode = st.session_state.mode
    if not mode:
        st.info("Kies hierboven **Exact**, **DocBase**, **Zoeken** (volledige CSV) of **Internet** (algemene vraag).")
        return

    # INTERNET (placeholder zonder AI)
    if mode == "Internet":
        q = st.text_input("Waarover gaat uw vraag?", placeholder="Algemene vraag (geen CSV nodig)")
        if not q: return
        ans = "Kunt u uw vraag iets concreter maken (bijv. 'DocBase wachtwoord resetten' of 'Exact bankkoppeling')?"
        final = ans + "\n\n" + AI_INFO
        st.markdown(final)
        pdf = make_pdf(q, final)
        st.download_button("📄 Download PDF", data=pdf, file_name="antwoord.pdf", mime="application/pdf")
        copy_button(final, hashlib.md5(final.encode("utf-8")).hexdigest()[:8])
        return

    # ZOEKEN (hele CSV)
    if mode == "Zoeken":
        q = st.text_input("Zoek in volledige CSV…", placeholder="Bijv. 'bankkoppeling', 'SILA', 'Scan & Herken'")
        if not q: return
        results = rank_rows(faq_df, q)
        st.caption(f"Gevonden: {len(results)}")
        if results.empty:
            st.info("Geen resultaten — pas je zoekterm aan.")
            return
        page_size = st.slider("Per pagina", 10, 100, 50, 10)
        total = len(results); pages = max(1, int(np.ceil(total/page_size)))
        page = st.number_input("Pagina", 1, pages, 1)
        s, e = (page-1)*page_size, min(page*page_size, total)
        st.caption(f"Toont {s+1}–{e} van {total}")
        for _, row in results.iloc[s:e].iterrows():
            render_result(row)
        return

    # EXACT / DOCBASE gerichte zoek UI
    scope_df = faq_df[faq_df["Systeem"] == mode]
    st.subheader(f"{mode} — zoeken")
    q = st.text_input(f"Zoek in {mode}…", placeholder="Typ een onderwerp")

    # categorie
    if mode == "Exact":
        cat = st.selectbox("Categorie", ["Alle"] + EIGHT_CATS, index=0)
    else:
        dyn = sorted([x for x in scope_df["Categorie"].dropna().unique().tolist() if x])
        cat = st.selectbox("Categorie", ["Alle"] + dyn, index=0)

    data = scope_df if cat == "Alle" else scope_df[scope_df["Categorie"] == cat]
    results = rank_rows(data, q) if q else data.copy()

    page_size = st.slider("Per pagina", 10, 100, 50, 10, key=f"ps_{mode}")
    total = len(results); pages = max(1, int(np.ceil(total/page_size)))
    page = st.number_input("Pagina", 1, pages, 1, key=f"pg_{mode}")
    s, e = (page-1)*page_size, min(page*page_size, total)
    st.caption(f"Toont {s+1 if total else 0}–{e} van {total}")
    for _, row in results.iloc[s:e].iterrows():
        render_result(row)

if __name__ == "__main__":
    main()
