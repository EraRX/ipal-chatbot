# IPAL Chatbox — main.py (stabiele complete versie)
# -------------------------------------------------
# - 4 knoppen: Exact | DocBase | Zoeken | Internet (+ Reset)
# - CSV robuust, cascade 1..6, 'Antwoord of oplossing' NIET opschonen
# - Index op Systeem/Subthema/Categorie maar kolommen blijven bestaan
# - Kopieer-knop, PDF-download, optionele AI (fallback zonder sleutel)
# -------------------------------------------------

from __future__ import annotations
import os, re, io, json, hashlib, logging
from datetime import datetime
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# Optioneel (AI)
from dotenv import load_dotenv
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Optioneel (web-fallback)
try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests = None
    BeautifulSoup = None

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, ListFlowable, ListItem
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import cm
from reportlab.lib import colors

# ── UI-config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="IPAL Chatbox", page_icon="💬", layout="centered")
st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; }
      html, body, [class*="css"] { font-size: 18px; }
      button[kind="primary"] { font-size: 20px !important; padding:.65em 1.2em; }
      .card { border:1px solid #e5e7eb;border-radius:16px;background:#fff;
              padding:1rem 1.1rem;margin:.5rem 0;box-shadow:0 3px 10px rgba(0,0,0,.04); }
      .badge { display:inline-block;background:#eef2ff;border:1px solid #e5e7eb;
               border-radius:999px;padding:.2rem .55rem;font-size:.78rem;margin-right:.4rem }
      .muted { color:#6b7280 }
    </style>
    """,
    unsafe_allow_html=True,
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ── Optionele AI ─────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY) if (OpenAI and OPENAI_KEY) else None
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Constantes ───────────────────────────────────────────────────────────────
EIGHT_CATEGORIES = [
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
CASCADE_ORDER = [
    "ID",
    "Systeem",                # 1
    "Subthema",               # 2
    "Categorie",              # 3
    "Omschrijving melding",   # 4
    "Toelichting melding",    # 5
    "Antwoord of oplossing",  # 6  (NIET opschonen)
    "Afbeelding",
]
SEARCH_COLS = ["Systeem","Subthema","Categorie","Omschrijving melding","Toelichting melding"]

AI_INFO = (
    "AI-Antwoord Info:\n"
    "1. Dit antwoord komt uit de IPAL chatbox. Controleer bij twijfel altijd de officiële bron.\n"
    "2. Hulp nodig met DocBase of Exact? Kijk eerst in de FAQ of maak een ticket in DocBase.\n"
    "- Veelgestelde vragen DocBase nieuw 2024\n"
    "- Veelgestelde vragen Exact Online"
)

# ── Helpers ──────────────────────────────────────────────────────────────────
def clean_text(s: str) -> str:
    if s is None: return ""
    s = str(s).replace("\u00A0", " ")
    repl = {
        "\u0091": "'", "\u0092": "'", "\u0093": '"', "\u0094": '"',
        "\u0096": "-", "\u0097": "-", "\u0085": "...",
        "\u2018": "'", "\u2019": "'", "\u201A": ",", "\u201B": "'",
        "\u201C": '"', "\u201D": '"', "\u201E": '"',
        "\u00AB": '"', "\u00BB": '"', "\u2039": "'", "\u203A": "'",
        "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00AD": "", "\u2026": "...",
        "\u00B4": "'", "\u02BC": "'", "\u02BB": "'",
    }
    for k, v in repl.items(): s = s.replace(k, v)
    return re.sub(r"\s+", " ", s).strip()

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

def _strip_md(s: str) -> str:
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"#+\s*([^\n]+)", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)
    return s

def make_pdf(question: str, answer: str) -> bytes:
    question = clean_text(question or "")
    answer   = clean_text(_strip_md(answer or ""))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leading=16, spaceAfter=10, alignment=TA_LEFT)
    head = ParagraphStyle("Heading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=colors.HexColor("#333"))

    story = []
    if os.path.exists("logopdf.png"):
        try:
            banner = Image("logopdf.png"); banner._restrictSize(A4[0]-4*cm, 10000); banner.hAlign = "LEFT"; story += [banner, Spacer(1, 8)]
        except Exception:
            pass
    story.append(Paragraph(f"Vraag: {question}", head)); story.append(Spacer(1, 8))
    story.append(Paragraph("Antwoord:", head))
    for line in (answer.split("\n") if answer else []):
        if line.strip(): story.append(Paragraph(line.strip(), body))
    story.append(Spacer(1, 8)); story.append(Paragraph(_strip_md(AI_INFO).replace("\n","<br/>"), body))
    doc.build(story); buf.seek(0); return buf.getvalue()

def copy_button(text: str, key_suffix: str):
    payload = text or ""
    js_text = json.dumps(payload)
    html_code = """
<div style="margin-top:8px;">
  <button id="COPY_BTN_ID" style="padding:6px 10px;font-size:16px;">Kopieer antwoord</button>
  <span id="COPY_STATE_ID" style="margin-left:8px;font-size:14px;"></span>
  <script>
    (function(){
      const btn = document.getElementById('COPY_BTN_ID');
      const state = document.getElementById('COPY_STATE_ID');
      if (btn) {
        btn.addEventListener('click', async () => {
          try { await navigator.clipboard.writeText(JS_TEXT);
                state.textContent='Gekopieerd!'; setTimeout(()=>state.textContent='',1500);
          } catch(e){ state.textContent='Niet gelukt — gebruik de tekst hieronder.'; setTimeout(()=>state.textContent='',3000); }
        });
      }
    })();
  </script>
</div>
""".replace("COPY_BTN_ID", f"copybtn-{key_suffix}") \
   .replace("COPY_STATE_ID", f"copystate-{key_suffix}") \
   .replace("JS_TEXT", js_text)
    components.html(html_code, height=70)
    with st.expander("Kopiëren lukt niet? Toon tekst om handmatig te kopiëren."):
        st.text_area("Tekst", payload, height=150, key=f"copy_fallback_{key_suffix}")

# ── CSV laden + normaliseren ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_faq(path: str = "faq.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        st.warning(f"FAQ-bestand '{path}' niet gevonden.")
        return pd.DataFrame(columns=CASCADE_ORDER).set_index(["Systeem","Subthema","Categorie"], drop=False)

    df, _meta = read_csv_smart(path)
    df.columns = [str(c).strip().replace("\ufeff","") for c in df.columns]

    # kolommen afdwingen
    for c in CASCADE_ORDER:
        if c not in df.columns: df[c] = ""

    # overbodige kolom verwijderen
    if "Soort melding" in df.columns:
        df.drop(columns=["Soort melding"], inplace=True)

    # normaliseren (NIET op 'Antwoord of oplossing')
    norm_cols = ["ID","Systeem","Subthema","Categorie","Omschrijving melding","Toelichting melding","Afbeelding"]
    for c in norm_cols:
        df[c] = df[c].astype(str).map(clean_text)

    # systeemnaam normaliseren
    def normalize_system(s: str) -> str:
        t = (s or "").strip().lower()
        if t.startswith("exact"): return "Exact"
        if t.startswith("docbase"): return "DocBase"
        if t.startswith("sila"): return "SILA"
        if t.startswith("algemeen"): return "Algemeen"
        return s or ""
    df["Systeem"] = df["Systeem"].map(normalize_system)

    # Exact: mappen naar 8 vaste categorieën
    def eight_name(s: str) -> str:
        base = re.sub(r"^\s*\d+(?:\.\d+)*\s*", "", (s or "").strip())
        base = re.sub(r"\s*\(.*?\)\s*$", "", base)
        for canon in EIGHT_CATEGORIES:
            if canon.lower() in base.lower() or base.lower() in canon.lower():
                return canon
        return ""

    def map_exact_category(row):
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
    df["Categorie"] = df.apply(map_exact_category, axis=1)

    # Subthema fix (Exact Online)
    df["Subthema"] = df.apply(lambda r: ("Exact Online" if r["Systeem"]=="Exact" and eight_name(r.get("Subthema","")) else (r.get("Subthema","") or ("Exact Online" if r["Systeem"]=="Exact" else ""))), axis=1)

    # combined voor zoeken
    df["combined"] = df[SEARCH_COLS].fillna("").agg(" ".join, axis=1)

    # index zetten maar kolommen behouden
    return df.set_index(["Systeem","Subthema","Categorie"], drop=False)

faq_df = load_faq(os.getenv("FAQ_CSV", "/mnt/data/faq (1).csv") if os.path.exists("/mnt/data/faq (1).csv") else os.getenv("FAQ_CSV", "faq.csv"))

# ── Zoeken / ranking ─────────────────────────────────────────────────────────
STOPWORDS_NL = {"de","het","een","en","of","maar","want","dus","als","dan","dat","die","dit","deze",
"ik","jij","hij","zij","wij","jullie","u","ze","je","mijn","jouw","zijn","haar","ons","hun",
"van","voor","naar","met","bij","op","in","aan","om","tot","uit","over","onder","boven","zonder",
"ook","nog","al","wel","niet","nooit","altijd","hier","daar","ergens","niets","iets","alles",
"is","was","wordt","zijn","heeft","heb","hebben","doe","doet","doen","kan","kunnen","moet","moeten"}

def _tok(text: str) -> list[str]:
    return [w for w in re.findall(r"[0-9A-Za-zÀ-ÿ_]+", (text or "").lower()) if len(w)>2 and w not in STOPWORDS_NL]

def token_score(q: str, text: str) -> int:
    qs = set(_tok(q)); ts = set(_tok(text)); return len(qs & ts)

def rank_rows(df: pd.DataFrame, q: str) -> pd.DataFrame:
    ql = (q or "").strip().lower()
    if not ql: return df.assign(_score=0)
    out = df.copy()
    out["_score"] = out["combined"].apply(lambda t: token_score(ql, t))
    out = out.sort_values("_score", ascending=False)
    return out[out["_score"]>0]

# ── Eenvoudige uitleg (AI of fallback) ───────────────────────────────────────
def simplify_text(txt: str, max_bullets: int = 5) -> str:
    text = clean_text(txt or "")
    if not text: return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    keys = ["klik","open","ga naar","instellingen","fout","oplossing","stap","menu","rapport","boek","opslaan","zoeken"]
    scored = []
    for s in sentences:
        sc = sum(1 for k in keys if k in s.lower()) + min(2, len(re.findall(r"\d+", s)))
        if len(s) <= 200: sc += 1
        scored.append((sc, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    bullets = [s for _, s in scored[:max_bullets]] or sentences[:min(max_bullets,3)]
    out = "### In het kort\n" + "\n".join(f"- {b}" for b in bullets)
    steps = [s for s in sentences if re.search(r"^(Klik|Open|Ga|Kies|Vul|Controleer|Selecteer)\b", s.strip(), re.I)]
    if steps:
        out += "\n\n### Stappenplan\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(steps[:max_bullets], 1))
    return out.strip()

def simple_from_source(text: str) -> str:
    txt = (text or "").strip()
    if not txt: return ""
    if client:
        try:
            resp = client.chat.completions.create(
                model=MODEL, temperature=0.2, max_tokens=500,
                messages=[
                    {"role":"system","content":"Leg simpel uit in NL voor vrijwilligers. Max 5 bullets; eventueel kort stappenplan. Baseer ALLES op de meegegeven bron, geen aannames."},
                    {"role":"user","content":f"Bron:\n{txt}\n\nMaak het eenvoudig."}
                ]
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass
    return simplify_text(txt)

def enrich_with_simple(answer: str) -> str:
    simp = simple_from_source(answer)
    return f"{answer}\n\n---\n\n{simp}" if simp else answer

# ── UI rendering helpers ─────────────────────────────────────────────────────
def render_result(row: pd.Series):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<span class='badge'>{row['Systeem']} › {row['Categorie']}</span>", unsafe_allow_html=True)
    st.markdown(f"**{row['Omschrijving melding']}**")
    if str(row.get("Toelichting melding","")).strip():
        st.caption(row["Toelichting melding"])
    with st.expander("Antwoord tonen", expanded=False):
        ans = str(row.get("Antwoord of oplossing","") or "")
        final = enrich_with_simple(ans)
        st.markdown(final)
        # PDF & kopie
        pdf = make_pdf(row.get("Omschrijving melding","Vraag"), final)
        st.download_button("📄 Download PDF", data=pdf, file_name="antwoord.pdf", mime="application/pdf")
        copy_button(final, hashlib.md5(final.encode("utf-8")).hexdigest()[:8])
    st.markdown("</div>", unsafe_allow_html=True)

# ── App ──────────────────────────────────────────────────────────────────────
def main():
    st.header("Welkom bij IPAL Chatbox")

    # Snelkeuzes
    c1, c2, c3, c4, c5 = st.columns(5)
    if "mode" not in st.session_state: st.session_state.mode = None
    if c1.button("Exact", use_container_width=True):   st.session_state.mode = "Exact"
    if c2.button("DocBase", use_container_width=True): st.session_state.mode = "DocBase"
    if c3.button("Zoeken", use_container_width=True):  st.session_state.mode = "Zoeken"
    if c4.button("Internet", use_container_width=True):st.session_state.mode = "Internet"
    if c5.button("🔄 Reset", use_container_width=True):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.session_state.mode = None
        st.cache_data.clear()
        st.rerun()

    mode = st.session_state.mode

    # Geen keuze? Toon korte uitleg
    if not mode:
        st.info("Kies hierboven **Exact**, **DocBase**, **Zoeken** (volledige CSV) of **Internet** (algemene vraag).")
        return

    # INTERNET
    if mode == "Internet":
        q = st.text_input("Waarover gaat uw vraag?", placeholder="Algemene vraag (geen CSV nodig)")
        if not q: return
        if client:
            try:
                resp = client.chat.completions.create(
                    model=MODEL, temperature=0.2, max_tokens=500,
                    messages=[{"role":"system","content":"Antwoord kort en praktisch, NL, max 8 zinnen."},
                              {"role":"user","content":q}]
                )
                ans = resp.choices[0].message.content.strip()
            except Exception as e:
                logging.error(f"AI fout: {e}"); ans = "Kunt u uw vraag iets concreter maken?"
        else:
            ans = "Kunt u uw vraag iets concreter maken (bijv. 'DocBase wachtwoord resetten' of 'Exact bankkoppeling')?"
        final = ans + "\n\n" + AI_INFO
        st.markdown(final)
        pdf = make_pdf(q, final); st.download_button("📄 Download PDF", data=pdf, file_name="antwoord.pdf", mime="application/pdf")
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

    # EXACT/DOCBASE gerichte zoek UI
    scope_df = faq_df[faq_df["Systeem"] == mode]
    st.subheader(f"{mode} — zoeken")
    q = st.text_input(f"Zoek in {mode}…", placeholder="Typ een onderwerp")
    # categorie-keuze
    if mode == "Exact":
        cat = st.selectbox("Categorie", ["Alle"] + EIGHT_CATEGORIES, index=0)
    else:
        dyn = sorted([x for x in scope_df["Categorie"].dropna().unique().tolist() if x])
        cat = st.selectbox("Categorie", ["Alle"] + dyn, index=0)
    data = scope_df if cat=="Alle" else scope_df[scope_df["Categorie"]==cat]
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
