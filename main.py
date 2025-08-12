"""
IPAL Chatbox — Definitieve main.py (4 knoppen)
- Start: Exact | DocBase | Zoeken (hele CSV) | Algemeen
- Exact/DocBase: cascade → item → PDF + vervolgvraag (op basis van dat item)
- Zoeken (hele CSV): vrije zoekterm over hele CSV → kies item → PDF + vervolgvraag
- Algemeen: géén CSV, alleen AI (en optioneel Web-fallback)
- CSV-robustheid: trim, NBSP→spatie, multi-spaces→één, casefold-matches
- Altijd PDF-knop (unieke key) + “Kopieer antwoord” + optionele Afbeelding + scope teller
- Algemeen: stelt max. 1 verhelderende vraag, maar blijft niet doorvragen uit zichzelf
"""

import os
import re
import io
import logging
import hashlib
from datetime import datetime
from typing import Optional, List

import streamlit as st
import pandas as pd
import pytz
from dotenv import load_dotenv
from openai import OpenAI

# Web-fallback
import requests
from bs4 import BeautifulSoup

try:
    from openai.error import RateLimitError
except ImportError:
    RateLimitError = Exception

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, ListFlowable, ListItem, Table, TableStyle
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ── UI-config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="IPAL Chatbox", layout="centered")
st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-size:20px; }
      button[kind="primary"] { font-size:22px !important; padding:.75em 1.5em; }
      video { width: 600px !important; height: auto !important; max-width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ── OpenAI (optioneel) ───────────────────────────────────────────────────────
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(1, 8), retry=retry_if_exception_type(RateLimitError))
@st.cache_data(show_spinner=False)
def chatgpt_cached(messages, temperature=0.2, max_tokens=700) -> str:
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()


# ── PDF helpers ──────────────────────────────────────────────────────────────
if os.path.exists("Calibri.ttf"):
    pdfmetrics.registerFont(TTFont("Calibri", "Calibri.ttf"))

FAQ_LINKS = [
    ("Veelgestelde vragen DocBase nieuw 2024", "https://parochie-automatisering.nl/docbase/Templates/docbase?action=SelOpenDocument&DetailsMode=2&Docname=00328526&Type=INSTR_DOCS&LoginMode=1&LinkToVersion=1&OpenFileMode=2&password=%3Auzt7hs%23qL%2A%28&username=Externehyperlink&ID=0.07961651005089099&EC=1"),
    ("Veelgestelde vragen Exact Online", "https://parochie-automatisering.nl/docbase/Templates/docbase?action=SelOpenDocument&DetailsMode=2&Docname=00328522&Type=INSTR_DOCS&LoginMode=1&LinkToVersion=1&OpenFileMode=2&password=%3Auzt7hs%23qL%2A%28&username=Externehyperlink&ID=0.8756321684738348&EC=1"),
]

def _strip_md(s: str) -> str:
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)                      # **bold** → plain
    s = re.sub(r"#+\s*([^\n]+)", r"\1", s)                        # # heading → plain
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)              # [label](url) → label
    return s

def make_pdf(question: str, answer: str) -> bytes:
    answer = _strip_md(answer or "")
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leading=16, spaceAfter=12, alignment=TA_LEFT)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=colors.HexColor("#333"), spaceBefore=12, spaceAfter=6)

    story = []

    if os.path.exists("logopdf.png"):
        logo = Image("logopdf.png", width=124, height=52)
        story.append(Table([[logo]], colWidths=[124], style=TableStyle([
            ("ALIGN", (0,0), (-1,-1), "LEFT"),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 0),
            ("RIGHTPADDING", (0,0), (-1,-1), 0),
            ("TOPPADDING", (0,0), (-1,-1), 0),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ])))

    story.append(Paragraph(f"Vraag: {question or ''}", heading_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Antwoord:", heading_style))

    for line in (answer.split("\n") if answer else []):
        line = line.strip()
        if line:
            story.append(Paragraph(line, body_style))

    # FAQ-links
    story.append(Spacer(1, 12))
    story.append(Paragraph("Klik hieronder om de FAQ te openen:", heading_style))
    items = []
    for label, url in FAQ_LINKS:
        p = Paragraph(f'<link href="{url}" color="blue">{label}</link>', body_style)
        items.append(ListItem(p, leftIndent=12))
    story.append(ListFlowable(items, bulletType="bullet"))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ── CSV laden + normaliseren ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_faq(path: str = "faq.csv") -> pd.DataFrame:
    cols = [
        "ID","Systeem","Subthema","Categorie",
        "Omschrijving melding","Toelichting melding","Soort melding",
        "Antwoord of oplossing","Afbeelding"
    ]
    if not os.path.exists(path):
        st.error(f"FAQ-bestand '{path}' niet gevonden.")
        return pd.DataFrame(columns=cols).set_index(["Systeem","Subthema","Categorie"])

    try:
        df = pd.read_csv(path, encoding="utf-8", sep=";")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="windows-1252", sep=";")

    for c in cols:
        if c not in df.columns:
            df[c] = None

    # Normalize
    norm_cols = ["Systeem","Subthema","Categorie","Omschrijving melding","Toelichting melding","Soort melding"]
    for c in norm_cols:
        df[c] = (df[c]
                 .fillna("")
                 .astype(str)
                 .str.replace("\u00A0", " ", regex=False)   # NBSP
                 .str.strip()
                 .str.replace(r"\s+", " ", regex=True))     # meerdere spaties → 1

    # Standaardiseer Systeem
    mapping = {"exact": "Exact", "docbase": "DocBase", "algemeen": "Algemeen"}
    df["Systeem"] = df["Systeem"].str.lower().map(mapping).fillna(df["Systeem"]).astype(str)

    # Combined voor ranking
    keep = ["Systeem","Subthema","Categorie","Omschrijving melding","Toelichting melding"]
    df["combined"] = df[keep].fillna("").agg(" ".join, axis=1)

    return df.set_index(["Systeem","Subthema","Categorie"], drop=True)

faq_df = load_faq()


# ── Cascade helpers ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def list_subthema(systeem: str) -> List[str]:
    try:
        return sorted(faq_df.xs(systeem, level="Systeem").index.get_level_values("Subthema").dropna().unique())
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def list_categorieen(systeem: str, subthema: str) -> List[str]:
    try:
        subset = faq_df.xs((systeem, subthema), level=["Systeem","Subthema"], drop_level=False)
        return sorted(subset.index.get_level_values("Categorie").dropna().unique())
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def list_toelichtingen(systeem: str, subthema: str, categorie: Optional[str]) -> List[str]:
    try:
        if not categorie or str(categorie).lower() == "alles":
            scope = faq_df.xs((systeem, subthema), level=["Systeem","Subthema"], drop_level=False)
        else:
            scope = faq_df.xs((systeem, subthema, categorie), level=["Systeem","Subthema","Categorie"], drop_level=False)
        vals = (scope["Toelichting melding"]
                .dropna()
                .astype(str)
                .str.replace("\u00A0", " ", regex=False)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .unique())
        return sorted(vals)
    except Exception:
        return []


# ── Veiligheidsfilter & relevance helpers ────────────────────────────────────
BLACKLIST = ["persoonlijke gegevens","medische gegevens","gezondheid","privacy schending"]

def filter_topics(msg: str):
    found = [t for t in BLACKLIST if re.search(r"\b" + re.escape(t) + r"\b", (msg or "").lower())]
    return (False, f"Je bericht bevat gevoelige onderwerpen: {', '.join(found)}.") if found else (True, "")

# Stopwoorden (NL)
STOPWORDS_NL = {
    "de","het","een","en","of","maar","want","dus","als","dan","dat","die","dit","deze",
    "ik","jij","hij","zij","wij","jullie","u","ze","je","mijn","jouw","zijn","haar","ons","hun",
    "van","voor","naar","met","bij","op","in","aan","om","tot","uit","over","onder","boven","zonder",
    "ook","nog","al","wel","niet","nooit","altijd","hier","daar","ergens","niets","iets","alles",
    "is","was","wordt","zijn","heeft","heb","hebben","doe","doet","doen","kan","kunnen","moet","moeten"
}

def _tokenize_clean(text: str) -> list[str]:
    return [
        w for w in re.findall(r"[0-9A-Za-zÀ-ÿ_]+", (text or "").lower())
        if len(w) > 2 and w not in STOPWORDS_NL
    ]

def _relevance(q: str, t: str) -> tuple[int, float]:
    qs = set(_tokenize_clean(q))
    ts = set(_tokenize_clean(t))
    hits = len(qs & ts)
    coverage = hits / max(1, len(qs))
    return hits, coverage

def _token_score(q: str, text: str) -> int:
    qs = set(_tokenize_clean(q))
    ts = set(_tokenize_clean(text))
    return len(qs & ts)


# ── Zoekfuncties ─────────────────────────────────────────────────────────────
def find_answer_by_codeword(df: pd.DataFrame, codeword: str = "[UNIEKECODE123]") -> Optional[str]:
    try:
        mask = df["Antwoord of oplossing"].astype(str).str.contains(codeword, case=False, na=False)
        if mask.any():
            return str(df.loc[mask].iloc[0]["Antwoord of oplossing"]).strip()
    except Exception:
        pass
    return None

def zoek_hele_csv(vraag: str, min_hits: int = 2, min_cov: float = 0.25, fallback_rows: int = 50) -> pd.DataFrame:
    """Zoek over hele CSV, met adaptieve drempels en zinnige fallback."""
    if faq_df.empty:
        return pd.DataFrame()

    df = faq_df.reset_index().copy()

    # Adaptieve drempel: nooit meer hits eisen dan er tokens in de vraag zitten (minimaal 1)
    q_tokens = set(_tokenize_clean(vraag))
    eff_min_hits = max(1, min(min_hits, len(q_tokens)))

    # Basisscore
    df["_score"] = df["combined"].apply(lambda t: _token_score(vraag, t))
    df = df.sort_values("_score", ascending=False)
    if df.empty:
        return df

    # Filter op relevantie
    def _ok(row):
        hits, cov = _relevance(vraag, str(row["combined"]))
        return hits >= eff_min_hits and cov >= min_cov

    filtered = df[df.apply(_ok, axis=1)]

    # Fallbacks bij lege hitlijst
    if filtered.empty:
        q_lower = (vraag or "").strip().lower()
        sys_map = {"exact": "Exact", "docbase": "DocBase", "algemeen": "Algemeen"}

        # 1) Speciaal: als vraag precies een systeemnaam is → toon top N van dat systeem
        if q_lower in sys_map:
            sys = sys_map[q_lower]
            try:
                subset = faq_df.xs(sys, level="Systeem", drop_level=False).reset_index()
                subset["_score"] = subset["combined"].apply(lambda t: _token_score(vraag, t))
                return subset.sort_values("_score", ascending=False).head(fallback_rows)
            except KeyError:
                pass

        # 2) Anders: top N met _score > 0
        nonzero = df[df["_score"] > 0].head(fallback_rows)
        if not nonzero.empty:
            return nonzero

        # 3) Allerlaatste redmiddel: gewoon top N
        return df.head(fallback_rows)

    return filtered

def vind_best_algemeen_AI(vraag: str) -> str:
    """Algemeen: géén CSV. Alleen AI + één verhelderende vraag max (of suggesties)."""
    if client is None:
        # Zonder AI: basic echo/suggestie
        return "Kunt u uw vraag iets concreter maken (bijv. ‘DocBase wachtwoord resetten’ of ‘Exact bankkoppeling’)?"
    sys = (
        "Je helpt vrijwilligers van parochies. "
        "Beantwoord kort en concreet. Stel maximaal één verhelderende vraag als dat echt nodig is. "
        "Noem geen niet-bestaande bronnen."
    )
    user = (
        f"Vraag: {vraag}\n\n"
        "Als de vraag niet direct over DocBase/Exact/IPAL-onderwerpen gaat, geef dan 1-2 praktische vervolgsuggesties "
        "of verwijs vriendelijk naar het juiste kanaal. Stel hoogstens één verhelderende vraag."
    )
    try:
        return chatgpt_cached(
            [{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0.2, max_tokens=500
        )
    except Exception as e:
        logging.error(f"AI (Algemeen) fout: {e}")
        return "Kunt u uw vraag iets concreter maken?"

# ── Web-fallback ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_web_info_cached(query: str) -> Optional[str]:
    result = []
    try:
        r = requests.get("https://docbase.nl", timeout=5); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        txt = " ".join(p.get_text(strip=True) for p in soup.find_all(['p','h1','h2','h3']))
        if txt and (query or "").lower() in txt.lower():
            result.append(f"Vanuit docbase.nl: {txt[:200]}... (verkort)")
    except Exception:
        pass
    try:
        r = requests.get("https://support.exactonline.com/community/s/knowledge-base", timeout=5); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        txt = " ".join(p.get_text(strip=True) for p in soup.find_all(['p','h1','h2','h3']))
        if txt and (query or "").lower() in txt.lower():
            result.append(f"Vanuit Exact Online Knowledge Base: {txt[:200]}... (verkort)")
    except Exception:
        pass
    return "\n".join(result) if result else None


# ── UI helpers & state ───────────────────────────────────────────────────────
AI_INFO = """
AI-Antwoord Info:  
1. Dit is het AI-antwoord vanuit de IPAL chatbox van het Interdiocesaan Platform Automatisering & Ledenadministratie. Het is altijd een goed idee om de meest recente informatie te controleren via officiële bronnen.  
2. Heeft u hulp nodig met DocBase of Exact? Dan kunt u eenvoudig een melding maken door een ticket aan te maken in DocBase. Maar voordat u een ticket invult, hebben we een handige tip: controleer eerst onze FAQ (het document met veelgestelde vragen en antwoorden). Dit document vindt u op onze site. Klik hieronder om de FAQ te openen en te kijken of uw vraag al beantwoord is:

- [Veelgestelde vragen DocBase nieuw 2024](https://parochie-automatisering.nl/docbase/Templates/docbase?action=SelOpenDocument&DetailsMode=2&Docname=00328526&Type=INSTR_DOCS&LoginMode=1&LinkToVersion=1&OpenFileMode=2&password=%3Auzt7hs%23qL%2A%28&username=Externehyperlink&ID=0.07961651005089099&EC=1)
- [Veelgestelde vragen Exact Online](https://parochie-automatisering.nl/docbase/Templates/docbase?action=SelOpenDocument&DetailsMode=2&Docname=00328522&Type=INSTR_DOCS&LoginMode=1&LinkToVersion=1&OpenFileMode=2&password=%3Auzt7hs%23qL%2A%28&username=Externehyperlink&ID=0.8756321684738348&EC=1)
"""

TIMEZONE = pytz.timezone("Europe/Amsterdam")
ASSISTANT_AVATAR = "aichatbox.png" if os.path.exists("aichatbox.png") else None
USER_AVATAR = "parochie.png" if os.path.exists("parochie.png") else None

DEFAULT_STATE = {
    "history": [],
    "selected_product": None,     # "Exact" | "DocBase" | "Zoeken" | "Algemeen"
    "selected_module": None,
    "selected_category": None,
    "selected_toelichting": None,
    "selected_answer_id": None,
    "selected_answer_text": None,
    "selected_image": None,
    "last_question": "",
    "last_item_label": "",
    "debug": False,
    "allow_ai": False,
    "allow_web": False,
    # Zoeken (hele CSV) drempels:
    "min_hits": 2,
    "min_cov": 0.25,
    # Voor Zoeken-mode:
    "search_query": "",
    "search_selection_index": None,
}
for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v

MAX_HISTORY = 12

def get_avatar(role: str):
    return ASSISTANT_AVATAR if role == "assistant" and ASSISTANT_AVATAR else USER_AVATAR

def add_msg(role: str, content: str):
    ts = datetime.now(TIMEZONE).strftime("%d-%m-%Y %H:%M")
    st.session_state.history = (st.session_state.history + [{"role": role, "content": content, "time": ts}])[-MAX_HISTORY:]

def with_info(text: str) -> str:
    return (text or "").strip() + "\n\n" + AI_INFO

def _copy_button(text: str, key_suffix: str):
    # Client-side kopiëren naar klembord
    safe = (text or "")
    safe = safe.replace("\\", "\\\\").replace("`", "\\`").replace("\n", "\\n")
    btn_id = f"copybtn-{key_suffix}"
    st.markdown(
        f"""
        <button id="{btn_id}" style="margin-top:8px;padding:6px 10px;font-size:16px;">
          Kopieer antwoord
        </button>
        <script>
        const _btn = document.getElementById("{btn_id}");
        if (_btn) {{
          _btn.onclick = async () => {{
            try {{
              await navigator.clipboard.writeText(`{safe}`);
              const old = _btn.innerText;
              _btn.innerText = "Gekopieerd!";
              setTimeout(()=>{{ _btn.innerText = old; }}, 1500);
            }} catch(e) {{ console.log(e); }}
          }};
        }}
        </script>
        """,
        unsafe_allow_html=True
    )

def render_breadcrumbs():
    syst = st.session_state.get("selected_product") or ""
    sub = st.session_state.get("selected_module") or ""
    cat = st.session_state.get("selected_category") or ""
    toe = st.session_state.get("selected_toelichting") or ""
    parts = [p for p in [syst, sub, (None if cat in ("", None, "alles") else cat), (toe or None)] if p]
    if parts:
        st.caption(" › ".join(parts))

def render_chat():
    for i, m in enumerate(st.session_state.history):
        st.chat_message(m["role"], avatar=get_avatar(m["role"])).markdown(f"{m['content']}\n\n_{m['time']}_")
        # ✅ Altijd een PDF-knop + copy-knop bij het laatste assistentbericht
        if m["role"] == "assistant" and i == len(st.session_state.history) - 1:
            q = (
                st.session_state.get("last_question")
                or st.session_state.get("last_item_label")
                or "Gekozen item"
            )
            pdf = make_pdf(q, m["content"])
            btn_key = f"pdf_{i}_{m['time'].replace(':','-')}"
            st.download_button("📄 Download PDF", data=pdf, file_name="antwoord.pdf", mime="application/pdf", key=btn_key)

            # Copy knop (client-side)
            hash_key = hashlib.md5((m["time"] + m["content"]).encode("utf-8")).hexdigest()[:8]
            _copy_button(m["content"], hash_key)

            # Afbeelding tonen indien gekozen record er één had
            img = st.session_state.get("selected_image")
            if img and isinstance(img, str) and img.strip():
                try:
                    st.image(img, caption="Afbeelding bij dit antwoord", use_column_width=True)
                except Exception:
                    pass


# ── App ──────────────────────────────────────────────────────────────────────
def main():
    with st.sidebar:
        if st.button("🔄 Nieuw gesprek", use_container_width=True):
            st.session_state.clear()
            for k, v in DEFAULT_STATE.items():
                if k not in st.session_state:
                    st.session_state[k] = v
            st.rerun()

        st.session_state["debug"] = st.toggle("Debug info", value=st.session_state.get("debug", False))
        st.session_state["allow_ai"] = st.toggle("AI-QA aan", value=st.session_state.get("allow_ai", False))
        st.session_state["allow_web"] = st.toggle("Web-fallback aan (Algemeen)", value=st.session_state.get("allow_web", False))

        # Drempels voor Zoeken (hele CSV)
        st.session_state["min_hits"] = st.slider(
            "CSV minimum treffers (Zoeken)", min_value=0, max_value=6,
            value=int(st.session_state.get("min_hits", 2)), step=1
        )
        st.session_state["min_cov"] = st.slider(
            "CSV minimale dekking (Zoeken)", min_value=0.0, max_value=1.0,
            value=float(st.session_state.get("min_cov", 0.25)), step=0.05
        )

        if st.button("🧹 Cache legen", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        if st.session_state["allow_ai"] and not OPENAI_KEY:
            st.warning("AI-QA staat aan maar er is geen OPENAI_API_KEY.")

        if st.session_state["debug"]:
            try:
                cnt_exact = len(faq_df.xs("Exact", level="Systeem", drop_level=False))
            except Exception:
                cnt_exact = 0
            try:
                cnt_doc = len(faq_df.xs("DocBase", level="Systeem", drop_level=False))
            except Exception:
                cnt_doc = 0
            st.caption(f"CSV records: {len(faq_df.reset_index())} | Exact: {cnt_exact} | DocBase: {cnt_doc}")

    # Startscherm
    if not st.session_state.get("selected_product"):
        video_path = "helpdesk.mp4"
        if os.path.exists(video_path):
            try:
                with open(video_path, "rb") as f:
                    st.video(f.read(), format="video/mp4", start_time=0, autoplay=True)
            except Exception as e:
                logging.error(f"Introvideo kon niet worden afgespeeld: {e}")
        elif os.path.exists("logo.png"):
            st.image("logo.png", width=244)
        else:
            st.info("Welkom bij IPAL Chatbox")

        st.header("Welkom bij IPAL Chatbox")
        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)
        if c1.button("Exact", use_container_width=True):
            st.session_state.update({
                "selected_product": "Exact",
                "selected_image": None,
                "selected_module": None,
                "selected_category": None,
                "selected_toelichting": None,
                "selected_answer_id": None,
                "selected_answer_text": None,
            })
            st.toast("Gekozen: Exact")
            st.rerun()
        if c2.button("DocBase", use_container_width=True):
            st.session_state.update({
                "selected_product": "DocBase",
                "selected_image": None,
                "selected_module": None,
                "selected_category": None,
                "selected_toelichting": None,
                "selected_answer_id": None,
                "selected_answer_text": None,
            })
            st.toast("Gekozen: DocBase")
            st.rerun()
        if c3.button("Zoeken (hele CSV)", use_container_width=True):
            st.session_state.update({
                "selected_product": "Zoeken",
                "selected_image": None,
                "search_query": "",
                "search_selection_index": None,
                "selected_answer_id": None,
                "selected_answer_text": None,
                "last_item_label": "",
            })
            st.toast("Gekozen: Zoeken (hele CSV)")
            st.rerun()
        if c4.button("Algemeen", use_container_width=True):
            st.session_state.update({
                "selected_product": "Algemeen",
                "selected_image": None,
                "selected_module": None,
                "selected_category": None,
                "selected_toelichting": None,
                "selected_answer_id": None,
                "selected_answer_text": None,
            })
            st.toast("Gekozen: Algemeen (vrije vraag)")
            st.rerun()
        render_chat()
        return

    # ── ALGEMEEN (géén CSV) ────────────────────────────────────────────────
    if st.session_state.get("selected_product") == "Algemeen":
        render_chat()
        vraag = st.chat_input("Stel uw algemene vraag (geen CSV):")
        if not vraag:
            return

        # UNIEKECODE123 direct (optioneel)
        if (vraag or "").strip().upper() == "UNIEKECODE123":
            cw = find_answer_by_codeword(faq_df.reset_index())
            if cw:
                st.session_state["last_question"] = vraag
                add_msg("user", vraag)
                add_msg("assistant", with_info(cw))
                st.rerun()
                return

        st.session_state["last_question"] = vraag
        add_msg("user", vraag)

        ok, warn = filter_topics(vraag)
        if not ok:
            add_msg("assistant", warn)
            st.rerun()
            return

        # Alleen AI (+ optioneel web ter aanvulling)
        antwoord = vind_best_algemeen_AI(vraag)
        if not antwoord and st.session_state.get("allow_web"):
            webbits = fetch_web_info_cached(vraag)
            if webbits:
                antwoord = webbits

        add_msg("assistant", with_info(antwoord or "Kunt u uw vraag iets concreter maken?"))
        st.rerun()
        return

    # ── ZOEKEN (hele CSV) ──────────────────────────────────────────────────
    if st.session_state.get("selected_product") == "Zoeken":
        render_chat()

        # 1) Zoekterm
        st.session_state["search_query"] = st.text_input(
            "Waar wil je in de volledige CSV op zoeken?",
            value=st.session_state.get("search_query", "")
        )
        q = st.session_state["search_query"].strip()
        if not q:
            return

        # 2) Resultaten ophalen met adaptieve drempels + fallback
        results = zoek_hele_csv(q, min_hits=st.session_state["min_hits"], min_cov=st.session_state["min_cov"])
        st.caption(f"Gevonden resultaten: {len(results)}")
        if results.empty:
            st.info("Geen resultaten gevonden. Pas je zoekterm aan of verlaag de drempels (sliders in de sidebar).")
            return

        df_reset = results.reset_index(drop=True)

        def mk_label(i, row):
            oms = str(row.get("Omschrijving melding", "")).strip()
            toel = str(row.get("Toelichting melding", "")).strip()
            preview = oms or toel or str(row.get("Antwoord of oplossing", "")).strip()
            preview = re.sub(r"\s+", " ", preview)[:140]
            return f"{i+1:02d}. {preview}"

        opties = [mk_label(i, r) for i, r in df_reset.iterrows()]
        keuze = st.selectbox("Kies een item uit de zoekresultaten:", ["(Kies)"] + opties)
        if keuze == "(Kies)":
            return

        idx = int(keuze.split(".")[0]) - 1
        row = df_reset.iloc[idx]
        row_id = row.get("ID", idx)
        ans = str(row.get("Antwoord of oplossing", "") or "").strip()
        if not ans:
            oms = str(row.get("Omschrijving melding", "") or "").strip()
            ans = f"(Geen uitgewerkt antwoord in CSV voor: {oms})"
        label = mk_label(idx, row)
        img = str(row.get("Afbeelding", "") or "").strip()
        st.session_state["selected_image"] = img if img else None

        if st.session_state.get("selected_answer_id") != row_id:
            st.session_state["selected_answer_id"] = row_id
            st.session_state["selected_answer_text"] = ans
            st.session_state["last_item_label"] = label
            st.session_state["last_question"] = f"Gekozen item: {label}"
            add_msg("assistant", with_info(ans))
            st.rerun()
            return

        # 3) Vervolgvraag over dit item
        vraag2 = st.chat_input("Stel uw vraag over dit antwoord:")
        if not vraag2:
            return

        st.session_state["last_question"] = vraag2
        add_msg("user", vraag2)

        ok, warn = filter_topics(vraag2)
        if not ok:
            add_msg("assistant", warn)
            st.rerun()
            return

        bron = str(st.session_state.get("selected_answer_text") or "")
        reactie = None
        if st.session_state.get("allow_ai") and client is not None:
            try:
                reactie = chatgpt_cached(
                    [
                        {"role": "system", "content": "Beantwoord uitsluitend op basis van de meegegeven bron. Geen aannames buiten de bron. Schrijf kort en duidelijk in het Nederlands."},
                        {"role": "user", "content": f"Bron:\n{bron}\n\nVraag: {vraag2}"},
                    ],
                    temperature=0.1, max_tokens=600,
                )
            except Exception as e:
                logging.error(f"AI-QA fout: {e}")
                reactie = None

        if not reactie:
            # Kleine extractieve fallback
            zinnen = re.split(r"(?<=[.!?])\s+", bron)
            scores = [(_token_score(vraag2, z), z) for z in zinnen]
            scores.sort(key=lambda x: x[0], reverse=True)
            top = [z for s, z in scores if s > 0][:3]
            reactie = "\n".join(top) if top else "Ik kan zonder AI geen betere toelichting uit het gekozen antwoord halen."

        add_msg("assistant", with_info(reactie))
        st.rerun()
        return

    # ── Exact/DocBase cascade ────────────────────────────────────────────────
    render_chat()
    # Breadcrumbs
    syst = st.session_state.get("selected_product")
    sub = st.session_state.get("selected_module") or ""
    cat = st.session_state.get("selected_category") or ""
    toe = st.session_state.get("selected_toelichting") or ""
    parts = [p for p in [syst, sub, (None if cat in ("", None, "alles") else cat), (toe or None)] if p]
    if parts:
        st.caption(" › ".join(parts))

    # 1) Subthema
    if not st.session_state.get("selected_module"):
        opts = list_subthema(syst)
        sel = st.selectbox("Kies subthema:", ["(Kies)"] + list(opts))
        if sel != "(Kies)":
            st.session_state["selected_module"] = sel
            st.session_state["selected_category"] = None
            st.session_state["selected_toelichting"] = None
            st.session_state["selected_answer_id"] = None
            st.session_state["selected_answer_text"] = None
            st.session_state["selected_image"] = None
            st.toast(f"Gekozen subthema: {sel}")
            st.rerun()
        return

    # 2) Categorie
    if not st.session_state.get("selected_category"):
        cats = list_categorieen(syst, st.session_state["selected_module"])
        if len(cats) == 0:
            st.info("Geen categorieën voor dit subthema — stap wordt overgeslagen.")
            st.session_state["selected_category"] = "alles"
            st.session_state["selected_toelichting"] = None
            st.session_state["selected_answer_id"] = None
            st.session_state["selected_answer_text"] = None
            st.session_state["selected_image"] = None
            st.rerun()
        selc = st.selectbox("Kies categorie:", ["(Kies)"] + list(cats))
        if selc != "(Kies)":
            st.session_state["selected_category"] = selc
            st.session_state["selected_toelichting"] = None
            st.session_state["selected_answer_id"] = None
            st.session_state["selected_answer_text"] = None
            st.session_state["selected_image"] = None
            st.toast(f"Gekozen categorie: {selc}")
            st.rerun()
        return

    # 3) Toelichting (optioneel)
    if st.session_state.get("selected_toelichting") is None:
        toes = list_toelichtingen(
            syst,
            st.session_state["selected_module"],
            st.session_state.get("selected_category"),
        )
        if len(toes) == 0:
            st.info("Geen toelichtingen gevonden — stap wordt overgeslagen.")
            st.session_state["selected_toelichting"] = ""  # markeer als afgehandeld
        else:
            toe_sel = st.selectbox("Kies toelichting:", ["(Kies)"] + list(toes))
            if toe_sel != "(Kies)":
                st.session_state["selected_toelichting"] = toe_sel
                st.session_state["selected_answer_id"] = None
                st.session_state["selected_answer_text"] = None
                st.session_state["selected_image"] = None
                st.toast(f"Gekozen toelichting: {toe_sel}")
                st.rerun()
            return  # wachten op keuze/overslaan

    # 4) Recordselectie binnen scope
    df_scope = faq_df
    cat = st.session_state["selected_category"]
    toe = st.session_state.get("selected_toelichting", "")

    try:
        df_scope = df_scope.xs(syst, level="Systeem", drop_level=False)
        df_scope = df_scope.xs(sub, level="Subthema", drop_level=False)
        if cat and str(cat).lower() != "alles":
            df_scope = df_scope.xs(cat, level="Categorie", drop_level=False)
    except KeyError:
        df_scope = pd.DataFrame(columns=faq_df.reset_index().columns)

    if not df_scope.empty and toe is not None and str(toe) != "":
        tm = (df_scope["Toelichting melding"]
              .astype(str)
              .str.replace("\u00A0", " ", regex=False)
              .str.strip()
              .str.replace(r"\s+", " ", regex=True)
              .str.casefold())
        sel = re.sub(r"\s+", " ", str(toe).replace("\u00A0", " ").strip()).casefold()
        df_scope = df_scope[tm == sel]

    try:
        st.sidebar.metric("Records in scope", len(df_scope))
    except Exception:
        pass

    if st.session_state.get("debug"):
        st.caption(f"Scope rows vóór itemkeuze: {len(df_scope)}")

    if df_scope.empty:
        st.info("Geen records gevonden binnen de gekozen Systeem/Subthema/Categorie/Toelichting.")
        return

    df_reset = df_scope.reset_index()

    def mk_label(i, row):
        oms = str(row.get("Omschrijving melding", "")).strip()
        toel = str(row.get("Toelichting melding", "")).strip()
        preview = oms or toel or str(row.get("Antwoord of oplossing", "")).strip()
        preview = re.sub(r"\s+", " ", preview)[:140]
        return f"{i+1:02d}. {preview}"

    opties = [mk_label(i, r) for i, r in df_reset.iterrows()]
    keuze = st.selectbox("Kies een item:", ["(Kies)"] + opties)
    if keuze != "(Kies)":
        i = int(keuze.split(".")[0]) - 1
        row = df_reset.iloc[i]
        row_id = row.get("ID", i)

        ans = str(row.get("Antwoord of oplossing", "") or "").strip()
        if not ans:
            oms = str(row.get("Omschrijving melding", "") or "").strip()
            ans = f"(Geen uitgewerkt antwoord in CSV voor: {oms})"
        label = mk_label(i, row)

        img = str(row.get("Afbeelding", "") or "").strip()
        st.session_state["selected_image"] = img if img else None

        if st.session_state.get("selected_answer_id") != row_id:
            st.session_state["selected_answer_id"] = row_id
            st.session_state["selected_answer_text"] = ans
            st.session_state["last_item_label"] = label
            st.session_state["last_question"] = f"Gekozen item: {label}"
            add_msg("assistant", with_info(ans))
            st.rerun()
            return

    # Vervolgvraag over gekozen antwoord
    vraag = st.chat_input("Stel uw vraag over dit antwoord:")
    if not vraag:
        return

    # UNIEKECODE123 ook hier toestaan
    if (vraag or "").strip().upper() == "UNIEKECODE123":
        cw = find_answer_by_codeword(faq_df.reset_index())
        if cw:
            st.session_state["last_question"] = vraag
            add_msg("user", vraag)
            add_msg("assistant", with_info(cw))
            st.rerun()
            return

    st.session_state["last_question"] = vraag
    add_msg("user", vraag)

    ok, warn = filter_topics(vraag)
    if not ok:
        add_msg("assistant", warn)
        st.rerun()
        return

    bron = str(st.session_state.get("selected_answer_text") or "")
    reactie = None
    if st.session_state.get("allow_ai") and client is not None:
        try:
            reactie = chatgpt_cached(
                [
                    {"role": "system", "content": "Beantwoord uitsluitend op basis van de meegegeven bron. Geen aannames buiten de bron. Schrijf kort en duidelijk in het Nederlands."},
                    {"role": "user", "content": f"Bron:\n{bron}\n\nVraag: {vraag}"},
                ],
                temperature=0.1, max_tokens=600,
            )
        except Exception as e:
            logging.error(f"AI-QA fout: {e}")
            reactie = None

    if not reactie:
        zinnen = re.split(r"(?<=[.!?])\s+", bron)
        scores = [(_token_score(vraag, z), z) for z in zinnen]
        scores.sort(key=lambda x: x[0], reverse=True)
        top = [z for s, z in scores if s > 0][:3]
        reactie = "\n".join(top) if top else "Ik kan zonder AI geen betere toelichting uit het gekozen antwoord halen."

    add_msg("assistant", with_info(reactie))
    st.rerun()


if __name__ == "__main__":
    main()
