# """
# IPAL Chatbox — main.py
# - Chat-wizard met 4 knoppen: Exact | DocBase | Zoeken | Internet
# - Klassieke cascade via expander
# - PDF met banner/logo over de volle contentbreedte (hoogte schaalt)
# - CSV-robustheid + smart quotes fix + werkende “Kopieer antwoord”
# - AI standaard AAN + “dit snap ik niet” → eenvoudige uitleg + stappenplan
# """

import os
import re
import io
import json
import hashlib
import logging
from datetime import datetime
from typing import Optional, List

import streamlit as st
import pandas as pd
import pytz
from dotenv import load_dotenv
from openai import OpenAI
import streamlit.components.v1 as components

# Web-fallback
import requests
from bs4 import BeautifulSoup

try:
    from openai.error import RateLimitError
except ImportError:
    RateLimitError = Exception

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# PDF
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, ListFlowable, ListItem
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


# ── Smart punctuation / Windows-1252 opschonen ──────────────────────────────
def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u00A0", " ")
    repl = {
        "\u0091": "'", "\u0092": "'", "\u0093": '"', "\u0094": '"',
        "\u0096": "-", "\u0097": "-", "\u0085": "...",
        "\u2018": "'", "\u2019": "'", "\u201A": ",", "\u201B": "'",
        "\u201C": '"', "\u201D": '"', "\u201E": '"',
        "\u00AB": '"', "\u00BB": '"', "\u2039": "'", "\u203A": "'",
        "\u2013": "-", "\u2014": "-", "\u2212": "-",
        "\u00AD": "", "\u2026": "...",
        "\u00B4": "'", "\u02BC": "'", "\u02BB": "'",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ── PDF/AI-INFO constants ───────────────────────────────────────────────────
FAQ_LINKS = [
    ("Veelgestelde vragen DocBase nieuw 2024", "https://parochie-automatisering.nl/docbase/Templates/docbase?action=SelOpenDocument&DetailsMode=2&Docname=00328526&Type=INSTR_DOCS&LoginMode=1&LinkToVersion=1&OpenFileMode=2&password=%3Auzt7hs%23qL%2A%28&username=Externehyperlink&ID=0.07961651005089099&EC=1"),
    ("Veelgestelde vragen Exact Online", "https://parochie-automatisering.nl/docbase/Templates/docbase?action=SelOpenDocument&DetailsMode=2&Docname=00328522&Type=INSTR_DOCS&LoginMode=1&LinkToVersion=1&OpenFileMode=2&password=%3Auzt7hs%23qL%2A%28&username=Externehyperlink&ID=0.8756321684738348&EC=1"),
]

AI_INFO = """
AI-Antwoord Info:  
1. Dit is het AI-antwoord vanuit de IPAL chatbox van het Interdiocesaan Platform Automatisering & Ledenadministratie. Het is altijd een goed idee om de meest recente informatie te controleren via officiële bronnen.  
2. Heeft u hulp nodig met DocBase of Exact? Dan kunt u eenvoudig een melding maken door een ticket aan te maken in DocBase. Maar voordat u een ticket invult, controleer eerst onze FAQ (veelgestelde vragen en antwoorden). Klik hieronder om de FAQ te openen:

- [Veelgestelde vragen DocBase nieuw 2024](https://parochie-automatisering.nl/docbase/Templates/docbase?action=SelOpenDocument&DetailsMode=2&Docname=00328526&Type=INSTR_DOCS&LoginMode=1&LinkToVersion=1&OpenFileMode=2&password=%3Auzt7hs%23qL%2A%28&username=Externehyperlink&ID=0.07961651005089099&EC=1)
- [Veelgestelde vragen Exact Online](https://parochie-automatisering.nl/docbase/Templates/docbase?action=SelOpenDocument&DetailsMode=2&Docname=00328522&Type=INSTR_DOCS&LoginMode=1&LinkToVersion=1&OpenFileMode=2&password=%3Auzt7hs%23qL%2A%28&username=Externehyperlink&ID=0.8756321684738348&EC=1)
"""


# ── PDF helpers ──────────────────────────────────────────────────────────────
if os.path.exists("Calibri.ttf"):
    pdfmetrics.registerFont(TTFont("Calibri", "Calibri.ttf"))

def _strip_md(s: str) -> str:
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"#+\s*([^\n]+)", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)
    return s

def _parse_ai_info(ai_info: str) -> tuple[list[str], bool]:
    numbered: list[str] = []
    show_click = False
    for raw in (ai_info or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("- ["):
            break
        if line.lower().startswith("ai-antwoord info"):
            continue
        m = re.match(r"^(\d+)\.\s*(.*)$", line)
        if m:
            txt = m.group(2)
            key = "Klik hieronder om de FAQ te openen"
            if key in txt:
                txt = txt.split(key, 1)[0].strip()
                show_click = True
            txt = clean_text(_strip_md(txt))
            if txt:
                numbered.append(txt)
        else:
            if numbered:
                extra = clean_text(_strip_md(line))
                if extra:
                    numbered[-1] += " + " + extra
    if not show_click and "Klik hieronder om de FAQ te openen" in (ai_info or ""):
        show_click = True
    return numbered, show_click

def make_pdf(question: str, answer: str) -> bytes:
    question = clean_text(question or "")
    answer   = clean_text(_strip_md(answer or ""))

    numbered_items, show_click = _parse_ai_info(AI_INFO)

    buffer = io.BytesIO()
    left = right = top = bottom = 2 * cm
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=left, rightMargin=right, topMargin=top, bottomMargin=bottom
    )
    content_width = A4[0] - left - right

    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontName="Helvetica",
        fontSize=11, leading=16, spaceAfter=12, alignment=TA_LEFT
    )
    heading_style = ParagraphStyle(
        "Heading", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=14, leading=18, textColor=colors.HexColor("#333"),
        spaceBefore=12, spaceAfter=6
    )

    story = []
    if os.path.exists("logopdf.png"):
        try:
            banner = Image("logopdf.png")
            banner._restrictSize(content_width, 10000)  # breedte = content_width; hoogte schaalt
            banner.hAlign = "LEFT"
            story.append(banner); story.append(Spacer(1, 8))
        except Exception as e:
            logging.error(f"Kon banner niet laden: {e}")

    story.append(Paragraph(f"Vraag: {question}", heading_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Antwoord:", heading_style))
    for line in (answer.split("\n") if answer else []):
        line = line.strip()
        if line:
            story.append(Paragraph(line, body_style))

    if numbered_items:
        story.append(Spacer(1, 12))
        story.append(Paragraph("AI-Antwoord Info:", heading_style))
        list_items = [ListItem(Paragraph(item, body_style), leftIndent=12) for item in numbered_items]
        story.append(ListFlowable(list_items, bulletType="1"))

    story.append(Spacer(1, 12))
    if show_click:
        story.append(Paragraph("Klik hieronder om de FAQ te openen:", heading_style))
    link_items = []
    for label, url in FAQ_LINKS:
        p = Paragraph(f'<link href="{url}" color="blue">{clean_text(label)}</link>', body_style)
        link_items.append(ListItem(p, leftIndent=12))
    story.append(ListFlowable(link_items, bulletType="bullet"))

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

    norm_cols = ["Systeem","Subthema","Categorie","Omschrijving melding","Toelichting melding","Soort melding","Antwoord of oplossing","Afbeelding"]
    for c in norm_cols:
        df[c] = (df[c].fillna("").astype(str)
                 .str.replace("\u00A0"," ", regex=False)
                 .str.strip()
                 .str.replace(r"\s+"," ", regex=True))
        df[c] = df[c].apply(clean_text)

    mapping = {"exact":"Exact","docbase":"DocBase","algemeen":"Algemeen"}
    df["Systeem"] = df["Systeem"].str.lower().map(mapping).fillna(df["Systeem"]).astype(str)

    keep = ["Systeem","Subthema","Categorie","Omschrijving melding","Toelichting melding"]
    df["combined"] = df[keep].fillna("").agg(" ".join, axis=1)

    return df.set_index(["Systeem","Subthema","Categorie"], drop=True)

faq_df = load_faq()


# ── Cascade helpers ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def list_toelichtingen(systeem: str, subthema: str, categorie: Optional[str]) -> List[str]:
    try:
        if not categorie or str(categorie).lower() == "alles":
            scope = faq_df.xs((systeem, subthema), level=["Systeem","Subthema"], drop_level=False)
        else:
            scope = faq_df.xs((systeem, subthema, categorie), level=["Systeem","Subthema","Categorie"], drop_level=False)
        vals = (scope["Toelichting melding"].dropna().astype(str).apply(clean_text).unique())
        return sorted(vals)
    except Exception:
        return []


# ── Veiligheidsfilter & relevance helpers ────────────────────────────────────
BLACKLIST = ["persoonlijke gegevens","medische gegevens","gezondheid","privacy schending"]

def filter_topics(msg: str):
    found = [t for t in BLACKLIST if re.search(r"\b"+re.escape(t)+r"\b", (msg or "").lower())]
    return (False, f"Je bericht bevat gevoelige onderwerpen: {', '.join(found)}.") if found else (True, "")

STOPWORDS_NL = {
    "de","het","een","en","of","maar","want","dus","als","dan","dat","die","dit","deze",
    "ik","jij","hij","zij","wij","jullie","u","ze","je","mijn","jouw","zijn","haar","ons","hun",
    "van","voor","naar","met","bij","op","in","aan","om","tot","uit","over","onder","boven","zonder",
    "ook","nog","al","wel","niet","nooit","altijd","hier","daar","ergens","niets","iets","alles",
    "is","was","wordt","zijn","heeft","heb","hebben","doe","doet","doen","kan","kunnen","moet","moeten"
}

def _tokenize_clean(text: str) -> list[str]:
    return [w for w in re.findall(r"[0-9A-Za-zÀ-ÿ_]+", (text or "").lower())
            if len(w) > 2 and w not in STOPWORDS_NL]

def _relevance(q: str, t: str) -> tuple[int, float]:
    qs = set(_tokenize_clean(q)); ts = set(_tokenize_clean(t))
    hits = len(qs & ts); coverage = hits / max(1, len(qs))
    return hits, coverage

def _token_score(q: str, text: str) -> int:
    qs = set(_tokenize_clean(q)); ts = set(_tokenize_clean(text))
    return len(qs & ts)


# ── Clarify intent helpers ───────────────────────────────────────────────────
CLARIFY_TRIGGERS = [
    "dit snap ik niet","ik snap het niet","onduidelijk","leg uit","uitleg",
    "begrijp ik niet","simpel uitleggen","simpel uitleg","in eenvoudige woorden",
    "stappenplan","kan je dit uitleggen","ik begrijp dit niet","wat bedoel je",
    "kunt u dit uitleggen","kunt u het uitleggen","kan je het uitleggen"
]

def is_clarify(msg: str) -> bool:
    t = (msg or "").strip().lower()
    return any(k in t for k in CLARIFY_TRIGGERS)

def simplify_text(text: str) -> str:
    """Zonder AI: korte, eenvoudige uitleg + stappenplan vanuit bron."""
    txt = clean_text(text or "")
    if not txt:
        return "Er staat geen aanvullende detailuitleg in dit item."
    zinnen = re.split(r"(?<=[.!?])\s+", txt)
    bullets = []
    for z in zinnen:
        z = re.sub(r"\([^)]*\)", "", z)
        z = re.sub(r"\s+", " ", z).strip()
        if 0 < len(z) <= 160:
            bullets.append(f"- {z}")
        if len(bullets) >= 5:
            break
    if not bullets:
        bullets = ["- In het kort: " + (txt[:150] + ("…" if len(txt) > 150 else ""))]
    stappen = []
    for z in zinnen:
        if any(x in z.lower() for x in ["klik","ga naar","open","selecteer","vul","druk","kies","bewaar","opslaan"]):
            stappen.append(z.strip())
        if len(stappen) >= 6:
            break
    if not stappen:
        stappen = [
            "Open de juiste functie/pagina.",
            "Volg de aanwijzingen in het scherm voor dit onderwerp.",
            "Controleer of het resultaat is wat u verwacht."
        ]
    stappen_md = "\n".join([f"{i+1}. {s}" for i, s in enumerate(stappen)])
    return f"**In eenvoudige woorden:**\n\n" + "\n".join(bullets) + "\n\n**Stappenplan:**\n" + stappen_md


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
    if faq_df.empty:
        return pd.DataFrame()
    df = faq_df.reset_index().copy()
    q_tokens = set(_tokenize_clean(vraag))
    eff_min_hits = max(1, min(min_hits, len(q_tokens)))
    df["_score"] = df["combined"].apply(lambda t: _token_score(vraag, t))
    df = df.sort_values("_score", ascending=False)
    if df.empty:
        return df
    def _ok(row):
        hits, cov = _relevance(vraag, str(row["combined"]))
        return hits >= eff_min_hits and cov >= min_cov
    filtered = df[df.apply(_ok, axis=1)]
    if filtered.empty:
        q_lower = (vraag or "").strip().lower()
        sys_map = {"exact":"Exact","docbase":"DocBase","algemeen":"Algemeen"}
        if q_lower in sys_map:
            sys = sys_map[q_lower]
            try:
                subset = faq_df.xs(sys, level="Systeem", drop_level=False).reset_index()
                subset["_score"] = subset["combined"].apply(lambda t: _token_score(vraag, t))
                return subset.sort_values("_score", ascending=False).head(fallback_rows)
            except KeyError:
                pass
        nonzero = df[df["_score"] > 0].head(fallback_rows)
        return nonzero if not nonzero.empty else df.head(fallback_rows)
    return filtered

def zoek_in_scope(scope: Optional[str], vraag: str, topn: int = 8) -> pd.DataFrame:
    base = faq_df.reset_index()
    if scope in ("Exact","DocBase"):
        base = base[base["Systeem"].astype(str).str.lower() == scope.lower()]
    if base.empty:
        return base
    base = base.copy()
    base["_score"] = base["combined"].apply(lambda t: _token_score(vraag, t))
    base = base.sort_values(["_score"], ascending=False)
    good = base[base["_score"] > 0]
    return (good if not good.empty else base).head(topn)

def vind_best_algemeen_AI(vraag: str) -> str:
    if client is None:
        return "Kunt u uw vraag iets concreter maken (bijv. ‘DocBase wachtwoord resetten’ of ‘Exact bankkoppeling’)?"
    sys = ("Je helpt vrijwilligers van parochies. "
           "Beantwoord kort en concreet. Stel maximaal één verhelderende vraag als dat echt nodig is. "
           "Noem geen niet-bestaande bronnen.")
    user = (f"Vraag: {vraag}\n\n"
            "Als de vraag niet direct over DocBase/Exact/IPAL-onderwerpen gaat, geef dan 1- of 2 praktische vervolgsuggesties "
            "of verwijs vriendelijk naar het juiste kanaal.")
    try:
        return chatgpt_cached(
            [{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0.2, max_tokens=500
        )
    except Exception as e:
        logging.error(f"AI (Internet) fout: {e}")
        return "Kunt u uw vraag iets concreter maken?"


# ── UI helpers & state ───────────────────────────────────────────────────────
TIMEZONE = pytz.timezone("Europe/Amsterdam")
ASSISTANT_AVATAR = "aichatbox.png" if os.path.exists("aichatbox.png") else None
USER_AVATAR = "parochie.png" if os.path.exists("parochie.png") else None

DEFAULT_STATE = {
    "history": [],
    "selected_product": None,
    "selected_module": None,
    "selected_category": None,
    "selected_toelichting": None,
    "selected_answer_id": None,
    "selected_answer_text": None,
    "selected_image": None,
    "last_question": "",
    "last_item_label": "",
    "debug": False,
    "allow_ai": True,          # <— AI standaard AAN
    "allow_web": False,
    "min_hits": 2,
    "min_cov": 0.25,
    "search_query": "",
    "search_selection_index": None,
    "last_processed_algemeen": "",
    # Conversatie-wizard
    "chat_mode": True,
    "chat_step": "greet",
    "chat_scope": None,        # "Exact" | "DocBase" | "Zoeken" | "Algemeen"
    "chat_results": [],
    "chat_greeted": False,
    "pdf_ready": False,
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

AI_INFO_MD = AI_INFO
def with_info(text: str) -> str:
    return clean_text((text or "").strip()) + "\n\n" + AI_INFO_MD

def _copy_button(text: str, key_suffix: str):
    payload = text or ""
    js_text = json.dumps(payload)
    html_code = """
<div style="margin-top:8px;">
  <button id="COPY_BTN_ID" style="padding:6px 10px;font-size:16px;">
    Kopieer antwoord
  </button>
  <span id="COPY_STATE_ID" style="margin-left:8px;font-size:14px;"></span>
  <script>
    (function(){
      const btn = document.getElementById('COPY_BTN_ID');
      const state = document.getElementById('COPY_STATE_ID');
      if (btn) {
        btn.addEventListener('click', async () => {
          try {
            await navigator.clipboard.writeText(JS_TEXT);
            state.textContent = 'Gekopieerd!';
            setTimeout(() => { state.textContent = ''; }, 1500);
          } catch (e) {
            state.textContent = 'Niet gelukt — gebruik de tekst hieronder.';
            setTimeout(() => { state.textContent = ''; }, 3000);
          }
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

def render_chat():
    for i, m in enumerate(st.session_state.history):
        st.chat_message(m["role"], avatar=get_avatar(m["role"]))\
            .markdown(f"{m['content']}\n\n_{m['time']}_")
        if m["role"] == "assistant" and i == len(st.session_state.history) - 1 and st.session_state.get("pdf_ready", False):
            q = (st.session_state.get("last_question") or st.session_state.get("last_item_label") or "Vraag")
            pdf = make_pdf(q, m["content"])
            btn_key = f"pdf_{i}_{m['time'].replace(':','-')}"
            st.download_button("📄 Download PDF", data=pdf, file_name="antwoord.pdf", mime="application/pdf", key=btn_key)
            hash_key = hashlib.md5((m["time"] + m["content"]).encode("utf-8")).hexdigest()[:8]
            _copy_button(m["content"], hash_key)
            img = st.session_state.get("selected_image")
            if img and isinstance(img, str) and img.strip():
                try:
                    st.image(img, caption="Afbeelding bij dit antwoord", use_column_width=True)
                except Exception:
                    pass


# ── Conversatie-wizard (échte chatbox) ───────────────────────────────────────
def _detect_scope(msg: str) -> Optional[str]:
    t = (msg or "").lower()
    if any(w in t for w in ["exact","eol","e-online","exact online"]): return "Exact"
    if any(w in t for w in ["docbase","doc base"]): return "DocBase"
    if any(w in t for w in ["csv","zoeken intern","zoek in csv","zoeken in csv","zoeken"]): return "Zoeken"
    if any(w in t for w in ["internet","web","algemeen","overig","anders","ik weet het niet"]): return "Algemeen"
    return None

def _mk_label(i: int, row: pd.Series) -> str:
    oms = clean_text(str(row.get('Omschrijving melding', '')).strip())
    toel = clean_text(str(row.get('Toelichting melding', '')).strip())
    preview = oms or toel or clean_text(str(row.get('Antwoord of oplossing', '')).strip())
    preview = re.sub(r"\s+", " ", preview)[:140]
    return f"{i+1:02d}. {preview}"

def chat_wizard():
    render_chat()

    # Snelkeuzes
    with st.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        if c1.button("Exact", key="wizard_exact", use_container_width=True):
            st.session_state.update({"chat_scope": "Exact", "chat_step": "ask_topic"})
            add_msg("assistant", "Prima. Kunt u in één zin beschrijven waar uw vraag over Exact Online over gaat?")
            st.session_state["pdf_ready"] = False; st.rerun()
        if c2.button("DocBase", key="wizard_docbase", use_container_width=True):
            st.session_state.update({"chat_scope": "DocBase", "chat_step": "ask_topic"})
            add_msg("assistant", "Dank u. Kunt u in één zin beschrijven waar uw vraag over DocBase over gaat?")
            st.session_state["pdf_ready"] = False; st.rerun()
        if c3.button("Zoeken", key="wizard_search", use_container_width=True):
            st.session_state.update({"chat_scope": "Zoeken", "chat_step": "ask_topic"})
            add_msg("assistant", "Waar wilt u in de CSV op zoeken? Typ een korte zoekterm.")
            st.session_state["pdf_ready"] = False; st.rerun()
        if c4.button("Internet", key="wizard_internet", use_container_width=True):
            st.session_state.update({"chat_scope": "Algemeen", "chat_step": "ask_topic"})
            add_msg("assistant", "Waarover gaat uw vraag? Beschrijf dit kort in één zin.")
            st.session_state["pdf_ready"] = False; st.rerun()
        if c5.button("🔄 Reset", key="wizard_reset", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            for k, v in DEFAULT_STATE.items():
                if k not in st.session_state:
                    st.session_state[k] = v
            try: st.cache_data.clear()
            except Exception: pass
            st.rerun()

    if not st.session_state.get("chat_greeted", False):
        add_msg("assistant", "👋 Waarmee kan ik u van dienst zijn? U kunt hieronder typen of een snelkeuze gebruiken (Exact, DocBase, Zoeken of Internet).")
        st.session_state["chat_greeted"] = True
        st.session_state["pdf_ready"] = False
        render_chat()

    step = st.session_state.get("chat_step", "greet")
    scope = st.session_state.get("chat_scope")
    placeholders = {
        "greet": "Typ uw bericht…",
        "ask_scope": "Gaat uw vraag over Exact, DocBase of iets anders?",
        "ask_topic": f"Uw vraag over {scope or '…'} in één zin…",
        "pick_item": "Kies een van de opties of stel uw vraag anders.",
        "followup": "Heeft u een vervolgvraag over dit antwoord?",
    }

    user_text = st.chat_input(placeholders.get(step, "Stel uw vraag…"))
    if not user_text:
        if step == "pick_item" and st.session_state.get("chat_results"):
            opts = st.session_state["chat_results"]
            labels = [_mk_label(i, pd.Series(r)) for i, r in enumerate(opts)]
            chosen = st.radio("Kies het beste passende item:", labels, index=0)
            if st.button("Toon antwoord", key="wizard_show_answer"):
                idx = labels.index(chosen); row = pd.Series(opts[idx])
                ans = clean_text(str(row.get('Antwoord of oplossing', '') or '').strip())
                if not ans:
                    oms = clean_text(str(row.get('Omschrijving melding', '') or '').strip())
                    ans = f"(Geen uitgewerkt antwoord in CSV voor: {oms})"
                label = _mk_label(idx, row)
                img = clean_text(str(row.get('Afbeelding', '') or '').strip())
                st.session_state["selected_image"] = img if img else None
                st.session_state["last_item_label"] = label
                st.session_state["last_question"] = f"Gekozen item: {label}"
                st.session_state["selected_answer_text"] = ans
                st.session_state["pdf_ready"] = True
                add_msg("assistant", with_info(ans))
                st.session_state["chat_step"] = "followup"; st.rerun()
        return

    add_msg("user", user_text)

    if (user_text or "").strip().upper() == "UNIEKECODE123":
        cw = find_answer_by_codeword(faq_df.reset_index())
        if cw:
            st.session_state["last_question"] = user_text
            st.session_state["pdf_ready"] = True
            add_msg("assistant", with_info(cw))
            st.session_state["chat_step"] = "followup"; st.rerun(); return

    ok, warn = filter_topics(user_text)
    if not ok:
        st.session_state["pdf_ready"] = False; add_msg("assistant", warn); st.rerun(); return

    if step in ("greet","ask_scope"):
        scope_guess = _detect_scope(user_text)
        if scope_guess is None and user_text.strip().lower() in ("ik heb een vraag","ik heb een vraag.","vraag","hallo","goedemiddag"):
            st.session_state["chat_step"] = "ask_scope"; st.session_state["pdf_ready"] = False
            add_msg("assistant","Gaat uw vraag over **Exact**, **DocBase** of **iets anders**?"); st.rerun(); return
        if scope_guess is None:
            hits = zoek_in_scope(None, user_text, topn=6)
            if not hits.empty:
                st.session_state["chat_scope"] = "Zoeken"
                st.session_state["chat_results"] = hits.to_dict("records")
                st.session_state["chat_step"] = "pick_item"
                st.session_state["pdf_ready"] = False
                add_msg("assistant","Ik heb een aantal mogelijke matches gevonden in onze CSV. Kies er één hieronder.")
                st.rerun(); return
            else:
                st.session_state["chat_scope"] = "Algemeen"
                st.session_state["chat_step"] = "ask_topic"
                st.session_state["pdf_ready"] = False
                add_msg("assistant","Kunt u in één zin beschrijven waar uw internetvraag over gaat?")
                st.rerun(); return
        else:
            st.session_state["chat_scope"] = scope_guess
            st.session_state["chat_step"] = "ask_topic"
            st.session_state["pdf_ready"] = False
            add_msg("assistant", f"Prima. Kunt u in één zin beschrijven waar uw vraag over **{scope_guess}** over gaat?")
            st.rerun(); return

    if step == "ask_topic":
        scope = st.session_state.get("chat_scope")
        if scope == "Algemeen":
            st.session_state["last_question"] = user_text
            antwoord = vind_best_algemeen_AI(user_text)
            if not antwoord and st.session_state.get("allow_web"):
                webbits = fetch_web_info_cached(user_text)
                if webbits: antwoord = webbits
            st.session_state["pdf_ready"] = True
            add_msg("assistant", with_info(antwoord or "Kunt u uw vraag iets concreter maken?"))
            st.session_state["chat_step"] = "followup"; st.rerun(); return
        else:
            st.session_state["last_question"] = user_text
            hits = zoek_in_scope(None if scope == "Zoeken" else scope, user_text, topn=8)
            if hits.empty:
                st.session_state["pdf_ready"] = False
                add_msg("assistant","Ik vond geen goede match in de CSV. Formuleer het iets anders of kies **Internet**.")
                st.rerun(); return
            st.session_state["chat_results"] = hits.to_dict("records")
            st.session_state["chat_step"] = "pick_item"
            st.session_state["pdf_ready"] = False
            add_msg("assistant","Ik heb een aantal mogelijke matches gevonden. Kies hieronder het beste passende item.")
            st.rerun(); return

    if step == "pick_item":
        m = re.search(r"\b(\d{1,2})\b", user_text)
        if m and st.session_state.get("chat_results"):
            idx = int(m.group(1)) - 1; opts = st.session_state["chat_results"]
            if 0 <= idx < len(opts):
                row = pd.Series(opts[idx])
                ans = clean_text(str(row.get('Antwoord of oplossing', '') or '').strip())
                if not ans:
                    oms = clean_text(str(row.get('Omschrijving melding', '') or '').strip())
                    ans = f"(Geen uitgewerkt antwoord in CSV voor: {oms})"
                label = _mk_label(idx, row)
                img = clean_text(str(row.get('Afbeelding', '') or '').strip())
                st.session_state["selected_image"] = img if img else None
                st.session_state["last_item_label"] = label
                st.session_state["last_question"] = f"Gekozen item: {label}"
                st.session_state["selected_answer_text"] = ans
                st.session_state["pdf_ready"] = True
                add_msg("assistant", with_info(ans))
                st.session_state["chat_step"] = "followup"; st.rerun(); return
        st.session_state["pdf_ready"] = False
        add_msg("assistant","Gebruik de selectie hierboven om een item te kiezen, of typ het nummer (bijv. 2).")
        st.rerun(); return

    if step == "followup":
        vraag2 = user_text
        st.session_state["last_question"] = vraag2

        # Verduidelijkings-intent: eenvoudige uitleg + stappenplan
        bron = str(st.session_state.get("selected_answer_text") or "").strip()
        if is_clarify(vraag2):
            reactie = None
            if client is not None:
                try:
                    reactie = chatgpt_cached(
                        [
                            {"role":"system","content":
                             "Leg in eenvoudige Nederlandse woorden uit voor een vrijwilliger zonder technische kennis. "
                             "Gebruik maximaal 5 bullets en, indien nuttig, een kort stappenplan. "
                             "Baseer ALLES uitsluitend op de gegeven bron; geen aannames."},
                            {"role":"user","content": f"Bron:\n{bron}\n\nMaak het simpel en concreet; voeg een stappenplan toe als dat helpt."}
                        ],
                        temperature=0.2, max_tokens=600
                    )
                except Exception:
                    reactie = None
            if not reactie:
                reactie = simplify_text(bron)
            st.session_state["pdf_ready"] = True
            add_msg("assistant", with_info(reactie))
            st.rerun(); return

        ok, warn = filter_topics(vraag2)
        if not ok:
            st.session_state["pdf_ready"] = False; add_msg("assistant", warn); st.rerun(); return

        reactie = None
        if st.session_state.get("allow_ai") and client is not None and bron:
            try:
                reactie = chatgpt_cached(
                    [
                        {"role":"system","content":"Beantwoord uitsluitend op basis van de meegegeven bron. Geen aannames buiten de bron. Schrijf kort en duidelijk in het Nederlands."},
                        {"role":"user","content":f"Bron:\n{bron}\n\nVraag: {vraag2}"}
                    ],
                    temperature=0.1, max_tokens=600,
                )
            except Exception as e:
                logging.error(f"AI-QA fout: {e}")
                reactie = None
        if not reactie:
            zinnen = re.split(r"(?<=[.!?])\s+", bron)
            scores = [(_token_score(vraag2, z), z) for z in zinnen]
            scores.sort(key=lambda x: x[0], reverse=True)
            top = [z for s, z in scores if s > 0][:3]
            reactie = "\n".join(top) if top else "Ik kan zonder AI geen betere toelichting uit het gekozen antwoord halen."
        st.session_state["pdf_ready"] = True
        add_msg("assistant", with_info(reactie))
        st.rerun(); return


# ── App ──────────────────────────────────────────────────────────────────────
def main():
    # Intro (video of logo)
    video_path = "helpdesk.mp4"
    if os.path.exists(video_path):
        try:
            with open(video_path, "rb") as f: st.video(f.read(), format="video/mp4", start_time=0)
        except Exception as e:
            logging.error(f"Introvideo kon niet worden afgespeeld: {e}")
    elif os.path.exists("logo.png"):
        st.image("logo.png", width=244)
    else:
        st.info("Welkom bij IPAL Chatbox")

    st.header("Welkom bij IPAL Chatbox")

    # Expander: klassieke cascade starten (optioneel)
    with st.expander("Liever de klassieke cascade openen?"):
        keuze = st.radio("Kies cascade:", ["Exact","DocBase","Zoeken","Internet"], horizontal=True, index=0, key="cascade_radio")
        if st.button("Start cascade", use_container_width=True, key="cascade_start"):
            if keuze == "Exact":
                st.session_state.update({
                    "chat_mode": False, "selected_product": "Exact",
                    "selected_image": None, "selected_module": None, "selected_category": None,
                    "selected_toelichting": None, "selected_answer_id": None, "selected_answer_text": None,
                    "last_item_label": "", "last_question": ""
                })
            elif keuze == "DocBase":
                st.session_state.update({
                    "chat_mode": False, "selected_product": "DocBase",
                    "selected_image": None, "selected_module": None, "selected_category": None,
                    "selected_toelichting": None, "selected_answer_id": None, "selected_answer_text": None,
                    "last_item_label": "", "last_question": ""
                })
            elif keuze == "Zoeken":
                st.session_state.update({
                    "chat_mode": False, "selected_product": "Zoeken",
                    "selected_image": None, "search_query": "", "search_selection_index": None,
                    "selected_answer_id": None, "selected_answer_text": None,
                    "last_item_label": "", "last_question": ""
                })
            else:
                st.session_state.update({
                    "chat_mode": False, "selected_product": "Algemeen",
                    "selected_image": None, "selected_module": None, "selected_category": None,
                    "selected_toelichting": None, "selected_answer_id": None, "selected_answer_text": None,
                    "last_item_label": "", "last_question": ""
                })
            st.rerun()

    # Wizard modus
    if st.session_state.get("chat_mode", True):
        chat_wizard(); return

    # ------ Klassieke flows ------
    render_chat()

    # ALGEMEEN (Internet)
    if st.session_state.get("selected_product") == "Algemeen":
        st.caption("Stel hier uw internetvraag (niet direct onder DocBase of Exact Online):")
        algemeen_vraag = st.text_input(" ", placeholder="Stel uw internetvraag:", key="algemeen_top_input", label_visibility="collapsed")
        last = st.session_state.get("last_processed_algemeen", "")
        if not algemeen_vraag or algemeen_vraag == last:
            return
        if (algemeen_vraag or "").strip().upper() == "UNIEKECODE123":
            cw = find_answer_by_codeword(faq_df.reset_index())
            if cw:
                st.session_state["last_question"] = algemeen_vraag
                st.session_state["last_processed_algemeen"] = algemeen_vraag
                add_msg("user", algemeen_vraag)
                st.session_state["pdf_ready"] = True
                add_msg("assistant", with_info(cw)); st.rerun(); return
        st.session_state["last_processed_algemeen"] = algemeen_vraag
        st.session_state["last_question"] = algemeen_vraag
        add_msg("user", algemeen_vraag)
        ok, warn = filter_topics(algemeen_vraag)
        if not ok:
            st.session_state["pdf_ready"] = False; add_msg("assistant", warn); st.rerun(); return
        antwoord = vind_best_algemeen_AI(algemeen_vraag)
        if not antwoord and st.session_state.get("allow_web"):
            webbits = fetch_web_info_cached(algemeen_vraag)
            if webbits: antwoord = webbits
        st.session_state["pdf_ready"] = True; add_msg("assistant", with_info(antwoord or "Kunt u uw vraag iets concreter maken?")); st.rerun(); return

    # ZOEKEN (hele CSV)
    if st.session_state.get("selected_product") == "Zoeken":
        st.session_state["search_query"] = st.text_input("Waar wil je in de volledige CSV op zoeken?", value=st.session_state.get("search_query",""))
        q = st.session_state["search_query"].strip()
        if not q: return
        results = zoek_hele_csv(q, min_hits=st.session_state["min_hits"], min_cov=st.session_state["min_cov"])
        st.caption(f"Gevonden resultaten: {len(results)}")
        if results.empty:
            st.info("Geen resultaten gevonden. Pas je zoekterm aan of verlaag de drempels (Geavanceerd)."); return
        df_reset = results.reset_index(drop=True)
        def mk_label(i, row):
            oms = clean_text(str(row.get('Omschrijving melding','')).strip())
            toel = clean_text(str(row.get('Toelichting melding','')).strip())
            preview = oms or toel or clean_text(str(row.get('Antwoord of oplossing','')).strip())
            preview = re.sub(r"\s+"," ", preview)[:140]
            return f"{i+1:02d}. {preview}"
        opties = [mk_label(i, r) for i, r in df_reset.iterrows()]
        keuze = st.selectbox("Kies een item uit de zoekresultaten:", ["(Kies)"] + opties)
        if keuze == "(Kies)": return
        idx = int(keuze.split(".")[0]) - 1
        row = df_reset.iloc[idx]; row_id = row.get("ID", idx)
        ans = clean_text(str(row.get('Antwoord of oplossing','') or '').strip())
        if not ans:
            oms = clean_text(str(row.get('Omschrijving melding','') or '').strip())
            ans = f"(Geen uitgewerkt antwoord in CSV voor: {oms})"
        label = mk_label(idx, row)
        img = clean_text(str(row.get('Afbeelding','') or '').strip())
        st.session_state["selected_image"] = img if img else None
        if st.session_state.get("selected_answer_id") != row_id:
            st.session_state["selected_answer_id"] = row_id
            st.session_state["selected_answer_text"] = ans
            st.session_state["last_item_label"] = label
            st.session_state["last_question"] = f"Gekozen item: {label}"
            st.session_state["pdf_ready"] = True
            add_msg("assistant", with_info(ans)); st.rerun(); return

        vraag2 = st.chat_input("Stel uw vraag over dit antwoord:")
        if not vraag2: return
        st.session_state["last_question"] = vraag2; add_msg("user", vraag2)

        bron = str(st.session_state.get("selected_answer_text") or "").strip()
        if is_clarify(vraag2):
            reactie = None
            if client is not None:
                try:
                    reactie = chatgpt_cached(
                        [
                            {"role":"system","content":
                             "Leg in eenvoudige Nederlandse woorden uit voor een vrijwilliger zonder technische kennis. "
                             "Gebruik maximaal 5 bullets en, indien nuttig, een kort stappenplan. "
                             "Baseer ALLES uitsluitend op de gegeven bron; geen aannames."},
                            {"role":"user","content": f"Bron:\n{bron}\n\nMaak het simpel en concreet; voeg een stappenplan toe als dat helpt."}
                        ],
                        temperature=0.2, max_tokens=600
                    )
                except Exception:
                    reactie = None
            if not reactie:
                reactie = simplify_text(bron)
            st.session_state["pdf_ready"] = True; add_msg("assistant", with_info(reactie)); st.rerun(); return

        ok, warn = filter_topics(vraag2)
        if not ok:
            st.session_state["pdf_ready"] = False; add_msg("assistant", warn); st.rerun(); return

        reactie = None
        if st.session_state.get("allow_ai") and client is not None:
            try:
                reactie = chatgpt_cached(
                    [
                        {"role":"system","content":"Beantwoord uitsluitend op basis van de meegegeven bron. Geen aannames buiten de bron. Schrijf kort en duidelijk in het Nederlands."},
                        {"role":"user","content":f"Bron:\n{bron}\n\nVraag: {vraag2}"}
                    ],
                    temperature=0.1, max_tokens=600,
                )
            except Exception as e:
                logging.error(f"AI-QA fout: {e}")
                reactie = None
        if not reactie:
            zinnen = re.split(r"(?<=[.!?])\s+", bron)
            scores = [(_token_score(vraag2, z), z) for z in zinnen]
            scores.sort(key=lambda x: x[0], reverse=True)
            top = [z for s, z in scores if s > 0][:3]
            reactie = "\n".join(top) if top else "Ik kan zonder AI geen betere toelichting uit het gekozen antwoord halen."
        st.session_state["pdf_ready"] = True; add_msg("assistant", with_info(reactie)); st.rerun(); return

    # Exact/DocBase (klassiek)
    syst = st.session_state.get("selected_product")
    sub  = st.session_state.get("selected_module") or ""
    cat  = st.session_state.get("selected_category") or ""
    toe  = st.session_state.get("selected_toelichting") or ""
    parts = [p for p in [syst, sub, (None if cat in ("", None, "alles") else cat), (toe or None)] if p]
    if parts: st.caption(" › ".join(parts))

    if not st.session_state.get("selected_module"):
        try: opts = sorted(faq_df.xs(syst, level="Systeem").index.get_level_values("Subthema").dropna().unique())
        except Exception: opts = []
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

    if not st.session_state.get("selected_category"):
        try:
            cats = sorted(
                faq_df.xs((syst, st.session_state["selected_module"]), level=["Systeem","Subthema"], drop_level=False)
                .index.get_level_values("Categorie").dropna().unique()
            )
        except Exception:
            cats = []
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

    if st.session_state.get("selected_toelichting") is None:
        toes = list_toelichtingen(syst, st.session_state["selected_module"], st.session_state.get("selected_category"))
        if len(toes) == 0:
            st.info("Geen toelichtingen gevonden — stap wordt overgeslagen.")
            st.session_state["selected_toelichting"] = ""
        else:
            toe_sel = st.selectbox("Kies toelichting:", ["(Kies)"] + list(toes))
            if toe_sel != "(Kies)":
                st.session_state["selected_toelichting"] = toe_sel
                st.session_state["selected_answer_id"] = None
                st.session_state["selected_answer_text"] = None
                st.session_state["selected_image"] = None
                st.toast(f"Gekozen toelichting: {toe_sel}")
                st.rerun()
            return

    df_scope = faq_df
    try:
        df_scope = df_scope.xs(syst, level="Systeem", drop_level=False)
        df_scope = df_scope.xs(sub, level="Subthema", drop_level=False)
        if cat and str(cat).lower() != "alles":
            df_scope = df_scope.xs(cat, level="Categorie", drop_level=False)
    except KeyError:
        df_scope = pd.DataFrame(columns=faq_df.reset_index().columns)

    if not df_scope.empty and toe is not None and str(toe) != "":
        tm = (df_scope["Toelichting melding"].astype(str).apply(clean_text))
        sel = clean_text(str(toe))
        df_scope = df_scope[tm == sel]

    if df_scope.empty:
        st.info("Geen records gevonden binnen de gekozen Systeem/Subthema/Categorie/Toelichting.")
        return

    df_reset = df_scope.reset_index()
    def mk_label(i, row):
        oms = clean_text(str(row.get('Omschrijving melding','')).strip())
        toel = clean_text(str(row.get('Toelichting melding','')).strip())
        preview = oms or toel or clean_text(str(row.get('Antwoord of oplossing','')).strip())
        preview = re.sub(r"\s+"," ", preview)[:140]
        return f"{i+1:02d}. {preview}"
    opties = [mk_label(i, r) for i, r in df_reset.iterrows()]
    keuze = st.selectbox("Kies een item:", ["(Kies)"] + opties)
    if keuze != "(Kies)":
        i = int(keuze.split(".")[0]) - 1
        row = df_reset.iloc[i]; row_id = row.get("ID", i)
        ans = clean_text(str(row.get('Antwoord of oplossing','') or '').strip())
        if not ans:
            oms = clean_text(str(row.get('Omschrijving melding','') or '').strip())
            ans = f"(Geen uitgewerkt antwoord in CSV voor: {oms})"
        label = mk_label(i, row)
        img = clean_text(str(row.get('Afbeelding','') or '').strip())
        st.session_state["selected_image"] = img if img else None
        if st.session_state.get("selected_answer_id") != row_id:
            st.session_state["selected_answer_id"] = row_id
            st.session_state["selected_answer_text"] = ans
            st.session_state["last_item_label"] = label
            st.session_state["last_question"] = f"Gekozen item: {label}"
            st.session_state["pdf_ready"] = True
            add_msg("assistant", with_info(ans)); st.rerun(); return

    vraag = st.chat_input("Stel uw vraag over dit antwoord:")
    if not vraag: return
    if (vraag or "").strip().upper() == "UNIEKECODE123":
        cw = find_answer_by_codeword(faq_df.reset_index())
        if cw:
            st.session_state["last_question"] = vraag
            add_msg("user", vraag)
            st.session_state["pdf_ready"] = True
            add_msg("assistant", with_info(cw)); st.rerun(); return

    st.session_state["last_question"] = vraag
    add_msg("user", vraag)
    ok, warn = filter_topics(vraag)
    if not ok:
        st.session_state["pdf_ready"] = False; add_msg("assistant", warn); st.rerun(); return

    bron = str(st.session_state.get("selected_answer_text") or "").strip()
    if is_clarify(vraag):
        reactie = None
        if client is not None:
            try:
                reactie = chatgpt_cached(
                    [
                        {"role":"system","content":
                         "Leg in eenvoudige Nederlandse woorden uit voor een vrijwilliger zonder technische kennis. "
                         "Gebruik maximaal 5 bullets en, indien nuttig, een kort stappenplan. "
                         "Baseer ALLES uitsluitend op de gegeven bron; geen aannames."},
                        {"role":"user","content": f"Bron:\n{bron}\n\nMaak het simpel en concreet; voeg een stappenplan toe als dat helpt."}
                    ],
                    temperature=0.2, max_tokens=600
                )
            except Exception:
                reactie = None
        if not reactie:
            reactie = simplify_text(bron)
        st.session_state["pdf_ready"] = True
        add_msg("assistant", with_info(reactie)); st.rerun(); return

    reactie = None
    if st.session_state.get("allow_ai") and client is not None:
        try:
            reactie = chatgpt_cached(
                [
                    {"role":"system","content":"Beantwoord uitsluitend op basis van de meegegeven bron. Geen aannames buiten de bron. Schrijf kort en duidelijk in het Nederlands."},
                    {"role":"user","content": f"Bron:\n{bron}\n\nVraag: {vraag}"}
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
    st.session_state["pdf_ready"] = True
    add_msg("assistant", with_info(reactie))
    st.rerun()


if __name__ == "__main__":
    main()
