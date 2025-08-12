"""
IPAL Chatbox — Definitieve main.py
- Cascade: Exact/DocBase → Subthema → Categorie → (Toelichting) → item → vraag
- Algemeen: géén cascade, directe vraag (CSV + API, en optioneel Web)
- UNIEKECODE123, Web-fallback (toggle), FAQ-links in PDF
- CSV-robustheid: trim, NBSP→spatie, multi-spaces→één, casefold-matches
- Geen chatspam (selecties via st.toast)
- Patches: altijd antwoord + altijd PDF-knop (met unieke key)
- Extra: Breadcrumbs, Kopieer-antwoord knop, Afbeelding tonen, Scope teller
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
    s = re.sub(r"\*\*([^*]+)\*\*", lambda m: m.group(1), s)          # **bold** → plain
    s = re.sub(r"#+\s*([^\n]+)", lambda m: m.group(1), s)            # # heading → plain
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: m.group(1), s)  # [label](url) → label
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

    # Normalize: strip + NBSP→space + collapse spaces
    norm_cols = ["Systeem","Subthema","Categorie","Omschrijving melding","Toelichting melding","Soort melding"]
    for c in norm_cols:
        df[c] = (df[c]
                 .fillna("")
                 .astype(str)
                 .str.replace("\u00A0", " ", regex=False)   # NBSP
                 .str.strip()
                 .str.replace(r"\s+", " ", regex=True))     # meerdere spaties/linebreaks → 1 spatie

    # Standaardiseer Systeem
    mapping = {"exact": "Exact", "docbase": "DocBase", "algemeen": "Algemeen"}
    df["Systeem"] = df["Systeem"].str.lower().map(mapping).fillna(df["Systeem"]).astype(str)

    # Combined voor eenvoudige ranking
    keep = ["Systeem","Subthema","Categorie","Omschrijving melding","Toelichting melding"]
    df["combined"] = df[keep].fillna("").agg(" ".join, axis=1)

    return df.set_index(["Systeem","Subthema","Categorie"], drop=True)

faq_df = load_faq()
PRODUCTEN = ["Exact","DocBase","Algemeen"]


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


# ── Veiligheidsfilter & ranking ──────────────────────────────────────────────
BLACKLIST = ["persoonlijke gegevens","medische gegevens","gezondheid","privacy schending"]

def filter_topics(msg: str):
    found = [t for t in BLACKLIST if re.search(r"\b" + re.escape(t) + r"\b", (msg or "").lower())]
    return (False, f"Je bericht bevat gevoelige onderwerpen: {', '.join(found)}.") if found else (True, "")

def _token_score(q: str, text: str) -> int:
    qs = [w for w in re.findall(r"\w+", (q or "").lower()) if len(w) > 2]
    ts = set(re.findall(r"\w+", str(text).lower()))
    return sum(1 for w in qs if w in ts)


# ── Zoekfuncties ─────────────────────────────────────────────────────────────
def find_answer_by_codeword(df: pd.DataFrame, codeword: str = "[UNIEKECODE123]") -> Optional[str]:
    try:
        mask = df["Antwoord of oplossing"].astype(str).str.contains(codeword, case=False, na=False)
        if mask.any():
            return str(df.loc[mask].iloc[0]["Antwoord of oplossing"]).strip()
    except Exception:
        pass
    return None

def vind_best_algemeen_antwoord(vraag: str) -> Optional[str]:
    try:
        if faq_df.empty:
            return None
        df_all = faq_df.reset_index().copy()
        vraag_norm = (vraag or "").strip().lower()

        # 1) UNIEKECODE123 direct
        if vraag_norm == "uniekecode123":
            cw = find_answer_by_codeword(df_all)
            if cw:
                return cw

        # 2) Exacte match op Omschrijving
        exact = df_all[df_all["Omschrijving melding"].astype(str).str.strip().str.lower() == vraag_norm]
        if not exact.empty:
            return str(exact.iloc[0].get("Antwoord of oplossing", "")).strip()

        # 3) Ranking
        df_all["_score"] = df_all["combined"].apply(lambda t: _token_score(vraag, t))
        df_all = df_all.sort_values("_score", ascending=False)
        if len(df_all) == 0 or int(df_all.iloc[0]["_score"]) <= 0:
            return None
        return str(df_all.iloc[0].get("Antwoord of oplossing", "")).strip()
    except Exception as e:
        logging.error(f"Algemeen-zoekfout: {e}")
        return None


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
    "selected_product": None,
    "selected_module": None,
    "selected_category": None,
    "selected_toelichting": None,
    "selected_answer_id": None,
    "selected_answer_text": None,
    "selected_image": None,      # nieuw
    "last_question": "",
    "last_item_label": "",
    "debug": False,
    "allow_ai": False,
    "allow_web": False,
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
    # Client-side copy naar klembord
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
        c1, c2, c3 = st.columns(3)
        if c1.button("Exact", use_container_width=True):
            st.session_state["selected_product"] = "Exact"
            st.session_state["selected_image"] = None
            st.toast("Gekozen: Exact")
            st.rerun()
        if c2.button("DocBase", use_container_width=True):
            st.session_state["selected_product"] = "DocBase"
            st.session_state["selected_image"] = None
            st.toast("Gekozen: DocBase")
            st.rerun()
        if c3.button("Algemeen", use_container_width=True):
            st.session_state["selected_product"] = "Algemeen"
            st.session_state["selected_image"] = None
            st.toast("Gekozen: Algemeen (vrije vraag)")
            st.rerun()
        render_chat()
        return

    # ALGEMEEN: geen cascade maar wél CSV + API (en optioneel web)
    if st.session_state.get("selected_product") == "Algemeen":
        render_chat()
        vraag = st.chat_input("Stel uw algemene vraag:")
        if not vraag:
            return

        # UNIEKECODE123 direct
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

        # 1) CSV altijd proberen
        csv_ans = vind_best_algemeen_antwoord(vraag)

        # 2) (optioneel) web-fallback verzamelen
        webbits = fetch_web_info_cached(vraag) if st.session_state.get("allow_web") else None

        # 3) API altijd raadplegen (als key aanwezig), ook als CSV iets geeft
        ai_ans = None
        if client is not None:
            try:
                sys_prompt = (
                    "Je bent een behulpzame Nederlandse assistent voor parochies. "
                    "Gebruik CSV-informatie als primaire bron wanneer aanwezig. "
                    "Wees kort, concreet en vriendelijk. Als iets onduidelijk is, stel 1 korte verhelderende vraag. "
                    "Als CSV leeg is, geef 1-3 praktische suggesties of vervolgstappen."
                )
                if csv_ans:
                    user_prompt = (
                        f"Vraag: {vraag}\n\n"
                        f"CSV-fragment (leidend indien relevant):\n{csv_ans}\n\n"
                        "Taak: Leg kort uit in 2-4 zinnen of bullets. Voeg 1 tip of waarschuwing toe als nuttig."
                    )
                else:
                    user_prompt = (
                        f"Vraag: {vraag}\n\n"
                        "CSV bevat geen directe match.\n"
                        "Taak: Stel 1 korte verhelderende vraag OF geef 1-3 praktische suggesties. "
                        "Noem geen niet-bestaande bronnen."
                    )

                ai_ans = chatgpt_cached(
                    [{"role": "system", "content": sys_prompt},
                     {"role": "user", "content": user_prompt}],
                    temperature=0.2, max_tokens=700,
                )
            except Exception as e:
                logging.error(f"AI (Algemeen) fout: {e}")
                ai_ans = None

        # 4) Antwoord samenstellen
        parts = []
        if csv_ans:
            parts.append(csv_ans.strip())
        if ai_ans:
            parts.append("AI-toelichting:\n" + ai_ans.strip())
        if not parts and webbits:
            parts.append(webbits.strip())

        final = "\n\n".join(parts) if parts else "Ik vond geen passend antwoord in de CSV. Kunt u uw vraag iets specifieker maken?"
        add_msg("assistant", with_info(final))
        st.rerun()
        return

    # Vanaf hier: Exact/DocBase cascade
    render_chat()
    render_breadcrumbs()

    # 1) Subthema
    if not st.session_state.get("selected_module"):
        opts = list_subthema(st.session_state["selected_product"])
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
        cats = list_categorieen(st.session_state["selected_product"], st.session_state["selected_module"])
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
            st.session_state["selected_product"],
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
            return  # wacht op keuze of overslaan

    # 4) Recordselectie binnen scope
    df_scope = faq_df
    syst = st.session_state["selected_product"]
    sub = st.session_state["selected_module"]
    cat = st.session_state["selected_category"]
    toe = st.session_state.get("selected_toelichting", "")

    try:
        df_scope = df_scope.xs(syst, level="Systeem", drop_level=False)
        df_scope = df_scope.xs(sub, level="Subthema", drop_level=False)
        if cat and str(cat).lower() != "alles":
            df_scope = df_scope.xs(cat, level="Categorie", drop_level=False)
    except KeyError:
        df_scope = pd.DataFrame(columns=faq_df.reset_index().columns)

    # Robuuste toelichting-match (trim, NBSP→spatie, collapse, casefold)
    if not df_scope.empty and toe is not None and str(toe) != "":
        tm = (df_scope["Toelichting melding"]
              .astype(str)
              .str.replace("\u00A0", " ", regex=False)
              .str.strip()
              .str.replace(r"\s+", " ", regex=True)
              .str.casefold())
        sel = re.sub(r"\s+", " ", str(toe).replace("\u00A0", " ").strip()).casefold()
        df_scope = df_scope[tm == sel]

    # Scope teller in sidebar
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

        # ✅ Altijd een echte tekst als antwoord + label voor PDF
        ans = str(row.get("Antwoord of oplossing", "") or "").strip()
        if not ans:
            oms = str(row.get("Omschrijving melding", "") or "").strip()
            ans = f"(Geen uitgewerkt antwoord in CSV voor: {oms})"
        label = mk_label(i, row)

        # Afbeelding opslaan in state (voor weergave onder antwoord)
        img = str(row.get("Afbeelding", "") or "").strip()
        st.session_state["selected_image"] = img if img else None

        if st.session_state.get("selected_answer_id") != row_id:
            st.session_state["selected_answer_id"] = row_id
            st.session_state["selected_answer_text"] = ans
            st.session_state["last_item_label"] = label
            st.session_state["last_question"] = f"Gekozen item: {label}"
            add_msg("assistant", with_info(ans))
            st.rerun()

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
