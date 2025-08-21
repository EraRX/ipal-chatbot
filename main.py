# """
# IPAL Chatbox — main.py
# - Chat-wizard met 4 knoppen: Exact | DocBase | Zoeken | Internet
# - Klassieke cascade via expander
# - Geen sidebar
# - PDF met banner/logo over de volle contentbreedte (hoogte schaalt)
# - CSV-robustheid + smart quotes fix + werkende “Kopieer antwoord”
# - NIEUW: Automatische eenvoudige uitleg onder elk CSV-antwoord
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
return


# ── Permanente knoppenbalk (blijft altijd zichtbaar) ────────────────────────
def render_top_buttons():
    with st.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        if c1.button("Exact", key="top_exact", use_container_width=True):
            st.session_state.update({"chat_mode": True, "chat_scope": "Exact", "chat_step": "ask_topic"})
            st.session_state["pdf_ready"] = False
            add_msg("assistant", "Prima. Kunt u in één zin beschrijven waar uw vraag over Exact Online gaat?")
            st.rerun()
        if c2.button("DocBase", key="top_docbase", use_container_width=True):
            st.session_state.update({"chat_mode": True, "chat_scope": "DocBase", "chat_step": "ask_topic"})
            st.session_state["pdf_ready"] = False
            add_msg("assistant", "Dank u. Kunt u in één zin beschrijven waar uw vraag over DocBase gaat?")
            st.rerun()
        if c3.button("Zoeken", key="top_search", use_container_width=True):
            st.session_state.update({"chat_mode": True, "chat_scope": "Zoeken", "chat_step": "ask_topic"})
            st.session_state["pdf_ready"] = False
            add_msg("assistant", "Waar wilt u in de CSV op zoeken? Typ een korte zoekterm.")
            st.rerun()
        if c4.button("Internet", key="top_internet", use_container_width=True):
            st.session_state.update({"chat_mode": True, "chat_scope": "Algemeen", "chat_step": "ask_topic"})
            st.session_state["pdf_ready"] = False
            add_msg("assistant", "Waarover gaat uw vraag? Beschrijf dit kort in één zin.")
            st.rerun()
        if c5.button("🔄 Reset", key="top_reset", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            for k, v in DEFAULT_STATE.items():
                if k not in st.session_state:
                    st.session_state[k] = v
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.rerun()


# ── App ──────────────────────────────────────────────────────────────────────
def main():
    # Intro (video of logo)
    video_path = "helpdesk.mp4"
    if os.path.exists(video_path):
        try:
            with open(video_path, "rb") as f:
                st.video(f.read(), format="video/mp4", start_time=0)
        except Exception as e:
            logging.error(f"Introvideo kon niet worden afgespeeld: {e}")
    elif os.path.exists("logo.png"):
        st.image("logo.png", width=244)
    else:
        st.info("Welkom bij IPAL Chatbox")

    st.header("Welkom bij IPAL Chatbox")

    # Permanente knoppenbalk
    render_top_buttons()

    # Klassieke cascade (optioneel)
    with st.expander("Liever de klassieke cascade openen?"):
        keuze = st.radio(
            "Kies cascade:", ["Exact", "DocBase", "Zoeken", "Internet"],
            horizontal=True, index=0, key="cascade_radio"
        )
        if st.button("Start cascade", use_container_width=True, key="cascade_start"):
            if keuze == "Exact":
                st.session_state.update({
                    "chat_mode": False, "selected_product": "Exact",
                    "selected_image": None, "selected_module": None,
                    "selected_category": None, "selected_omschrijving": None,
                    "selected_toelichting": None, "selected_answer_id": None,
                    "selected_answer_text": None, "last_item_label": "", "last_question": "",
                })
            elif keuze == "DocBase":
                st.session_state.update({
                    "chat_mode": False, "selected_product": "DocBase",
                    "selected_image": None, "selected_module": None,
                    "selected_category": None, "selected_omschrijving": None,
                    "selected_toelichting": None, "selected_answer_id": None,
                    "selected_answer_text": None, "last_item_label": "", "last_question": "",
                })
            elif keuze == "Zoeken":
                st.session_state.update({
                    "chat_mode": False, "selected_product": "Zoeken",
                    "selected_image": None, "search_query": "", "search_selection_index": None,
                    "selected_answer_id": None, "selected_answer_text": None,
                    "last_item_label": "", "last_question": "",
                })
            else:
                st.session_state.update({
                    "chat_mode": False, "selected_product": "Algemeen",
                    "selected_image": None, "selected_module": None,
                    "selected_category": None, "selected_omschrijving": None,
                    "selected_toelichting": None, "selected_answer_id": None,
                    "selected_answer_text": None, "last_item_label": "", "last_question": "",
                })
            st.rerun()

    # Wizard actief?
    if st.session_state.get("chat_mode", True):
        chat_wizard()
        return

    # ------ Klassieke flows ------
    if not st.session_state.get("selected_product"):
        c1, c2 = st.columns(2); c3, c4 = st.columns(2)
        if c1.button("Exact", use_container_width=True, key="classic_exact"):
            st.session_state.update({"selected_product": "Exact",
                                     "selected_image": None, "selected_module": None,
                                     "selected_category": None, "selected_omschrijving": None,
                                     "selected_toelichting": None, "selected_answer_id": None,
                                     "selected_answer_text": None, "last_item_label": "", "last_question": ""})
            st.rerun()
        if c2.button("DocBase", use_container_width=True, key="classic_docbase"):
            st.session_state.update({"selected_product": "DocBase",
                                     "selected_image": None, "selected_module": None,
                                     "selected_category": None, "selected_omschrijving": None,
                                     "selected_toelichting": None, "selected_answer_id": None,
                                     "selected_answer_text": None, "last_item_label": "", "last_question": ""})
            st.rerun()
        if c3.button("Zoeken", use_container_width=True, key="classic_zoeken"):
            st.session_state.update({"selected_product": "Zoeken",
                                     "selected_image": None, "search_query": "", "search_selection_index": None,
                                     "selected_answer_id": None, "selected_answer_text": None,
                                     "last_item_label": "", "last_question": ""})
            st.rerun()
        if c4.button("Internet", use_container_width=True, key="classic_internet"):
            st.session_state.update({"selected_product": "Algemeen",
                                     "selected_image": None, "selected_module": None,
                                     "selected_category": None, "selected_omschrijving": None,
                                     "selected_toelichting": None, "selected_answer_id": None,
                                     "selected_answer_text": None, "last_item_label": "", "last_question": ""})
            st.rerun()
        render_chat()
        return

    # INTERNET (algemeen)
    if st.session_state.get("selected_product") == "Algemeen":
        render_chat()
        st.caption("Stel hier uw internetvraag (niet direct onder DocBase of Exact Online):")
        algemeen_vraag = st.text_input(" ", placeholder="Stel uw internetvraag:",
                                       key="algemeen_top_input", label_visibility="collapsed")
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
                add_msg("assistant", with_info(cw))
                st.rerun()
                return
        st.session_state["last_processed_algemeen"] = algemeen_vraag
        st.session_state["last_question"] = algemeen_vraag
        add_msg("user", algemeen_vraag)
        ok, warn = filter_topics(algemeen_vraag)
        if not ok:
            st.session_state["pdf_ready"] = False
            add_msg("assistant", warn)
            st.rerun()
            return
        antwoord = vind_best_algemeen_AI(algemeen_vraag)
        if not antwoord and st.session_state.get("allow_web"):
            webbits = fetch_web_info_cached(algemeen_vraag)
            if webbits:
                antwoord = webbits
        st.session_state["pdf_ready"] = True
        add_msg("assistant", with_info(antwoord or "Kunt u uw vraag iets concreter maken?"))
        st.rerun()
        return

    # ZOEKEN (hele CSV)
    if st.session_state.get("selected_product") == "Zoeken":
        render_chat()
        st.session_state["search_query"] = st.text_input("Waar wil je in de volledige CSV op zoeken?",
                                                         value=st.session_state.get("search_query",""))
        q = st.session_state["search_query"].strip()
        if not q:
            return
        results = zoek_hele_csv(q, min_hits=st.session_state["min_hits"], min_cov=st.session_state["min_cov"])
        st.caption(f"Gevonden resultaten: {len(results)}")
        if results.empty:
            st.info("Geen resultaten gevonden. Pas je zoekterm aan of verlaag de drempels (Geavanceerd).")
            return
        df_reset = results.reset_index(drop=True)

        def mk_label(i, row):
            oms = clean_text(str(row.get('Omschrijving melding','')).strip())
            toel = clean_text(str(row.get('Toelichting melding','')).strip())
            preview = oms or toel or clean_text(str(row.get('Antwoord of oplossing','')).strip())
            preview = re.sub(r"\s+"," ", preview)[:140]
            return f"{i+1:02d}. {preview}"

        opties = [mk_label(i, r) for i, r in df_reset.iterrows()]
        keuze = st.selectbox("Kies een item uit de zoekresultaten:", ["(Kies)"] + opties)
        if keuze == "(Kies)":
            return
        idx = int(keuze.split(".")[0]) - 1
        row = df_reset.iloc[idx]
        row_id = row.get("ID", idx)
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
            final_ans = enrich_with_simple(ans) if st.session_state.get("auto_simple", True) else ans
            st.session_state["pdf_ready"] = True
            add_msg("assistant", with_info(final_ans))
            st.rerun()
            return
        vraag2 = st.chat_input("Stel uw vraag over dit antwoord:")
        if not vraag2:
            return
        st.session_state["last_question"] = vraag2
        add_msg("user", vraag2)
        ok, warn = filter_topics(vraag2)
        if not ok:
            st.session_state["pdf_ready"] = False
            add_msg("assistant", warn)
            st.rerun()
            return
        bron = str(st.session_state.get("selected_answer_text") or "")
        reactie = None
        if st.session_state.get("allow_ai") and client is not None:
            try:
                reactie = chatgpt_cached(
                    [{"role":"system","content":"Beantwoord uitsluitend op basis van de meegegeven bron. Geen aannames buiten de bron. Schrijf kort en duidelijk in het Nederlands."},
                     {"role":"user","content":f"Bron:\n{bron}\n\nVraag: {vraag2}"}],
                    temperature=0.1, max_tokens=600,
                )
            except Exception as e:
                logging.error(f"AI-QA fout: {e}")
                reactie = None
        if not reactie:
            reactie = simplify_text(bron) if bron else "Ik kan zonder AI geen betere toelichting uit het gekozen antwoord halen."
        st.session_state["pdf_ready"] = True
        add_msg("assistant", with_info(reactie))
        st.rerun()
        return

    # Exact/DocBase cascade (klassiek) — volgorde 1-2-3-4-5-6
    render_chat()
    syst = st.session_state.get("selected_product")
    sub  = st.session_state.get("selected_module") or ""
    cat  = st.session_state.get("selected_category") or ""
    oms  = st.session_state.get("selected_omschrijving") or ""
    toe  = st.session_state.get("selected_toelichting") or ""
    parts = [p for p in [syst, sub, (None if cat in ("", None, "alles") else cat), (oms or None), (toe or None)] if p]
    if parts:
        st.caption(" › ".join(parts))

    # 1. Subthema
    if not st.session_state.get("selected_module"):
        try:
            opts = sorted(faq_df.xs(syst, level="Systeem").index.get_level_values("Subthema").dropna().unique())
        except Exception:
            opts = []
        sel = st.selectbox("Kies subthema:", ["(Kies)"] + list(opts))
        if sel != "(Kies)":
            st.session_state["selected_module"] = sel
            st.session_state["selected_category"] = None
            st.session_state["selected_omschrijving"] = None
            st.session_state["selected_toelichting"] = None
            st.session_state["selected_answer_id"] = None
            st.session_state["selected_answer_text"] = None
            st.session_state["selected_image"] = None
            st.toast(f"Gekozen subthema: {sel}")
            st.rerun()
        return

    # 2. Categorie
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
            st.session_state["selected_omschrijving"] = None
            st.session_state["selected_toelichting"] = None
            st.session_state["selected_answer_id"] = None
            st.session_state["selected_answer_text"] = None
            st.session_state["selected_image"] = None
            st.rerun()
        selc = st.selectbox("Kies categorie:", ["(Kies)"] + list(cats))
        if selc != "(Kies)":
            st.session_state["selected_category"] = selc
            st.session_state["selected_omschrijving"] = None
            st.session_state["selected_toelichting"] = None
            st.session_state["selected_answer_id"] = None
            st.session_state["selected_answer_text"] = None
            st.session_state["selected_image"] = None
            st.toast(f"Gekozen categorie: {selc}")
            st.rerun()
        return

    # 3½. scope voor volgende stappen
    df_scope = faq_df
    try:
        df_scope = df_scope.xs(syst, level="Systeem", drop_level=False)
        df_scope = df_scope.xs(sub, level="Subthema", drop_level=False)
        if cat and str(cat).lower() != "alles":
            df_scope = df_scope.xs(cat, level="Categorie", drop_level=False)
    except KeyError:
        df_scope = pd.DataFrame(columns=faq_df.reset_index().columns)

    # 4. Omschrijving melding (NIEUW stap expliciet)
    if st.session_state.get("selected_omschrijving") is None:
        try:
            omsen = sorted(
                df_scope["Omschrijving melding"].dropna().astype(str).apply(clean_text).unique()
            )
        except Exception:
            omsen = []
        if len(omsen) == 0:
            st.info("Geen omschrijvingen gevonden — stap wordt overgeslagen.")
            st.session_state["selected_omschrijving"] = ""
        else:
            oms_sel = st.selectbox("Kies omschrijving:", ["(Kies)"] + list(omsen))
            if oms_sel != "(Kies)":
                st.session_state["selected_omschrijving"] = oms_sel
                st.session_state["selected_toelichting"] = None
                st.session_state["selected_answer_id"] = None
                st.session_state["selected_answer_text"] = None
                st.session_state["selected_image"] = None
                st.toast(f"Gekozen omschrijving: {oms_sel}")
                st.rerun()
            return

    # filter op omschrijving (indien gezet)
    sel_oms = st.session_state.get("selected_omschrijving", "")
    if df_scope is not None and not df_scope.empty and sel_oms not in (None, ""):
        omcol = df_scope["Omschrijving melding"].astype(str).apply(clean_text)
        df_scope = df_scope[omcol == clean_text(sel_oms)]

    # 5. Toelichting melding
    if st.session_state.get("selected_toelichting") is None:
        try:
            toes = sorted(
                df_scope["Toelichting melding"].dropna().astype(str).apply(clean_text).unique()
            )
        except Exception:
            toes = []
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

    # filter op toelichting (indien gezet)
    sel_toe = st.session_state.get("selected_toelichting", "")
    if df_scope is not None and not df_scope.empty and sel_toe not in (None, ""):
        tm = df_scope["Toelichting melding"].astype(str).apply(clean_text)
        df_scope = df_scope[tm == clean_text(sel_toe)]

    if df_scope is None or df_scope.empty:
        st.info("Geen records gevonden binnen de gekozen Systeem/Subthema/Categorie/Omschrijving/Toelichting.")
        return

    # 6. Antwoord of oplossing (selecteer item en toon)
    df_reset = df_scope.reset_index()

    # Als er nog maar één record is, toon direct het antwoord
    if len(df_reset) == 1:
        row = df_reset.iloc[0]
        row_id = row.get("ID", 0)
        ans = clean_text(str(row.get('Antwoord of oplossing', '') or '').strip())
        if not ans:
            oms_txt = clean_text(str(row.get('Omschrijving melding', '') or '').strip())
            ans = f"(Geen uitgewerkt antwoord in CSV voor: {oms_txt})"
        label = f"01. {re.sub(r'\\s+', ' ', clean_text(ans))[:140]}"
        img = clean_text(str(row.get('Afbeelding', '') or '').strip())
        st.session_state["selected_image"] = img if img else None
        if st.session_state.get("selected_answer_id") != row_id:
            st.session_state["selected_answer_id"] = row_id
            st.session_state["selected_answer_text"] = ans
            st.session_state["last_item_label"] = label
            st.session_state["last_question"] = f"Gekozen item: {label}"
            final_ans = enrich_with_simple(ans) if st.session_state.get("auto_simple", True) else ans
            st.session_state["pdf_ready"] = True
            add_msg("assistant", with_info(final_ans))
            st.rerun()
        # Als hetzelfde record al getoond was, gewoon door naar de vervolgvraag hieronder
    else:
        def mk_label(i, row):
            # Belangrijk: Antwoord → Toelichting → Omschrijving
            ansx  = clean_text(str(row.get('Antwoord of oplossing', '')).strip())
            toelx = clean_text(str(row.get('Toelichting melding', '')).strip())
            omsx  = clean_text(str(row.get('Omschrijving melding', '')).strip())
            preview = ansx or toelx or omsx
            preview = re.sub(r"\s+", " ", preview)[:140]
            return f"{i+1:02d}. {preview}"

        opties = [mk_label(i, r) for i, r in df_reset.iterrows()]
        keuze = st.selectbox("Kies een item:", ["(Kies)"] + opties)
        if keuze != "(Kies)":
            i = int(keuze.split(".")[0]) - 1
            row = df_reset.iloc[i]
            row_id = row.get("ID", i)
            ans = clean_text(str(row.get('Antwoord of oplossing', '') or '').strip())
            if not ans:
                oms_txt = clean_text(str(row.get('Omschrijving melding', '') or '').strip())
                ans = f"(Geen uitgewerkt antwoord in CSV voor: {oms_txt})"
            label = mk_label(i, row)
            img = clean_text(str(row.get('Afbeelding', '') or '').strip())
            st.session_state["selected_image"] = img if img else None
            if st.session_state.get("selected_answer_id") != row_id:
                st.session_state["selected_answer_id"] = row_id
                st.session_state["selected_answer_text"] = ans
                st.session_state["last_item_label"] = label
                st.session_state["last_question"] = f"Gekozen item: {label}"
                final_ans = enrich_with_simple(ans) if st.session_state.get("auto_simple", True) else ans
                st.session_state["pdf_ready"] = True
                add_msg("assistant", with_info(final_ans))
                st.rerun()
                return

    # Vervolgvraag over getoond antwoord
    vraag = st.chat_input("Stel uw vraag over dit antwoord:")
    if not vraag:
        return

    if (vraag or "").strip().upper() == "UNIEKECODE123":
        cw = find_answer_by_codeword(faq_df.reset_index())
        if cw:
            st.session_state["last_question"] = vraag
            add_msg("user", vraag)
            st.session_state["pdf_ready"] = True
            add_msg("assistant", with_info(cw))
            st.rerun()
            return

    st.session_state["last_question"] = vraag
    add_msg("user", vraag)
    ok, warn = filter_topics(vraag)
    if not ok:
        st.session_state["pdf_ready"] = False
        add_msg("assistant", warn)
        st.rerun()
        return

    bron = str(st.session_state.get("selected_answer_text") or "")
    reactie = None
    if st.session_state.get("allow_ai") and client is not None:
        try:
            reactie = chatgpt_cached(
                [
                    {"role":"system","content":"Beantwoord uitsluitend op basis van de meegegeven bron. Geen aannames buiten de bron. Schrijf kort en duidelijk in het Nederlands."},
                    {"role":"user","content":f"Bron:\n{bron}\n\nVraag: {vraag}"}
                ],
                temperature=0.1, max_tokens=600,
            )
        except Exception as e:
            logging.error(f"AI-QA fout: {e}")
            reactie = None

    if not reactie:
        reactie = simplify_text(bron) if bron else "Ik kan zonder AI geen betere toelichting uit het gekozen antwoord halen."

    st.session_state["pdf_ready"] = True
    add_msg("assistant", with_info(reactie))
    st.rerun()


if __name__ == "__main__":
    main()

