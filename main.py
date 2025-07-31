"""
IPAL Chatbox voor oudere vrijwilligers
- Python 3, Streamlit
- Groot lettertype, eenvoudige bediening
- Antwoorden uit FAQ aangevuld met AI voor specifieke modules
- Topicfiltering (blacklist + herstelde fallback op geselecteerde module)
- Logging en foutafhandeling
- Antwoorden downloaden als PDF met AI-Antwoord Info
"""
import os
import re
import logging
import io
from datetime import datetime
import streamlit as st
import pandas as pd
import pytz
import requests
from bs4 import BeautifulSoup
from PIL import Image as PILImage
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Disable proxy environment variables
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
logging.info(f"Proxy settings: HTTP_PROXY={os.environ.get('HTTP_PROXY', 'None')}, HTTPS_PROXY={os.environ.get('HTTPS_PROXY', 'None')}")

# Load environment variables
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Streamlit page config
st.set_page_config(page_title='IPAL Chatbox', layout='centered')
st.markdown(
    '<style>html, body, [class*="css"] { font-size:20px; } button[kind="primary"] { font-size:22px !important; padding:.75em 1.5em; }</style>',
    unsafe_allow_html=True
)

# Sidebar API key input
if not OPENAI_KEY:
    api_key_input = st.sidebar.text_input('🔑 Voer uw OpenAI API-sleutel in:', type='password')
    if api_key_input:
        st.session_state.api_key_input = api_key_input
    else:
        st.sidebar.error('🔑 Voeg je OpenAI API-key toe in .env, Secrets, of hierboven.')
        st.stop()

# Validate API key
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), retry=retry_if_exception_type(Exception))
def validate_api_key():
    api_key = st.session_state.get('api_key_input', OPENAI_KEY)
    if not api_key:
        st.error('⚠️ Geen API-sleutel gevonden.')
        st.stop()
    try:
        client = OpenAI(api_key=api_key, http_client=None)
        client.models.list()
        logging.info("OpenAI API-sleutel succesvol gevalideerd")
        return client
    except Exception as e:
        logging.error(f"API-validatie fout: {str(e)}")
        st.error(f'⚠️ Fout bij API-validatie: {str(e)}')
        st.stop()

client = validate_api_key()

# Register Calibri font
if os.path.exists("Calibri.ttf"):
    pdfmetrics.registerFont(TTFont("Calibri", "Calibri.ttf"))
else:
    logging.info("Calibri.ttf niet gevonden, gebruik Helvetica")

# PDF generation
def make_pdf(question: str, answer: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leading=16, spaceAfter=12, alignment=TA_LEFT)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=colors.HexColor("#333333"), spaceBefore=12, spaceAfter=6)

    story = []
    if os.path.exists("logo.png"):
        logo = Image("logo.png", width=124, height=52)
        logo_table = Table([[logo]], colWidths=[124])
        logo_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
        ]))
        story.append(logo_table)

    story.append(Paragraph(f"Vraag: {question}", heading_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Antwoord:", heading_style))
    for line in answer.split("\n"):
        if line.strip():
            story.append(Paragraph(line.strip(), body_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>AI-Antwoord Info:</b>", body_style))
    story.append(Paragraph("1. Dit is het AI-antwoord vanuit de IPAL chatbox van het Interdiocesaan Platform Automatisering & Ledenadministratie. Het is altijd een goed idee om de meest recente informatie te controleren via officiële bronnen.", body_style))
    story.append(Paragraph("2. Heeft u hulp nodig met DocBase of Exact? Dan kunt u eenvoudig een melding maken door een ticket aan te maken in DocBase. Maar voordat u een ticket invult, hebben we een handige tip: controleer eerst onze FAQ (het document met veelgestelde vragen en antwoorden). Dit document vindt u op onze site.", body_style))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

@st.cache_data
def load_faq(path="faq.xlsx"):
    """Load FAQ from Excel file."""
    if not os.path.exists(path):
        logging.error(f"FAQ niet gevonden: {path}")
        st.error(f"FAQ-bestand '{path}' niet gevonden.")
        return pd.DataFrame(columns=['Systeem','Subthema','Omschrijving melding','Toelichting melding','Antwoord of oplossing','Afbeelding'])
    df = pd.read_excel(path, engine="openpyxl")
    if 'Afbeelding' not in df.columns:
        df['Afbeelding'] = None
    df['Antwoord'] = df['Antwoord of oplossing']
    df['combined'] = df[['Systeem','Subthema','Omschrijving melding','Toelichting melding']].fillna('').agg(' '.join, axis=1)
    return df

faq_df = load_faq()
producten = ['Algemeen', 'Exact', 'DocBase']
subthema_dict = {p: sorted(faq_df.loc[faq_df['Systeem']==p,'Subthema'].dropna().unique()) for p in ['Exact', 'DocBase']}
BLACKLIST = ["persoonlijke gegevens", "medische gegevens", "gezondheid", "privacy schending"]

def filter_topics(msg: str):
    """Filter messages for blacklisted content."""
    found = [t for t in BLACKLIST if re.search(rf"\b{re.escape(t)}\b", msg.lower())]
    return (False, f"Je bericht bevat gevoelige onderwerpen: {', '.join(found)}.") if found else (True, "")

def fetch_bishop_from_rkkerk(loc: str):
    """Fetch bishop name from rkkerk.nl."""
    slug = loc.lower().replace(" ", "-")
    url = f"https://www.rkkerk.nl/bisdom-{slug}/"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        h1 = soup.find("h1")
        if h1 and "bisschop" in h1.text.lower():
            return h1.text.split("—")[0].strip()
    except:
        pass
    return None

def fetch_bishop_from_rkk_online(loc: str):
    """Fetch bishop name from rkk-online.nl."""
    query = loc.replace(" ", "+")
    url = f"https://www.rkk-online.nl/?s={query}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in ("h1", "h2", "h3"):
            h = soup.find(tag, string=re.compile(r"bisschop", re.I))
            if h:
                return h.text.split("–")[0].strip()
    except:
        pass
    return None

def fetch_all_bishops_nl():
    """Fetch all Dutch bishops."""
    dioceses = ["Utrecht", "Haarlem-Amsterdam", "Rotterdam", "Groningen-Leeuwarden", "’s-Hertogenbosch", "Roermond", "Breda"]
    result = {}
    for d in dioceses:
        name = fetch_bishop_from_rkkerk(d) or fetch_bishop_from_rkk_online(d)
        if name:
            result[d] = name
    return result

AVATARS = {"assistant": "aichatbox.jpg", "user": "parochie.jpg"}
def get_avatar(role: str):
    """Get avatar for chat message."""
    path = AVATARS.get(role)
    return PILImage.open(path).resize((64, 64)) if path and os.path.exists(path) else "🙂"

TIMEZONE = pytz.timezone("Europe/Amsterdam")
MAX_HISTORY = 20
def add_msg(role: str, content: str):
    """Add message to chat history."""
    ts = datetime.now(TIMEZONE).strftime('%d-%m-%Y %H:%M')
    st.session_state.history = (st.session_state.history + [{'role': role, 'content': content, 'time': ts}])[-MAX_HISTORY:]

def render_chat():
    """Render chat history."""
    for m in st.session_state.history:
        st.chat_message(m['role'], avatar=get_avatar(m['role'])).markdown(f"{m['content']}\n\n_{m['time']}_")

if 'history' not in st.session_state:
    st.session_state.history = []
    st.session_state.selected_product = None
    st.session_state.selected_module = None
    st.session_state.last_question = ''
    st.session_state.api_key_input = None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), retry=retry_if_exception_type(Exception))
def chatgpt(messages, temperature=0.3, max_tokens=800):
    """Call OpenAI API with retries."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

def main():
    """Main application logic."""
    if st.sidebar.button('🔄 Nieuw gesprek'):
        st.session_state.clear()
        st.rerun()

    if st.session_state.history and st.session_state.history[-1]['role'] == 'assistant':
        pdf_data = make_pdf(
            question=st.session_state.last_question,
            answer=st.session_state.history[-1]['content']
        )
        st.sidebar.download_button('📄 Download PDF', data=pdf_data, file_name='antwoord.pdf', mime='application/pdf')

    if not st.session_state.selected_product:
        st.header('Welkom bij IPAL Chatbox')
        c1, c2, c3 = st.columns(3)
        if c1.button('Algemeen', use_container_width=True):
            st.session_state.selected_product = 'Algemeen'
            st.session_state.selected_module = 'alles'
            add_msg('assistant', 'Gekozen: Algemeen')
            st.rerun()
        if c2.button('DocBase', use_container_width=True):
            st.session_state.selected_product = 'DocBase'
            add_msg('assistant', 'Gekozen: DocBase')
            st.rerun()
        if c3.button('Exact', use_container_width=True):
            st.session_state.selected_product = 'Exact'
            add_msg('assistant', 'Gekozen: Exact')
            st.rerun()
        render_chat()
        return

    if st.session_state.selected_product in ['Exact', 'DocBase'] and not st.session_state.selected_module:
        opts = subthema_dict.get(st.session_state.selected_product, [])
        sel = st.selectbox('Kies onderwerp:', ['(Kies)'] + opts)
        if sel != '(Kies)':
            st.session_state.selected_module = sel
            add_msg('assistant', f'Gekozen: {sel}')
            st.rerun()
        render_chat()
        return

    render_chat()
    vraag = st.chat_input('Stel uw vraag:')
    if not vraag:
        return

    st.session_state.last_question = vraag
    add_msg('user', vraag)

    ok, warn = filter_topics(vraag)
    if not ok:
        add_msg('assistant', warn)
        st.rerun()

    m = re.match(r'(?i)(?:wie is\s+)?(?:de\s+)?bisschop(?:\s+van)?\s+(.+)', vraag)
    if m:
        loc = m.group(1).strip()
        bishop = fetch_bishop_from_rkkerk(loc) or fetch_bishop_from_rkk_online(loc)
        if bishop:
            ans = f"De huidige bisschop van {loc} is {bishop}."
            add_msg('assistant', ans + '\n\n**AI-Antwoord Info:**\n**1. Dit is het AI-antwoord vanuit de IPAL chatbox van het Interdiocesaan Platform Automatisering & Ledenadministratie.** Het is altijd een goed idee om de meest recente informatie te controleren via officiële bronnen.\n**2. Heeft u hulp nodig met DocBase of Exact?** Dan kunt u eenvoudig een melding maken door een ticket aan te maken in DocBase. Maar voordat u een ticket invult, hebben we een handige tip: controleer eerst onze FAQ (het document met veelgestelde vragen en antwoorden). Dit document vindt u op onze site.')
            st.rerun()

    if re.search(r'(?i)bisschoppen\s+(?:van\s+)?nederland', vraag):
        allb = fetch_all_bishops_nl()
        if allb:
            lines = [f"Mgr. {n} – Bisschop van {d}" for d, n in allb.items()]
            ans = "Huidige Nederlandse bisschoppen:\n" + "\n".join(lines)
            add_msg('assistant', ans + '\n\n**AI-Antwoord Info:**\n**1. Dit is het AI-antwoord vanuit de IPAL chatbox van het Interdiocesaan Platform Automatisering & Ledenadministratie.** Het is altijd een goed idee om de meest recente informatie te controleren via officiële bronnen.\n**2. Heeft u hulp nodig met DocBase of Exact?** Dan kunt u eenvoudig een melding maken door een ticket aan te maken in DocBase. Maar voordat u een ticket invult, hebben we een handige tip: controleer eerst onze FAQ (het document met veelgestelde vragen en antwoorden). Dit document vindt u op onze site.')
            st.rerun()

    dfm = faq_df[faq_df['combined'].str.contains(re.escape(vraag), case=False, na=False)]
    if not dfm.empty:
        row = dfm.iloc[0]
        ans = row['Antwoord']
        try:
            ans = chatgpt([
                {'role': 'system', 'content': 'Herschrijf eenvoudig en vriendelijk.'},
                {'role': 'user', 'content': ans}
            ], temperature=0.2)
        except Exception as e:
            logging.warning(f"Herschrijf mislukt: {e}")
        if isinstance(row['Afbeelding'], str) and os.path.exists(row['Afbeelding']):
            st.image(PILImage.open(row['Afbeelding']), caption='Voorbeeld', use_column_width=True)
        add_msg('assistant', ans + '\n\n**AI-Antwoord Info:**\n**1. Dit is het AI-antwoord vanuit de IPAL chatbox van het Interdiocesaan Platform Automatisering & Ledenadministratie.** Het is altijd een goed idee om de meest recente informatie te controleren via officiële bronnen.\n**2. Heeft u hulp nodig met DocBase of Exact?** Dan kunt u eenvoudig een melding maken door een ticket aan te maken in DocBase. Maar voordat u een ticket invult, hebben we een handige tip: controleer eerst onze FAQ (het document met veelgestelde vragen en antwoorden). Dit document vindt u op onze site.')
        st.rerun()

    with st.spinner('ChatGPT even aan het werk…'):
        try:
            ai = chatgpt([
                {'role': 'system', 'content': 'Je bent een behulpzame Nederlandse assistent.'},
                {'role': 'user', 'content': vraag}
            ])
            add_msg('assistant', ai + '\n\n**AI-Antwoord Info:**\n**1. Dit is het AI-antwoord vanuit de IPAL chatbox van het Interdiocesaan Platform Automatisering & Ledenadministratie.** Het is altijd een goed idee om de meest recente informatie te controleren via officiële bronnen.\n**2. Heeft u hulp nodig met DocBase of Exact?** Dan kunt u eenvoudig een melding maken door een ticket aan te maken in DocBase. Maar voordat u een ticket invult, hebben we een handige tip: controleer eerst onze FAQ (het document met veelgestelde vragen en antwoorden). Dit document vindt u op onze site.')
        except Exception as e:
            logging.error(f'AI-fallback mislukt: {e}')
            add_msg('assistant', f'⚠️ AI-fallback mislukt: {e}')
    st.rerun()

if __name__ == '__main__':
    main()
