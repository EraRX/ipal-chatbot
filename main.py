
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

try:
    from openai.error import RateLimitError
except ImportError:
    RateLimitError = Exception

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, ListFlowable, ListItem, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

st.set_page_config(page_title='IPAL Chatbox', layout='centered')
st.markdown(
    '<style>html, body, [class*="css"] { font-size:20px; } button[kind="primary"] { font-size:22px !important; padding:.75em 1.5em; }</style>',
    unsafe_allow_html=True
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.sidebar.error("🔑 Voeg je OpenAI API-key toe in .env of Secrets.")
    st.stop()
client = OpenAI(api_key=OPENAI_KEY)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(1,10), retry=retry_if_exception_type(RateLimitError))
def chatgpt(messages, temperature=0.3, max_tokens=800):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

if os.path.exists("Calibri.ttf"):
    pdfmetrics.registerFont(TTFont("Calibri", "Calibri.ttf"))
else:
    logging.info("Calibri.ttf niet gevonden, gebruik ingebouwde Helvetica")

# PDF generation with chat-style layout and logo top-left
def make_pdf(question: str, answer: str, ai_info: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leading=16, spaceAfter=12, alignment=TA_LEFT)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=colors.HexColor("#333333"), spaceBefore=12, spaceAfter=6)
    bullet_style = ParagraphStyle("Bullet", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leftIndent=12, bulletIndent=0, leading=16)

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

    avatar_path = "aichatbox.jpg"
    if os.path.exists(avatar_path):
        avatar = Image(avatar_path, width=30, height=30)
        intro_text = Paragraph(answer.split("\n")[0], body_style)
        story.append(Table([[avatar, intro_text]], colWidths=[30, 440], style=TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')])))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Antwoord:", heading_style))
    for line in answer.split("\n")[1:]:
        line = line.strip()
        if line.startswith("•") or line.startswith("-"):
            bullets = ListFlowable([ListItem(Paragraph(line[1:].strip(), bullet_style))], bulletType="bullet")
            story.append(bullets)
        elif line:
            story.append(Paragraph(line, body_style))

    doc.build(story)

    story.append(Paragraph("<b>AI-Antwoord Info:</b>", body_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>1. Dit is het AI-antwoord vanuit de IPAL chatbox van het Interdiocesaan Platform Automatisering & Ledenadministratie.</b> Het is altijd een goed idee om de meest recente informatie te controleren via officiële bronnen.", body_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>2. Heeft u hulp nodig met DocBase of Exact?</b> Dan kunt u eenvoudig een melding maken door een ticket aan te maken in DocBase. Maar voordat u een ticket invult, hebben we een handige tip: controleer eerst onze FAQ (het document met veelgestelde vragen en antwoorden). Dit document vindt u op onze site.", body_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Waarom de FAQ gebruiken?</b>", body_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph("In het document met veelgestelde vragen vindt u snel en eenvoudig antwoorden op veelvoorkomende vragen, zonder dat u hoeft te wachten op hulp.", body_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Klik hieronder om de FAQ te openen en te kijken of uw vraag al beantwoord is:", body_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph("– Veel gestelde vragen Docbase nieuw 2024", body_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph("– Veel gestelde vragen Exact Online", body_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Instructie: Ticket aanmaken in DocBase</b>", body_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Dat is eenvoudig! Zorg ervoor dat uw melding duidelijk is:", body_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph("• Beschrijf het probleem zo gedetailleerd mogelijk.", body_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph("• Voegt u geen document toe, zet dan het documentformaat in het ticket op “geen bijlage”.", body_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph("• Geef uw telefoonnummer op waarop wij u kunnen bereiken, zodat de helpdesk contact met u kan opnemen. STEL HIER UW VRAAG:", body_style))
    story.append(Spacer(1, 6))
    buffer.seek(0)
    return buffer.getvalue()

@st.cache_data
def load_faq(path="faq.xlsx"):
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
producten = ['Exact','DocBase']
subthema_dict = {p: sorted(faq_df.loc[faq_df['Systeem']==p,'Subthema'].dropna().unique()) for p in producten}
BLACKLIST = ["persoonlijke gegevens","medische gegevens","gezondheid","privacy schending"]

def filter_topics(msg: str):
    found = [t for t in BLACKLIST if re.search(rf"\b{re.escape(t)}\b", msg.lower())]
    return (False, f"Je bericht bevat gevoelige onderwerpen: {', '.join(found)}.") if found else (True, "")

def fetch_bishop_from_rkkerk(loc: str):
    slug = loc.lower().replace(" ", "-")
    url = f"https://www.rkkerk.nl/bisdom-{slug}/"
    try:
        r = requests.get(url, timeout=10); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        h1 = soup.find("h1")
        if h1 and "bisschop" in h1.text.lower():
            return h1.text.split("—")[0].strip()
    except:
        pass
    return None

def fetch_bishop_from_rkk_online(loc: str):
    query = loc.replace(" ", "+")
    url = f"https://www.rkk-online.nl/?s={query}"
    try:
        r = requests.get(url, timeout=10); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in ("h1","h2","h3"):
            h = soup.find(tag, string=re.compile(r"bisschop", re.I))
            if h:
                return h.text.split("–")[0].strip()
    except:
        pass
    return None

def fetch_all_bishops_nl():
    dioceses = ["Utrecht","Haarlem-Amsterdam","Rotterdam","Groningen-Leeuwarden","’s-Hertogenbosch","Roermond","Breda"]
    result = {}
    for d in dioceses:
        name = fetch_bishop_from_rkkerk(d) or fetch_bishop_from_rkk_online(d)
        if name:
            result[d] = name
    return result

AVATARS = {"assistant":"aichatbox.jpg","user":"parochie.jpg"}
def get_avatar(role: str):
    path = AVATARS.get(role)
    return PILImage.open(path).resize((64,64)) if path and os.path.exists(path) else "🙂"

TIMEZONE = pytz.timezone("Europe/Amsterdam")
MAX_HISTORY = 20
def add_msg(role: str, content: str):
    ts = datetime.now(TIMEZONE).strftime('%d-%m-%Y %H:%M')
    st.session_state.history = (st.session_state.history + [{'role':role,'content':content,'time':ts}])[-MAX_HISTORY:]

def render_chat():
    for m in st.session_state.history:
        st.chat_message(m['role'], avatar=get_avatar(m['role'])).markdown(f"{m['content']}\n\n_{m['time']}_")

if 'history' not in st.session_state:
    st.session_state.history = []
    st.session_state.selected_product = None
    st.session_state.selected_module = None
    st.session_state.last_question = ''

def main():
    if st.sidebar.button('🔄 Nieuw gesprek'):
        st.session_state.clear()
        st.rerun()

    if st.session_state.history and st.session_state.history[-1]['role']=='assistant':
        st.session_state.history[-1]['content'] += '''

**AI-Antwoord Info:**  
**1. Dit is het AI-antwoord vanuit de IPAL chatbox van het Interdiocesaan Platform Automatisering & Ledenadministratie.** Het is altijd een goed idee om de meest recente informatie te controleren via officiële bronnen.  
**2. Heeft u hulp nodig met DocBase of Exact?** Dan kunt u eenvoudig een melding maken door een ticket aan te maken in DocBase. Maar voordat u een ticket invult, hebben we een handige tip: controleer eerst onze FAQ (het document met veelgestelde vragen en antwoorden). Dit document vindt u op onze site.  

**Waarom de FAQ gebruiken?**  
In het document met veelgestelde vragen vindt u snel en eenvoudig antwoorden op veelvoorkomende vragen, zonder dat u hoeft te wachten op hulp.  
Klik hieronder om de FAQ te openen en te kijken of uw vraag al beantwoord is:  
– Veel gestelde vragen Docbase nieuw 2024  
– Veel gestelde vragen Exact Online  

**Instructie: Ticket aanmaken in DocBase**  
Dat is eenvoudig! Zorg ervoor dat uw melding duidelijk is:  
• Beschrijf het probleem zo gedetailleerd mogelijk.  
• Voegt u geen document toe, zet dan het documentformaat in het ticket op “geen bijlage”.  
• Geef uw telefoonnummer op waarop wij u kunnen bereiken, zodat de helpdesk contact met u kan opnemen.
'''
        pdf_data = make_pdf(
            question=st.session_state.last_question,
            answer=st.session_state.history[-1]['content'],
            ai_info=st.session_state.history[-1]['content']
        )
        st.sidebar.download_button('📄 Download PDF', data=pdf_data, file_name='antwoord.pdf', mime='application/pdf')

    if not st.session_state.selected_product:
        st.header('Welkom bij IPAL Chatbox')
        c1, c2, c3 = st.columns(3)
        if c1.button('Exact', use_container_width=True):
            st.session_state.selected_product='Exact'
            add_msg('assistant','Gekozen: Exact')
            st.rerun()
        if c2.button('DocBase', use_container_width=True):
            st.session_state.selected_product='DocBase'
            add_msg('assistant','Gekozen: DocBase')
            st.rerun()
        if c3.button('Algemeen', use_container_width=True):
            st.session_state.selected_product='Algemeen'
            st.session_state.selected_module='alles'
            add_msg('assistant','Gekozen: Algemeen')
            st.rerun()
        render_chat()
        return

    if st.session_state.selected_product in ['Exact','DocBase'] and not st.session_state.selected_module:
        opts = subthema_dict.get(st.session_state.selected_product,[])
        sel = st.selectbox('Kies onderwerp:', ['(Kies)']+opts)
        if sel!='(Kies)':
            st.session_state.selected_module=sel
            add_msg('assistant',f'Gekozen: {sel}')
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

    m = re.match(r'(?i)wie is bisschop(?: van)?\s+(.+)', vraag)
    if m:
        loc = m.group(1).strip()
        bishop = fetch_bishop_from_rkkerk(loc) or fetch_bishop_from_rkk_online(loc)
        if bishop:
            add_msg('assistant', f"De huidige bisschop van {loc} is {bishop}.")
            st.rerun()

    if re.search(r'(?i)bisschoppen nederland', vraag):
        allb = fetch_all_bishops_nl()
        if allb:
            lines = [f"Mgr. {n} – Bisschop van {d}" for d,n in allb.items()]
            add_msg('assistant', "Huidige Nederlandse bisschoppen:\n" + "\n".join(lines))
            st.rerun()

    dfm = faq_df[faq_df['combined'].str.contains(re.escape(vraag), case=False, na=False)]
    if not dfm.empty:
        row = dfm.iloc[0]
        ans = row['Antwoord']
        try:
            ans = chatgpt([
                {'role':'system','content':'Herschrijf eenvoudig en vriendelijk.'},
                {'role':'user','content':ans}
            ], temperature=0.2)
        except:
            pass
        if isinstance(row['Afbeelding'], str) and os.path.exists(row['Afbeelding']):
            st.image(PILImage.open(row['Afbeelding']), caption='Voorbeeld', use_column_width=True)
        add_msg('assistant', ans)
        st.rerun()

    with st.spinner('ChatGPT even aan het werk…'):
        try:
            ai = chatgpt([
                {'role':'system','content':'Je bent een behulpzame Nederlandse assistent.'},
                {'role':'user','content':vraag}
            ])
            add_msg('assistant', ai)
        except Exception as e:
            logging.exception('AI-fallback mislukt')
            add_msg('assistant', f'⚠️ AI-fallback mislukt: {e}')
    st.rerun()

if __name__ == '__main__':
    main()
