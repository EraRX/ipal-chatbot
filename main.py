# Dit is een aangepaste versie van main.py die speciaal is ontworpen voor oudere vrijwilligers
# met beperkte computervaardigheden. Het bevat grote knoppen, groter lettertype,
# eenvoudige taal en duidelijke visuele structuur.
# Antwoorden worden beperkt tot FAQ-items, aangevuld met een AI-herschrijving voor leesbaarheid.

import os
from datetime import datetime
import streamlit as st
import openai
import pandas as pd
import re
from dotenv import load_dotenv
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pytz

# Laad omgevingsvariabelen
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

# Pagina-configuratie
st.set_page_config(page_title='IPAL Helpdesk', layout='centered')

# Globale stijl voor leesbaarheid
st.markdown('''
<style>
  html, body, [class*="css"] { font-size: 20px; }
  button[kind="primary"] { font-size: 22px !important; padding: 0.75em 1.5em; }
</style>
''', unsafe_allow_html=True)

# Controleer API-sleutel
def validate_api_key():
    if not openai.api_key:
        st.error('⚠️ Stel uw OPENAI_API_KEY in via .env-bestand')
        st.stop()
    try:
        openai.models.list()
    except openai.AuthenticationError:
        st.error('⚠️ Ongeldige API-sleutel')
        st.stop()
    except openai.RateLimitError:
        st.error('⚠️ API-limiet bereikt, probeer later opnieuw')
        st.stop()

validate_api_key()

# Laad en verwerk FAQ
@st.cache_data
def load_faq(path: str = 'faq.xlsx') -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"FAQ-bestand '{path}' niet gevonden.")
        return pd.DataFrame(columns=['combined','Antwoord of oplossing'])
    df = pd.read_excel(path)
    required = ['Systeem','Subthema','Omschrijving melding','Toelichting melding','Antwoord of oplossing']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"FAQ-bestand mist kolommen: {missing}")
        return pd.DataFrame(columns=['combined','Antwoord of oplossing'])
    # Voeg optionele kolom Afbeelding toe
    if 'Afbeelding' not in df.columns:
        df['Afbeelding'] = None
    # Converteer eventuele HYPERLINK-formules naar markdown links
    def convert_link(cell):
        if isinstance(cell,str) and cell.startswith('=HYPERLINK'):
            m = re.match(r'=HYPERLINK\("([^"]+)","([^"]+)"\)',cell)
            if m:
                return f"[{m.group(2)}]({m.group(1)})"
        return cell
    df['Antwoord of oplossing'] = df['Antwoord of oplossing'].apply(convert_link)
    # Combineer kolommen voor zoekindex
    df['combined'] = df[required].fillna('').agg(' '.join,axis=1)
    return df

faq_df = load_faq()
producten = ['Exact','DocBase']

# Bereid subthema's per product
subthema_dict = {}
if not faq_df.empty:
    for prod in producten:
        subthema_dict[prod] = sorted(
            faq_df.loc[faq_df['Systeem']==prod,'Subthema'].
            dropna().unique().tolist()
        )

# Initialiseer sessiestatus
defaults = {
    'history': [],
    'selected_product': None,
    'selected_module': None,
    'reset_triggered': False
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

timezone = pytz.timezone('Europe/Amsterdam')

# Voeg berichten toe aan geschiedenis
def add_message(role: str, text: str):
    timestamp = datetime.now(timezone).strftime('%d-%m-%Y %H:%M')
    st.session_state.history.append({'role':role,'content':text,'time':timestamp})
    if len(st.session_state.history) > 100:
        st.session_state.history = st.session_state.history[-100:]

# Render chatgeschiedenis
def render_chat():
    for msg in st.session_state.history:
        if msg['role']=='assistant' and os.path.exists('aichatbox.jpg'):
            avatar = Image.open('aichatbox.jpg').resize((64,64))
        else:
            avatar = '🙂'
        st.chat_message(msg['role'],avatar=avatar).markdown(
            f"{msg['content']}\n\n_{msg['time']}_"
        )

# Reset alle sessiestatus
def on_reset():
    for k,v in defaults.items():
        st.session_state[k] = v

# Haal AI-antwoord op voor herschrijven
@retry(stop=stop_after_attempt(3),wait=wait_exponential(min=1,max=10),retry=retry_if_exception_type(openai.RateLimitError))
def rewrite_answer(text: str) -> str:
    messages = [
        {'role':'system','content':'Herschrijf dit antwoord eenvoudig en vriendelijk.'},
        {'role':'user','content':text}
    ]
    resp = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=300
    )
    return resp.choices[0].message.content.strip()

# Haal antwoord uit FAQ en herschrijf
def get_answer(text: str) -> str:
    # Probeer eerst antwoord uit de FAQ
    if st.session_state.selected_module and not faq_df.empty:
        subset = faq_df[faq_df['Subthema'] == st.session_state.selected_module]
        pattern = re.escape(text)
        matches = subset[subset['combined'].str.contains(pattern, case=False, na=False)]
        if not matches.empty:
            row = matches.iloc[0]
            ans = row['Antwoord of oplossing']
            img = row.get('Afbeelding')
            # Herschrijf via AI voor leesbaarheid
            try:
                ans = rewrite_answer(ans)
            except:
                pass
            # Toon afbeelding als beschikbaar
            if isinstance(img, str) and img and os.path.exists(img):
                st.image(img, caption='Voorbeeld', use_column_width=True)
            return ans
    # Geen FAQ-match: fallback naar AI binnen context
    if st.session_state.selected_module:
        ai_resp = get_ai_answer(text)
        if ai_resp:
            return f"IPAL-Helpdesk antwoord:
{ai_resp}"
    # Anders melding
    return '⚠️ Ik kan uw vraag niet beantwoorden. Neem contact op alsjeblieft.'

# Hoofdapplicatie
def main():
    if st.session_state.reset_triggered:
        on_reset()
    st.sidebar.button('🔄 Nieuw gesprek',on_click=on_reset)

    # Stap 1: product selecteren
    if not st.session_state.selected_product:
        st.header('Welkom bij de IPAL Helpdesk')
        st.write('Klik hieronder op het systeem waarover u een vraag heeft')
        col1,col2 = st.columns(2)
        if col1.button('DocBase',use_container_width=True):
            st.session_state.selected_product='DocBase'
            add_message('assistant','U heeft gekozen voor DocBase')
            st.rerun()
        if col2.button('Exact',use_container_width=True):
            st.session_state.selected_product='Exact'
            add_message('assistant','U heeft gekozen voor Exact')
            st.rerun()
        render_chat()
        return

    # Stap 2: module selecteren
    if not st.session_state.selected_module:
        opties = subthema_dict.get(st.session_state.selected_product, [])
        keuze = st.selectbox('Kies een onderwerp:', ['(Kies)'] + opties)
        if keuze != '(Kies)':
            st.session_state.selected_module = keuze
            add_message('assistant',f"U heeft gekozen voor '{keuze}'")
            st.rerun()
        render_chat()
        return

    # Stap 3: chat interactie
    render_chat()
    vraag = st.chat_input('Stel hier uw vraag:')
    if vraag:
        add_message('user',vraag)
        with st.spinner('Even zoeken...'):
            antwoord = get_answer(vraag)
            add_message('assistant',antwoord)
        st.rerun()

if __name__ == '__main__':
    main()
