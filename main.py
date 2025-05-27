# Dit is een aangepaste versie van main.py die speciaal is ontworpen voor oudere vrijwilligers
# met beperkte computervaardigheden. Het bevat grote knoppen, grotere letters,
# eenvoudige taal en duidelijke structuur. Antwoorden worden alleen gehaald uit de FAQ,
# aangevuld met een AI-herschrijving om leesbaarheid te verbeteren.

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
dotenv_path = '.env'
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
openai.api_key = os.getenv('OPENAI_API_KEY')
MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

# Pagina-configuratie
st.set_page_config(page_title='IPAL Helpdesk', layout='centered')

# Globale stijlen voor betere leesbaarheid
st.markdown('''
<style>
  html, body, [class*="css"] { font-size: 20px; }
  button[kind="primary"] { font-size: 22px !important; padding: 0.75em 1.5em; }
</style>
''', unsafe_allow_html=True)

# Controleer of de API-sleutel correct is ingesteld
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

# Laad en verwerk de FAQ uit een Excel-bestand
@st.cache_data
def load_faq(path: str = 'faq.xlsx') -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"FAQ-bestand '{path}' niet gevonden.")
        return pd.DataFrame(columns=['combined', 'Antwoord of oplossing'])
    df = pd.read_excel(path)
    required = [
        'Systeem', 'Subthema', 'Omschrijving melding',
        'Toelichting melding', 'Antwoord of oplossing'
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"FAQ-bestand mist kolommen: {missing}")
        return pd.DataFrame(columns=['combined', 'Antwoord of oplossing'])
    if 'Afbeelding' not in df.columns:
        df['Afbeelding'] = None
    def convert_link(cell):
        if isinstance(cell, str) and cell.startswith('=HYPERLINK'):
            m = re.match(r'=HYPERLINK\("([^"]+)","([^"]+)"\)', cell)
            if m:
                return f"[{m.group(2)}]({m.group(1)})"
        return cell
    df['Antwoord of oplossing'] = df['Antwoord of oplossing'].apply(convert_link)
    df['combined'] = df[required].fillna('').agg(' '.join, axis=1)
    return df

faq_df = load_faq()
producten = ['Exact', 'DocBase']

# Maak dictionary van subthema's per product
subthema_dict = {}
if not faq_df.empty:
    for prod in producten:
        subthema_dict[prod] = sorted(
            faq_df.loc[faq_df['Systeem'] == prod, 'Subthema']
                  .dropna().unique().tolist()
        )

# Initialiseer session state defaults
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

# Voeg bericht toe aan geschiedenis
def add_message(role: str, text: str):
    ts = datetime.now(timezone).strftime('%d-%m-%Y %H:%M')
    st.session_state.history.append({'role': role, 'content': text, 'time': ts})
    if len(st.session_state.history) > 100:
        st.session_state.history = st.session_state.history[-100:]

# Render chat
def render_chat():
    for msg in st.session_state.history:
        if msg['role'] == 'assistant' and os.path.exists('aichatbox.jpg'):
            avatar = Image.open('aichatbox.jpg').resize((64, 64))
        else:
            avatar = '🙂'
        st.chat_message(msg['role'], avatar=avatar).markdown(
            f"{msg['content']}\n\n_{msg['time']}_"
        )

# Reset helper
def on_reset():
    for k, v in defaults.items():
        st.session_state[k] = v

# AI herschrijving
def rewrite_answer(text: str) -> str:
    messages = [
        {'role': 'system', 'content': 'Herschrijf dit antwoord eenvoudig en vriendelijk.'},
        {'role': 'user', 'content': text}
    ]
    resp = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=300
    )
    return resp.choices[0].message.content.strip()

# AI fallback
def get_ai_answer(text: str) -> str:
    system = 'You are IPAL Chatbox, a Dutch helpdesk assistant. Antwoord kort.'
    messages = [{'role': 'system', 'content': system}]
    for m in st.session_state.history[-10:]:
        messages.append({'role': m['role'], 'content': m['content']})
    messages.append({'role': 'user', 'content': f"[{st.session_state.selected_module}] {text}"})
    resp = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=300
    )
    return resp.choices[0].message.content.strip()

# Antwoord logica
def get_answer(text: str) -> str:
    # FAQ lookup
    if st.session_state.selected_module and not faq_df.empty:
        df_mod = faq_df[faq_df['Subthema'] == st.session_state.selected_module]
        pat = re.escape(text)
        matches = df_mod[df_mod['combined'].str.contains(pat, case=False, na=False)]
        if not matches.empty:
            row = matches.iloc[0]
            ans = row['Antwoord of oplossing']
            img = row.get('Afbeelding')
            try:
                ans = rewrite_answer(ans)
            except:
                pass
            if isinstance(img, str) and img and os.path.exists(img):
                st.image(img, caption='Voorbeeld', use_column_width=True)
            return ans
    # AI fallback
    if st.session_state.selected_module:
        ai_resp = get_ai_answer(text)
        if ai_resp:
            return f"IPAL-Helpdesk antwoord:\n{ai_resp}"
    # Geen antwoord
    return '⚠️ Ik kan uw vraag niet beantwoorden. Neem contact op.'

# Main functie
def main():
    if st.session_state.reset_triggered:
        on_reset()
    st.sidebar.button('🔄 Nieuw gesprek', on_click=on_reset)
    # Kies product
    if not st.session_state.selected_product:
        st.header('Welkom bij de IPAL Helpdesk')
        c1, c2 = st.columns(2)
        if c1.button('DocBase', use_container_width=True):
            st.session_state.selected_product = 'DocBase'
            add_message('assistant', 'U heeft gekozen voor DocBase')
            st.rerun()
        if c2.button('Exact', use_container_width=True):
            st.session_state.selected_product = 'Exact'
            add_message('assistant', 'U heeft gekozen voor Exact')
            st.rerun()
        render_chat()
        return
    # Kies module
    if not st.session_state.selected_module:
        opties = subthema_dict.get(st.session_state.selected_product, [])
        keuze = st.selectbox('Kies een onderwerp:', ['(Kies)'] + opties)
        if keuze != '(Kies)':
            st.session_state.selected_module = keuze
            add_message('assistant', f"U heeft gekozen voor '{keuze}'")
            st.rerun()
        render_chat()
        return
    # Chat input\    render_chat()
    vraag = st.chat_input('Stel uw vraag:')
    if vraag:
        add_message('user', vraag)
        with st.spinner('Even zoeken...'):
            antwoord = get_answer(vraag)
            add_message('assistant', antwoord)
        st.rerun()

# Kick-off
if __name__ == '__main__':
    main()
