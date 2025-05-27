# Dit is een aangepaste versie van main.py die speciaal is ontworpen voor oudere vrijwilligers
# met beperkte computervaardigheden. Het bevat grote knoppen, groter lettertype, eenvoudige taal
# en duidelijke visuele structuur. Afbeeldingsondersteuning wordt ook toegevoegd via een Excel-kolom.

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
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Pagina-configuratie
st.set_page_config(page_title="IPAL Chatbox", layout="centered")

# Stijlen voor grotere tekst/knoppen
st.markdown("""
<style>
  html, body, [class*=\"css\"] { font-size: 20px; }
  button[kind=\"primary\"] { font-size: 22px !important; padding: 0.75em 1.5em; }
</style>
""", unsafe_allow_html=True)

# Valideer API-sleutel
def validate_api_key():
    if not openai.api_key:
        st.error("⚠️ Stel uw OPENAI_API_KEY in via .env.")
        st.stop()
    try:
        openai.models.list()
    except openai.AuthenticationError:
        st.error("⚠️ Ongeldige API-sleutel.")
        st.stop()
    except openai.RateLimitError:
        st.error("⚠️ API-limiet bereikt, probeer later opnieuw.")
        st.stop()

validate_api_key()

# Laad FAQ uit Excel
def load_faq(path: str = 'faq.xlsx') -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"FAQ-bestand '{path}' niet gevonden.")
        return pd.DataFrame(columns=['combined', 'Antwoord of oplossing'])
    try:
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
        # Converteer HYPERLINK-formules
        def convert_link(cell):
            if isinstance(cell, str) and cell.startswith('=HYPERLINK'):
                m = re.match(r'=HYPERLINK\("([^"]+)","([^"]+)"\)', cell)
                if m:
                    url, text = m.groups()
                    return f"[{text}]({url})"
            return cell
        df['Antwoord of oplossing'] = df['Antwoord of oplossing'].apply(convert_link)
        df['combined'] = df[required].fillna('').agg(' '.join, axis=1)
        return df
    except Exception as e:
        st.error(f"Fout bij laden FAQ: {e}")
        return pd.DataFrame(columns=['combined', 'Antwoord of oplossing'])

faq_df = load_faq()
producten = ["Exact", "DocBase"]

# Bouw subthema-dict
subthema_dict = {
    prod: sorted(
        faq_df.loc[faq_df['Systeem']==prod, 'Subthema']
              .dropna().unique().tolist()
    )
    for prod in producten
    if not faq_df.empty
}

# Initialiseer session state
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

def add_message(role: str, text: str):
    timestamp = datetime.now(timezone).strftime('%d-%m-%Y %H:%M')
    st.session_state.history.append({'role': role, 'content': text, 'time': timestamp})
    if len(st.session_state.history) > 100:
        st.session_state.history = st.session_state.history[-100:]

def render_chat():
    for msg in st.session_state.history:
        if msg['role']=='assistant' and os.path.exists('aichatbox.jpg'):
            img = Image.open('aichatbox.jpg').resize((64,64))
            avatar = img
        else:
            avatar = '🙂'
        st.chat_message(msg['role'], avatar=avatar).markdown(
            f"{msg['content']}\n\n_{msg['time']}_"
        )

def on_reset():
    for key in defaults:
        st.session_state[key] = defaults[key]

# AI antwoord (Chat completion)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type(openai.RateLimitError)
)
def get_ai_answer(text: str) -> str:
    system = 'You are IPAL Chatbox, a helpful Dutch helpdesk assistant. Answer concisely.'
    history = st.session_state.history[-10:]
    messages = [{'role':'system','content':system}] + [
        {'role':m['role'],'content':m['content']} for m in history
    ] + [
        {'role':'user','content':f"[{st.session_state.selected_module}] {text}"}
    ]
    try:
        res = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=400
        )
        return res.choices[0].message.content.strip()
    except:
        return None

# Combineer FAQ en AI antwoorden
def get_answer(text: str) -> str:
    if st.session_state.selected_module and not faq_df.empty:
        df_mod = faq_df[faq_df['Subthema']==st.session_state.selected_module]
        pat = re.escape(text)
        matches = df_mod[df_mod['combined'].str.contains(pat, case=False, na=False)]
        if not matches.empty:
            row = matches.iloc[0]
            ans = row['Antwoord of oplossing']
            img = row.get('Afbeelding')
            # Verduidelijk via AI
            try:
                ai_resp = openai.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {'role':'system','content':'Herschrijf dit antwoord eenvoudig en vriendelijk.'},
                        {'role':'user','content':ans}
                    ],
                    temperature=0.2,
                    max_tokens=300
                )
                ans = ai_resp.choices[0].message.content.strip()
            except:
                pass
            if isinstance(img, str) and img and os.path.exists(img):
                st.image(img, caption='Voorbeeld', use_column_width=True)
            return ans
    # Fallback AI
    ai_answer = get_ai_answer(text)
    if ai_answer:
        return f"IPAL-Helpdesk antwoord:\n{ai_answer}"
    return "Er is geen antwoord gevonden. Formuleer uw vraag anders of klik op 'Nieuw gesprek'."

# Hoofdapplicatie
 def main():
    if st.session_state.reset_triggered:
        on_reset()
    st.sidebar.button('🔄 Nieuw gesprek', on_click=on_reset)

    # Stap 1: kies product
    if not st.session_state.selected_product:
        st.header('Welkom bij de IPAL Helpdesk')
        st.write('Klik hieronder op het systeem waarover u een vraag heeft.')
        c1, c2 = st.columns(2)
        if c1.button('DocBase', use_container_width=True):
            st.session_state.selected_product = 'DocBase'
            add_message('assistant','U heeft gekozen voor DocBase.')
            st.rerun()
        if c2.button('Exact', use_container_width=True):
            st.session_state.selected_product = 'Exact'
            add_message('assistant','U heeft gekozen voor Exact.')
            st.rerun()
        render_chat()
        return

    # Stap 2: kies module
    if not st.session_state.selected_module:
        opties = subthema_dict.get(st.session_state.selected_product, [])
        sel = st.selectbox('Kies een onderwerp:', ['(Kies)'] + opties)
        if sel != '(Kies)':
            st.session_state.selected_module = sel
            add_message('assistant',f"U heeft gekozen voor '{sel}'. Wat wilt u hierover weten?")
            st.rerun()
        render_chat()
        return

    # Stap 3: chat input
    render_chat()
    prompt = st.chat_input('Stel hier uw vraag:')
    if prompt:
        add_message('user',prompt)
        with st.spinner('Even zoeken...'):
            antwoord = get_answer(prompt)
            add_message('assistant',antwoord)
        st.rerun()

if __name__ == '__main__':
    main()
