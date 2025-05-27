# Dit is een aangepaste versie van main.py, speciaal voor oudere vrijwilligers met beperkte computervaardigheden.
# Grote knoppen, groter lettertype, eenvoudige taal, duidelijke visuele structuur.
# Antwoorden uit de FAQ, aangevuld met AI-fallback voor specifieke modules.

import os
import re
from datetime import datetime
import streamlit as st
import openai
import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pytz

# Topic filter definitions (whitelist and blacklist)
whitelist_topics = {
    "ledenadministratie": [
        "parochiaan", "lidmaatschap", "inschrijving", "uitschrijving", "doopregister",
        "sila", "parochie", "postcode", "adreswijziging", "verhuizing", "mutatie",
        "kerkledenadministratie", "doopdatum", "ledenbeheer", "registratie"
    ],
    "exact_online_boekhouding_financien": [
        "boekhouding", "financiën", "kerkbijdrage", "factuur", "betaling", "budget",
        "financieel beheer", "exact online", "administratie", "rekening", "kosten",
        "inkomsten", "uitgaven", "begraafplaatsadministratie", "donatie"
    ],
    "rooms_katholieke_kerk_administratief": [
        "parochiebeheer", "bisdom", "kerkprovincie", "kerkelijke administratie",
        "parochieblad", "vrijwilligers", "functionaris gegevensbescherming",
        "verwerkersovereenkomst", "avg-compliance", "statuten"
    ]
}
blacklist_categories = [
    "politiek", "religie", "persoonlijke gegevens", "gezondheid", "gokken",
    "adult content", "geweld", "haatzaaiende taal", "desinformatie",
    "geloofsovertuiging", "medische gegevens", "strafrechtelijk verleden",
    "seksuele inhoud", "complottheorie", "nepnieuws", "discriminatie",
    "racisme", "extremisme", "godsdienstige leer", "persoonlijke overtuiging"
]

def filter_chatbot_topics(message: str):
    """
    Return (allowed: bool, reason: str).
    Blokkeer bij blacklist match.
    Sta AI-fallback alleen toe als geselecteerde module whitelisted is en message bevat een keyword.
    """
    text = message.lower()
    # Blacklist
    for blocked in blacklist_categories:
        if re.search(r'\b' + re.escape(blocked) + r'\b', text):
            return False, f"Geblokkeerd: bevat verboden onderwerp '{blocked}'"
    # Whitelist
    mod = (st.session_state.selected_module or '').lower().replace(' ', '_')
    if mod in whitelist_topics:
        for kw in whitelist_topics[mod]:
            if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text):
                return True, ''
        return False, '⚠️ Geen geldig onderwerp voor AI-ondersteuning'
    return False, '⚠️ AI-fallback niet toegestaan voor dit onderwerp'

# Laad omgevingsvariabelen en configureer API
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

# Streamlit pagina instellingen
st.set_page_config(page_title='IPAL Helpdesk', layout='centered')
# Globale stijlen
st.markdown('''
<style>
  html, body, [class*="css"] { font-size: 20px; }
  button[kind="primary"] { font-size: 22px !important; padding: 0.75em 1.5em; }
</style>
''', unsafe_allow_html=True)

# Validatie API sleutel
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

# FAQ laden
@st.cache_data
def load_faq(path: str='faq.xlsx') -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"FAQ-bestand '{path}' niet gevonden.")
        return pd.DataFrame(columns=['combined','Antwoord'])
    df = pd.read_excel(path)
    required = ['Systeem','Subthema','Omschrijving melding','Toelichting melding','Antwoord of oplossing']
    if any(col not in df.columns for col in required):
        missing = [col for col in required if col not in df.columns]
        st.error(f"FAQ mist kolommen: {missing}")
        return pd.DataFrame(columns=['combined','Antwoord'])
    if 'Afbeelding' not in df.columns:
        df['Afbeelding'] = None
    df['Antwoord'] = df['Antwoord of oplossing']
    df['combined'] = df[required].fillna('').agg(' '.join,axis=1)
    return df
faq_df = load_faq()
producten = ['Exact','DocBase']
subthema_dict = {p: sorted(faq_df[faq_df['Systeem']==p]['Subthema'].dropna().unique().tolist()) for p in producten}

# Session state defaults
defaults = {'history':[],'selected_product':None,'selected_module':None,'reset_triggered':False}
for k,v in defaults.items(): st.session_state.setdefault(k,v)
timezone = pytz.timezone('Europe/Amsterdam')

# Berichten beheren
def add_message(role: str, content: str):
    ts = datetime.now(timezone).strftime('%d-%m-%Y %H:%M')
    st.session_state.history.append({'role': role,'content': content,'time': ts})
    if len(st.session_state.history)>100:
        st.session_state.history = st.session_state.history[-100:]

def render_chat():
    for msg in st.session_state.history:
        avatar = Image.open('aichatbox.jpg').resize((64,64)) if (msg['role']=='assistant' and os.path.exists('aichatbox.jpg')) else '🙂'
        st.chat_message(msg['role'], avatar=avatar).markdown(f"{msg['content']}\n\n_{msg['time']}_")

def on_reset():
    for k in defaults: st.session_state[k]=defaults[k]

# AI herschrijven
def rewrite_answer(text: str) -> str:
    resp = openai.chat.completions.create(
        model=MODEL,
        messages=[{'role':'system','content':'Herschrijf dit antwoord eenvoudig en vriendelijk.'},
                  {'role':'user','content':text}],
        temperature=0.2,max_tokens=300
    )
    return resp.choices[0].message.content.strip()

# AI fallback
def get_ai_answer(text: str) -> str:
    messages = [{'role':'system','content':'You are IPAL Chatbox, helpful Dutch helpdesk assistant.'}]
    messages += [{'role':m['role'],'content':m['content']} for m in st.session_state.history[-10:]]
    messages.append({'role':'user','content':f"[{st.session_state.selected_module}] {text}"})
    resp = openai.chat.completions.create(model=MODEL,messages=messages,temperature=0.3,max_tokens=300)
    return resp.choices[0].message.content.strip()

# Antwoord logica
def get_answer(text: str) -> str:
    # 1) FAQ lookup
    if st.session_state.selected_module and not faq_df.empty:
        dfm = faq_df[faq_df['Subthema'] == st.session_state.selected_module]
        pat = re.escape(text)
        matches = dfm[dfm['combined'].str.contains(pat, case=False, na=False)]
        if not matches.empty:
            row = matches.iloc[0]
            ans = row['Antwoord']
            img = row.get('Afbeelding')
            # AI-herschrijving voor leesbaarheid
            try:
                ans = rewrite_answer(ans)
            except:
                pass
            # Toon afbeelding als beschikbaar
            if isinstance(img, str) and img and os.path.exists(img):
                st.image(img, caption='Voorbeeld', use_column_width=True)
            return ans
    # 2) AI fallback als topic toegestaan
    allowed, reason = filter_chatbot_topics(text)
    if allowed:
        ai_resp = get_ai_answer(text)
        if ai_resp:
            return f"IPAL-Helpdesk antwoord:
{ai_resp}"
    # 3) Anders geen antwoord
    return reason

# Main functie
def main():
    if st.session_state.reset_triggered: on_reset()
    st.sidebar.button('🔄 Nieuw gesprek', on_click=on_reset)
    # Kies product
    if not st.session_state.selected_product:
        st.header('Welkom bij de IPAL Helpdesk')
        c1,c2 = st.columns(2)
        if c1.button('DocBase',use_container_width=True): st.session_state.selected_product='DocBase'; add_message('assistant','U heeft gekozen voor DocBase'); st.rerun()
        if c2.button('Exact',use_container_width=True): st.session_state.selected_product='Exact'; add_message('assistant','U heeft gekozen voor Exact'); st.rerun()
        render_chat(); return
    # Kies module
    if not st.session_state.selected_module:
        opts = subthema_dict.get(st.session_state.selected_product,[])
        sel = st.selectbox('Kies een onderwerp:', ['(Kies)']+opts)
        if sel!='(Kies)': st.session_state.selected_module=sel; add_message('assistant',f"U heeft gekozen voor '{sel}'"); st.rerun()
        render_chat(); return
    # Chat interactie
    render_chat()
    vraag = st.chat_input('Stel hier uw vraag:')
    if vraag:
        add_message('user',vraag)
        allowed, reason = filter_chatbot_topics(vraag)
        if not allowed:
            add_message('assistant',reason)
            st.rerun()
        with st.spinner('Even zoeken...'):
            antwoord = get_answer(vraag)
            add_message('assistant',antwoord)
        st.rerun()

# Kick-off
if __name__=='__main__': main()
