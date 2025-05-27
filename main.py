# Dit is een aangepaste versie van main.py die speciaal is ontworpen voor oudere vrijwilligers
# met beperkte computervaardigheden. Het bevat grote knoppen, grotere letters,
# eenvoudige taal en duidelijke structuur. Antwoorden worden alleen gehaald uit de FAQ,
# aangevuld met een AI-fallback voor specifieke modules via een whitelist.

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

# Stijlen voor leesbaarheid
st.markdown('''
<style>
  html, body, [class*="css"] { font-size: 20px; }
  button[kind="primary"] { font-size: 22px !important; padding: 0.75em 1.5em; }
</style>
''', unsafe_allow_html=True)

# Controleer API-sleutel
def validate_api_key():
    if not openai.api_key:
        st.error('⚠️ Stel uw OPENAI_API_KEY in.')
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

# Laad FAQ
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
    if 'Afbeelding' not in df.columns:
        df['Afbeelding'] = None
    def convert_link(cell):
        if isinstance(cell,str) and cell.startswith('=HYPERLINK'):
            m = re.match(r'=HYPERLINK\("([^"]+)","([^"]+)"\)',cell)
            if m:
                return f"[{m.group(2)}]({m.group(1)})"
        return cell
    df['Antwoord of oplossing'] = df['Antwoord of oplossing'].apply(convert_link)
    df['combined'] = df[required].fillna('').agg(' '.join,axis=1)
    return df
faq_df = load_faq()
producten = ['Exact','DocBase']

# Subthema's per product
subthema_dict = {}
if not faq_df.empty:
    for prod in producten:
        subthema_dict[prod] = sorted(
            faq_df.loc[faq_df['Systeem']==prod,'Subthema'].dropna().unique().tolist()
        )

# Session state defaults
defaults = {'history':[],'selected_product':None,'selected_module':None,'reset_triggered':False}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

timezone = pytz.timezone('Europe/Amsterdam')

# Bericht toevoegen
def add_message(role:str, text:str):
    ts = datetime.now(timezone).strftime('%d-%m-%Y %H:%M')
    st.session_state.history.append({'role':role,'content':text,'time':ts})
    if len(st.session_state.history)>100:
        st.session_state.history = st.session_state.history[-100:]

# Chat renderen
def render_chat():
    for msg in st.session_state.history:
        if msg['role']=='assistant' and os.path.exists('aichatbox.jpg'):
            avatar = Image.open('aichatbox.jpg').resize((64,64))
        else:
            avatar = '🙂'
        st.chat_message(msg['role'],avatar=avatar).markdown(f"{msg['content']}\n\n_{msg['time']}_")

# Reset
def on_reset():
    for k,v in defaults.items(): st.session_state[k]=v

# AI herschrijven
@retry(stop=stop_after_attempt(3),wait=wait_exponential(min=1,max=10),retry=retry_if_exception_type(openai.RateLimitError))
def rewrite_answer(text:str)->str:
    msgs=[{'role':'system','content':'Herschrijf dit antwoord eenvoudig en vriendelijk.'},{'role':'user','content':text}]
    resp=openai.chat.completions.create(model=MODEL,messages=msgs,temperature=0.2,max_tokens=300)
    return resp.choices[0].message.content.strip()

# AI fallback
@retry(stop=stop_after_attempt(3),wait=wait_exponential(min=1,max=10),retry=retry_if_exception_type(openai.RateLimitError))
def get_ai_answer(text:str)->str:
    system='You are IPAL Chatbox, a Dutch helpdesk assistant. Antwoord kort.'
    msgs=[{'role':'system','content':system}]
    msgs+=[{'role':m['role'],'content':m['content']} for m in st.session_state.history[-10:]]
    msgs.append({'role':'user','content':f"[{st.session_state.selected_module}] {text}"})
    resp=openai.chat.completions.create(model=MODEL,messages=msgs,temperature=0.3,max_tokens=300)
    return resp.choices[0].message.content.strip()

# Whitelist modules voor AI fallback
AI_WHITELIST = ['Financiële administratie','Ledenadministratie','Rooms Katholieke Kerk']

# Antwoord logica
def get_answer(text:str)->str:
    # 1) FAQ
    if st.session_state.selected_module and not faq_df.empty:
        dfm=faq_df[faq_df['Subthema']==st.session_state.selected_module]
        pat=re.escape(text)
        m=dfm[dfm['combined'].str.contains(pat,case=False,na=False)]
        if not m.empty:
            row=m.iloc[0]
            ans=row['Antwoord of oplossing']
            img=row.get('Afbeelding')
            try: ans=rewrite_answer(ans)
            except: pass
            if isinstance(img,str) and img and os.path.exists(img): st.image(img,caption='Voorbeeld',use_column_width=True)
            return ans
    # 2) AI fallback voor whitelisted modules
    mod=st.session_state.selected_module
    if mod in AI_WHITELIST:
        ai_resp=get_ai_answer(text)
        if ai_resp:
            return f"IPAL-Helpdesk antwoord:\n{ai_resp}"
    # 3) Anders geen antwoord
    return '⚠️ Ik kan uw vraag niet beantwoorden. Neem contact op.'

# Main
 def main():
    if st.session_state.reset_triggered: on_reset()
    st.sidebar.button('🔄 Nieuw gesprek',on_click=on_reset)
    if not st.session_state.selected_product:
        st.header('Welkom bij de IPAL Helpdesk')
        c1,c2=st.columns(2)
        if c1.button('DocBase',use_container_width=True):st.session_state.selected_product='DocBase';add_message('assistant','U heeft gekozen voor DocBase');st.rerun()
        if c2.button('Exact',use_container_width=True):st.session_state.selected_product='Exact';add_message('assistant','U heeft gekozen voor Exact');st.rerun()
        render_chat();return
    if not st.session_state.selected_module:
        opts=subthema_dict.get(st.session_state.selected_product,[])
        sel=st.selectbox('Kies onderwerp:', ['(Kies)']+opts)
        if sel!='(Kies)':st.session_state.selected_module=sel;add_message('assistant',f"U heeft gekozen voor '{sel}'");st.rerun()
        render_chat();return
    render_chat()
    q=st.chat_input('Stel uw vraag:')
    if q: add_message('user',q);ans=get_answer(q);add_message('assistant',ans);st.rerun()

# Kick-off
if __name__=='__main__': main()
