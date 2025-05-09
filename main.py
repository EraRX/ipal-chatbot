import os
from datetime import datetime
import streamlit as st
import openai
import pandas as pd
import re
from dotenv import load_dotenv
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import base64

# ---------------------------------------------
# IPAL Directe Interactieve Chatbox
# ---------------------------------------------
# Vereisten: streamlit, openai, pandas, pillow, python-dotenv, tenacity, openpyxl

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def validate_api_key():
    if not openai.api_key:
        st.error("⚠️ Stel je OPENAI_API_KEY in in een .env-bestand of Streamlit Cloud Secrets.")
        st.stop()
    try:
        openai.models.list()
    except openai.AuthenticationError:
        st.error("⚠️ Ongeldige OPENAI_API_KEY. Controleer je .env-bestand of Streamlit Cloud Secrets.")
        st.stop()
    except openai.RateLimitError as e:
        st.error("⚠️ API-limiet bereikt bij validatie. Controleer je account op https://platform.openai.com/usage.")
        print(f"RateLimitError bij validatie: {str(e)}")
        st.stop()

validate_api_key()

st.set_page_config(page_title="IPAL Chatbox", layout="centered")

logo_path = "logo.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=160)

assistant_avatar = None
avatar_path = 'avatar.png'
if os.path.exists(avatar_path):
    try:
        img = Image.open(avatar_path)
        assistant_avatar = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
    except Exception:
        assistant_avatar = None

@st.cache_data
def load_faq(path: str = 'faq.xlsx') -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"⚠️ FAQ-bestand '{path}' niet gevonden in {os.getcwd()}. Plaats faq.xlsx in de juiste map.")
        return pd.DataFrame(columns=['combined', 'Antwoord of oplossing'])
    try:
        df = pd.read_excel(path)
        required_columns = ['Systeem', 'Subthema', 'Omschrijving melding', 'Toelichting melding', 'Antwoord of oplossing']
        if not all(col in df.columns for col in required_columns):
            st.error(f"⚠️ FAQ-bestand mist vereiste kolommen. Verwachte kolommen: {required_columns}")
            return pd.DataFrame(columns=['combined', 'Antwoord of oplossing'])
        def convert_hyperlink(text):
            if isinstance(text, str) and text.startswith('=HYPERLINK'):
                match = re.match(r'=HYPERLINK\("([^"]+)","([^"]+)"\)', text)
                if match:
                    url, display_text = match.groups()
                    return f"[{display_text}]({url})"
                return text
            return text
        df['Antwoord of oplossing'] = df['Antwoord of oplossing'].apply(convert_hyperlink)
        search_columns = [col for col in required_columns if col != 'Antwoord of oplossing']
        df['combined'] = df[search_columns].fillna('').agg(' '.join, axis=1)
        return df
    except Exception as e:
        st.error(f"⚠️ Fout bij het laden van FAQ-bestand: {str(e)}")
        return pd.DataFrame(columns=['combined', 'Antwoord of oplossing'])

faq_df = load_faq()

producten = sorted([p for p in faq_df['Systeem'].dropna().unique() if isinstance(p, str) and p.strip()]) if not faq_df.empty else []
if faq_df.empty or not producten:
    st.error("⚠️ FAQ-bestand is niet correct geladen of bevat geen geldige producten. Controleer het bestand en probeer opnieuw.")
    st.stop()

if 'history' not in st.session_state:
    st.session_state.history = [
        {'role': 'assistant', 'content': '👋 Hallo! Ik ben de IPAL Chatbox. Hoe kan ik je helpen vandaag?', 'time': datetime.now().strftime('%Y-%m-%d %H:%M')}
    ]
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None
if 'selected_subthema' not in st.session_state:
    st.session_state.selected_subthema = None
if 'reset_triggered' not in st.session_state:
    st.session_state.reset_triggered = False

def add_message(role: str, content: str):
    st.session_state.history.append({'role': role, 'content': content, 'time': datetime.now().strftime('%Y-%m-%d %H:%M')})
    if len(st.session_state.history) > 100:
        st.session_state.history = st.session_state.history[-100:]

def render_chat():
    for msg in st.session_state.history:
        avatar = assistant_avatar if msg['role'] == 'assistant' and assistant_avatar else ('🤖' if msg['role'] == 'assistant' else '🙂')
        st.chat_message(msg['role'], avatar=avatar).markdown(f"{msg['content']}\n*{msg['time']}*")
        if msg['role'] == 'assistant' and 'bmw x3' in msg['content'].lower():
            st.image("bmw-x3.jpg", caption="Afbeelding van de nieuwste BMW X3", use_column_width=True)

def on_reset():
    st.session_state.reset_triggered = True
    st.session_state.selected_product = None
    st.session_state.selected_subthema = None

def get_faq_answer(user_text: str) -> str:
    if not faq_df.empty:
        try:
            pattern = re.escape(user_text)
            matches = faq_df[faq_df['combined'].str.contains(pattern, case=False, na=False, regex=True)]
            if not matches.empty:
                top = matches.head(3)['Antwoord of oplossing'].tolist()
                return "📌 FAQ-resultaten:\n" + "\n".join(f"- {ans}" for ans in top)
        except Exception as e:
            print(f"FAQ search error: {str(e)}")
    return "⚠️ Geen antwoord gevonden in FAQ. Probeer je vraag specifieker te stellen."

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), retry=retry_if_exception_type(openai.RateLimitError))
def get_ai_answer(user_text: str) -> str:
    system_prompt = "You are IPAL Chatbox, a helpful Dutch helpdesk assistant. Answer questions briefly and clearly."
    messages = [{'role': 'system', 'content': system_prompt}]
    for m in st.session_state.history[-10:]:
        messages.append({'role': m['role'], 'content': m['content']})
    full_question = f"[{st.session_state.selected_subthema}] {user_text}"
    messages.append({'role': 'user', 'content': full_question})
    try:
        with st.spinner("Bezig met het genereren van een IPAL-Helpdesk antwoord..."):
            resp = openai.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=150
            )
        return "IPAL-Helpdesk antwoord: " + resp.choices[0].message.content.strip()
    except Exception as e:
        return get_faq_answer(user_text)

def get_answer(user_text: str) -> str:
    ai_answer = get_ai_answer(user_text)
    faq_answer = get_faq_answer(user_text)
    if ai_answer and faq_answer.startswith("📌"):
        return f"{ai_answer}\n\n{faq_answer}"
    elif ai_answer:
        return ai_answer
    else:
        return faq_answer

def main():
    if st.session_state.reset_triggered:
        st.session_state.history = [
            {'role': 'assistant', 'content': '👋 Hallo! Ik ben de IPAL Chatbox. Hoe kan ik je helpen vandaag?', 'time': datetime.now().strftime('%Y-%m-%d %H:%M')}
        ]
        st.session_state.selected_product = None
        st.session_state.selected_subthema = None
        st.session_state.reset_triggered = False

    st.sidebar.button('🔄 Nieuw gesprek', on_click=on_reset)

    if not st.session_state.selected_product:
        st.markdown("### Kies een product om mee te beginnen:")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("logo-docbase-icon.png"):
                st.image("logo-docbase-icon.png", use_container_width=False, width=120)
                if st.button("Klik hier", key="docbase_button"):
                    st.session_state.selected_product = "DocBase"
                    st.session_state.history = [
                        {'role': 'assistant', 'content': '👋 Je hebt DocBase gekozen. Kies nu een subthema om verder te gaan.', 'time': datetime.now().strftime('%Y-%m-%d %H:%M')}
                    ]
                    st.rerun()
        with col2:
            if os.path.exists("Exact.png"):
                st.image("Exact.png", use_container_width=False, width=120)
                if st.button("Klik hier", key="exact_button"):
                    st.session_state.selected_product = "Exact"
                    st.session_state.history = [
                        {'role': 'assistant', 'content': '👋 Je hebt Exact gekozen. Kies nu een subthema om verder te gaan.', 'time': datetime.now().strftime('%Y-%m-%d %H:%M')}
                    ]
                    st.rerun()
        return

    subthema_opties = sorted([s for s in faq_df[faq_df['Systeem'] == st.session_state.selected_product]['Subthema'].dropna().unique() if isinstance(s, str) and s.strip()])
    if not subthema_opties:
        st.error(f"⚠️ Geen subthema's beschikbaar voor {st.session_state.selected_product}. Controleer of het FAQ-bestand goed geladen is.")
        st.stop()

    st.session_state.selected_subthema = st.selectbox(
        f"📁 Kies een subthema voor {st.session_state.selected_product}",
        ["(Kies een subthema)"] + subthema_opties
    )

    if st.session_state.selected_subthema == "(Kies een subthema)":
        st.warning("⚠️ Kies een subthema voordat je een vraag stelt.")
        st.stop()

    render_chat()

    vraag = st.chat_input("Stel je vraag over: " + (st.session_state.selected_subthema or "(geen subthema)"), key="chat_input_" + str(len(st.session_state.history)))
    if vraag:
        add_message('user', vraag)
        with st.spinner("Antwoord wordt gegenereerd..."):
            antwoord = get_answer(vraag)
            add_message('assistant', antwoord)
        st.rerun()

if __name__ == '__main__':
    main()
