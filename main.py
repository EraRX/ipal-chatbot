import os
from datetime import datetime
import streamlit as st
import openai
import pandas as pd
import re
from dotenv import load_dotenv
from PIL import Image

# ---------------------------------------------
# IPAL Directe Interactieve Chatbox
# ---------------------------------------------
# Vereisten: streamlit, openai, pandas, pillow, python-dotenv

# Laad API-sleutel en model uit .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Valideer API-sleutel
def validate_api_key():
    if not openai.api_key:
        st.error("⚠️ Stel je OPENAI_API_KEY in in een .env-bestand.")
        st.stop()
    try:
        openai.models.list()
    except openai.AuthenticationError:
        st.error("⚠️ Ongeldige OPENAI_API_KEY. Controleer je .env-bestand.")
        st.stop()

validate_api_key()

# Paginaconfiguratie
st.set_page_config(page_title="IPAL Chatbox", layout="centered")

# Sidebar logo
logo_path = "logo.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=160)

# Laad en schaal avatar.png
assistant_avatar = None
avatar_path = 'avatar.png'
if os.path.exists(avatar_path):
    try:
        img = Image.open(avatar_path)
        assistant_avatar = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
    except Exception:
        assistant_avatar = None

# Laad FAQ-data voor fallback
@st.cache_data
def load_faq(path: str = 'faq.xlsx') -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = pd.read_excel(path)
            required_columns = ['Systeem', 'Omschrijving melding', 'Toelichting melding', 'Antwoord of oplossing']
            if not all(col in df.columns for col in required_columns):
                st.warning("FAQ-bestand mist vereiste kolommen.")
                return pd.DataFrame(columns=['combined', 'Antwoord of oplossing'])
            def convert_hyperlink(text):
                if isinstance(text, str) and text.startswith('=HYPERLINK'):
                    match = re.match(r'=HYPERLINK\("([^"]+)","([^"]+)"\)', text)
                    if match:
                        url, display_text = match.groups()
                        return f"[{display_text}]({url})"
                return text
            df['Antwoord of oplossing'] = df['Antwoord of oplossing'].apply(convert_hyperlink)
            df['combined'] = df[required_columns].fillna('').agg(' '.join, axis=1)
            return df
        except Exception as e:
            print(f"Error loading FAQ: {str(e)}")
            return pd.DataFrame(columns=['combined', 'Antwoord of oplossing'])
    return pd.DataFrame(columns=['combined', 'Antwoord of oplossing'])

faq_df = load_faq()

def reset_history():
    st.session_state.history = [
        {
            'role': 'assistant',
            'content': '👋 Hallo! Ik ben de IPAL Chatbox. Hoe kan ik je helpen vandaag?',
            'time': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
    ]

if 'history' not in st.session_state:
    reset_history()

def add_message(role: str, content: str):
    st.session_state.history.append({
        'role': role,
        'content': content,
        'time': datetime.now().strftime('%Y-%m-%d %H:%M')
    })
    MAX_HISTORY = 100
    if len(st.session_state.history) > MAX_HISTORY:
        st.session_state.history = st.session_state.history[-MAX_HISTORY:]

def render_chat():
    for msg in st.session_state.history:
        avatar = assistant_avatar if (msg['role'] == 'assistant' and assistant_avatar) else ('🤖' if msg['role'] == 'assistant' else '🙂')
        content = msg['content']
        timestamp = msg['time']
        st.chat_message(msg['role'], avatar=avatar).markdown(f"{content}\n*{timestamp}*")

def on_reset():
    reset_history()
    st.rerun()

st.sidebar.button('🔄 Nieuw gesprek', on_click=on_reset)

def get_faq_samenvatting(vraag: str) -> str:
    if not faq_df.empty:
        try:
            pattern = re.escape(vraag)
            matches = faq_df[faq_df['combined'].str.contains(pattern, case=False, na=False, regex=True)]
            top = matches.head(3)['Antwoord of oplossing'].tolist()
            return "\n".join(top) if top else ""
        except Exception as e:
            print(f"FAQ search error: {str(e)}")
    return ""

def get_answer(user_text: str) -> str:
    faq_info = get_faq_samenvatting(user_text)
    system_prompt = (
        "Je bent IPAL Chatbox, een behulpzame Nederlandse helpdeskassistent. "
        "Gebruik de meegegeven FAQ-informatie om vragen zo goed mogelijk te beantwoorden."
    )
    user_prompt = f"Gebruikersvraag: {user_text}\n\nFAQ-informatie:\n{faq_info}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        with st.spinner("Bezig met het genereren van een antwoord..."):
            resp = openai.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=300
            )
        return resp.choices[0].message.content.strip()
    except openai.AuthenticationError:
        st.error("⚠️ Ongeldige OpenAI API-sleutel. Controleer je .env-bestand.")
        return faq_info or "⚠️ Ongeldige sleutel."
    except openai.RateLimitError:
        st.error("⚠️ Limiet van OpenAI API bereikt. Probeer later opnieuw.")
        return faq_info or "⚠️ Geen resultaat."
    except openai.APIConnectionError:
        st.error("⚠️ Verbindingsprobleem met OpenAI. Controleer je internetverbinding.")
        return faq_info or "⚠️ Geen antwoord mogelijk."
    except Exception as e:
        st.error("⚠️ Er ging iets mis bij het ophalen van het antwoord. Probeer opnieuw.")
        return faq_info or "⚠️ Geen antwoord gevonden."

def main():
    user_input = st.chat_input('Typ je vraag hier...')
    if user_input and user_input.strip() and len(user_input) <= 500:
        add_message('user', user_input)
        answer = get_answer(user_input)
        add_message('assistant', answer)
    elif user_input and len(user_input) > 500:
        add_message('assistant', '⚠️ Je bericht is te lang. Houd het onder 500 tekens.')
    render_chat()

if __name__ == '__main__':
    main()
