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

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

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

systeemopties = sorted(faq_df['Systeem'].dropna().unique()) if not faq_df.empty else []

if 'history' not in st.session_state:
    st.session_state.history = [
        {
            'role': 'assistant',
            'content': '👋 Hallo! Ik ben de IPAL Chatbox. Kies eerst een onderwerp hieronder en stel daarna je vraag.',
            'time': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
    ]

st.sidebar.button('🔄 Nieuw gesprek', on_click=lambda: st.experimental_rerun())

# Toon dropdown altijd
keuze = st.selectbox("📁 Kies een onderwerp", ["(Kies een onderwerp)"] + systeemopties)

# Stop als nog niets gekozen is
if keuze == "(Kies een onderwerp)":
    st.stop()

# Bewaar selectie
st.session_state.selected_systeem = keuze

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

def faq_fallback(user_text: str) -> str:
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

def get_answer(user_text: str) -> str:
    system_prompt = (
        "You are IPAL Chatbox, a helpful Dutch helpdesk assistant. "
        "Answer questions briefly and clearly."
    )
    messages = [{'role': 'system', 'content': system_prompt}]
    for m in st.session_state.history:
        messages.append({'role': m['role'], 'content': m['content']})
    messages.append({'role': 'user', 'content': user_text})
    try:
        with st.spinner("Bezig met het genereren van een antwoord..."):
            resp = openai.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=300
            )
        return "🤖 AI-antwoord: " + resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Fallback triggered: {str(e)}")
        return faq_fallback(user_text)

def main():
    render_chat()
    vraag = st.chat_input("Stel je vraag over: " + st.session_state.selected_systeem)
    if vraag:
        add_message('user', vraag)
        antwoord = get_answer(vraag)
        add_message('assistant', antwoord)

if __name__ == '__main__':
    main()
