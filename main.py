import os
from datetime import datetime
import streamlit as st
import openai
import pandas as pd
import re
from dotenv import load_dotenv
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ---------------------------------------------
# IPAL Directe Interactieve Chatbox
# ---------------------------------------------
# Vereisten: streamlit, openai, pandas, pillow, python-dotenv, tenacity, openpyxl

# Laad API-sleutel en model uit .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Valideer API-sleutel
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

# Paginaconfiguratie
st.set_page_config(page_title="IPAL Chatbox", layout="centered")

# Laad logo in sidebar
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

# Laad FAQ-data
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
        # Converteer hyperlink-formules naar Markdown
        def convert_hyperlink(text):
            if isinstance(text, str) and text.startswith('=HYPERLINK'):
                match = re.match(r'=HYPERLINK\("([^"]+)","([^"]+)"\)', text)
                if match:
                    url, display_text = match.groups()
                    return f"[{display_text}]({url})"
                return text
            return text
        df['Antwoord of oplossing'] = df['Antwoord of oplossing'].apply(convert_hyperlink)
        # Gebruik alle kolommen behalve 'Antwoord of oplossing' voor zoekfunctionaliteit
        search_columns = [col for col in required_columns if col != 'Antwoord of oplossing']
        df['combined'] = df[search_columns].fillna('').agg(' '.join, axis=1)
        return df
    except ImportError as e:
        if 'openpyxl' in str(e):
            st.error("⚠️ Python-bibliotheek 'openpyxl' ontbreekt. Zorg dat 'openpyxl' in requirements.txt staat.")
        else:
            st.error(f"⚠️ Fout bij het laden van FAQ-bestand: {str(e)}")
        print(f"ImportError: {str(e)}")
        return pd.DataFrame(columns=['combined', 'Antwoord of oplossing'])
    except Exception as e:
        st.error(f"⚠️ Fout bij het laden van FAQ-bestand: {str(e)}")
        print(f"Error loading FAQ: {str(e)}")
        return pd.DataFrame(columns=['combined', 'Antwoord of oplossing'])

faq_df = load_faq()

# Genereer subthema-opties
subthema_opties = sorted([s for s in faq_df['Subthema'].dropna().unique() if isinstance(s, str) and s.strip()]) if not faq_df.empty else []

# Controleer of FAQ correct is geladen
if faq_df.empty or not subthema_opties:
    st.error("⚠️ FAQ-bestand is niet correct geladen of bevat geen geldige subthema's. Controleer het bestand en probeer opnieuw.")
    st.stop()

# Initialiseren van sessiestatus
if 'history' not in st.session_state:
    st.session_state.history = [
        {
            'role': 'assistant',
            'content': '👋 Hallo! Ik ben de IPAL Chatbox. Hoe kan ik je helpen vandaag?',
            'time': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
    ]
    st.session_state.selected_subthema = None

# Voeg bericht toe aan geschiedenis
def add_message(role: str, content: str):
    st.session_state.history.append({
        'role': role,
        'content': content,
        'time': datetime.now().strftime('%Y-%m-%d %H:%M')
    })
    MAX_HISTORY = 100
    if len(st.session_state.history) > MAX_HISTORY:
        st.session_state.history = st.session_state.history[-MAX_HISTORY:]

# Toon chatgeschiedenis
def render_chat():
    for msg in st.session_state.history:
        avatar = assistant_avatar if (msg['role'] == 'assistant' and assistant_avatar) else ('🤖' if msg['role'] == 'assistant' else '🙂')
        content = msg['content']
        timestamp = msg['time']
        st.chat_message(msg['role'], avatar=avatar).markdown(f"{content}\n*{timestamp}*")

# Reset gesprek
def on_reset():
    st.session_state.history = [
        {
            'role': 'assistant',
            'content': '👋 Hallo! Ik ben de IPAL Chatbox. Hoe kan ik je helpen vandaag?',
            'time': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
    ]
    st.session_state.selected_subthema = None
    st.rerun()

st.sidebar.button('🔄 Nieuw gesprek', on_click=on_reset)

# Selectbox voor subthema-keuze
if subthema_opties:
    st.session_state.selected_subthema = st.selectbox("📁 Kies een subthema", ["(Kies een subthema)"] + subthema_opties)
    if st.session_state.selected_subthema == "(Kies een subthema)":
        st.warning("⚠️ Kies een subthema voordat je een vraag stelt.")
        st.stop()
else:
    st.error("⚠️ Geen subthema's beschikbaar. Controleer of het FAQ-bestand goed geladen is.")
    st.stop()

# FAQ zoekfunctie
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

# AI antwoordfunctie met retry-logica
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(openai.RateLimitError)
)
def get_ai_answer(user_text: str) -> str:
    system_prompt = (
        "You are IPAL Chatbox, a helpful Dutch helpdesk assistant. "
        "Answer questions briefly and clearly."
    )
    # Beperk chatgeschiedenis om tokens te besparen
    history_limit = 10
    messages = [{'role': 'system', 'content': system_prompt}]
    for m in st.session_state.history[-history_limit:]:
        messages.append({'role': m['role'], 'content': m['content']})
    full_question = f"[{st.session_state.selected_subthema}] {user_text}"
    messages.append({'role': 'user', 'content': full_question})
    try:
        with st.spinner("Bezig met het genereren van een AI-antwoord..."):
            resp = openai.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=150  # Verlaagd om tokens te besparen
            )
        return "🤖 AI-antwoord: " + resp.choices[0].message.content.strip()
    except openai.AuthenticationError:
        st.error("⚠️ Ongeldige OpenAI API-sleutel. Controleer je .env-bestand of Streamlit Cloud Secrets.")
        print("AuthenticationError: Invalid API key")
        return None
    except openai.RateLimitError as e:
        error_details = getattr(e, 'response', None)
        print(f"RateLimitError: {str(e)}")
        if error_details:
            headers = error_details.headers
            print(f"Rate Limit Headers: {dict(headers)}")
        st.error("⚠️ Limiet van OpenAI API bereikt, zelfs bij nul gebruik. Controleer je account op https://platform.openai.com/usage of neem contact op met OpenAI-support.")
        return None
    except openai.APIConnectionError:
        st.error("⚠️ Verbindingsprobleem met OpenAI. Controleer je internetverbinding.")
        print("APIConnectionError: Failed to connect to OpenAI")
        return None
    except Exception as e:
        st.error(f"⚠️ Er ging iets mis bij het ophalen van het AI-antwoord: {str(e)}")
        print(f"Unexpected error: {str(e)}")
        return None

# Gecombineerde antwoordfunctie
def get_answer(user_text: str) -> str:
    # Haal AI-antwoord op
    ai_answer = get_ai_answer(user_text)
    # Haal FAQ-antwoord op
    faq_answer = get_faq_answer(user_text)
    # Combineer antwoorden
    if ai_answer and faq_answer.startswith("📌"):
        return f"{ai_answer}\n\n{faq_answer}"
    elif ai_answer:
        return ai_answer
    else:
        return faq_answer

# Main UI
def main():
    render_chat()
    vraag = st.chat_input("Stel je vraag over: " + (st.session_state.selected_subthema or "(geen subthema)"))
    if vraag:
        add_message('user', vraag)
        antwoord = get_answer(vraag)
        add_message('assistant', antwoord)

if __name__ == '__main__':
    main()
