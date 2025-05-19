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
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------
# IPAL Directe Interactieve Chatbox
# ---------------------------------------------
# Vereisten: streamlit, openai, pandas, pillow, python-dotenv, tenacity, openpyxl, requests, beautifulsoup4

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

# Genereer product-opties
producten = sorted([p for p in faq_df['Systeem'].dropna().unique() if isinstance(p, str) and p.strip()]) if not faq_df.empty else []

# Controleer of FAQ correct is geladen
if faq_df.empty or not producten:
    st.error("⚠️ FAQ-bestand is niet correct geladen of bevat geen geldige producten. Controleer het bestand en probeer opnieuw.")
    st.stop()

# Initialiseren van sessiestatus
if 'history' not in st.session_state:
    st.session_state.history = []
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None
if 'reset_triggered' not in st.session_state:
    st.session_state.reset_triggered = False

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
    st.session_state.reset_triggered = True
    st.session_state.selected_product = None

# Functie om informatie van de Exact kennisbank te halen (placeholder)
def get_exact_knowledge_base_info(query: str) -> str:
    knowledge_base_url = "https://support.exactonline.com/community/s/knowledge-base#All-All-HNO-Landing-accounts-acc-lndngpgnwuil"
    try:
        response = requests.get(knowledge_base_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            content = soup.find_all('p')
            relevant_info = "Geen specifieke informatie gevonden."
            for p in content:
                if query.lower() in p.text.lower():
                    relevant_info = p.text.strip()[:500]
                    break
            return relevant_info
        else:
            return f"Kon de Exact kennisbank niet bereiken. Bezoek de [Exact Kennisbank]({knowledge_base_url}) voor meer informatie."
    except Exception as e:
        print(f"Error accessing Exact knowledge base: {str(e)}")
        return f"Kon de Exact kennisbank niet bereiken. Bezoek de [Exact Kennisbank]({knowledge_base_url}) voor meer informatie."

# FAQ zoekfunctie
def get_faq_answer(user_text: str) -> str:
    if not faq_df.empty:
        try:
            # Filter FAQ op basis van het geselecteerde product
            df_filtered = faq_df[faq_df['Systeem'] == st.session_state.selected_product]
            if df_filtered.empty:
                return "Geen FAQ-items gevonden voor dit product."
            
            pattern = re.escape(user_text)
            matches = df_filtered[df_filtered['combined'].str.contains(pattern, case=False, na=False, regex=True)]
            if not matches.empty:
                top = matches.head(3)['Antwoord of oplossing'].tolist()
                return "Hier zijn enkele relevante FAQ-antwoorden:\n" + "\n".join(f"- {ans}" for ans in top)
        except Exception as e:
            print(f"FAQ search error: {str(e)}")
    return "Geen antwoord gevonden in de FAQ. Probeer je vraag anders te formuleren."

# AI antwoordfunctie met retry-logica
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(openai.RateLimitError)
)
def get_ai_answer(user_text: str) -> str:
    keywords = ['exact', 'docbase', 'doc base', 'documentbase', 'doc-base']
    user_text_lower = user_text.lower()
    is_relevant = any(keyword in user_text_lower for keyword in keywords) or st.session_state.selected_product.lower() in user_text_lower

    if not is_relevant:
        return "Ik kan alleen vragen beantwoorden over Exact en DocBase. Waarmee kan ik u helpen?"

    system_prompt = (
        "You are IPAL Chatbox, a helpful Dutch helpdesk assistant. "
        "Answer questions in a friendly and conversational tone. "
        "Start your answers with a friendly phrase like 'Laat me je helpen!' or 'Goede vraag!'. "
        "Only answer questions related to Exact and DocBase software. "
        "If the question is not related to Exact or DocBase, respond with: "
        "'Ik kan alleen vragen beantwoorden over Exact en DocBase. Waarmee kan ik u helpen?'"
    )
    history_limit = 10
    messages = [{'role': 'system', 'content': system_prompt}]
    for m in st.session_state.history[-history_limit:]:
        messages.append({'role': m['role'], 'content': m['content']})
    full_question = f"[{st.session_state.selected_product}] {user_text}"
    messages.append({'role': 'user', 'content': full_question})
    try:
        with st.spinner("Even nadenken..."):
            resp = openai.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.5,
                max_tokens=1000
            )
        return resp.choices[0].message.content.strip()
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
        st.error(f"⚠️ Er ging iets mis bij het ophalen van het antwoord: {str(e)}")
        print(f"Unexpected error: {str(e)}")
        return None

# Gecombineerde antwoordfunctie
def get_answer(user_text: str) -> str:
    keywords = ['exact', 'docbase', 'doc base', 'documentbase', 'doc-base']
    user_text_lower = user_text.lower()
    is_relevant = any(keyword in user_text_lower for keyword in keywords) or st.session_state.selected_product.lower() in user_text_lower

    if not is_relevant:
        return "Ik kan alleen vragen beantwoorden over Exact en DocBase. Waarmee kan ik u helpen?"

    knowledge_base_answer = None
    if "exact" in st.session_state.selected_product.lower():
        knowledge_base_answer = get_exact_knowledge_base_info(user_text)

    faq_answer = get_faq_answer(user_text)
    ai_answer = get_ai_answer(user_text)

    final_answer = ""
    if knowledge_base_answer and "Kon de Exact kennisbank niet bereiken" not in knowledge_base_answer:
        final_answer += f"📚 Informatie uit de Exact Kennisbank:\n{knowledge_base_answer}\n\n"
    elif knowledge_base_answer:
        final_answer += f"{knowledge_base_answer}\n\n"
    
    if ai_answer and "Ik kan alleen vragen beantwoorden" not in ai_answer:
        final_answer += f"{ai_answer}\n\n"
    
    if "Hier zijn enkele relevante FAQ-antwoorden" in faq_answer:
        final_answer += faq_answer

    if not final_answer.strip():
        final_answer = "Ik kon helaas geen specifiek antwoord vinden. Probeer je vraag anders te formuleren of bezoek de [Exact Kennisbank](https://support.exactonline.com/community/s/knowledge-base#All-All-HNO-Landing-accounts-acc-lndngpgnwuil) voor meer informatie."

    return final_answer

# Main UI
def main():
    if st.session_state.reset_triggered:
        st.session_state.history = []
        st.session_state.selected_product = None
        st.session_state.reset_triggered = False

    st.sidebar.button('🔄 Nieuw gesprek', on_click=on_reset)

    if not st.session_state.selected_product:
        # Toon alleen de welkomstboodschap als er nog geen product is gekozen
        if not st.session_state.history:
            add_message('assistant', 'Hallo, ik ben de IPAL AI-assistent, waarmee kan ik u helpen?')

        st.markdown("### Welkom bij de IPAL-Helpdesk:")
        st.markdown("""
            <style>
            .logo-container {
                display: flex;
                justify-content: center;
                gap: 20px;
                margin-top: 20px;
            }
            .stButton > button {
                width: 120px;
                height: 30px;
                font-size: 14px;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                background-color: #f0f0f0;
                color: #333;
                cursor: pointer;
                margin-top: 5px;
            }
            .stButton > button:hover {
                background-color: #e0e0e0;
            }
            </style>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            if os.path.exists("logo-docbase-icon.png"):
                st.image("logo-docbase-icon.png", use_container_width=False, width=120)
                if st.button("Klik hier", key="docbase_button"):
                    st.session_state.selected_product = "DocBase"
                    st.session_state.history = []  # Reset geschiedenis
                    add_message('assistant', f"Hallo, ik ben de IPAL AI-assistent. Je hebt DocBase gekozen. Stel nu je vraag hieronder.")
                    st.rerun()
            else:
                st.warning("⚠️ Logo 'logo-docbase-icon.png' niet gevonden in de repository.")

        with col2:
            if os.path.exists("Exact.png"):
                st.image("Exact.png", use_container_width=False, width=120)
                if st.button("Klik hier", key="exact_button"):
                    st.session_state.selected_product = "Exact Online (Module: Bank)"  # Specifieke module
                    st.session_state.history = []  # Reset geschiedenis
                    add_message('assistant', f"Hallo, ik ben de IPAL AI-assistent. Je hebt Exact Online (Module: Bank) gekozen. Stel nu je vraag hieronder.")
                    st.rerun()
            else:
                st.warning("⚠️ Logo 'Exact.png' niet gevonden in de repository.")

        render_chat()
        return

    # Toon chat en verwerk vragen
    render_chat()
    
    vraag = st.chat_input(
        f"Stel je vraag over {st.session_state.selected_product}:",
        key="chat_input_" + str(len(st.session_state.history))
    )
    
    if vraag:
        add_message('user', vraag)
        with st.spinner("Even nadenken..."):
            antwoord = get_answer(vraag)
            add_message('assistant', antwoord)
        st.rerun()

if __name__ == '__main__':
    main()
