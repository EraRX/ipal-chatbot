# main.py

import os, re, logging, io, textwrap
from datetime import datetime

import streamlit as st
import pandas as pd
import pytz
import requests
from bs4 import BeautifulSoup
from PIL import Image as PILImage
from dotenv import load_dotenv
import openai

# Fallback import for RateLimitError
try:
    from openai.error import RateLimitError
except ImportError:
    RateLimitError = Exception

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# --- Config & Styling ---
st.set_page_config(page_title="IPAL Chatbox", layout="centered")
st.markdown("""
  <style>
    html, body, [class*="css"] { font-size:20px; }
    button[kind="primary"] { font-size:22px !important; padding:.75em 1.5em; }
  </style>
""", unsafe_allow_html=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- OpenAI Setup ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not openai.api_key:
    st.sidebar.error("🔑 Voeg je OpenAI API-key toe in .env of Secrets.")
    st.stop()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(1,10),
       retry=retry_if_exception_type(RateLimitError))
def chatgpt(messages, temperature=0.3, max_tokens=800):
    resp = openai.chat.completions.create(
        model=MODEL, messages=messages,
        temperature=temperature, max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

# --- FAQ Loader ---
@st.cache_data
def load_faq(path="faq.xlsx"):
    if not os.path.exists(path):
        st.error(f"⚠️ FAQ '{path}' niet gevonden")
        return pd.DataFrame(columns=["combined","Antwoord","Afbeelding"])
    df = pd.read_excel(path, engine="openpyxl")
    if "Afbeelding" not in df.columns: df["Afbeelding"]=None
    keys=["Systeem","Subthema","Omschrijving melding","Toelichting melding"]
    df["combined"]=df[keys].fillna("").agg(" ".join, axis=1)
    df["Antwoord"]=df["Antwoord of oplossing"]
    return df[["combined","Antwoord","Afbeelding"]]

faq_df = load_faq()

# --- Blacklist ---
BLACKLIST=["persoonlijke gegevens","medische gegevens","gezondheid","privacy schending"]
def filter_topics(msg):
    found=[t for t in BLACKLIST if re.search(rf"\b{re.escape(t)}\b", msg.lower())]
    return (False, f"Je bericht bevat gevoelige onderwerpen: {', '.join(found)}.") if found else (True,"")

# --- PDF Export ---
def make_pdf(text):
    buf=io.BytesIO()
    c=canvas.Canvas(buf,pagesize=A4)
    w,h=A4; m=40; uw=w-2*m
    logo="logo.png"; y0=h-50
    if os.path.exists(logo):
        img=PILImage.open(logo); ar=img.width/img.height; lh=50
        c.drawImage(logo,m,h-lh-10,width=lh*ar,height=lh,mask="auto"); y0=h-lh-30
    t=c.beginText(m,y0); t.setFont("Helvetica",12)
    maxc=int(uw/(12*0.6))
    for p in text.split("\n"):
        for l in textwrap.wrap(p,width=maxc):
            t.textLine(l)
    c.drawText(t); c.showPage(); c.save(); buf.seek(0)
    return buf.getvalue()

# --- RKK Scraping ---
def fetch_bishop_from_rkkerk(loc):
    slug=loc.lower().replace(" ","-")
    url=f"https://www.rkkerk.nl/bisdom-{slug}/"
    try:
        r=requests.get(url,timeout=10); r.raise_for_status()
        s=BeautifulSoup(r.text,"html.parser")
        h1=s.find("h1")
        if h1 and "bisschop" in h1.text.lower():
            return h1.text.strip().split("—")[0].strip()
    except: pass
    return None

def fetch_bishop_from_rkk_online(loc):
    # zoekt op rkk-online.nl via zoek-URL en parseert resultaat
    query=loc.replace(" ","+")
    url=f"https://www.rkk-online.nl/?s={query}"
    try:
        r=requests.get(url,timeout=10); r.raise_for_status()
        s=BeautifulSoup(r.text,"html.parser")
        # zoek eerste kop met bisschop
        h=s.find(lambda tag: tag.name in ["h1","h2","h3"] and "bisschop" in tag.text.lower())
        if h:
            return h.text.strip().split("–")[0].strip()
    except: pass
    return None

def fetch_all_bishops_nl():
    dioceses=["Utrecht","Haarlem-Amsterdam","Rotterdam","Groningen-Leeuwarden",
              "’s-Hertogenbosch","Roermond","Breda"]
    res={}
    for d in dioceses:
        name=fetch_bishop_from_rkkerk(d) or fetch_bishop_from_rkk_online(d)
        if name: res[d]=name
    return res

# --- Avatars & Chat Helpers ---
AVATARS={"assistant":"aichatbox.jpg","user":"parochie.jpg"}
def get_avatar(role):
    p=AVATARS.get(role)
    return (PILImage.open(p).resize((64,64)) if p and os.path.exists(p) else "🙂")

TIMEZONE=pytz.timezone("Europe/Amsterdam"); MAX_HISTORY=20
def add_msg(r,c):
    ts=datetime.now(TIMEZONE).strftime("%d-%m-%Y %H:%M")
    st.session_state.history=(st.session_state.history+[{"role":r,"content":c,"time":ts}])[-MAX_HISTORY:]

def render_chat():
    for m in st.session_state.history:
        st.chat_message(m["role"],avatar=get_avatar(m["role"])).markdown(
            f"{m['content']}\n\n_{m['time']}_"
        )

if "history" not in st.session_state:
    st.session_state.history=[]; st.session_state.selected_product=None

# --- Main ---
def main():
    if st.sidebar.button("🔄 Nieuw gesprek"):
        st.session_state.clear(); st.rerun()

    if st.session_state.history and st.session_state.history[-1]["role"]=="assistant":
        st.sidebar.download_button("📄 Download PDF",
            data=make_pdf(st.session_state.history[-1]["content"]),
            file_name="antwoord.pdf", mime="application/pdf")

    if not st.session_state.selected_product:
        st.header("Welkom bij IPAL Chatbox")
        c1,c2,c3=st.columns(3)
        if c1.button("Exact",use_container_width=True):
            st.session_state.selected_product="Exact"; add_msg("assistant","Gekozen: Exact"); st.rerun()
        if c2.button("DocBase",use_container_width=True):
            st.session_state.selected_product="DocBase"; add_msg("assistant","Gekozen: DocBase"); st.rerun()
        if c3.button("Algemeen",use_container_width=True):
            st.session_state.selected_product="Algemeen"; add_msg("assistant","Gekozen: Algemeen"); st.rerun()
        render_chat(); return

    render_chat()
    vraag=st.chat_input("Stel uw vraag:")
    if not vraag: return

    add_msg("user",vraag)
    ok,warn=filter_topics(vraag)
    if not ok:
        add_msg("assistant",warn); st.rerun()

    # specifieke bisschop-vraag
    m=re.match(r'(?i)wie is bisschop(?: van)?\s+(.+)\?*',vraag)
    if m:
        loc=m.group(1).strip()
        bishop=fetch_bishop_from_rkkerk(loc) or fetch_bishop_from_rkk_online(loc)
        if bishop:
            ans=f"De huidige bisschop van {loc} is {bishop}."
            add_msg("assistant",ans); st.rerun()

    # bisschoppen NL overzicht
    if re.search(r'(?i)bisschoppen nederland',vraag):
        allb=fetch_all_bishops_nl()
        if allb:
            lines=[f"Mgr. {n} – Bisschop van {d}" for d,n in allb.items()]
            add_msg("assistant","Huidige Nederlandse bisschoppen:\n"+"\n".join(lines))
            st.rerun()

    # FAQ lookup
    dfm=faq_df[faq_df["combined"].str.contains(re.escape(vraag),case=False,na=False)]
    if not dfm.empty:
        row=dfm.iloc[0]; ans=row["Antwoord"]
        try:
            ans=chatgpt([{"role":"system","content":"Herschrijf dit eenvoudig."},
                         {"role":"user","content":ans}], temperature=0.2)
        except: pass
        if img:=row["Afbeelding"]:
            st.image(img,caption="Voorbeeld",use_column_width=True)
        add_msg("assistant",ans); st.rerun()

    # AI fallback
    with st.spinner("ChatGPT even aan het werk…"):
        try:
            ai=chatgpt([{"role":"system","content":"Je bent een behulpzame assistent."},
                        {"role":"user","content":vraag}])
            add_msg("assistant",ai)
        except Exception as e:
            logging.exception("AI-fallback mislukt")
            add_msg("assistant",f"⚠️ AI-fallback mislukt: {e}")
    st.rerun()

if __name__=="__main__":
    main()
