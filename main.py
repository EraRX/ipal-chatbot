# main.py (test-imports)

import os
import streamlit as st
from faq import load_faq
from ai_client import get_ai_answer, rewrite_answer
from filters import filter_chatbot_topics
from pdf_utils import genereer_pdf

def main():
    st.write("Imports OK!")
    
if __name__ == "__main__":
    main()
