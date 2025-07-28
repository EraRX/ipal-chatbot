import os
import logging

import pandas as pd
import streamlit as st

class FAQLoadError(Exception):
    """Custom exception voor fouten bij het laden van de FAQ."""
    pass

def load_faq(path: str = "faq.xlsx") -> pd.DataFrame:
    """
    Laad de FAQ uit een Excel-bestand en maak de velden 'combined' en 'Antwoord' aan.
    Gooit FAQLoadError als lezen mislukt.
    """
    if not os.path.exists(path):
        logging.error(f"FAQ niet gevonden: {path}")
        st.error(f"FAQ-bestand '{path}' niet gevonden.")
        return pd.DataFrame(columns=["Systeem", "Subthema", "combined", "Antwoord", "Afbeelding"])

    try:
        df = pd.read_excel(path)
    except Exception as e:
        logging.error(f"Fout bij laden FAQ: {e}")
        raise FAQLoadError(f"Kan FAQ niet laden: {e}")

    # Zorg dat kolom 'Afbeelding' bestaat
    if "Afbeelding" not in df.columns:
        df["Afbeelding"] = None

    # Nieuwe kolommen
    df["Antwoord"] = df["Antwoord of oplossing"]
    required = ["Systeem", "Subthema", "Omschrijving melding", "Toelichting melding"]
    df["combined"] = df[required].fillna("").agg(" ".join, axis=1)

    return df
