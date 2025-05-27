python

import unittest
import pandas as pd
from unittest.mock import patch
from main import filter_chatbot_topics, load_faq, get_answer

class TestIPALChatbox(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        import streamlit as st
        st.session_state.clear()
        st.session_state.selected_module = "ledenadministratie"

    def test_filter_chatbot_topics_whitelist(self):
        """Test whitelist topic filtering."""
        allowed, reason = filter_chatbot_topics("Hoe wijzig ik een adres in de parochie?")
        self.assertTrue(allowed)
        self.assertEqual(reason, "")

    def test_filter_chatbot_topics_blacklist(self):
        """Test blacklist topic filtering."""
        allowed, reason = filter_chatbot_topics("Wat is de politieke situatie?")
        self.assertFalse(allowed)
        self.assertEqual(reason, "Geblokkeerd: bevat verboden onderwerp 'politiek'")

    def test_filter_chatbot_topics_invalid(self):
        """Test invalid topic filtering."""
        allowed, reason = filter_chatbot_topics("Random onderwerp")
        self.assertFalse(allowed)
        self.assertEqual(reason, "⚠️ Geen geldig onderwerp voor AI-ondersteuning")

    @patch('pandas.read_excel')
    def test_load_faq_missing_file(self, mock_read_excel):
        """Test FAQ loading with missing file."""
        import os
        with patch('os.path.exists', return_value=False):
            df = load_faq('faq.xlsx')
            self.assertTrue(df.empty)
            self.assertEqual(list(df.columns), ['combined', 'Antwoord'])

    @patch('pandas.read_excel')
    def test_load_faq_missing_columns(self, mock_read_excel):
        """Test FAQ loading with missing required columns."""
        mock_read_excel.return_value = pd.DataFrame({'Systeem': ['Exact']})
        df = load_faq('faq.xlsx')
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), ['combined', 'Antwoord'])

    @patch('main.get_ai_answer')
    def test_get_answer_module_definition(self, mock_get_ai_answer):
        """Test module definition fallback."""
        import streamlit as st
        st.session_state.selected_module = "ledenadministratie"
        mock_get_ai_answer.return_value = "Ledenadministratie beheert parochieleden."
        result = get_answer("ledenadministratie")
        self.assertEqual(result, "IPAL-Helpdesk antwoord:\nLedenadministratie beheert parochieleden.")

    @patch('main.get_ai_answer')
    def test_get_answer_whitelist(self, mock_get_ai_answer):
        """Test AI fallback for whitelisted topic."""
        import streamlit as st
        st.session_state.selected_module = "ledenadministratie"
        mock_get_ai_answer.return_value = "Adreswijziging wordt verwerkt in SILA."
        result = get_answer("Hoe wijzig ik een adres?")
        self.assertEqual(result, "IPAL-Helpdesk antwoord:\nAdreswijziging wordt verwerkt in SILA.")

if __name__ == '__main__':
    unittest.main()

