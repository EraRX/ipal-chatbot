import re

# Overgenomen blacklist-categorieën
BLACKLIST_CATEGORIES = [
    "persoonlijke gegevens", "medische gegevens", "gezondheid", "strafrechtelijk verleden",
    "financiële gegevens", "biometrische gegevens", "geboortedatum", "adresgegevens",
    "identiteitsbewijs", "burgerservicenummer", "persoonlijke overtuiging",
    "seksuele geaardheid", "etniciteit", "nationaliteit",
    "discriminatie", "racisme", "haatzaaiende taal", "xenofobie", "seksisme",
    "homofobie", "transfobie", "antisemitisme", "islamofobie", "vooroordelen",
    "stereotypering", "religie", "geloofsovertuiging", "godsdienstige leer", "religieuze extremisme",
    "sekten", "godslastering", "politiek", "politieke extremisme", "radicalisering", "terrorisme", "propaganda",
    "seksuele inhoud", "adult content", "pornografie", "seks", "sex", "seksueel",
    "seksualiteit", "erotiek", "prostitutie", "geweld", "fysiek geweld", "psychologisch geweld", "huiselijk geweld",
    "oorlog", "mishandeling", "misdaad", "illegale activiteiten", "drugs", "wapens", "smokkel",
    "desinformatie", "nepnieuws", "complottheorie", "misleiding", "fake news", "hoax",
    "gokken", "kansspelen", "verslaving", "online gokken", "casino",
    "zelfbeschadiging", "zelfmoord", "eetstoornissen", "kindermisbruik",
    "dierenmishandeling", "milieuschade", "exploitatie", "mensenhandel",
    "phishing", "malware", "hacking", "cybercriminaliteit", "doxing",
    "identiteitsdiefstal", "obsceniteit", "aanstootgevende inhoud", "schokkende inhoud",
    "gruwelijke inhoud", "sensatiezucht", "privacy schending"
]

# Één gecombineerde regex voor performance
BLACKLIST_PATTERN = re.compile(
    r"\b(" + "|".join(map(re.escape, BLACKLIST_CATEGORIES)) + r")\b",
    flags=re.IGNORECASE
)

def check_blacklist(text: str) -> list[str]:
    """Return lijst van gevonden blacklist-termen in `text`."""
    return list({m.group(0).lower() for m in BLACKLIST_PATTERN.finditer(text)})

def filter_chatbot_topics(message: str) -> tuple[bool, str]:
    """
    Checkt op verboden termen.
    Geeft (True, "") terug als alles ok is, anders (False, warning_message).
    """
    found = check_blacklist(message)
    if not found:
        return True, ""
    warning = (
        "Je bericht bevat inhoud die niet voldoet aan onze richtlijnen. "
        "Vermijd gevoelige onderwerpen en probeer het opnieuw."
    )
    return False, warning
