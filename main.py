# IPAL Helpdesk 2.0 — main.py (volledige versie)
# --------------------------------------------------
# Dit bestand is een compacte maar complete vervanging
# van je oude main.py (>1000 regels). Het dekt:
#  - Startopties: Exact | DocBase | Zoeken Intern | Zoeken Algemeen | Conversatie‑wizard
#  - Strikte scope (Exact/DocBase) + 8 vaste categorieën voor Exact (auto‑mapping)
#  - Robuuste CSV‑loader (UTF‑8/CP1252, "," of ";") met normalizers & smart‑quotes fix
#  - Zoeken met ranking (zwaarder op Categorie/Omschrijving/Toelichting)
#  - Paginering per 50 resultaten; “Toon volgende / vorige”
#  - Result‑kaarten met expander, kopieerbaar antwoord (st.code) en simpele ticket‑export
#  - Wizard: natuurlijke dialoog → scope → korte vraag → top‑matches → kies → antwoord + download
#  - “PDF”-download met automatische fallback naar HTML indien reportlab niet aanwezig is
#  - Kolom “Soort melding” wordt genegeerd (geen functie meer)
#  - Geen nieuwe kolomnamen nodig; volgt jouw bestaande structuur
# --------------------------------------------------

from __future__ import annotations
import os, re, json
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import streamlit as st

# ===============
# Pagina‑instellingen & stijl
# ===============
st.set_page_config(page_title="IPAL Helpdesk 2.0", page_icon="💬", layout="wide")

st.markdown(
    """
    <style>
      .block-container{padding-top:1.2rem;}
      .card{border:1px solid #e5e7eb;border-radius:16px;background:#fff;padding:1rem 1.1rem;margin:.5rem 0;box-shadow:0 3px 10px rgba(0,0,0,.04)}
      .badge{display:inline-block;background:#eef2ff;border:1px solid #e5e7eb;border-radius:999px;padding:.2rem .55rem;font-size:.78rem;margin-right:.4rem}
      .chip{display:inline-block;border:1px solid #e5e7eb;border-radius:999px;padding:.35rem .7rem;margin:.2rem .25rem}
      .kbd{background:#f6f7f9;border:1px solid #e5e7eb;border-bottom-width:2px;padding:.15rem .35rem;border-radius:6px;font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;font-size:.78rem}
      .muted{color:#6b7280}
    </style>
    """,
    unsafe_allow_html=True,
)

# ===============
# Constantes
# ===============
EIGHT_CATEGORIES = [
    "Inloggen & Beveiliging",
    "Navigatie & Administraties",
    "Koppelingen & Synchronisatie",
    "Ledenbeheer & Toezeggingen",
    "Financiële inrichting",
    "Inkoop & Verkoop",
    "Bank & Betalingen & Incasso",
    "Rapportages & Jaarafsluiting",
]

SEARCH_COLS = [
    "Systeem", "Subthema", "Categorie",
    "Omschrijving melding", "Toelichting melding",
    "Antwoord of oplossing",
]

# ===============
# CSV inlezen en opschonen
# ===============

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def read_csv_smart(path: str | Path) -> Tuple[pd.DataFrame, dict]:
    tries = [
        dict(encoding="utf-8", sep=","),
        dict(encoding="utf-8-sig", sep=","),
        dict(encoding="utf-8", sep=";"),
        dict(encoding="utf-8-sig", sep=";"),
        dict(encoding="cp1252", sep=";"),
        dict(encoding="latin1", sep=";"),
        dict(engine="python", encoding="utf-8", sep=None),
    ]
    last_err = None
    for kw in tries:
        try:
            df = pd.read_csv(path, **kw)
            return _normalize_cols(df), kw
        except Exception as e:
            last_err = e
    raise last_err


def clean_text(s: str) -> str:
    s = str(s or "")
    s = s.replace("\xa0", " ")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.replace("–", "-").replace("—", "-")
    s = (
        s.replace("Incasso?s", "Incasso's")
         .replace("creditnota?s", "creditnota's")
         .replace("financile", "financiële")
         .replace("beindig", "beëindig").replace("beindigen", "beëindigen")
         .replace("initiren", "initiëren").replace("verifiren", "verifiëren")
    )
    s = re.sub(r"^\s*\d+(?:\.\d+)*\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "Systeem","Subthema","Categorie","Omschrijving melding",
        "Toelichting melding","Antwoord of oplossing","Afbeelding",
    ]
    for c in required:
        if c not in df.columns:
            df[c] = ""
    for c in required:
        df[c] = df[c].apply(clean_text)
    # "Soort melding" wordt genegeerd
    if "Soort melding" in df.columns:
        df.drop(columns=["Soort melding"], inplace=True)
    return df

# Auto‑mapping voor Exact → 8 vaste categorieën
CAT_RULES = [
    ("Inloggen & Beveiliging", r"inloggen|wachtwoord|2fa|tweestaps|beveiliging|gebruikersrechten|autorisatie"),
    ("Navigatie & Administraties", r"hoofdmenu|navigeren|administraties|schakelen"),
    ("Koppelingen & Synchronisatie", r"koppeling|koppelen|integratie|synchronisatie|rel-?id|sila|docbase"),
    ("Ledenbeheer & Toezeggingen", r"ledenbeheer|[^a-z]lid[^a-z]|toezegging|kerkbijdrage"),
    ("Financiële inrichting", r"grootboek|dagboek|btw(?!-aang)|kostenplaats|kostendrager|rekeningschema|rekeningenschema|begroting|inrichting"),
    ("Inkoop & Verkoop", r"inkoop|verkoop|creditnota|digitale brievenbus|scan ?& ?herken|inkooporder|factuur"),
    ("Bank & Betalingen & Incasso", r"bank(?!et)|bankmut|banktrans|sepa|betaling|incasso|bic"),
    ("Rapportages & Jaarafsluiting", r"rapportage|rapporten|jaarafsluiting|jaaroverzicht|btw-?aang"),
]


def remap_exact_categories(df: pd.DataFrame) -> pd.DataFrame:
    def mapper(row):
        if str(row.get("Systeem","")) .strip().lower() != "exact":
            return row.get("Categorie", "")
        cur = row.get("Categorie", "" ).strip()
        if cur in EIGHT_CATEGORIES:
            return cur
        blob = " ".join([
            row.get("Subthema", ""), cur,
            row.get("Omschrijving melding", ""),
            row.get("Toelichting melding", ""),
        ]).lower()
        for label, pat in CAT_RULES:
            if re.search(pat, blob):
                return label
        return "Navigatie & Administraties"
    df["Categorie"] = df.apply(mapper, axis=1)
    return df

@st.cache_data(show_spinner=False)
def load_faq(path: str) -> Tuple[pd.DataFrame, dict]:
    df, meta = read_csv_smart(path)
    df = normalize_df(df)
    df = remap_exact_categories(df)
    return df, meta

# ===============
# Zoeken & ranking
# ===============

def scope_df(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope in ("Exact", "DocBase"):
        return df[df["Systeem"].str.strip().str.lower() == scope.lower()].copy()
    return df.copy()


def rank_rows(df: pd.DataFrame, q: str) -> pd.DataFrame:
    ql = (q or "").strip().lower()
    if not ql:
        return df.assign(_score=0)

    def score(row):
        s = 0
        c = str(row.get("Categorie", "" )).lower()
        om = str(row.get("Omschrijving melding", "" )).lower()
        tl = str(row.get("Toelichting melding", "" )).lower()
        sb = str(row.get("Subthema", "" )).lower()
        s += 6 if ql in c else 0
        s += 5 if ql in om else 0
        s += 3 if ql in tl else 0
        s += 1 if ql in sb else 0
        for t in [t for t in re.split(r"\W+", ql) if t]:
            if re.search(rf"\b{re.escape(t)}\b", c): s += 2
            if re.search(rf"\b{re.escape(t)}\b", om): s += 2
        return s

    out = df.copy()
    out["_score"] = out.apply(score, axis=1)
    out = out.sort_values("_score", ascending=False)
    return out[out["_score"] > 0]

# ===============
# Helpers UI / export
# ===============

def to_pdf_bytes(title: str, body: str) -> bytes | None:
    """Probeer reportlab; zo niet, return None."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm
        import io
        buff = io.BytesIO()
        c = canvas.Canvas(buff, pagesize=A4)
        width, height = A4
        y = height - 2*cm
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2*cm, y, title[:90])
        y -= 1*cm
        c.setFont("Helvetica", 10)
        for line in body.split("\n"):
            chunks = re.findall(r".{1,100}", line)
            for ch in chunks:
                if y < 2*cm:
                    c.showPage(); y = height - 2*cm; c.setFont("Helvetica",10)
                c.drawString(2*cm, y, ch)
                y -= 0.6*cm
        c.showPage(); c.save()
        return buff.getvalue()
    except Exception:
        return None


def render_hit(row: pd.Series, idx: int):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<span class='badge'>{row['Systeem']} › {row['Categorie']}</span>", unsafe_allow_html=True)
    st.markdown(f"**{row['Omschrijving melding']}**")
    if str(row.get("Toelichting melding", "" )).strip():
        st.caption(row["Toelichting melding"])
    with st.expander("Antwoord tonen", expanded=False):
        st.code(row["Antwoord of oplossing"], language="markdown")  # copy‑button aanwezig
        meta = {
            "systeem": row.get("Systeem", ""),
            "categorie": row.get("Categorie", ""),
            "omschrijving": row.get("Omschrijving melding", ""),
            "toelichting": row.get("Toelichting melding", ""),
        }
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            st.download_button(
                "Ticket‑JSON",
                data=json.dumps(meta, ensure_ascii=False, indent=2),
                file_name=f"ticket_{idx}.json",
                mime="application/json",
                use_container_width=True,
            )
        with col2:
            pdf = to_pdf_bytes(meta["omschrijving"] or "IPAL Helpdesk", row["Antwoord of oplossing"])
            if pdf:
                st.download_button("Download PDF", data=pdf, file_name=f"antwoord_{idx}.pdf", mime="application/pdf", use_container_width=True)
            else:
                html = f"<h2>{meta['omschrijving'] or 'IPAL Helpdesk'}</h2>" + f"<pre>{row['Antwoord of oplossing']}</pre>"
                st.download_button("Download HTML", data=html, file_name=f"antwoord_{idx}.html", mime="text/html", use_container_width=True)
        with col3:
            if str(row.get("Afbeelding", "" )).strip():
                st.caption("📎 Afbeelding beschikbaar in CSV")
    st.markdown("</div>", unsafe_allow_html=True)

# ===============
# Sidebar: bron, scope, admin
# ===============
DEFAULT_PATHS = ["/mnt/data/faq_fixed.csv", "/mnt/data/faq.csv"]
with st.sidebar:
    st.header("Bron & Scope")
    src = st.selectbox("CSV", DEFAULT_PATHS + ["Handmatig…"], index=0 if Path(DEFAULT_PATHS[0]).exists() else 1)
    if src == "Handmatig…":
        csv_path = st.text_input("Pad naar CSV", value="/mnt/data/faq_fixed.csv")
    else:
        csv_path = src
    up = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
    if up is not None:
        tmp = Path("/tmp/ipal_faq.csv")
        tmp.write_bytes(up.getvalue())
        csv_path = str(tmp)

    scope = st.radio("Systeem", ["Exact", "DocBase"], horizontal=True)
    admin = st.toggle("Admin", value=False)

# Inlezen CSV
try:
    DF, meta = load_faq(csv_path)
except Exception as e:
    st.error(f"Kon CSV niet inlezen: {e}")
    st.stop()

SCOPED = scope_df(DF, scope)

# ===============
# Tabs (Startopties)
# ===============
TAB_EXACT, TAB_DOCBASE, TAB_SEARCH, TAB_GENERAL, TAB_WIZ = st.tabs([
    "Exact", "DocBase", "Zoeken Intern", "Zoeken Algemeen", "Conversatie‑wizard"
])

# ---- Tab: Exact ----
with TAB_EXACT:
    left, right = st.columns([7,5])
    with left:
        st.subheader("Exact — zoeken")
        q = st.text_input("Zoek in Exact…", placeholder="Bijv. bankrekening koppelen, btw aangifte, Scan & Herken…")
        cat = st.selectbox("Categorie", options=["Alle"] + EIGHT_CATEGORIES, index=0)
        data = SCOPED
        if cat != "Alle":
            data = data[data["Categorie"] == cat]
        results = rank_rows(data, q) if q else data.head(100)

        # paginering
        key_page = "page_exact"
        if key_page not in st.session_state: st.session_state[key_page] = 0
        page_size = 50
        total = len(results)
        start = st.session_state[key_page] * page_size
        end = min(start + page_size, total)
        st.caption(f"{total} resultaten • toont {start+1 if total else 0}–{end}")
        for i, (_, row) in enumerate(results.iloc[start:end].iterrows(), start=start+1):
            render_hit(row, i)
        colA, colB, _ = st.columns([1,1,6])
        if colA.button("◀︎ Vorige", disabled=start==0):
            st.session_state[key_page] = max(0, st.session_state[key_page]-1); st.experimental_rerun()
        if colB.button("Volgende ▶︎", disabled=end>=total):
            st.session_state[key_page] += 1; st.experimental_rerun()

    with right:
        st.markdown("**Snelkoppelingen**")
        for c in ["Inloggen & Beveiliging","Bank & Betalingen & Incasso","Koppelingen & Synchronisatie","Rapportages & Jaarafsluiting"]:
            st.markdown(f"<span class='chip'>{c}</span>", unsafe_allow_html=True)
        if admin:
            st.divider()
            by_cat = SCOPED["Categorie"].value_counts().rename_axis("Categorie").reset_index(name="Aantal")
            st.dataframe(by_cat, use_container_width=True)

# ---- Tab: DocBase ----
with TAB_DOCBASE:
    left, right = st.columns([7,5])
    with left:
        st.subheader("DocBase — zoeken")
        q = st.text_input("Zoek in DocBase…", key="q_doc", placeholder="Bijv. ledenkaart, rapportages…")
        # dynamische categorieën
        dyn = sorted([x for x in SCOPED["Categorie"].dropna().unique().tolist() if x])
        cat = st.selectbox("Categorie", options=["Alle"] + dyn, index=0)
        data = SCOPED
        if cat != "Alle":
            data = data[data["Categorie"] == cat]
        results = rank_rows(data, q) if q else data.head(100)

        key_page = "page_doc"
        if key_page not in st.session_state: st.session_state[key_page] = 0
        page_size = 50
        total = len(results)
        start = st.session_state[key_page] * page_size
        end = min(start + page_size, total)
        st.caption(f"{total} resultaten • toont {start+1 if total else 0}–{end}")
        for i, (_, row) in enumerate(results.iloc[start:end].iterrows(), start=start+1):
            render_hit(row, i)
        colA, colB, _ = st.columns([1,1,6])
        if colA.button("◀︎ Vorige", disabled=start==0):
            st.session_state[key_page] = max(0, st.session_state[key_page]-1); st.experimental_rerun()
        if colB.button("Volgende ▶︎", disabled=end>=total):
            st.session_state[key_page] += 1; st.experimental_rerun()

    with right:
        if admin:
            st.markdown("**Datadiagnostiek**")
            by_cat = SCOPED["Categorie"].value_counts().rename_axis("Categorie").reset_index(name="Aantal")
            st.dataframe(by_cat, use_container_width=True)

# ---- Tab: Zoeken Intern (beide systemen) ----
with TAB_SEARCH:
    st.subheader("Zoeken Intern — Exact + DocBase")
    q = st.text_input("Zoeken in alle records…", key="q_all")
    # optionele systeemfilter
    sys_filter = st.multiselect("Beperk tot…", options=DF["Systeem"].dropna().unique().tolist(), default=["Exact","DocBase"]) 
    data = DF[DF["Systeem"].isin(sys_filter)] if sys_filter else DF
    results = rank_rows(data, q) if q else data.head(100)

    key_page = "page_all"
    if key_page not in st.session_state: st.session_state[key_page] = 0
    page_size = 50
    total = len(results)
    start = st.session_state[key_page] * page_size
    end = min(start + page_size, total)
    st.caption(f"{total} resultaten • toont {start+1 if total else 0}–{end}")
    for i, (_, row) in enumerate(results.iloc[start:end].iterrows(), start=start+1):
        render_hit(row, i)
    colA, colB, _ = st.columns([1,1,6])
    if colA.button("◀︎ Vorige", disabled=start==0):
        st.session_state[key_page] = max(0, st.session_state[key_page]-1); st.experimental_rerun()
    if colB.button("Volgende ▶︎", disabled=end>=total):
        st.session_state[key_page] += 1; st.experimental_rerun()

# ---- Tab: Zoeken Algemeen (AI‑only placeholder) ----
with TAB_GENERAL:
    st.subheader("Zoeken Algemeen (AI)")
    st.caption("Deze modus gebruikt bij voorkeur een LLM‑API. De app valt automatisch terug op CSV‑hits als er geen API‑sleutel is.")
    prompt = st.text_area("Uw vraag", height=120, placeholder="Stel uw algemene vraag hier…")

    def generate_ai_answer(q: str) -> str | None:
        # Plaatshouder — integreer hier OpenAI of Azure OpenAI.
        # Return None als er geen LLM beschikbaar is; dan val je terug op CSV‑zoek.
        return None

    if st.button("Beantwoord vraag"):
        ans = generate_ai_answer(prompt)
        if ans:
            st.success("AI‑antwoord:")
            st.write(ans)
        else:
            st.info("Geen AI‑sleutel gevonden — toon dichtstbijzijnde CSV‑hits")
            approx = rank_rows(DF, prompt).head(5)
            for i, (_, row) in enumerate(approx.iterrows(), start=1):
                render_hit(row, i)

# ---- Tab: Conversatie‑wizard ----
with TAB_WIZ:
    st.subheader("Conversatie‑wizard")

    if "wiz_step" not in st.session_state:
        st.session_state.wiz_step = 0
        st.session_state.wiz_scope = None
        st.session_state.wiz_query = ""
        st.session_state.wiz_choice_idx = None

    step = st.session_state.wiz_step

    if step == 0:
        st.write("Waarmee kan ik u van dienst zijn?")
        col1, col2, col3 = st.columns(3)
        if col1.button("Ik heb een vraag"):
            st.session_state.wiz_step = 1; st.experimental_rerun()
        if col2.button("Zoek in CSV"):
            st.session_state.wiz_step = 2; st.experimental_rerun()
        if col3.button("Algemene vraag (AI)"):
            st.session_state.wiz_step = 4; st.experimental_rerun()

    elif step == 1:
        st.write("Gaat uw vraag over **Exact**, **DocBase** of iets anders?")
        col1, col2, col3 = st.columns(3)
        if col1.button("Exact"):
            st.session_state.wiz_scope = "Exact"; st.session_state.wiz_step = 2; st.experimental_rerun()
        if col2.button("DocBase"):
            st.session_state.wiz_scope = "DocBase"; st.session_state.wiz_step = 2; st.experimental_rerun()
        if col3.button("Iets anders"):
            st.session_state.wiz_scope = None; st.session_state.wiz_step = 4; st.experimental_rerun()

    elif step == 2:
        scope_txt = st.session_state.wiz_scope or "(alle)"
        st.write(f"Geef één zin over uw onderwerp — scope: **{scope_txt}**")
        q = st.text_input("Onderwerp in één zin", key="wiz_q", value=st.session_state.wiz_query)
        if st.button("Zoek top‑matches"):
            st.session_state.wiz_query = q
            st.session_state.wiz_step = 3
            st.experimental_rerun()

    elif step == 3:
        q = st.session_state.wiz_query
        data = scope_df(DF, st.session_state.wiz_scope) if st.session_state.wiz_scope else DF
        hits = rank_rows(data, q).head(8)
        st.caption(f"Top {len(hits)} matches")
        idx_map: List[int] = []
        for i, (rid, row) in enumerate(hits.iterrows(), start=1):
            idx_map.append(rid)
            with st.expander(f"{i}. {row['Systeem']} › {row['Categorie']} — {row['Omschrijving melding']}"):
                st.caption(row["Toelichting melding"])
                if st.button(f"Kies deze", key=f"wiz_pick_{i}"):
                    st.session_state.wiz_choice_idx = rid
                    st.session_state.wiz_step = 5
                    st.experimental_rerun()
        if st.button("Terug"):
            st.session_state.wiz_step = 2; st.experimental_rerun()

    elif step == 5:
        rid = st.session_state.wiz_choice_idx
        if rid is None or rid not in DF.index:
            st.error("Keuze niet gevonden"); st.stop()
        row = DF.loc[rid]
        st.success(f"Gekozen: {row['Systeem']} › {row['Categorie']}")
        st.markdown(f"### {row['Omschrijving melding']}")
        if str(row.get("Toelichting melding", "" )).strip():
            st.caption(row["Toelichting melding"])
        st.code(row["Antwoord of oplossing"], language="markdown")
        col1, col2 = st.columns([1,1])
        with col1:
            pdf = to_pdf_bytes(row["Omschrijving melding"], row["Antwoord of oplossing"])
            if pdf:
                st.download_button("Download PDF", data=pdf, file_name="wizard_antwoord.pdf", mime="application/pdf")
            else:
                html = f"<h2>{row['Omschrijving melding']}</h2><pre>{row['Antwoord of oplossing']}</pre>"
                st.download_button("Download HTML", data=html, file_name="wizard_antwoord.html", mime="text/html")
        with col2:
            ticket = {
                "systeem": row.get("Systeem", ""),
                "categorie": row.get("Categorie", ""),
                "omschrijving": row.get("Omschrijving melding", ""),
                "toelichting": row.get("Toelichting melding", ""),
            }
            st.download_button("Ticket‑JSON", data=json.dumps(ticket, ensure_ascii=False, indent=2), file_name="wizard_ticket.json", mime="application/json")
        st.divider()
        if st.button("Nieuwe vraag"):
            for k in ["wiz_step","wiz_scope","wiz_query","wiz_choice_idx"]:
                st.session_state.pop(k, None)
            st.experimental_rerun()

# Einde bestand
