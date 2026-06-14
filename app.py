"""
GAIL's Bakery — Dataroom AI Assistant
======================================
Golborne Capital AI Engineer Case Study

Architecture: Full-context retrieval
  - All 13 dataroom documents loaded at startup (~7,000 words)
  - Every question sends the complete dataroom to Claude as context
  - No embeddings, no vector database, no second API key needed
  - One service (Anthropic), one key, nothing to break
"""

import os
import glob
import streamlit as st
import anthropic

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────

DATAROOM_DIR = "dataroom"
CHAT_MODEL   = "claude-sonnet-4-6"
MAX_TOKENS   = 1500

# ─────────────────────────────────────────────────────────
# LOAD DOCUMENTS
# ─────────────────────────────────────────────────────────

def load_documents(dataroom_dir: str) -> list[dict]:
    documents = []
    md_files = sorted(glob.glob(os.path.join(dataroom_dir, "*.md")))
    for filepath in md_files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        first_line = content.split("\n")[0].strip()
        title = first_line.lstrip("#").strip() if first_line.startswith("#") else filename
        documents.append({
            "filename": filename,
            "title":    title,
            "content":  content
        })
    return documents

# ─────────────────────────────────────────────────────────
# BUILD CONTEXT STRING
# ─────────────────────────────────────────────────────────

def build_context(documents: list[dict]) -> str:
    parts = []
    for doc in documents:
        parts.append(
            f"=== SOURCE DOCUMENT: {doc['filename']} ===\n"
            f"Title: {doc['title']}\n\n"
            f"{doc['content']}"
        )
    return "\n\n" + "="*60 + "\n\n".join(parts)

# ─────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a financial analyst assistant for Golborne Capital.
You have been given the complete dataroom for GAIL'S LIMITED (Companies House: 06055393),
a UK premium artisan bakery chain trading as GAIL's Bakery.

The dataroom contains 13 source documents covering financials, charges, ownership,
management, news, risks, and credit analysis — all sourced from public information
including Companies House filings and attributed press reporting.

RULES — follow these precisely:

1. ONLY use information from the dataroom documents provided. No outside knowledge.

2. CITE YOUR SOURCE for every factual claim using this exact format:
   [Source: filename.md]
   Example: GAIL's retail revenue was £219.8m in FY2025 [Source: 02_financials_fy2025.md]

3. FINANCIAL FIGURES must be quoted exactly as they appear in the source.
   Never round, estimate or approximate unless the source itself uses that language.
   Never invent a figure that is not explicitly stated in the documents.

4. IF THE ANSWER IS NOT IN THE DATAROOM, say exactly:
   "The dataroom does not contain sufficient information to answer this question."
   Do not guess. Do not use general knowledge to fill gaps.

5. FOR DRAFTING TASKS (credit summary, risk analysis etc.):
   Use only facts from the dataroom. Cite sources throughout.

6. FORMAT: Be clear and professional. Use bullet points for lists.
   Use tables where they add clarity. Keep answers focused and concise.

The complete dataroom follows below.

{dataroom}
"""

# ─────────────────────────────────────────────────────────
# GENERATE ANSWER
# ─────────────────────────────────────────────────────────

def generate_answer(question: str, system: str,
                    history: list[dict],
                    client: anthropic.Anthropic) -> str:
    messages = history + [{"role": "user", "content": question}]
    response = client.messages.create(
        model      = CHAT_MODEL,
        max_tokens = MAX_TOKENS,
        system     = system,
        messages   = messages
    )
    return response.content[0].text

# ─────────────────────────────────────────────────────────
# STREAMLIT UI — ChatGPT style
# ─────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title = "GAIL's Dataroom",
        page_icon  = "🍞",
        layout     = "wide",
        initial_sidebar_state = "expanded"
    )

    # ── Global CSS ────────────────────────────────────────
    st.markdown("""
    <style>
        /* Hide Streamlit default chrome — but NOT the whole header, because the
           ">" control that re-opens a collapsed sidebar lives inside it. */
        #MainMenu, footer { visibility: hidden; }
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] { display: none !important; }
        header[data-testid="stHeader"] { background: transparent !important; }
        /* The "<<" sidebar-collapse button is removed entirely, because once the
           sidebar is collapsed the re-open control is unreliable in this build.
           With no collapse button, the sidebar always stays open. */
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        .block-container {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            max-width: 100% !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #000000;
            border-right: 1px solid #1f1f1f;
        }
        [data-testid="stSidebar"] * {
            color: #d1d5db !important;
        }
        [data-testid="stSidebar"] h3 {
            color: #ffffff !important;
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
            margin-bottom: 0.5rem !important;
        }
        [data-testid="stSidebar"] small {
            color: #9ca3af !important;
            font-size: 0.78rem !important;
            line-height: 1.8 !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: #2a2a2a !important;
            margin: 0.8rem 0 !important;
        }

        /* Sidebar buttons — example questions */
        [data-testid="stSidebar"] .stButton > button {
            background: #1a1a1a !important;
            border: 1px solid #2a2a2a !important;
            color: #9ca3af !important;
            font-size: 0.78rem !important;
            text-align: left !important;
            padding: 0.4rem 0.6rem !important;
            border-radius: 6px !important;
            margin-bottom: 2px !important;
            transition: all 0.15s !important;
            white-space: normal !important;
            height: auto !important;
            line-height: 1.4 !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: #2a2a2a !important;
            border-color: #3a3a3a !important;
            color: #e5e7eb !important;
        }
        [data-testid="stSidebar"] .stButton > button:focus:not(:active) {
            background: #1a1a1a !important;
            color: #9ca3af !important;
            box-shadow: none !important;
        }
        /* Sidebar expanders — document viewer */
        [data-testid="stSidebar"] .streamlit-expanderHeader {
            background: #1a1a1a !important;
            border: 1px solid #2a2a2a !important;
            border-radius: 6px !important;
            color: #9ca3af !important;
            font-size: 0.78rem !important;
            padding: 0.4rem 0.6rem !important;
            margin-bottom: 2px !important;
        }
        [data-testid="stSidebar"] .streamlit-expanderHeader:hover {
            background: #2a2a2a !important;
            color: #e5e7eb !important;
        }
        [data-testid="stSidebar"] .streamlit-expanderHeader p {
            color: inherit !important;
            font-size: 0.78rem !important;
        }
        [data-testid="stSidebar"] .streamlit-expanderContent {
            background: #111111 !important;
            border: 1px solid #2a2a2a !important;
            border-top: none !important;
            border-radius: 0 0 6px 6px !important;
            padding: 0.6rem !important;
            margin-bottom: 2px !important;
        }
        [data-testid="stSidebar"] .streamlit-expanderContent p,
        [data-testid="stSidebar"] .streamlit-expanderContent li {
            color: #9ca3af !important;
            font-size: 0.75rem !important;
            line-height: 1.6 !important;
        }
        [data-testid="stSidebar"] .streamlit-expanderContent strong {
            color: #d1d5db !important;
        }

        /* Main area background */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stHeader"],
        .main .block-container {
            background-color: #000000 !important;
        }

        /* Bottom container that wraps the chat input */
        [data-testid="stBottom"],
        [data-testid="stBottomBlockContainer"],
        [data-testid="stBottom"] > div {
            background-color: #000000 !important;
        }

        /* Chat messages */
        [data-testid="stChatMessage"] {
            background: transparent !important;
            border: none !important;
            padding: 0.75rem 0 !important;
            max-width: 48rem;
            margin: 0 auto;
        }

        /* User message bubble */
        [data-testid="stChatMessage"][data-testid*="user"],
        .stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
            background: #2f2f2f !important;
            border-radius: 12px !important;
            padding: 0.75rem 1rem !important;
        }

        /* Message text */
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li,
        [data-testid="stChatMessage"] td,
        [data-testid="stChatMessage"] th {
            color: #ececec !important;
            font-size: 0.95rem !important;
            line-height: 1.7 !important;
        }

        /* Tables in answers */
        [data-testid="stChatMessage"] table {
            border-collapse: collapse !important;
            width: 100% !important;
            margin: 0.75rem 0 !important;
            font-size: 0.85rem !important;
        }
        [data-testid="stChatMessage"] th {
            background: #2f2f2f !important;
            color: #e5e7eb !important;
            font-weight: 600 !important;
            padding: 0.5rem 0.75rem !important;
            border: 1px solid #3a3a3a !important;
        }
        [data-testid="stChatMessage"] td {
            padding: 0.45rem 0.75rem !important;
            border: 1px solid #3a3a3a !important;
            color: #d1d5db !important;
        }
        [data-testid="stChatMessage"] tr:nth-child(even) td {
            background: #1a1a1a !important;
        }

        /* Code / source citations */
        [data-testid="stChatMessage"] code {
            background: #2f2f2f !important;
            color: #a78bfa !important;
            padding: 0.1rem 0.3rem !important;
            border-radius: 4px !important;
            font-size: 0.82rem !important;
        }

        /* Chat input */
        [data-testid="stChatInput"] {
            background: #000000 !important;
            border: 1px solid #2a2a2a !important;
            border-radius: 12px !important;
            max-width: 48rem !important;
            margin: 0 auto !important;
        }
        [data-testid="stChatInput"] textarea {
            color: #ececec !important;
            background: transparent !important;
            font-size: 0.95rem !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: #6b7280 !important;
        }

        /* Clear button */
        .clear-btn .stButton > button {
            background: transparent !important;
            border: 1px solid #3a3a3a !important;
            color: #6b7280 !important;
            font-size: 0.78rem !important;
            border-radius: 6px !important;
            padding: 0.3rem 0.8rem !important;
        }
        .clear-btn .stButton > button:hover {
            border-color: #ef4444 !important;
            color: #ef4444 !important;
        }

        /* Status / success messages */
        [data-testid="stAlert"] {
            max-width: 48rem;
            margin: 0 auto;
        }

        /* Metrics */
        [data-testid="stMetric"] {
            background: #2f2f2f;
            border-radius: 8px;
            padding: 0.6rem 1rem;
        }
        [data-testid="stMetric"] label {
            color: #9ca3af !important;
            font-size: 0.72rem !important;
        }
        [data-testid="stMetricValue"] {
            color: #e5e7eb !important;
            font-size: 1rem !important;
        }

        /* Spinner */
        .stSpinner > div {
            border-top-color: #7c3aed !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🍞 GAIL's Dataroom")
        st.markdown("<small>GAIL'S LIMITED · 06055393</small>", unsafe_allow_html=True)
        st.divider()

        st.markdown("### Documents")
        # Map display name → filename in the dataroom folder
        doc_map = [
            ("01 · Company Overview",        "01_company_overview.md"),
            ("02 · FY2025 Financials",       "02_financials_fy2025.md"),
            ("03 · FY2024 Financials",       "03_financials_fy2024.md"),
            ("04 · FY2023 & Historical",     "04_financials_fy2023_historical.md"),
            ("05 · Charges Register",        "05_charges_register.md"),
            ("06 · Ownership & Management",  "06_ownership_management.md"),
            ("07 · News & Events",           "07_news_events.md"),
            ("08 · Subsidiary Accounts Note","08_subsidiary_accounts_note.md"),
            ("09 · Lender Risks",            "09_lender_risks.md"),
            ("10 · Credit Summary",          "10_credit_summary.md"),
            ("11 · Bain Capital Acquisition","11_bain_capital_acquisition.md"),
            ("12 · Sale Process & Valuation","12_sale_process_valuation.md"),
            ("13 · Dataroom Index",          "13_dataroom_index.md"),
        ]
        for display_name, filename in doc_map:
            filepath = os.path.join(DATAROOM_DIR, filename)
            with st.expander(display_name):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                    st.markdown(content)
                except Exception:
                    st.markdown("_Document not found._")

        st.divider()
        st.markdown("### Try asking")
        example_qs = [
            "What was revenue and EBITDA in the last reported year?",
            "What charges are registered and who holds them?",
            "Who are the current directors?",
            "What are the key risks for a lender?",
            "Draft a short credit summary.",
            "Who owns GAIL's?",
            "How many sites does GAIL's operate?",
            "What is GAIL's net debt?",
        ]
        for q in example_qs:
            if st.button(q, key=f"eg_{q[:25]}", use_container_width=True):
                st.session_state.pending_question = q

        st.divider()
        st.markdown(
            "<small>Sources: Companies House · Grain Topco accounts · "
            "British Baker · The Grocer · Sky News · Bloomberg</small>",
            unsafe_allow_html=True
        )

    # ── API key ───────────────────────────────────────────
    api_key = None
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        st.error("⚠️ No API key. Set ANTHROPIC_API_KEY in Streamlit secrets.")
        st.stop()

    client = anthropic.Anthropic(api_key=api_key)

    # ── Session state ─────────────────────────────────────
    if "ready" not in st.session_state:
        st.session_state.ready            = False
        st.session_state.system_prompt    = None
        st.session_state.messages         = []
        st.session_state.pending_question = None
        st.session_state.doc_count        = 0
        st.session_state.word_count       = 0

    # ── Load documents once ───────────────────────────────
    if not st.session_state.ready:
        with st.spinner("Loading dataroom..."):
            try:
                docs    = load_documents(DATAROOM_DIR)
                context = build_context(docs)
                system  = SYSTEM_PROMPT.format(dataroom=context)
                st.session_state.system_prompt = system
                st.session_state.ready         = True
                st.session_state.doc_count     = len(docs)
                st.session_state.word_count    = len(context.split())
            except Exception as e:
                st.error(f"Failed to load documents: {e}")
                st.stop()

    # ── Welcome screen (empty state) ──────────────────────
    if not st.session_state.messages:
        st.markdown("""
        <div style='text-align:center; padding: 5rem 1rem 2rem 1rem;'>
            <div style='font-size:2.5rem; margin-bottom:0.5rem;'>🍞</div>
            <h2 style='color:#ececec; font-size:1.4rem; font-weight:600; margin:0 0 0.5rem 0;'>
                GAIL's Bakery Dataroom
            </h2>
            <p style='color:#6b7280; font-size:0.9rem; max-width:28rem; margin:0 auto;'>
                Ask questions about GAIL'S LIMITED — financials, charges, ownership,
                risks, or request a credit summary. All answers cite their source.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Stats row
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            st.metric("Documents", st.session_state.doc_count)
        with col2:
            st.metric("Words in context", f"{st.session_state.word_count:,}")
        with col3:
            st.metric("Company", "06055393")

    # ── Chat history ──────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Pending question from sidebar ─────────────────────
    if st.session_state.get("pending_question"):
        question = st.session_state.pending_question
        st.session_state.pending_question = None
    else:
        question = st.chat_input("Ask anything about GAIL's Bakery...")

    # ── Process question ──────────────────────────────────
    if question and st.session_state.ready:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner(""):
                try:
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
                    answer = generate_answer(
                        question,
                        st.session_state.system_prompt,
                        history,
                        client
                    )
                    st.markdown(answer)
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": answer
                    })
                except Exception as e:
                    err = f"Error: {e}"
                    st.error(err)
                    st.session_state.messages.append({
                        "role": "assistant", "content": err
                    })

    # ── Clear conversation ────────────────────────────────
    if st.session_state.messages:
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([3,1,3])
        with col2:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
