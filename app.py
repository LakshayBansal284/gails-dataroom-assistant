"""
GAIL's Bakery — Dataroom AI Assistant
======================================
Golborne Capital AI Engineer Case Study

Architecture: Full-context retrieval
  - All 13 dataroom documents are loaded at startup (~7,000 words total)
  - Every question sends the complete dataroom to Claude as context
  - No embeddings, no vector database, no second API key needed
  - One service (Anthropic), one key, nothing to break

Why not RAG?
  - The dataroom is ~7,500 words — under 5% of Claude's 200k context window
  - RAG solves a scale problem that doesn't exist here
  - Full context means Claude always has every document available
  - Financial figures are always sourced from the actual filed text
  - Simpler architecture = more reliable in production and in a live demo
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
# Read every .md file in the dataroom/ folder.
# Returns a list of dicts with filename, title, and content.

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
# Concatenate all documents into one string.
# Each document is labelled with its filename so Claude
# can cite it accurately in answers.

def build_context(documents: list[dict]) -> str:
    parts = []
    for doc in documents:
        parts.append(
            f"=== SOURCE DOCUMENT: {doc['filename']} ===\n"
            f"Title: {doc['title']}\n\n"
            f"{doc['content']}"
        )
    return "\n\n{'='*60}\n\n".join(parts)


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

def generate_answer(question: str,
                    system: str,
                    history: list[dict],
                    client: anthropic.Anthropic) -> str:
    """
    Send the question (plus conversation history) to Claude.
    The entire dataroom is already in the system prompt.
    """
    # Build messages: prior conversation turns + new question
    messages = history + [{"role": "user", "content": question}]

    response = client.messages.create(
        model      = CHAT_MODEL,
        max_tokens = MAX_TOKENS,
        system     = system,
        messages   = messages
    )
    return response.content[0].text


# ─────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title = "GAIL's Dataroom Assistant",
        page_icon  = "🍞",
        layout     = "wide"
    )

    # ── Header ────────────────────────────────────────────
    st.markdown("""
    <div style='padding: 1.5rem 0 0.5rem 0;'>
        <h1 style='margin:0; font-size:1.6rem; font-weight:700;'>
            🍞 GAIL's Bakery — Dataroom Assistant
        </h1>
        <p style='margin:0.3rem 0 0 0; color:#666; font-size:0.9rem;'>
            Ask questions about GAIL'S LIMITED (06055393) · Powered by Claude · Golborne Capital
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📁 Dataroom Sources")
        st.markdown("13 documents loaded:")
        doc_names = [
            "01 · Company Overview",
            "02 · FY2025 Financials",
            "03 · FY2024 Financials",
            "04 · FY2023 & Historical",
            "05 · Charges Register",
            "06 · Ownership & Management",
            "07 · News & Events",
            "08 · Subsidiary Accounts Note",
            "09 · Lender Risks",
            "10 · Credit Summary",
            "11 · Bain Capital Acquisition",
            "12 · Sale Process & Valuation",
            "13 · Dataroom Index",
        ]
        for name in doc_names:
            st.markdown(f"<small>{name}</small>", unsafe_allow_html=True)

        st.divider()
        st.markdown("### 💡 Try asking:")
        example_qs = [
            "What was revenue and EBITDA in the last reported year?",
            "What charges are registered against the company and who holds them?",
            "Who are the current directors?",
            "What are the key risks for a lender?",
            "Draft a short credit summary.",
            "Who owns GAIL's?",
            "How many sites does GAIL's operate?",
            "What happened with the Goldman Sachs sale process?",
            "What was the pre-tax loss in FY2025?",
            "When was the Bain Capital acquisition and for how much?",
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
        st.error("⚠️ No API key found. Set ANTHROPIC_API_KEY in Streamlit secrets.")
        st.stop()

    client = anthropic.Anthropic(api_key=api_key)

    # ── Session state ─────────────────────────────────────
    if "ready" not in st.session_state:
        st.session_state.ready            = False
        st.session_state.system_prompt    = None
        st.session_state.messages         = []   # full chat history
        st.session_state.pending_question = None
        st.session_state.doc_count        = 0
        st.session_state.word_count       = 0

    # ── Load documents (once at startup) ─────────────────
    # This is fast — just reading text files from disk.
    # No API calls needed at startup at all.
    if not st.session_state.ready:
        with st.spinner("📚 Loading dataroom documents..."):
            try:
                docs    = load_documents(DATAROOM_DIR)
                context = build_context(docs)
                system  = SYSTEM_PROMPT.format(dataroom=context)

                st.session_state.system_prompt = system
                st.session_state.ready         = True
                st.session_state.doc_count     = len(docs)
                st.session_state.word_count    = len(context.split())

                st.success(
                    f"✅ {len(docs)} documents loaded "
                    f"({len(context.split()):,} words in context). "
                    f"Ready to answer questions."
                )
            except Exception as e:
                st.error(f"Failed to load documents: {e}")
                st.stop()

    # ── Stats bar ─────────────────────────────────────────
    if st.session_state.ready:
        col1, col2, col3 = st.columns(3)
        col1.metric("Documents", st.session_state.doc_count)
        col2.metric("Words in context", f"{st.session_state.word_count:,}")
        col3.metric("Model", "claude-sonnet-4-6")
        st.divider()

    # ── Chat history ──────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Handle sidebar button questions ──────────────────
    if st.session_state.get("pending_question"):
        question = st.session_state.pending_question
        st.session_state.pending_question = None
    else:
        question = st.chat_input("Ask anything about GAIL's Bakery...")

    # ── Process question ──────────────────────────────────
    if question and st.session_state.ready:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate and show answer
        with st.chat_message("assistant"):
            with st.spinner("Reading dataroom and generating answer..."):
                try:
                    # Build history in Anthropic format (exclude current question
                    # as it's added inside generate_answer)
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

                    # Save assistant response to history
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": answer
                    })

                except Exception as e:
                    err = f"Error generating answer: {e}"
                    st.error(err)
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": err
                    })

    # ── Clear chat button ─────────────────────────────────
    if st.session_state.messages:
        if st.button("🗑️ Clear conversation", type="secondary"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
