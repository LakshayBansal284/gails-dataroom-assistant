"""
GAIL's Bakery — Dataroom AI Assistant (RAG Version)
=====================================================
Golborne Capital AI Engineer Case Study

This file shows the RAG (Retrieval Augmented Generation) architecture
as an alternative to the full-context approach used in app.py.

WHY THIS FILE EXISTS:
The production app (app.py) uses full-context retrieval — sending all
13 documents to Claude on every query. This works because the dataroom
is only ~7,500 words, well within Claude's 150,000-word context window.

RAG would be the correct architecture if the dataroom grew beyond
~50 documents (~100,000+ words). This file demonstrates that
architecture for reference.

RAG PIPELINE OVERVIEW:
  Startup:
    1. Load all .md files from dataroom/
    2. Split each document into ~400-word chunks with 50-word overlap
    3. Convert each chunk to a vector (embedding) using Voyage AI
    4. Store all vectors + text in ChromaDB (in-memory vector database)

  Per query:
    5. Convert the user's question to a vector (same embedding model)
    6. ChromaDB finds the 5 most similar chunk vectors (cosine similarity)
    7. Send only those 5 chunks + question to Claude
    8. Claude answers citing sources

DEPENDENCIES (add to requirements.txt if using this file):
  anthropic>=0.40.0
  chromadb>=0.5.0
  streamlit>=1.40.0
  requests>=2.31.0

IMPORTANT: This file requires a VOYAGE_API_KEY in Streamlit secrets.
Voyage AI (voyageai.com) provides the embedding model.
A separate free account is needed — distinct from the Anthropic API key.
"""

import os
import glob
import requests
import streamlit as st
import chromadb
import anthropic

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATAROOM_DIR  = "dataroom"          # folder containing the 13 .md files

CHUNK_SIZE    = 400                 # words per chunk
                                    # Too small = chunks lack context
                                    # Too large = retrieval becomes imprecise
                                    # 400 words ≈ one coherent topic/section

CHUNK_OVERLAP = 50                  # words shared between adjacent chunks
                                    # Prevents important sentences being split
                                    # across two chunks where neither has
                                    # the complete context

TOP_K         = 5                   # how many chunks to retrieve per question
                                    # More chunks = more context for Claude
                                    # but also more tokens per call
                                    # 5 is a good balance for this dataroom size

EMBED_MODEL   = "voyage-3"          # Voyage AI embedding model
                                    # voyage-3 is optimised for retrieval tasks
                                    # Produces 1024-dimensional vectors
                                    # Better than generic models for financial text

CHAT_MODEL    = "claude-sonnet-4-6" # Claude model for answer generation
                                    # Same model as the full-context version

MAX_TOKENS    = 1500                # maximum length of Claude's answer


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DOCUMENTS
# ─────────────────────────────────────────────────────────────────────────────
# Read every .md file from the dataroom/ folder.
# Returns a list of dicts: {filename, title, content}
# The filename is used later for source citations in answers.

def load_documents(dataroom_dir: str) -> list[dict]:
    """
    Reads all markdown files from the dataroom folder.
    Each file represents one source document.
    """
    documents = []

    # glob.glob finds all files matching the pattern dataroom/*.md
    # sorted() ensures consistent ordering (01, 02, 03...)
    md_files = sorted(glob.glob(os.path.join(dataroom_dir, "*.md")))

    for filepath in md_files:
        filename = os.path.basename(filepath)   # e.g. "02_financials_fy2025.md"

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()                  # full text of the document

        # Extract title from the first line (each .md starts with # Title)
        first_line = content.split("\n")[0].strip()
        title = first_line.lstrip("#").strip() if first_line.startswith("#") else filename

        documents.append({
            "filename": filename,
            "title":    title,
            "content":  content
        })

    return documents


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CHUNKING
# ─────────────────────────────────────────────────────────────────────────────
# Split each document into smaller overlapping pieces called "chunks".
#
# WHY CHUNK?
# Embedding a full document as one unit makes retrieval imprecise.
# If document 2 (FY2025 financials) is 600 words covering revenue,
# EBITDA, site count, and staff numbers, embedding it as one unit
# means all those topics get blended into one vector. A question about
# revenue and a question about staff count would both retrieve the same
# chunk — but you only need part of it.
#
# By chunking into 400-word pieces, each chunk covers roughly one topic,
# making retrieval much more precise.
#
# WHY OVERLAP?
# Without overlap, a sentence at the boundary between chunk 1 and chunk 2
# might be split: the first half in chunk 1, the second half in chunk 2,
# making neither chunk usable for that sentence.
# Overlap of 50 words means the last 50 words of chunk 1 are repeated
# at the start of chunk 2, so boundary sentences always appear whole.

def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits a string of text into overlapping word-count chunks.

    Example with chunk_size=5, overlap=2 and text "A B C D E F G H":
      Chunk 1: "A B C D E"      (words 0-4)
      Chunk 2: "D E F G H"      (words 3-7, overlapping D E from chunk 1)
    """
    words  = text.split()       # split into individual words
    chunks = []
    start  = 0

    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])      # rejoin words into a string
        chunks.append(chunk)

        if end == len(words):   # reached the end of the document
            break

        # Move forward by (chunk_size - overlap) so chunks overlap
        # e.g. chunk_size=400, overlap=50 → move forward 350 words each time
        start += chunk_size - overlap

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunks all documents and returns a flat list of chunk dicts.
    Each chunk carries its source metadata for citations.

    A chunk dict looks like:
    {
        "chunk_id":  "02_financials_fy2025.md_chunk_3",
        "text":      "...400 words of text...",
        "filename":  "02_financials_fy2025.md",
        "title":     "Grain Topco — Financial Results FY2025",
        "chunk_num": 3
    }
    """
    all_chunks = []

    for doc in documents:
        text_chunks = chunk_text(doc["content"])

        for i, chunk_text_str in enumerate(text_chunks):
            all_chunks.append({
                "chunk_id":  f"{doc['filename']}_chunk_{i}",
                "text":      chunk_text_str,
                "filename":  doc["filename"],
                "title":     doc["title"],
                "chunk_num": i      # which chunk within this document
            })

    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────
# Convert text into vectors (lists of numbers) using the Voyage AI API.
#
# WHAT IS AN EMBEDDING?
# An embedding is a list of ~1024 numbers that represents the MEANING
# of a piece of text. Texts with similar meaning produce similar numbers.
#
# Example:
#   "GAIL's retail revenue was £219m"  → [0.23, -0.87, 0.45, ...]
#   "the company's turnover"           → [0.21, -0.85, 0.44, ...]  ← similar
#   "charges held by Glas Trust"       → [0.91,  0.34, -0.67, ...] ← different
#
# This is what makes semantic search possible — we can find passages
# that MEAN the same thing as a question, not just share the same words.
#
# WHY VOYAGE AI?
# voyage-3 is specifically optimised for retrieval tasks (finding relevant
# passages given a query). It outperforms generic embedding models on
# financial and legal text benchmarks.
#
# NOTE: Voyage AI is a separate service from Anthropic. Same API key format
# but a different account at voyageai.com. The key starts with "pa-".

def embed_texts(texts: list[str],
                voyage_api_key: str,
                input_type: str = "document") -> list[list[float]]:
    """
    Calls the Voyage AI REST API to convert texts to embedding vectors.

    Args:
        texts:          list of strings to embed (can be multiple at once)
        voyage_api_key: Voyage AI API key (from voyageai.com, starts with pa-)
        input_type:     "document" when embedding dataroom chunks at startup
                        "query"    when embedding the user's question at query time
                        The model performs better when told which type it's seeing.

    Returns:
        list of vectors — one 1024-element list of floats per input text
    """
    response = requests.post(
        "https://api.voyageai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {voyage_api_key}",
            "Content-Type":  "application/json"
        },
        json={
            "model":      EMBED_MODEL,
            "input":      texts,
            "input_type": input_type
        },
        timeout=60      # 60 second timeout — embedding batches can take time
    )

    # Raise an exception if the API returned an error (4xx or 5xx status)
    response.raise_for_status()

    data = response.json()

    # The response structure is:
    # { "data": [ {"embedding": [0.23, -0.87, ...]}, {...}, ... ] }
    # We extract just the embedding vectors
    return [item["embedding"] for item in data["data"]]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — VECTOR DATABASE (ChromaDB)
# ─────────────────────────────────────────────────────────────────────────────
# Store all chunk embeddings in ChromaDB for fast similarity search.
#
# WHAT IS A VECTOR DATABASE?
# A normal database searches by exact keyword match.
# A vector database searches by SIMILARITY of meaning.
#
# ChromaDB stores each chunk as:
#   - The text itself (so we can send it to Claude later)
#   - The embedding vector (1024 numbers representing meaning)
#   - Metadata (filename, title — for source citations)
#
# When a question comes in, ChromaDB compares the question's vector
# against all stored vectors and returns the most similar ones.
# This is called "cosine similarity" — it measures the angle between
# vectors in 1024-dimensional space. Smaller angle = more similar meaning.
#
# WHY IN-MEMORY?
# ChromaDB runs inside the Python process — no external server, no signup,
# no cost. For 13 documents (~120 chunks) this is perfectly sufficient.
# At thousands of documents you'd switch to a persistent store (ChromaDB
# also supports this, or cloud options like Pinecone).

def build_vector_store(chunks: list[dict],
                       voyage_api_key: str,
                       progress_bar=None) -> chromadb.Collection:
    """
    Embeds all chunks and stores them in a ChromaDB in-memory collection.

    Args:
        chunks:         flat list of chunk dicts from chunk_documents()
        voyage_api_key: Voyage AI key for calling the embedding API
        progress_bar:   optional Streamlit progress bar widget

    Returns:
        ChromaDB Collection object — used later for similarity search
    """
    # Create an in-memory ChromaDB client
    # "In-memory" means it lives inside the running Python process
    # No files are written to disk — the index is rebuilt on each app start
    chroma_client = chromadb.Client()

    # Delete any existing collection from a previous run (e.g. Streamlit rerun)
    try:
        chroma_client.delete_collection("gails_dataroom")
    except Exception:
        pass    # Collection didn't exist — that's fine

    # Create a new collection
    # hnsw:space = "cosine" tells ChromaDB to use cosine similarity
    # when comparing vectors (standard for text embeddings)
    collection = chroma_client.create_collection(
        name     = "gails_dataroom",
        metadata = {"hnsw:space": "cosine"}
    )

    # Process chunks in batches of 20
    # Why batches? The Voyage API accepts multiple texts per call,
    # but very large batches can hit rate limits or timeouts.
    # 20 chunks ≈ 8,000 words per API call — safe and efficient.
    batch_size = 20
    total      = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]

        # Extract the fields we need for this batch
        texts      = [c["text"]      for c in batch]    # the actual text
        ids        = [c["chunk_id"]  for c in batch]    # unique ID per chunk
        metadatas  = [                                   # source info for citations
            {
                "filename":  c["filename"],
                "title":     c["title"],
                "chunk_num": c["chunk_num"]
            }
            for c in batch
        ]

        # Call Voyage AI to get embeddings for this batch of texts
        # input_type="document" because these are knowledge base documents
        embeddings = embed_texts(texts, voyage_api_key, input_type="document")

        # Store everything in ChromaDB
        # After this call, ChromaDB can find these chunks by similarity search
        collection.add(
            ids        = ids,           # unique identifiers
            documents  = texts,         # the text (returned in query results)
            embeddings = embeddings,    # the vectors (used for similarity search)
            metadatas  = metadatas      # source info (returned in query results)
        )

        # Update progress bar if one was passed in
        if progress_bar:
            progress = min((i + batch_size) / total, 1.0)
            progress_bar.progress(
                progress,
                text=f"Embedding chunks {i+1}–{min(i+batch_size, total)} of {total}..."
            )

    return collection


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────
# Given a user's question, find the most relevant chunks from ChromaDB.
#
# THE RETRIEVAL PROCESS:
#   1. Embed the question using the same Voyage model
#      (critically: use input_type="query" not "document")
#   2. ChromaDB compares the question vector to all ~120 stored chunk vectors
#   3. Returns the TOP_K chunks with smallest cosine distance
#      (distance of 0 = identical meaning, distance of 2 = opposite meaning)
#
# SEMANTIC vs KEYWORD SEARCH:
# If someone asks "what did the company earn?" — keyword search finds nothing
# (the word "earn" doesn't appear in the documents).
# Semantic search finds the revenue/EBITDA passages because they MEAN
# the same thing as "what the company earned."

def retrieve_relevant_chunks(question: str,
                              collection: chromadb.Collection,
                              voyage_api_key: str,
                              top_k: int = TOP_K) -> list[dict]:
    """
    Finds the top_k most semantically similar chunks to the question.

    Args:
        question:       the user's natural language question
        collection:     ChromaDB collection containing all embedded chunks
        voyage_api_key: for embedding the question
        top_k:          number of chunks to retrieve (default: TOP_K = 5)

    Returns:
        list of chunk dicts with keys: text, filename, title, distance
        Ordered by similarity (most similar first)
    """
    # Step 1: Embed the question
    # IMPORTANT: use input_type="query" not "document"
    # The Voyage model was trained knowing the difference —
    # "query" produces vectors optimised for finding relevant passages,
    # "document" produces vectors optimised for being found.
    question_embedding = embed_texts(
        [question],
        voyage_api_key,
        input_type="query"
    )[0]   # [0] because embed_texts returns a list; we only sent one text

    # Step 2: Query ChromaDB for the most similar chunks
    results = collection.query(
        query_embeddings = [question_embedding],    # our question as a vector
        n_results        = top_k,                  # how many to return
        include          = ["documents", "metadatas", "distances"]
        # "distances" are the cosine distances — lower = more similar
    )

    # Step 3: Unpack results into a clean list
    # ChromaDB returns results nested in lists (supports batch queries)
    # [0] accesses the first (and only) query's results
    chunks = []
    for text, meta, dist in zip(
        results["documents"][0],    # the actual text of each chunk
        results["metadatas"][0],    # filename, title, chunk_num
        results["distances"][0]     # cosine distance (0 = identical, 2 = opposite)
    ):
        chunks.append({
            "text":     text,
            "filename": meta["filename"],
            "title":    meta["title"],
            "distance": dist
        })

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — GENERATION
# ─────────────────────────────────────────────────────────────────────────────
# Send the retrieved chunks + question to Claude and get an answer.
#
# WHAT GOES INTO THE PROMPT:
#   System prompt contains:
#     - Claude's role and instructions
#     - The citation rules (cite every claim with [Source: filename])
#     - The accuracy rules (quote figures exactly, say you don't know)
#     - The retrieved chunks (NOT the full dataroom — just the relevant bits)
#
# User message contains:
#     - The question
#
# DIFFERENCE FROM FULL-CONTEXT VERSION:
# In app.py, the system prompt contains ALL 13 documents (~7,500 words).
# In this RAG version, the system prompt contains only the 5 retrieved
# chunks (~2,000 words). Same rules, same citation format — just
# operating on a subset of the dataroom rather than everything.
#
# This matters at scale: if the dataroom had 500 documents, you couldn't
# send them all. RAG lets you send only what's relevant.

RAG_SYSTEM_PROMPT = """You are a financial analyst assistant for Golborne Capital.
You have been given SELECTED PASSAGES from a dataroom on GAIL'S LIMITED 
(Companies House: 06055393), a UK premium artisan bakery chain.

These passages were retrieved because they are the most relevant to the 
user's question. They come from a 13-document dataroom covering financials,
charges, ownership, management, news, risks, and credit analysis.

RULES — follow these precisely:

1. ONLY use information from the context passages provided below.
   Do not use outside knowledge or information not in these passages.

2. CITE YOUR SOURCE for every factual claim using this exact format:
   [Source: filename.md]
   Example: Revenue was £219,828,000 [Source: 02_financials_fy2025.md]

3. FINANCIAL FIGURES must be quoted exactly as they appear in the source.
   Never round, estimate, or approximate unless the source itself does.
   Never invent a figure not stated in the passages.

4. IF THE ANSWER IS NOT IN THE PROVIDED PASSAGES, say:
   "The retrieved passages do not contain sufficient information to 
   answer this question. You may want to rephrase the question or 
   ask about a related topic."
   Do NOT use general knowledge to fill gaps.

5. FOR DRAFTING TASKS (credit summary, risk analysis etc.):
   Use only facts from the passages. Cite throughout.

6. FORMAT: Professional, clear, concise. Use bullet points and tables
   where they add clarity.

The retrieved context passages follow below:
{context}
"""


def generate_answer_rag(question: str,
                         chunks: list[dict],
                         history: list[dict],
                         client: anthropic.Anthropic) -> str:
    """
    Sends the retrieved chunks + question to Claude and returns the answer.

    The key difference from the full-context version:
    - Full context: system prompt contains ALL 13 documents
    - RAG: system prompt contains only the TOP_K retrieved chunks

    Args:
        question: user's question
        chunks:   retrieved chunks from ChromaDB (already filtered to relevant ones)
        history:  prior conversation turns (for follow-up questions)
        client:   Anthropic API client
    """
    # Format the retrieved chunks into the context string
    # Each chunk is labelled with its source filename so Claude can cite it
    context_parts = []
    for i, chunk in enumerate(chunks):
        label = f"[Source: {chunk['filename']} | Relevance rank: {i+1} of {TOP_K}]"
        context_parts.append(f"{label}\n{chunk['text']}")

    # Join chunks with a separator so Claude can tell them apart
    context_str = "\n\n---\n\n".join(context_parts)

    # Fill in the system prompt with the actual retrieved context
    system = RAG_SYSTEM_PROMPT.format(context=context_str)

    # Build the message list: prior conversation + current question
    messages = history + [{"role": "user", "content": question}]

    # Call Claude
    response = client.messages.create(
        model      = CHAT_MODEL,
        max_tokens = MAX_TOKENS,
        system     = system,
        messages   = messages
    )

    return response.content[0].text


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
# Same dark ChatGPT-style interface as app.py, with one addition:
# the "Sources retrieved" expander shows which chunks were retrieved
# and their similarity distances — making the RAG process visible.

def main():
    st.set_page_config(
        page_title = "GAIL's Dataroom (RAG)",
        page_icon  = "🍞",
        layout     = "wide",
        initial_sidebar_state = "expanded"
    )

    # ── CSS (same dark theme as app.py) ──────────────────────────────────────
    st.markdown("""
    <style>
        #MainMenu, footer, header { visibility: hidden; }
        .block-container { padding-top: 0 !important; padding-bottom: 0 !important; max-width: 100% !important; }
        .stApp, .stApp > div, [data-testid="stAppViewContainer"] { background-color: #212121 !important; }
        [data-testid="stSidebar"] { background-color: #171717 !important; border-right: 1px solid #2a2a2a; }
        [data-testid="stSidebar"] * { color: #d1d5db !important; }
        [data-testid="stSidebar"] h3 { color: #ffffff !important; font-size: 0.75rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; }
        [data-testid="stSidebar"] hr { border-color: #2a2a2a !important; }
        [data-testid="stSidebar"] .stButton > button { background: #1a1a1a !important; border: 1px solid #2a2a2a !important; color: #9ca3af !important; font-size: 0.78rem !important; text-align: left !important; padding: 0.4rem 0.6rem !important; border-radius: 6px !important; white-space: normal !important; height: auto !important; line-height: 1.4 !important; }
        [data-testid="stSidebar"] .stButton > button:hover { background: #2a2a2a !important; color: #e5e7eb !important; }
        [data-testid="stChatMessage"] { background: transparent !important; border: none !important; padding: 0.75rem 0 !important; max-width: 48rem; margin: 0 auto; }
        .stChatMessage:has([data-testid="chatAvatarIcon-user"]) { background: #2f2f2f !important; border-radius: 12px !important; padding: 0.75rem 1rem !important; }
        [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li, [data-testid="stChatMessage"] td, [data-testid="stChatMessage"] th { color: #ececec !important; font-size: 0.95rem !important; line-height: 1.7 !important; }
        [data-testid="stChatMessage"] table { border-collapse: collapse !important; width: 100% !important; font-size: 0.85rem !important; }
        [data-testid="stChatMessage"] th { background: #2f2f2f !important; color: #e5e7eb !important; padding: 0.5rem 0.75rem !important; border: 1px solid #3a3a3a !important; }
        [data-testid="stChatMessage"] td { padding: 0.45rem 0.75rem !important; border: 1px solid #3a3a3a !important; color: #d1d5db !important; }
        [data-testid="stBottom"], [data-testid="stBottom"] > div, .stChatFloatingInputContainer, .stChatFloatingInputContainer > div { background-color: #212121 !important; border-color: transparent !important; }
        [data-testid="stChatInput"] { background: #2f2f2f !important; border: 1px solid #3a3a3a !important; border-radius: 12px !important; max-width: 48rem !important; margin: 0 auto !important; }
        [data-testid="stChatInput"] textarea { color: #ececec !important; background: transparent !important; font-size: 0.95rem !important; }
        [data-testid="stMetric"] { background: #2f2f2f; border-radius: 8px; padding: 0.6rem 1rem; }
        [data-testid="stMetric"] label { color: #9ca3af !important; font-size: 0.72rem !important; }
        [data-testid="stMetricValue"] { color: #e5e7eb !important; font-size: 1rem !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🍞 GAIL's Dataroom")
        st.markdown("<small>GAIL'S LIMITED · 06055393</small>", unsafe_allow_html=True)
        st.markdown("<small>Architecture: RAG (Voyage-3 + ChromaDB)</small>",
                    unsafe_allow_html=True)
        st.divider()

        st.markdown("### Documents")
        doc_map = [
            ("01 · Company Overview",         "01_company_overview.md"),
            ("02 · FY2025 Financials",        "02_financials_fy2025.md"),
            ("03 · FY2024 Financials",        "03_financials_fy2024.md"),
            ("04 · FY2023 & Historical",      "04_financials_fy2023_historical.md"),
            ("05 · Charges Register",         "05_charges_register.md"),
            ("06 · Ownership & Management",   "06_ownership_management.md"),
            ("07 · News & Events",            "07_news_events.md"),
            ("08 · Subsidiary Accounts Note", "08_subsidiary_accounts_note.md"),
            ("09 · Lender Risks",             "09_lender_risks.md"),
            ("10 · Credit Summary",           "10_credit_summary.md"),
            ("11 · Bain Capital Acquisition", "11_bain_capital_acquisition.md"),
            ("12 · Sale Process & Valuation", "12_sale_process_valuation.md"),
            ("13 · Dataroom Index",           "13_dataroom_index.md"),
        ]
        for display_name, filename in doc_map:
            filepath = os.path.join(DATAROOM_DIR, filename)
            with st.expander(display_name):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        st.markdown(f.read())
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
            "What is the net debt and leverage ratio?",
            "Who owns the preference shares?",
            "What is the bank loan amount and when does it mature?",
        ]
        for q in example_qs:
            if st.button(q, key=f"eg_{q[:25]}", use_container_width=True):
                st.session_state.pending_question = q

        st.divider()
        st.markdown(
            "<small>Sources: Companies House · Grain Topco FY2025 accounts · "
            "British Baker · The Grocer · Sky News · Bloomberg</small>",
            unsafe_allow_html=True
        )

    # ── API keys ──────────────────────────────────────────────────────────────
    # Two API keys needed for RAG:
    #   1. ANTHROPIC_API_KEY — for Claude (answer generation)
    #   2. VOYAGE_API_KEY    — for Voyage embeddings (retrieval)
    #
    # In Streamlit Cloud secrets, set both:
    #   ANTHROPIC_API_KEY = "sk-ant-..."
    #   VOYAGE_API_KEY = "pa-..."

    anthropic_key = None
    voyage_key    = None

    try:
        anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
        voyage_key    = st.secrets["VOYAGE_API_KEY"]
    except Exception:
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        voyage_key    = os.environ.get("VOYAGE_API_KEY")

    if not anthropic_key:
        st.error("⚠️ ANTHROPIC_API_KEY not found in Streamlit secrets.")
        st.stop()

    if not voyage_key:
        st.error(
            "⚠️ VOYAGE_API_KEY not found. "
            "Sign up free at voyageai.com, get an API key, "
            "and add it to Streamlit secrets as VOYAGE_API_KEY."
        )
        st.stop()

    client = anthropic.Anthropic(api_key=anthropic_key)

    # ── Session state ─────────────────────────────────────────────────────────
    # Streamlit reruns the entire script on every user interaction.
    # Session state persists data between reruns so we don't rebuild
    # the vector store on every click.
    if "rag_ready" not in st.session_state:
        st.session_state.rag_ready        = False
        st.session_state.collection       = None   # ChromaDB collection
        st.session_state.messages         = []     # conversation history
        st.session_state.pending_question = None
        st.session_state.chunk_count      = 0
        st.session_state.doc_count        = 0

    # ── Build the RAG index (once at startup) ─────────────────────────────────
    # This runs once the first time the app loads.
    # It reads all documents, chunks them, embeds them via Voyage API,
    # and stores everything in ChromaDB.
    # On subsequent Streamlit reruns, rag_ready = True so this is skipped.
    if not st.session_state.rag_ready:
        with st.spinner("📚 Building RAG index — chunking and embedding dataroom..."):
            try:
                # Step 1: Load documents
                docs = load_documents(DATAROOM_DIR)

                # Step 2: Chunk all documents
                # This is fast — just string operations, no API calls
                chunks = chunk_documents(docs)

                # Step 3 & 4: Embed chunks and build ChromaDB index
                # This calls the Voyage API — takes 15-30 seconds for ~120 chunks
                progress_bar = st.progress(0, text="Starting embedding...")
                collection   = build_vector_store(chunks, voyage_key, progress_bar)
                progress_bar.empty()

                # Save to session state so we don't rebuild on every rerun
                st.session_state.collection  = collection
                st.session_state.rag_ready   = True
                st.session_state.chunk_count = len(chunks)
                st.session_state.doc_count   = len(docs)

                st.success(
                    f"✅ RAG index built — {len(chunks)} chunks from {len(docs)} documents. "
                    f"Using voyage-3 embeddings + ChromaDB."
                )

            except Exception as e:
                st.error(f"Failed to build RAG index: {e}")
                st.stop()

    # ── Welcome screen ────────────────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("""
        <div style='text-align:center; padding: 5rem 1rem 2rem 1rem;'>
            <div style='font-size:2.5rem; margin-bottom:0.5rem;'>🍞</div>
            <h2 style='color:#ececec; font-size:1.4rem; font-weight:600; margin:0 0 0.5rem 0;'>
                GAIL's Bakery Dataroom — RAG Version
            </h2>
            <p style='color:#6b7280; font-size:0.9rem; max-width:32rem; margin:0 auto;'>
                Questions are answered by retrieving the most relevant passages
                from the dataroom using semantic search (Voyage-3 + ChromaDB),
                then generating a cited answer with Claude.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Documents", st.session_state.doc_count)
        col2.metric("Chunks indexed", st.session_state.chunk_count)
        col3.metric("Chunks retrieved/query", TOP_K)
        col4.metric("Embedding model", "voyage-3")

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # For assistant messages, show which chunks were retrieved
            # This makes the RAG process transparent and verifiable
            if msg["role"] == "assistant" and "retrieved_chunks" in msg:
                with st.expander(
                    f"📎 {TOP_K} chunks retrieved (click to see sources and similarity scores)",
                    expanded=False
                ):
                    for i, chunk in enumerate(msg["retrieved_chunks"]):
                        st.markdown(
                            f"**Rank {i+1}:** `{chunk['filename']}` — "
                            f"*{chunk['title']}*"
                        )
                        # Distance: 0 = identical, 2 = completely different
                        # Good retrieval typically shows distances < 0.5
                        similarity_pct = max(0, (1 - chunk['distance']) * 100)
                        st.markdown(
                            f"<small style='color:#9ca3af'>"
                            f"Cosine distance: {chunk['distance']:.3f} | "
                            f"Similarity: ~{similarity_pct:.0f}%"
                            f"</small>",
                            unsafe_allow_html=True
                        )
                        # Show the first 300 characters of the retrieved chunk
                        st.markdown(f"> {chunk['text'][:300]}...")
                        st.divider()

    # ── Handle sidebar button questions ───────────────────────────────────────
    if st.session_state.get("pending_question"):
        question = st.session_state.pending_question
        st.session_state.pending_question = None
    else:
        question = st.chat_input("Ask anything about GAIL's Bakery...")

    # ── Process question ──────────────────────────────────────────────────────
    if question and st.session_state.rag_ready:

        # Add user message to display and history
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching dataroom and generating answer..."):
                try:
                    # STEP 5: RETRIEVAL
                    # Embed the question, find the top-5 most similar chunks
                    retrieved_chunks = retrieve_relevant_chunks(
                        question,
                        st.session_state.collection,
                        voyage_key
                    )

                    # STEP 6: GENERATION
                    # Build history for conversation memory
                    # (exclude the current question — it's added inside generate_answer_rag)
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]

                    # Send retrieved chunks + question to Claude
                    answer = generate_answer_rag(
                        question,
                        retrieved_chunks,
                        history,
                        client
                    )

                    # Display the answer
                    st.markdown(answer)

                    # Show retrieved sources in an expander
                    # This is the key differentiator of RAG — you can see
                    # exactly which passages the answer was drawn from
                    with st.expander(
                        f"📎 {TOP_K} chunks retrieved (click to see sources)",
                        expanded=False
                    ):
                        for i, chunk in enumerate(retrieved_chunks):
                            st.markdown(
                                f"**Rank {i+1}:** `{chunk['filename']}` — "
                                f"*{chunk['title']}*"
                            )
                            similarity_pct = max(0, (1 - chunk['distance']) * 100)
                            st.markdown(
                                f"<small style='color:#9ca3af'>"
                                f"Cosine distance: {chunk['distance']:.3f} | "
                                f"Similarity: ~{similarity_pct:.0f}%"
                                f"</small>",
                                unsafe_allow_html=True
                            )
                            st.markdown(f"> {chunk['text'][:300]}...")
                            st.divider()

                    # Save to conversation history
                    st.session_state.messages.append({
                        "role":             "assistant",
                        "content":          answer,
                        "retrieved_chunks": retrieved_chunks  # saved for replay
                    })

                except Exception as e:
                    err = f"Error: {e}"
                    st.error(err)
                    st.session_state.messages.append({
                        "role": "assistant", "content": err
                    })

    # ── Clear conversation ────────────────────────────────────────────────────
    if st.session_state.messages:
        col1, col2, col3 = st.columns([3, 1, 3])
        with col2:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()


if __name__ == "__main__":
    main()
