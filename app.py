"""
GAIL's Bakery — Dataroom AI Assistant
======================================
Built for: Golborne Capital AI Engineer Case Study

Architecture:
  1. At startup: read all 13 dataroom documents, split into chunks,
     embed each chunk using Anthropic's embedding model, store in ChromaDB
  2. At query time: embed the user's question, find the most similar
     chunks via ChromaDB, send those chunks + question to Claude,
     return an answer with source citations
"""

import os
import glob
import textwrap
import streamlit as st
import chromadb
import anthropic

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────

DATAROOM_DIR = "dataroom"          # folder containing the .md files
CHUNK_SIZE   = 400                 # words per chunk — small enough to be precise,
                                   # large enough to have context
CHUNK_OVERLAP = 50                 # words of overlap between chunks so we don't
                                   # accidentally cut a sentence at the boundary
TOP_K        = 5                   # how many chunks to retrieve per question
EMBED_MODEL  = "voyage-3"          # Anthropic's embedding model via voyage API
CHAT_MODEL   = "claude-sonnet-4-6" # model that generates the answer

# ─────────────────────────────────────────────────────────
# STEP 1 — LOAD DOCUMENTS FROM THE DATAROOM FOLDER
# ─────────────────────────────────────────────────────────
# We read every .md file in the dataroom/ directory.
# Each file is one source document. We keep the filename
# so we can later cite it in answers.

def load_documents(dataroom_dir: str) -> list[dict]:
    """
    Returns a list of dicts, one per document:
      { "filename": "01_company_overview.md",
        "title":    "GAIL'S LIMITED — Company Overview",
        "content":  "# GAIL'S LIMITED — Company Ov..." }
    """
    documents = []
    md_files = sorted(glob.glob(os.path.join(dataroom_dir, "*.md")))

    for filepath in md_files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # The first line of each .md file is the document title (e.g. "# GAIL'S LIMITED — ...")
        # We strip the leading "# " to get a clean title for citations
        first_line = content.split("\n")[0].strip()
        title = first_line.lstrip("#").strip() if first_line.startswith("#") else filename

        documents.append({
            "filename": filename,
            "title":    title,
            "content":  content
        })

    return documents


# ─────────────────────────────────────────────────────────
# STEP 2 — CHUNKING
# ─────────────────────────────────────────────────────────
# We can't send a whole document to the embedding model in one go —
# it would make retrieval imprecise. Instead we split each document
# into overlapping chunks of ~400 words.
#
# Example: a 1200-word document with CHUNK_SIZE=400 and OVERLAP=50
# becomes roughly 3 chunks: words 0-400, 350-750, 700-1100
#
# The overlap ensures that if a key sentence sits at the boundary
# between two chunks, it appears in at least one of them fully.

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-count chunks."""
    words  = text.split()
    chunks = []
    start  = 0

    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap   # move forward but overlap a bit

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Takes the list of documents and returns a flat list of chunks.
    Each chunk dict carries:
      - chunk_id  : unique string like "01_company_overview.md_chunk_3"
      - text      : the actual text of this chunk
      - filename  : which file it came from
      - title     : human-readable document title
      - chunk_num : which chunk within the document (for debugging)
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
                "chunk_num": i
            })
    return all_chunks


# ─────────────────────────────────────────────────────────
# STEP 3 — EMBEDDING
# ─────────────────────────────────────────────────────────
# Embedding converts text into a list of numbers (a "vector")
# that represents the meaning of that text.
#
# Similar meaning → similar vectors → close together in vector space
#
# We use Anthropic's Voyage embedding API.
# The embed_texts() function takes a list of strings and returns
# a list of vectors (one vector per string).

def embed_texts(texts: list[str], client: anthropic.Anthropic,
                input_type: str = "document") -> list[list[float]]:
    """
    Call Anthropic's embedding API.
    input_type = "document" when embedding the dataroom chunks
    input_type = "query"    when embedding the user's question
    (The model performs slightly better when it knows what type of text it's seeing.)
    """
    response = client.beta.messages.batches  # not used — we use voyage directly
    # Anthropic routes embedding through their messages API via voyage
    # We use the dedicated embeddings endpoint:
    embed_response = client._client.post(
        "/v1/embeddings",
        json={
            "model":      EMBED_MODEL,
            "input":      texts,
            "input_type": input_type
        }
    )
    data = embed_response.json()
    # The response has a "data" list, each item has an "embedding" key
    return [item["embedding"] for item in data["data"]]


# ─────────────────────────────────────────────────────────
# STEP 4 — VECTOR DATABASE (ChromaDB)
# ─────────────────────────────────────────────────────────
# ChromaDB is an in-memory vector database.
# "In-memory" means it lives inside the running app — no external
# server needed, no signup, completely free.
#
# We store each chunk as a record with:
#   - the chunk text (so we can send it to Claude later)
#   - the embedding vector (so ChromaDB can do similarity search)
#   - metadata (filename, title — for citations)

def build_vector_store(chunks: list[dict],
                       client: anthropic.Anthropic) -> chromadb.Collection:
    """
    Creates a ChromaDB collection, embeds all chunks, and stores them.
    Returns the collection object (which we'll query later).
    """
    # Create an in-memory ChromaDB client
    chroma_client = chromadb.Client()

    # Create a collection — think of it as a table in a database
    # If it already exists (e.g. on a Streamlit rerun), delete and recreate
    try:
        chroma_client.delete_collection("gails_dataroom")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="gails_dataroom",
        # We tell ChromaDB NOT to do its own embedding — we supply our own vectors
        metadata={"hnsw:space": "cosine"}  # use cosine similarity (standard for text)
    )

    # Process in batches of 50 to avoid hitting API rate limits
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        texts      = [c["text"]      for c in batch]
        ids        = [c["chunk_id"]  for c in batch]
        metadatas  = [{"filename": c["filename"],
                       "title":    c["title"],
                       "chunk_num": c["chunk_num"]} for c in batch]

        # Get embeddings from Anthropic for this batch
        embeddings = embed_texts(texts, client, input_type="document")

        # Add to ChromaDB
        collection.add(
            ids        = ids,
            documents  = texts,
            embeddings = embeddings,
            metadatas  = metadatas
        )

    return collection


# ─────────────────────────────────────────────────────────
# STEP 5 — RETRIEVAL
# ─────────────────────────────────────────────────────────
# When a user asks a question, we:
#   1. Embed the question (same model, input_type="query")
#   2. Ask ChromaDB: "find the TOP_K chunks most similar to this vector"
#   3. ChromaDB compares the question vector against all stored chunk vectors
#      using cosine similarity and returns the closest ones
#
# This gives us the most relevant passages from the dataroom —
# without sending the entire dataroom to Claude every time.

def retrieve_relevant_chunks(question: str,
                              collection: chromadb.Collection,
                              client: anthropic.Anthropic,
                              top_k: int = TOP_K) -> list[dict]:
    """
    Embeds the question, queries ChromaDB, returns the top-k chunks
    as a list of dicts with keys: text, filename, title, distance
    """
    # Embed the question
    question_embedding = embed_texts([question], client, input_type="query")[0]

    # Query ChromaDB
    results = collection.query(
        query_embeddings = [question_embedding],
        n_results        = top_k,
        include          = ["documents", "metadatas", "distances"]
    )

    # Unpack results into a clean list
    chunks = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text":     text,
            "filename": meta["filename"],
            "title":    meta["title"],
            "distance": dist  # lower = more similar (cosine distance)
        })

    return chunks


# ─────────────────────────────────────────────────────────
# STEP 6 — GENERATION
# ─────────────────────────────────────────────────────────
# We send Claude:
#   - A system prompt explaining its role and rules
#   - The retrieved chunks as "context" (with source labels)
#   - The user's question
#
# Claude is instructed to:
#   - Answer ONLY from the provided context
#   - Say "the dataroom does not contain this information" if unsure
#   - Cite the source document for every factual claim

SYSTEM_PROMPT = """You are a financial analyst assistant for Golborne Capital. 
You have been given access to a structured dataroom on GAIL'S LIMITED (Companies House number 06055393), 
a UK premium artisan bakery chain trading as GAIL's Bakery.

Your job is to answer questions accurately using ONLY the context passages provided below. 
Each passage is labelled with its source document.

CRITICAL RULES:
1. Only use information from the provided context passages. Do not use any outside knowledge.
2. For every factual claim — especially any financial figure (revenue, EBITDA, loss, margin, 
   headcount, site count, valuation) — you MUST cite the source document in brackets, 
   like this: [Source: 02_financials_fy2025.md]
3. If the answer cannot be found in the provided context, say clearly: 
   "The dataroom does not contain sufficient information to answer this question."
   Do NOT guess or invent figures.
4. Financial figures must be quoted exactly as they appear in the source. 
   Do not round, estimate or approximate unless the source itself says "approximately".
5. If a question asks you to draft something (e.g. a credit summary), 
   draft it using only facts from the context, citing sources throughout.
6. Keep answers clear, professional and concise. Use bullet points for lists.
   Use tables where they add clarity.

Context passages from the GAIL's dataroom:
{context}
"""

def generate_answer(question: str,
                    chunks: list[dict],
                    client: anthropic.Anthropic) -> str:
    """
    Sends the retrieved chunks + question to Claude and returns the answer.
    """
    # Format the context — each chunk is labelled with its source
    context_parts = []
    seen_sources = set()
    for chunk in chunks:
        label = f"[Source: {chunk['filename']}]"
        context_parts.append(f"{label}\n{chunk['text']}")
        seen_sources.add(chunk['filename'])

    context_str = "\n\n---\n\n".join(context_parts)

    # Fill in the system prompt with the actual context
    system = SYSTEM_PROMPT.format(context=context_str)

    # Call Claude
    response = client.messages.create(
        model      = CHAT_MODEL,
        max_tokens = 1500,
        system     = system,
        messages   = [{"role": "user", "content": question}]
    )

    return response.content[0].text


# ─────────────────────────────────────────────────────────
# STEP 7 — STREAMLIT UI
# ─────────────────────────────────────────────────────────
# Streamlit turns this Python script into a web app.
# Every time a user interacts with the page, the script reruns
# from top to bottom — but st.session_state persists data
# between reruns (like the chat history and the vector store).

def main():
    # ── Page config ──────────────────────────────────────
    st.set_page_config(
        page_title = "GAIL's Dataroom Assistant",
        page_icon  = "🍞",
        layout     = "wide"
    )

    # ── Header ───────────────────────────────────────────
    st.markdown("""
    <div style='padding: 1.5rem 0 0.5rem 0;'>
        <h1 style='margin:0; font-size:1.6rem; font-weight:700;'>🍞 GAIL's Bakery — Dataroom Assistant</h1>
        <p style='margin:0.3rem 0 0 0; color:#666; font-size:0.9rem;'>
            Ask questions about GAIL'S LIMITED (06055393) · Powered by RAG + Claude · Golborne Capital
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Sidebar — source documents ────────────────────────
    with st.sidebar:
        st.markdown("### 📁 Dataroom Sources")
        st.markdown("13 documents indexed:")
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
        ]
        for q in example_qs:
            if st.button(q, key=f"eg_{q[:20]}", use_container_width=True):
                st.session_state.pending_question = q

        st.divider()
        st.markdown("<small>Sources: Companies House · Grain Topco accounts · British Baker · The Grocer · Sky News · Bloomberg</small>",
                    unsafe_allow_html=True)

    # ── API key handling ──────────────────────────────────
    # On Streamlit Cloud, keys come from st.secrets (set in the dashboard)
    # Locally, they come from environment variables
    api_key = None
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        st.error("⚠️ No API key found. Set ANTHROPIC_API_KEY in Streamlit secrets or as an environment variable.")
        st.code('export ANTHROPIC_API_KEY="sk-ant-..."', language="bash")
        st.stop()

    client = anthropic.Anthropic(api_key=api_key)

    # ── Initialise session state ──────────────────────────
    # st.session_state is a dictionary that persists across reruns.
    # We use it to store:
    #   - the vector store (so we don't re-embed on every message)
    #   - the chat history (so we can display the conversation)
    if "vector_store_ready" not in st.session_state:
        st.session_state.vector_store_ready = False
        st.session_state.collection         = None
        st.session_state.messages           = []
        st.session_state.pending_question   = None

    # ── Build the vector store (once at startup) ──────────
    # This runs ONCE — the first time the app loads (or on refresh).
    # It reads all 13 documents, chunks them, embeds them, and
    # stores them in ChromaDB. After that, st.session_state.vector_store_ready
    # is True and we skip this block.
    if not st.session_state.vector_store_ready:
        with st.spinner("📚 Indexing dataroom — loading and embedding 13 documents..."):
            try:
                docs   = load_documents(DATAROOM_DIR)
                chunks = chunk_documents(docs)

                # Show progress
                progress_bar = st.progress(0, text="Embedding document chunks...")

                # We'll embed in small batches and update the progress bar
                chroma_client = chromadb.Client()
                try:
                    chroma_client.delete_collection("gails_dataroom")
                except Exception:
                    pass
                collection = chroma_client.create_collection(
                    name     = "gails_dataroom",
                    metadata = {"hnsw:space": "cosine"}
                )

                batch_size = 20
                for i in range(0, len(chunks), batch_size):
                    batch      = chunks[i : i + batch_size]
                    texts      = [c["text"]      for c in batch]
                    ids        = [c["chunk_id"]  for c in batch]
                    metadatas  = [{"filename": c["filename"],
                                   "title":    c["title"],
                                   "chunk_num": c["chunk_num"]} for c in batch]
                    embeddings = embed_texts(texts, client, input_type="document")
                    collection.add(
                        ids        = ids,
                        documents  = texts,
                        embeddings = embeddings,
                        metadatas  = metadatas
                    )
                    progress = min((i + batch_size) / len(chunks), 1.0)
                    progress_bar.progress(progress,
                        text=f"Embedding chunks {i+1}–{min(i+batch_size, len(chunks))} of {len(chunks)}...")

                progress_bar.empty()
                st.session_state.collection         = collection
                st.session_state.vector_store_ready = True
                st.success(f"✅ Dataroom indexed — {len(chunks)} chunks from {len(docs)} documents ready.")

            except Exception as e:
                st.error(f"Failed to build vector store: {e}")
                st.stop()

    # ── Chat interface ────────────────────────────────────
    # Display all previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # If this is an assistant message, show which sources were used
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📎 Sources retrieved", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(f"**{src['filename']}** — {src['title']}")
                        st.markdown(f"<small style='color:#888'>Similarity distance: {src['distance']:.3f} (lower = more relevant)</small>",
                                    unsafe_allow_html=True)
                        st.markdown(f"> {src['text'][:300]}...")
                        st.divider()

    # Handle a question from the sidebar example buttons
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None
    else:
        question = st.chat_input("Ask anything about GAIL's Bakery...")

    # ── Process question ──────────────────────────────────
    if question:
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching dataroom and generating answer..."):
                try:
                    # RETRIEVAL: find relevant chunks
                    chunks = retrieve_relevant_chunks(
                        question,
                        st.session_state.collection,
                        client
                    )

                    # GENERATION: send chunks + question to Claude
                    answer = generate_answer(question, chunks, client)

                    # Display the answer
                    st.markdown(answer)

                    # Show sources in an expander
                    with st.expander("📎 Sources retrieved", expanded=False):
                        for chunk in chunks:
                            st.markdown(f"**{chunk['filename']}** — {chunk['title']}")
                            st.markdown(
                                f"<small style='color:#888'>Similarity distance: {chunk['distance']:.3f} (lower = more relevant)</small>",
                                unsafe_allow_html=True)
                            st.markdown(f"> {chunk['text'][:300]}...")
                            st.divider()

                    # Save to history
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": answer,
                        "sources": chunks
                    })

                except Exception as e:
                    err_msg = f"Error generating answer: {e}"
                    st.error(err_msg)
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": err_msg
                    })


if __name__ == "__main__":
    main()
