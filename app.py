"""
GAIL's Bakery — Dataroom AI Assistant
======================================
Built for: Golborne Capital AI Engineer Case Study

Architecture:
  1. At startup: read all 13 dataroom documents, split into chunks,
     embed each chunk using Voyage AI embedding model, store in ChromaDB
  2. At query time: embed the user's question, find the most similar
     chunks via ChromaDB, send those chunks + question to Claude,
     return an answer with source citations
"""

import os
import glob
import requests
import streamlit as st
import chromadb
import anthropic

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────

DATAROOM_DIR  = "dataroom"
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 50
TOP_K         = 5
EMBED_MODEL   = "voyage-3"
CHAT_MODEL    = "claude-sonnet-4-6"


# ─────────────────────────────────────────────────────────
# STEP 1 — LOAD DOCUMENTS
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
# STEP 2 — CHUNKING
# ─────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
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
# STEP 3 — EMBEDDING via Voyage AI API
# ─────────────────────────────────────────────────────────

def embed_texts(texts: list[str], api_key: str,
                input_type: str = "document") -> list[list[float]]:
    """
    Call Voyage AI embedding API directly via requests.
    Voyage AI is Anthropic's embedding partner — same API key works.
    input_type = "document" for dataroom chunks
    input_type = "query"    for user questions
    """
    response = requests.post(
        "https://api.voyageai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json"
        },
        json={
            "model":      EMBED_MODEL,
            "input":      texts,
            "input_type": input_type
        },
        timeout=60
    )
    response.raise_for_status()
    data = response.json()
    return [item["embedding"] for item in data["data"]]


# ─────────────────────────────────────────────────────────
# STEP 4 — VECTOR DATABASE (ChromaDB)
# ─────────────────────────────────────────────────────────

def build_vector_store(chunks: list[dict], api_key: str,
                       progress_bar=None) -> chromadb.Collection:
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
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch      = chunks[i : i + batch_size]
        texts      = [c["text"]      for c in batch]
        ids        = [c["chunk_id"]  for c in batch]
        metadatas  = [{"filename": c["filename"],
                       "title":    c["title"],
                       "chunk_num": c["chunk_num"]} for c in batch]

        embeddings = embed_texts(texts, api_key, input_type="document")

        collection.add(
            ids        = ids,
            documents  = texts,
            embeddings = embeddings,
            metadatas  = metadatas
        )

        if progress_bar:
            progress = min((i + batch_size) / total, 1.0)
            progress_bar.progress(
                progress,
                text=f"Embedding chunks {i+1}–{min(i+batch_size, total)} of {total}..."
            )

    return collection


# ─────────────────────────────────────────────────────────
# STEP 5 — RETRIEVAL
# ─────────────────────────────────────────────────────────

def retrieve_relevant_chunks(question: str,
                              collection: chromadb.Collection,
                              api_key: str,
                              top_k: int = TOP_K) -> list[dict]:
    question_embedding = embed_texts([question], api_key, input_type="query")[0]

    results = collection.query(
        query_embeddings = [question_embedding],
        n_results        = top_k,
        include          = ["documents", "metadatas", "distances"]
    )

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
            "distance": dist
        })
    return chunks


# ─────────────────────────────────────────────────────────
# STEP 6 — GENERATION
# ─────────────────────────────────────────────────────────

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


def generate_answer(question: str, chunks: list[dict],
                    client: anthropic.Anthropic) -> str:
    context_parts = []
    for chunk in chunks:
        label = f"[Source: {chunk['filename']}]"
        context_parts.append(f"{label}\n{chunk['text']}")
    context_str = "\n\n---\n\n".join(context_parts)

    system = SYSTEM_PROMPT.format(context=context_str)

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

def main():
    st.set_page_config(
        page_title = "GAIL's Dataroom Assistant",
        page_icon  = "🍞",
        layout     = "wide"
    )

    st.markdown("""
    <div style='padding: 1.5rem 0 0.5rem 0;'>
        <h1 style='margin:0; font-size:1.6rem; font-weight:700;'>🍞 GAIL's Bakery — Dataroom Assistant</h1>
        <p style='margin:0.3rem 0 0 0; color:#666; font-size:0.9rem;'>
            Ask questions about GAIL'S LIMITED (06055393) · Powered by RAG + Claude · Golborne Capital
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Sidebar ───────────────────────────────────────────
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
        st.error("⚠️ No API key found. Set ANTHROPIC_API_KEY in Streamlit secrets or as an environment variable.")
        st.code('export ANTHROPIC_API_KEY="sk-ant-..."', language="bash")
        st.stop()

    client = anthropic.Anthropic(api_key=api_key)

    # ── Session state ─────────────────────────────────────
    if "vector_store_ready" not in st.session_state:
        st.session_state.vector_store_ready = False
        st.session_state.collection         = None
        st.session_state.messages           = []
        st.session_state.pending_question   = None

    # ── Build vector store (once) ─────────────────────────
    if not st.session_state.vector_store_ready:
        with st.spinner("📚 Indexing dataroom — loading and embedding 13 documents..."):
            try:
                docs   = load_documents(DATAROOM_DIR)
                chunks = chunk_documents(docs)

                progress_bar = st.progress(0, text="Starting embedding...")

                collection = build_vector_store(chunks, api_key, progress_bar)

                progress_bar.empty()
                st.session_state.collection         = collection
                st.session_state.vector_store_ready = True
                st.success(f"✅ Dataroom indexed — {len(chunks)} chunks from {len(docs)} documents ready.")

            except Exception as e:
                st.error(f"Failed to build vector store: {e}")
                st.stop()

    # ── Chat history ──────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📎 Sources retrieved", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(f"**{src['filename']}** — {src['title']}")
                        st.markdown(
                            f"<small style='color:#888'>Distance: {src['distance']:.3f} (lower = more relevant)</small>",
                            unsafe_allow_html=True)
                        st.markdown(f"> {src['text'][:300]}...")
                        st.divider()

    # ── Handle sidebar button questions ──────────────────
    if st.session_state.get("pending_question"):
        question = st.session_state.pending_question
        st.session_state.pending_question = None
    else:
        question = st.chat_input("Ask anything about GAIL's Bakery...")

    # ── Process question ──────────────────────────────────
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching dataroom and generating answer..."):
                try:
                    chunks = retrieve_relevant_chunks(
                        question,
                        st.session_state.collection,
                        api_key
                    )
                    answer = generate_answer(question, chunks, client)

                    st.markdown(answer)

                    with st.expander("📎 Sources retrieved", expanded=False):
                        for chunk in chunks:
                            st.markdown(f"**{chunk['filename']}** — {chunk['title']}")
                            st.markdown(
                                f"<small style='color:#888'>Distance: {chunk['distance']:.3f} (lower = more relevant)</small>",
                                unsafe_allow_html=True)
                            st.markdown(f"> {chunk['text'][:300]}...")
                            st.divider()

                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": answer,
                        "sources": chunks
                    })

                except Exception as e:
                    err_msg = f"Error: {e}"
                    st.error(err_msg)
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": err_msg
                    })


if __name__ == "__main__":
    main()
