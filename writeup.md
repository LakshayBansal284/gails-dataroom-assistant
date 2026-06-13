# Technical Write-Up — GAIL's Dataroom Assistant
**Golborne Capital AI Engineer Case Study**

---

## Approach

The task has two parts: build a structured dataroom, then build an AI assistant over it.

For the **dataroom**, I sourced 13 documents from Companies House, Grain Topco's filed consolidated accounts (FY2023–FY2025), and attributed press coverage. The filing entity GAIL'S LIMITED files as an audit-exempt subsidiary, which means its standalone accounts contain minimal P&L data. The commercially meaningful financials live at the Grain Topco parent level — recognising this distinction was the most important analytical decision in building the dataroom.

For the **assistant**, I implemented a RAG (Retrieval Augmented Generation) pipeline rather than prompt-stuffing the full dataroom into every call. The architecture is: chunk → embed → store → retrieve → generate.

---

## Key Technical Decisions

**1. RAG over prompt-stuffing**
The full dataroom is approximately 25,000 words — within Claude's context window, but sending it all on every query is wasteful and imprecise. RAG retrieves only the 5 most semantically relevant chunks per question. More importantly, it grounds financial figures in specific source passages, making hallucination much harder.

**2. Anthropic Voyage embeddings**
I used the `voyage-3` embedding model via Anthropic's API. This keeps the entire stack on one API key and one provider. Voyage-3 is specifically optimised for retrieval tasks and outperforms generic embedding models on document search benchmarks.

**3. ChromaDB in-memory**
For a 13-document dataroom producing approximately 120 chunks, an in-memory vector database is entirely sufficient. Using ChromaDB in-memory avoids any external service dependency, keeps the deployment to a single file, and starts in seconds. At production scale (thousands of documents), I would migrate to a persistent store.

**4. Source citation enforced via system prompt**
Claude is instructed to attach `[Source: filename]` to every factual claim and to say "the dataroom does not contain this information" rather than guess. Financial figures are required to be quoted exactly as they appear in source documents — not rounded or approximated unless the source itself uses that language.

**5. Chunk size of 400 words with 50-word overlap**
This size balances precision (small enough to retrieve a specific figure) with context (large enough that the surrounding sentence is preserved). Overlap prevents key sentences at chunk boundaries from being split across two chunks with neither containing the full context.

---

## What I Would Develop Further

**Hybrid search** — combining the current semantic similarity search with keyword/BM25 search. For exact figure lookups ("£53.6m"), keyword matching is more reliable than semantic search.

**PDF ingestion pipeline** — the current dataroom uses structured markdown transcriptions. Ingesting the actual Companies House PDFs directly (using pdfplumber or pypdf) would remove the transcription step and allow the dataroom to be refreshed from filed documents automatically.

**Conversation memory** — currently each question is answered independently. Adding prior turns to the message history would allow follow-up questions ("and how does that compare to FY2024?").

**Persistent vector store** — saving the ChromaDB index to disk would eliminate the ~15-second cold-start rebuild on each deployment restart.

**Structured data extraction** — key financial tables (three-year P&L, charges register) should be stored as structured JSON alongside the text chunks, enabling the assistant to answer table-lookup questions with guaranteed numerical precision.
