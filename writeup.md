# Technical Write-Up — GAIL's Dataroom Assistant
**Golborne Capital AI Engineer Case Study**

---

## Approach

The task has two parts: build a structured dataroom, then build an AI assistant over it.

For the **dataroom**, I sourced 13 documents from Companies House, Grain Topco's filed
consolidated accounts (FY2023–FY2025), and attributed press coverage. The key analytical
decision was recognising that GAIL'S LIMITED files as an audit-exempt subsidiary — its
standalone accounts show only ~£636k EBITDA due to intercompany transfer pricing. The
commercially meaningful financials are at Grain Topco parent level. The dataroom is built
around group-level figures accordingly.

For the **assistant**, I used full-context retrieval rather than RAG.

---

## Key Technical Decisions

**1. Full-context over RAG**

The entire dataroom is ~7,500 words. Claude's context window is 200,000 tokens.
The dataroom fits in under 5% of the available context.

RAG (chunking, embedding, vector search) exists to solve a scale problem — when your
knowledge base is too large to fit in a prompt. That problem does not exist here.
Sending the complete dataroom on every query means:
- Claude always has access to every document simultaneously
- Financial figures are always read from the actual source text
- No embedding model, no vector database, no second API key
- Nothing to break during a live demo
- Startup is instant (just reading text files, no API calls)

This is a deliberate architectural choice, not a shortcut.

**2. Source citation enforced via system prompt**

Claude is instructed to attach [Source: filename.md] to every factual claim and to say
"the dataroom does not contain sufficient information" rather than guess. Financial figures
must be quoted exactly as they appear — not rounded or approximated.

**3. Conversation memory**

Prior turns are passed back to Claude on each call, allowing follow-up questions
("and how does that compare to FY2024?") without the user needing to repeat context.

**4. Single API key, single service**

The entire stack runs on one Anthropic key. No secondary services, no configuration
complexity, nothing to rotate or manage separately.

---

## What I Would Develop Further

**Hybrid search for scale** — if the dataroom grew to hundreds of documents, I would
introduce RAG at that point: chunk, embed with Voyage-3, store in ChromaDB, retrieve
top-k per query. The current architecture would not scale beyond ~50,000 words.

**PDF ingestion pipeline** — ingest Companies House PDFs directly using pdfplumber,
rather than transcribed markdown. Would allow automatic refresh when new filings appear.

**Structured data layer** — store key financial tables as JSON alongside the text.
For exact figure lookups, querying structured data is more reliable than asking an LLM
to extract a number from prose.

**Streaming responses** — use Claude's streaming API so the answer appears word by word
rather than after a delay. Better user experience for longer answers like credit summaries.

**Document refresh alert** — a daily script checking Companies House for new filings
(confirmation statements, new accounts) and flagging when the dataroom is out of date.
