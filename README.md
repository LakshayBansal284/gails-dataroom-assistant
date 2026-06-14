# GAIL's Bakery — Dataroom AI Assistant
**Golborne Capital · AI Engineer Intern Case Study | Lakshay Bansal**

---

## Live Demo
🔗https://gails-dataroom-assistant-7za9gapsmkkxzjppzuxprw.streamlit.app/

---

## What It Does

A natural language assistant over a structured 13-document dataroom for GAIL'S LIMITED
(Companies House: 06055393), a UK premium artisan bakery chain trading as GAIL's Bakery.

Ask questions in plain English and receive accurate answers with source citations drawn
directly from public filings and attributed press reporting.

**Example questions:**
- What was revenue and EBITDA in the last reported year?
- What charges are registered against the company and who holds them?
- Who are the current directors and when were they appointed?
- What are the key risks for a lender?
- Draft a short credit summary of the business.
- What is GAIL's net debt? ← correctly says the dataroom doesn't contain this

---

## Architecture

```
User types a question
        ↓
All 13 dataroom documents loaded into Claude's context window (~7,500 words)
        ↓
Claude reads the complete dataroom and generates an answer
        ↓
Answer returned with [Source: filename.md] citations
```

### Why full-context retrieval, not RAG

RAG exists to solve one problem: the knowledge base is too large to fit in a prompt.

The complete dataroom is approximately 7,500 words.
Claude's context window holds over 150,000 words.
The dataroom fits in under 5% of the available space.
The size problem does not exist.

Sending the complete dataroom on every query means:

- Claude always has access to every document simultaneously
- Financial figures are read directly from the filed source text — the actual
  accounts are in the prompt, making hallucination extremely difficult
- Cross-document reasoning works naturally — a question about lender risks
  draws from charges, financials, ownership and news all at once
- No embedding model, no vector database, no second API key required
- Startup is instant — just reading text files, no API calls at startup
- Nothing to break during a live demo

At the scale of this dataroom, full-context retrieval is strictly superior to RAG
on every dimension that matters: accuracy, reliability, simplicity, and cost.

RAG would be introduced if the dataroom grew beyond ~50 documents (~100,000+ words),
at which point chunking, embedding (Voyage-3), and vector search (ChromaDB) would
be added as a retrieval layer in front of the same generation logic.

---

## Dataroom (13 documents)

| # | Document | Primary Source |
|---|----------|----------------|
| 01 | Company Overview | Companies House (06055393) |
| 02 | FY2025 Financials | Grain Topco accounts (filed Nov 2025); British Baker; The Grocer |
| 03 | FY2024 Financials | Grain Topco accounts (filed Nov 2024) |
| 04 | FY2023 & Historical | Grain Topco accounts (filed Nov 2023); MCA Insight |
| 05 | Charges Register | Companies House charges register |
| 06 | Ownership & Management | Companies House PSC + officers; Bain Capital press release |
| 07 | News & Events | British Baker; Sky News; Bloomberg; Wikipedia |
| 08 | Subsidiary Accounts Note | Companies House; UK.GlobalDatabase |
| 09 | Lender Risks | Compiled from all sources |
| 10 | Credit Summary | Compiled from all sources |
| 11 | Bain Capital Acquisition | Bain Capital press release; Willkie Farr |
| 12 | Sale Process & Valuation | Sky News; Bloomberg; The Grocer |
| 13 | Dataroom Index | This research |

### Key dataroom decision
GAIL'S LIMITED files as an audit-exempt subsidiary. Its standalone accounts show
only ~£636k EBITDA — an artefact of intercompany transfer pricing with The Bread
Factory manufacturing entity. All financial analysis uses Grain Topco consolidated
accounts, which is the correct basis for credit underwriting. Document 08 explains
this distinction explicitly so the assistant can surface it when relevant.

---


## Tech Stack

| Component | Tool |
|-----------|------|
| LLM | Anthropic Claude (claude-sonnet-4-6) 
| Web framework | Streamlit |
| Deployment | Streamlit Cloud |
| Vector DB | None needed |
| Embeddings | None needed |


---

## What I Would Build Further

- **RAG at scale** — introduce chunking, Voyage-3 embeddings, and ChromaDB retrieval
  if the dataroom grew beyond ~50 documents
- **PDF ingestion** — ingest Companies House PDFs directly via pdfplumber rather than
  transcribed markdown; enables automatic refresh on new filings
- **Structured data layer** — store financial tables as JSON for guaranteed numerical
  precision on exact figure lookups
- **Streaming responses** — word-by-word answer rendering for better UX on longer outputs
- **Document refresh alerts** — daily check for new Companies House filings;
  next accounts due 30 November 2026 (year to 28 February 2026)
