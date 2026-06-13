# GAIL's Bakery — Dataroom AI Assistant
**Golborne Capital · AI Engineer Intern Case Study**

A RAG-powered AI assistant over a structured dataroom for GAIL'S LIMITED (Companies House: 06055393).

---

## Live Demo
🔗 [Insert your Streamlit Cloud URL here]

---

## What It Does

Ask natural language questions about GAIL's Bakery and receive accurate answers with source citations drawn from a 13-document dataroom of publicly available information.

**Example questions it can answer:**
- What was revenue and EBITDA in the last reported year?
- What charges are registered against the company and who holds them?
- Who are the current directors and when were they appointed?
- What are the key risks for a lender?
- Draft a short credit summary of the business.
- Who owns GAIL's and what is the ownership structure?

---

## Architecture

```
User question
     ↓
Anthropic Voyage embedding (voyage-3 model)
     ↓
ChromaDB cosine similarity search → top-5 relevant chunks
     ↓
Claude claude-sonnet-4-6 generates answer using only retrieved chunks
     ↓
Answer with source citations displayed in Streamlit UI
```

### Why RAG?
Rather than stuffing the entire dataroom into every prompt (expensive, imprecise), the app:
1. At startup: splits all 13 documents into ~400-word chunks with 50-word overlap, embeds each chunk, stores in ChromaDB
2. At query time: embeds the question, retrieves the 5 most semantically similar chunks, sends only those to Claude

This ensures financial figures come from the exact filed source, not from model memory — critical for accuracy.

### Why Anthropic Voyage embeddings?
- Free tier covers all development and demo use
- Same API key as Claude — no separate service
- voyage-3 is specifically optimised for retrieval tasks

### Handling financial accuracy
- All numerical figures are sourced from filed Companies House accounts (Grain Topco consolidated) or attributed press
- Claude is instructed via system prompt to quote figures exactly as they appear in source documents
- Claude is instructed to say "the dataroom does not contain this information" rather than guess
- The `[Source: filename]` citation pattern makes every figure traceable

---

## Dataroom (13 documents)

| # | Document | Source |
|---|----------|--------|
| 01 | Company Overview | Companies House (06055393) |
| 02 | FY2025 Financials | Grain Topco accounts (filed Nov 2025); British Baker; The Grocer |
| 03 | FY2024 Financials | Grain Topco accounts (filed Nov 2024) |
| 04 | FY2023 & Historical | Grain Topco accounts (filed Nov 2023); MCA Insight |
| 05 | Charges Register | Companies House charges register |
| 06 | Ownership & Management | Companies House PSC + officers; Bain Capital press release |
| 07 | News & Events | British Baker; Sky News; Bloomberg; Wikipedia |
| 08 | Subsidiary Accounts Note | Companies House; UK.GlobalDatabase |
| 09 | Lender Risks | Compiled analysis |
| 10 | Credit Summary | Compiled analysis |
| 11 | Bain Capital Acquisition | Bain Capital press release; Willkie Farr |
| 12 | Sale Process & Valuation | Sky News; Bloomberg; The Grocer |
| 13 | Dataroom Index | This research |

---

## Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/gails-dataroom-assistant
cd gails-dataroom-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Anthropic API key
echo 'ANTHROPIC_API_KEY = "sk-ant-your-key-here"' > .streamlit/secrets.toml

# 4. Run
streamlit run app.py
```

The app opens at http://localhost:8501

---

## Deploying to Streamlit Cloud

1. Push this repo to GitHub (make sure it's public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub → New app → select this repo → `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-actual-key"
   ```
5. Click Deploy

---

## What I Would Build With More Time

1. **PDF ingestion** — ingest the actual Companies House PDFs directly rather than transcribed markdown, using a PDF parser (pypdf or pdfplumber) to extract text page by page
2. **Persistent vector store** — save the ChromaDB index to disk between sessions so it doesn't rebuild on every cold start
3. **Conversation memory** — pass prior turns back to Claude so the user can ask follow-up questions ("and what about FY2024?")
4. **Hybrid search** — combine semantic similarity search (current) with keyword/BM25 search for exact figure lookups (e.g. "£53.6m")
5. **Document refresh pipeline** — a script that checks Companies House for new filings and updates the dataroom automatically
6. **Confidence scoring** — surface the retrieval similarity scores more prominently so users can see how confident the retrieval was
7. **Table-aware chunking** — current word-count chunking can split markdown tables mid-row; a smarter chunker would keep tables intact

---

## Tech Stack
- **Language:** Python 3.11
- **Web framework:** Streamlit
- **LLM:** Anthropic Claude (claude-sonnet-4-6)
- **Embeddings:** Anthropic Voyage (voyage-3)
- **Vector database:** ChromaDB (in-memory)
- **Deployment:** Streamlit Cloud
- **Cost:** £0 (all free tiers)
