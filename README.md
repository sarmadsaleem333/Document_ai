# Local Document Classification and Retrieval

This project classifies documents in `data/` into `Invoice`, `Resume`, `Utility Bill`, or `Other / Unclassifiable`, extracts structured fields into `output/output.json`, and provides a local semantic search system for meaning-based document queries.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#install-dependencies)
- [Usage](#run-the-program)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Technologies](#libraries-and-methods-used)
- [Scalability](#scalability-notes)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Classify documents
python src/main.py

# 3. Search semantically
python src/main.py search "payments due in January"

# 4. Ask a question (optional with local LLM)
python src/main.py qa "What is the invoice total?"
```

## Install Dependencies

Install everything with:

```bash
pip install -r requirements.txt
```

**Core dependencies:**

- `pdfplumber` – PDF text extraction
- `sentence-transformers` – Text-to-vector embeddings (384-dim vectors)
- `numpy` – Vector operations and similarity scoring
- `scikit-learn` – Optional future machine learning

**Optional:**

- `transformers` – Needed only if using local QA with a Hugging Face text-generation model

## Run the Program

### 1. Document Classification (Default)

From the project root:

```bash
python src/main.py
```

This:

- Loads all documents from `data/` (PDF or TXT)
- Classifies each document using a weighted keyword + regex scorer
- Extracts structured fields (invoice number, email, account, etc.)
- Saves results to `output/output.json`

**Sample output:**

```json
{
  "invoice.pdf": {
    "class": "Invoice",
    "invoice_number": "INV-1234",
    "date": "2023-06-13",
    "company": "ACME Corp",
    "total_amount": 10620.0
  }
}
```

### 2. Semantic Search

```bash
python src/main.py search "Find all documents mentioning payments due in January"
```

This:

1. Chunks each document into overlapping 120-word segments
2. Encodes all chunks as 384-dimensional vectors using `SentenceTransformer("all-MiniLM-L6-v2")`
3. Encodes your query to the same vector space
4. Ranks chunks by cosine similarity
5. Returns top-3 most relevant chunks with scores

**Example output:**

```
- Sample_Utility_Bill.pdf [chunk 8] score=0.489
  within 20 days of the billing date, the bill is considered correct and returned payment. payable...

- Sample_Utility_Bill.pdf [chunk 4] score=0.423
  Box 3855; Seattle, WA 98124-3855 DELINQUENT BILLINGS On the last day of the month...
```

**Use with `--top-k` to change result count:**

```bash
python src/main.py search "invoice amount" --top-k 5
```

### 3. Optional Local QA

```bash
python src/main.py qa "What is the amount due in the utility bill?"
```

This retrieves relevant chunks (via semantic search) and displays them without generation.

**With a local Hugging Face model:**

```bash
python src/main.py qa "What is the amount due?" --llm-model mistral-7b-instruct-v0.1
```

The system will:

1. Retrieve the 3 most relevant chunks
2. Build a prompt: `"Context: [chunks]\n\nQuestion: ...\nAnswer:"`
3. Run the local model to generate an answer

## How It Works

### Classification Pipeline

**Step 1: Weighted Scoring**

- Each document is scored against keyword/regex patterns for Invoice, Resume, and Utility Bill
- Example: Finding "invoice" → +1 score, "invoice #" → +1 score, regex `INV-\d+` → +1 score
- The class with the highest score wins
- If scores are tied or all zeros → classified as `Other / Unclassifiable`

**Step 2: Field Extraction**

- Once classified, extraction functions use flexible regex patterns to pull structured data:
  - **Invoice:** invoice number, date (multiple formats), company, total amount
  - **Resume:** name, email, phone, years of experience
  - **Utility Bill:** account number, date, kWh usage, amount due

### Semantic Search Pipeline

**Step 1: Document Chunking**

```python
def _chunk_text(text, chunk_size_words=120, overlap_words=30):
    # Split text into 120-word chunks with 30-word overlap
    # Overlap prevents important context from being cut at boundaries
```

**Step 2: Encoding**

```python
# Each chunk → 384-dimensional vector using SentenceTransformer
# The model is a lightweight BERT variant trained on sentence similarity
embeddings = model.encode(chunk_texts, normalize_embeddings=True)
# Shape: (num_chunks, 384)
```

**Step 3: Similarity Search**

```python
# Query → same 384-dim vector space
query_embedding = model.encode([query], normalize_embeddings=True)

# Cosine similarity via dot product (vectors are normalized)
scores = embeddings @ query_embedding  # (num_chunks,) scores

# Rank and return top-k
top_indices = np.argsort(-scores)[:top_k]
```

**Why this approach?**

- **Chunking:** Preserves local context; allows fine-grained ranking
- **Normalized embeddings:** Cosine similarity via dot product (fast, interpretable 0–1 range)
- **384 dimensions:** Rich enough for meaning; small enough for speed
- **All local:** No API calls, no external dependencies

## Architecture

```
data/
├── invoice.pdf
├── Resume_Sarmad-Saleem.pdf
└── Sample_Utility_Bill.pdf

src/
├── main.py              # CLI entry point (classify, search, qa)
├── classifier.py        # Weighted keyword/regex scoring
├── parser.py            # Field extraction (invoice, resume, bill)
├── extractor.py         # PDF/TXT loading
└── retrieval.py         # Chunking, embedding, semantic search

output/
└── output.json          # Classification + extraction results
```

### Data Flow

```
Documents → Load & Chunk → Encode (SentenceTransformer)
              ↓
          384-dim vectors (stored in NumPy array)
              ↓
          Query → Encode → Cosine similarity → Rank → Return top-k
```

## Libraries and Methods Used

| Library                | Purpose                        | Why?                                   |
| ---------------------- | ------------------------------ | -------------------------------------- |
| `pdfplumber`           | PDF text extraction            | Standard, reliable, open-source        |
| `SentenceTransformers` | Text-to-vector embeddings      | Pretrained, fast, 384-dim output       |
| `NumPy`                | Vector similarity scoring      | Efficient dot product; no DB overhead  |
| `transformers`         | Optional local text generation | Hugging Face ecosystem; fully local QA |
| `scikit-learn`         | Available for future ML tasks  | Common baseline; not currently used    |

### Why NumPy Instead of FAISS / Vector Database?

**NumPy (chosen):**

- ✓ Simple: 1 dot product, O(n) time
- ✓ No extra dependencies (already have NumPy)
- ✓ Fast enough: 14 chunks → microseconds
- ✓ Clear code: `scores = embeddings @ query_embedding`

**FAISS / Vector DB (not needed here):**

- Better for 10,000+ chunks or real-time <5ms requirements
- Adds complexity, disk I/O, serialization overhead
- Overkill for a 3–4 document project

**Scalability:** If you expand to 10,000+ documents, switch to FAISS/Milvus/Weaviate.

## Output Format

### Classification Output: `output/output.json`

```json
{
  "filename.pdf": {
    "class": "Invoice | Resume | Utility Bill | Other / Unclassifiable",
    "invoice_number": "string or null",
    "date": "YYYY-MM-DD or MM/DD/YYYY or null",
    "company": "string or null",
    "total_amount": "number or null",
    "email": "string or null",
    "phone": "string or null",
    "experience_years": "number or null",
    "account_number": "string or null",
    "usage_kwh": "number or null",
    "amount_due": "number or null"
  }
}
```

Fields are populated only for their relevant document class.

## Scalability Notes

| Scale             | Approach        | Notes                                 |
| ----------------- | --------------- | ------------------------------------- |
| **1–100 docs**    | NumPy (current) | All embeddings in RAM; <10ms search   |
| **100–10K docs**  | NumPy + caching | Save/load embeddings as `.npy` file   |
| **10K–1M docs**   | FAISS or Milvus | GPU acceleration, disk-backed indexes |
| **Real-time API** | Milvus/Weaviate | High throughput, concurrent searches  |

Current setup handles **100s of documents** comfortably.

## Future Improvements

- [ ] Cache embeddings to disk (avoid re-encoding on restart)
- [ ] Add FAISS for faster exact nearest-neighbor search at scale
- [ ] Integrate a local LLM (Mistral, Llama 2) for QA without external APIs
- [ ] Add confidence scores to classification output
- [ ] REST API wrapper (FastAPI) for client–server deployment
