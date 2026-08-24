# 📚 DocuMind

> A multi-document Retrieval-Augmented Generation (RAG) application for asking questions across PDF documents using Gemini, ChromaDB, semantic vector search, BM25, and Reciprocal Rank Fusion.

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-LLM%20%26%20Embeddings-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Database-FF6B6B)](https://www.trychroma.com/)
[![BM25](https://img.shields.io/badge/BM25-Keyword%20Retrieval-6C5CE7)](https://pypi.org/project/rank-bm25/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

🔗 **GitHub:** https://github.com/Phawfull/DocuMind  
🌐 **Live Demo:** https://documind-rag.streamlit.app/

> **Version note:** The `main` branch preserves the original Document Q&A project built during my internship. The upgraded application is maintained in the `v2` branch.

---

# Overview

DocuMind started as a simple terminal-based Document Question Answering proof of concept that I built during my internship.

I later revisited the project and evolved it into a multi-document RAG application with a web interface, hybrid retrieval, source tracking, and public deployment.

Instead of sending an entire PDF directly to an LLM, DocuMind first retrieves relevant passages from the uploaded documents and then provides those passages to Gemini as context for answer generation.

The current V2 pipeline is:

```text
PDFs
  ↓
Text Extraction
  ↓
Chunking
  ↓
Gemini Embeddings
  ↓
ChromaDB
  ↓
Vector Search + BM25
  ↓
RRF Fusion
  ↓
Retrieved Context
  ↓
Gemini 2.5 Flash
  ↓
Answer + Sources
```

---

# Features

- 📄 Multi-document PDF ingestion
- 🧠 Gemini `gemini-embedding-001` embeddings
- 🗄️ ChromaDB vector storage
- 🔎 Semantic vector retrieval
- 🔤 BM25 keyword retrieval
- 🔗 Reciprocal Rank Fusion (RRF)
- 🤖 Gemini 2.5 Flash answer generation
- 📌 Document and page-level source tracking
- 💬 Streamlit web interface
- 🔐 Session-scoped document handling
- 🧹 Session document management
- 🛡️ Lightweight usage limits for the public demo
- 💻 Original CLI version preserved in `main.py`

---

# Architecture

```text
                         ┌─────────────────────────┐
                         │      Streamlit UI       │
                         │                         │
                         │  • Upload PDFs          │
                         │  • Document list        │
                         │  • Chat interface       │
                         │  • Source display       │
                         └────────────┬────────────┘
                                      │
                         Uploaded PDF / Question
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
             Document Ingestion                  Question Retrieval
                    │                                   │
                    ▼                                   ▼
              PyMuPDF Parser                    Gemini Query Embedding
                    │                                   │
                    ▼                         ┌─────────┴─────────┐
               Text Chunking                  │                   │
                    │                         ▼                   ▼
                    ▼                   ChromaDB Vector       BM25 Search
            Gemini Embeddings                Search               │
                    │                         │                   │
                    ▼                         └─────────┬─────────┘
                 ChromaDB                              ▼
                    │                            RRF Fusion
                    │                                │
                    │                                ▼
                    │                         Retrieved Chunks
                    │                                │
                    └────────────────────────────────┤
                                                     ▼
                                              Gemini 2.5 Flash
                                                     │
                                                     ▼
                                              Answer + Sources
```

---

# How It Works

## 1. Document Ingestion

The user uploads one or more PDF files through the Streamlit interface.

PyMuPDF extracts text page by page while preserving page information.

```text
PDF
 ↓
PyMuPDF
 ↓
Extracted Text
```

---

## 2. Text Chunking

The extracted text is divided into overlapping chunks.

Chunking allows the retrieval system to work with smaller, more relevant passages instead of an entire document at once.

Each chunk retains metadata such as:

```text
document_name
page_number
chunk_index
session_id
```

---

## 3. Embedding Generation

Each document chunk is converted into a semantic embedding using:

```text
gemini-embedding-001
```

Document chunks use:

```text
RETRIEVAL_DOCUMENT
```

User queries use:

```text
RETRIEVAL_QUERY
```

This allows the embedding model to distinguish between document content being indexed and a query being searched.

---

## 4. Vector Storage

The chunks, embeddings, and metadata are stored in ChromaDB.

```text
Chunk
  ↓
Embedding
  ↓
ChromaDB
```

---

## 5. Semantic Retrieval

When a user asks a question, the query is converted into an embedding and searched against the stored document vectors.

```text
Question
   ↓
Gemini Query Embedding
   ↓
ChromaDB Similarity Search
   ↓
Candidate Chunks
```

---

## 6. BM25 Keyword Retrieval

The same query is also searched using BM25.

BM25 provides a lexical retrieval path that is particularly useful for:

- exact terminology
- names
- identifiers
- technical phrases
- numbers and other keyword-sensitive queries

Example:

```text
ISO 27001
Model XJ-500
Project Alpha
```

---

## 7. Reciprocal Rank Fusion

The results from semantic retrieval and BM25 are combined using Reciprocal Rank Fusion.

```text
Semantic Search
       +
      BM25
       ↓
   RRF Fusion
       ↓
Final Retrieved Context
```

This allows the system to benefit from both semantic similarity and exact keyword matching.

---

## 8. Answer Generation

The retrieved chunks are passed to Gemini 2.5 Flash together with the original question.

```text
User Question
      +
Retrieved Context
      ↓
Gemini 2.5 Flash
      ↓
Generated Answer
```

The generation layer is designed to answer from the supplied document context rather than relying solely on general model knowledge.

---

## 9. Source Display

Retrieved chunks retain their document and page information.

Example:

```text
📄 Annual_Report.pdf · Page 14

Relevant passage:
Revenue increased by ...
```

This gives users a way to inspect the material that was retrieved for the answer.

---

# Why Hybrid Retrieval?

A single retrieval strategy is not equally effective for every query.

### Semantic Vector Search

Useful when the wording of the question differs from the wording used in the source.

```text
Question:
How many vehicles did the company deliver?

Document:
The manufacturer delivered 1.2 million vehicles.
```

The wording differs, but the meaning is closely related.

### BM25

Useful when exact terms matter.

```text
ISO 27001
Model XJ-500
Project Alpha
```

### RRF

RRF combines the rankings from both retrieval methods:

```text
Semantic Retrieval
       +
Keyword Retrieval
       ↓
   RRF Fusion
       ↓
Retrieved Context
```

---

# Session-Based Document Handling

The V2 application associates document chunks with the active Streamlit session.

Conceptually:

```text
Browser A
   ↓
Session A
   ↓
PDF A + PDF B

Browser B
   ↓
Session B
   ↓
PDF C
```

This is designed so uploaded documents are scoped to the current application session rather than being treated as one shared user corpus.

The application also allows the current session's documents to be cleared from the interface.

---

# Project Structure

```text
DocuMind/
│
├── app.py                 # Streamlit web application
├── main.py                # Original CLI application
│
├── ingestion.py           # PDF extraction, chunking, embeddings, storage
├── retrieval.py           # Query embeddings and vector retrieval
├── hybrid_retrieval.py    # BM25 + vector retrieval + RRF
├── generation.py          # Gemini answer generation
│
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Git exclusions
│
├── .env                   # Local secrets, not committed
└── chroma_db/             # Local ChromaDB data, not committed
```

---

# Core Components

## `app.py`

The Streamlit entry point.

Responsibilities include:

- PDF uploads
- indexed-document display
- chat interface
- session state
- source display
- usage limits
- session document management
- user-facing error handling

Run it using:

```bash
streamlit run app.py
```

---

## `ingestion.py`

Handles the document ingestion pipeline:

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Gemini Embeddings
 ↓
ChromaDB
```

It also stores document, page, chunk, and session metadata.

---

## `retrieval.py`

Handles:

```text
User Question
 ↓
Gemini Query Embedding
 ↓
ChromaDB Vector Search
 ↓
Retrieved Chunks
```

---

## `hybrid_retrieval.py`

Combines vector retrieval and BM25:

```text
Vector Search
      +
BM25
      ↓
RRF Fusion
      ↓
Final Retrieved Chunks
```

---

## `generation.py`

Builds the grounded prompt and uses Gemini 2.5 Flash to generate the final answer.

---

## `main.py`

The original terminal-based interface is preserved separately from the Streamlit application.

Run it with:

```bash
python main.py
```

---

# Technology Stack

| Technology | Purpose |
|------------|---------|
| **Python** | Core programming language |
| **Streamlit** | Web interface |
| **PyMuPDF** | PDF text extraction |
| **Google Gemini** | Embeddings and answer generation |
| **ChromaDB** | Vector storage and similarity search |
| **rank-bm25** | BM25 keyword retrieval |
| **python-dotenv** | Environment variable management |

### Models

**Embeddings**

```text
gemini-embedding-001
```

**Answer Generation**

```text
gemini-2.5-flash
```

---

# Running Locally

## Prerequisites

- Python 3.x
- Google Gemini API key
- Git

## 1. Clone the repository

```bash
git clone https://github.com/Phawfull/DocuMind.git
cd DocuMind
```

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Do not commit this file.

## 5. Run the web application

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## 6. Run the original CLI

```bash
python main.py
```

---

# 🌐 Live Deployment

The V2 application is deployed using Streamlit Community Cloud.

```text
GitHub
  ↓
v2 branch
  ↓
Streamlit Community Cloud
  ↓
app.py
  ↓
Public Demo
```

### Live Demo

https://documind-rag.streamlit.app/

The deployed application uses Streamlit Secrets for the Gemini API key rather than storing the credential in the repository.

---

# 🔐 Security

Never commit API keys or private credentials.

The following should remain outside version control:

```text
.env
.venv/
chroma_db/
__pycache__/
.idea/
```

The deployed application should configure:

```text
GEMINI_API_KEY
```

through Streamlit's secret management.

---

# ⚠️ Current Limitations

DocuMind is currently a portfolio and learning project rather than a production document platform.

Current limitations include:

- Text-based PDFs are the primary supported input.
- Scanned/image-only PDFs do not currently use OCR.
- ChromaDB is used as the local vector database.
- The public deployment is designed as a demo rather than a production SaaS application.
- Gemini usage is subject to the limits and policies of the configured Google project.
- Conversational follow-up questions are not yet rewritten into standalone retrieval queries.
- There is currently no formal retrieval benchmark in the repository.
- The project does not claim guaranteed factual correctness beyond the retrieved context.

---

# 🧪 Manual Verification

Before releasing a change, verify:

```text
[ ] PDF uploads successfully
[ ] Text extraction completes
[ ] Embeddings are generated
[ ] Document appears in the indexed-document list
[ ] A basic question returns an answer
[ ] Sources display the expected document/page
[ ] Multiple PDFs can be indexed
[ ] A question about the second PDF retrieves correctly
[ ] Cross-document questions work
[ ] Duplicate uploads are handled
[ ] Clear Chat only clears conversation history
[ ] Clear My Documents removes the current session's documents
[ ] .env is not committed
[ ] chroma_db/ is not committed
```

---

# 📈 Future Improvements

Possible future directions include:

- Retrieval evaluation and benchmarking
- Unit tests for chunking and RRF
- Batched embedding requests
- Retry and backoff for transient API failures
- More advanced BM25 processing
- Persistent BM25 indexing
- Query rewriting for conversational retrieval
- OCR for scanned PDFs
- Better document previews
- Streaming responses
- Managed vector storage
- Production authentication and multi-user infrastructure
- FastAPI backend
- React frontend

---

# 📚 Concepts Demonstrated

DocuMind demonstrates practical implementation of:

- Retrieval-Augmented Generation (RAG)
- Document chunking
- Embeddings
- Semantic search
- Vector databases
- Keyword retrieval
- BM25
- Reciprocal Rank Fusion
- Prompt construction
- Source metadata
- LLM application development
- Multi-document retrieval
- Streamlit development
- Session state
- Environment and secret management

---

# 📚 Version History

## V1 — Internship POC

The original Document Q&A project built during my internship.

```text
PDF
 ↓
Chunking
 ↓
Gemini Embeddings
 ↓
ChromaDB
 ↓
Vector Retrieval
 ↓
Gemini
```

The original implementation is preserved in the `main` branch.

## V2 — DocuMind

The upgraded version developed afterward.

```text
Multiple PDFs
 ↓
Gemini Embeddings
 ↓
ChromaDB
 ↓
Vector Search + BM25
 ↓
RRF
 ↓
Gemini
 ↓
Answer + Sources
```

V2 adds:

- multi-document ingestion
- hybrid retrieval
- source tracking
- Streamlit web interface
- public deployment
- session-scoped document handling
- usage controls for the public demo

---

# 👤 Author

**Arush Gupta**

B.Tech Computer Science Engineering — AI/ML

GitHub: https://github.com/Phawfull/DocuMind

---

# ⭐ Project Status

**V2 — Deployed and actively being improved**

DocuMind represents the evolution of a basic internship Document Q&A POC into a multi-document RAG application with hybrid retrieval and a public web interface.

The `main` branch preserves the original implementation, while the `v2` branch contains the upgraded version.

---

# 📄 License

This project is licensed under the [MIT License](LICENSE).
