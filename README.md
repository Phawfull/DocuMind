# 📚 DocuMind

> A multi-document Retrieval-Augmented Generation (RAG) application for asking grounded questions across PDF documents using Gemini, ChromaDB, semantic vector search, BM25 keyword retrieval, and Reciprocal Rank Fusion.

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-LLM%20%26%20Embeddings-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Database-FF6B6B)](https://www.trychroma.com/)
[![BM25](https://img.shields.io/badge/BM25-Keyword%20Retrieval-6C5CE7)](https://pypi.org/project/rank-bm25/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

DocuMind is a document question-answering application built around a **Retrieval-Augmented Generation (RAG)** architecture.

Instead of relying on an LLM's general knowledge alone, DocuMind first retrieves relevant passages from the user's documents and then provides those passages to Gemini as the answer context. This helps keep responses grounded in the uploaded material.

The application supports **multiple PDF documents**, allowing users to build a small document knowledge base and ask questions across the indexed collection through a Streamlit web interface.

The retrieval layer combines two complementary approaches:

- **Semantic vector search** using Gemini-generated embeddings and ChromaDB
- **Lexical keyword search** using BM25

The two ranked result sets are combined using **Reciprocal Rank Fusion (RRF)** before the most relevant context is passed to Gemini for answer generation.

---

## ✨ Key Features

### Multi-document PDF ingestion
Upload multiple PDF documents and keep them indexed in a shared ChromaDB collection.

### Semantic retrieval
Gemini embeddings are used to represent document chunks and user queries in vector space, enabling retrieval based on semantic similarity rather than exact wording.

### BM25 keyword retrieval
BM25 provides a complementary lexical retrieval path that is useful for exact names, terminology, identifiers, technical phrases, and other keyword-sensitive queries.

### Hybrid retrieval with RRF
Results from semantic search and BM25 are combined using Reciprocal Rank Fusion to produce a single ranked candidate list.

### Source-aware answers
Retrieved chunks retain document and page metadata so answers can be associated with the source document and page that supplied the context.

### Streamlit web interface
Users can upload documents, see indexed files, ask questions, view answers, and inspect the retrieved source passages directly from a browser.

### Duplicate document protection
Documents are tracked using their filename so the application can avoid repeatedly ingesting the same document.

### Persistent local vector storage
ChromaDB is persisted locally, allowing indexed documents to remain available across application sessions during local development.

---

## 🧠 Architecture

```text
                         ┌─────────────────────────┐
                         │      Streamlit UI        │
                         │                          │
                         │  • Upload PDFs           │
                         │  • Document list          │
                         │  • Chat interface         │
                         │  • Source display         │
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
                ChromaDB                                ▼
          ┌───────────────┐                       RRF Fusion
          │ Text          │                              │
          │ Embedding     │                              ▼
          │ Document Name │                       Top Retrieved
          │ Page Number   │                           Chunks
          │ Chunk Index   │                              │
          └───────────────┘                              ▼
                                                   Gemini 2.5 Flash
                                                          │
                                                          ▼
                                                   Answer + Sources
```

---

## 🔄 End-to-End Workflow

### 1. Upload

The user uploads one or more PDFs through the Streamlit interface.

### 2. Text extraction

PyMuPDF extracts text from each page.

### 3. Chunking

The extracted text is divided into overlapping chunks so that each retrieval unit remains small enough to search effectively.

### 4. Embedding generation

Each chunk is converted into an embedding using Google's `gemini-embedding-001` model.

### 5. Storage

Chunks, embeddings, and source metadata are stored in the persistent ChromaDB collection.

Each chunk retains metadata including:

```text
document_name
page_number
chunk_index
```

### 6. Query embedding

When a user asks a question, the question is embedded using the same Gemini embedding model.

### 7. Semantic retrieval

ChromaDB performs vector similarity search to find chunks that are semantically related to the query.

### 8. Keyword retrieval

BM25 performs lexical search across the indexed chunk text.

### 9. Reciprocal Rank Fusion

The ranked outputs from semantic retrieval and BM25 are combined using RRF.

The purpose is to benefit from both:

```text
Semantic search → meaning and related concepts
BM25            → exact words and phrases
```

### 10. Context construction

The highest-ranked retrieved chunks are passed to the generation layer with their source metadata.

### 11. Answer generation

Gemini 2.5 Flash generates an answer using the retrieved context.

The generation prompt instructs the model to remain grounded in the supplied document context.

### 12. Source display

The Streamlit application displays the document and page associated with retrieved chunks so users can inspect the supporting passages.

---

## 🗂️ Project Structure

```text
DOCUMENT_QA_POC/
│
├── app.py                 # Streamlit web application
├── main.py                # Original command-line interface
│
├── ingestion.py           # PDF extraction, chunking, embeddings, storage
├── retrieval.py           # Query embeddings and semantic vector search
├── hybrid_retrieval.py    # BM25 + vector retrieval + RRF fusion
├── generation.py           # Gemini answer generation
│
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git exclusions
├── .env                    # Local environment variables (not committed)
│
└── chroma_db/              # Local ChromaDB data (not committed)
```

---

## 🧩 Core Components

### `app.py`

The Streamlit entry point.

Responsibilities include:

- PDF uploads
- document status display
- chat UI
- session-level chat history
- source display
- user-friendly error handling

Run it with:

```bash
streamlit run app.py
```

### `ingestion.py`

Handles the document ingestion pipeline:

```text
PDF
→ page text extraction
→ chunking
→ Gemini embeddings
→ ChromaDB
```

It also attaches source metadata to every chunk.

### `retrieval.py`

Provides the core semantic retrieval functionality:

```text
question
→ Gemini embedding
→ ChromaDB similarity search
→ structured retrieved chunks
```

### `hybrid_retrieval.py`

Adds lexical retrieval and result fusion:

```text
Vector Search
      +
    BM25
      ↓
   RRF Fusion
      ↓
 Final retrieved chunks
```

### `generation.py`

Constructs the grounded context prompt and calls Gemini to generate the final answer.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python | Core application |
| UI | Streamlit | Web interface |
| PDF processing | PyMuPDF | Text extraction |
| Embeddings | Gemini `gemini-embedding-001` | Semantic document/query embeddings |
| Vector database | ChromaDB | Persistent vector storage and similarity search |
| Keyword retrieval | `rank-bm25` | BM25 lexical search |
| Fusion | Reciprocal Rank Fusion | Combine vector and lexical rankings |
| LLM | Gemini 2.5 Flash | Grounded answer generation |
| Configuration | `python-dotenv` | Load environment variables |

---

## 🚀 Getting Started

### Prerequisites

Install:

- Python 3.x
- A Google Gemini API key
- Git (optional but recommended)

Create a Gemini API key through Google's AI tooling and keep it private.

---

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd document_qa_poc
```

---

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

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Do **not** commit `.env` to GitHub.

---

## 5. Run the web application

```bash
streamlit run app.py
```

Then open the local URL provided by Streamlit, typically:

```text
http://localhost:8501
```

---

## 6. Optional: Run the CLI version

The original command-line interface is still available:

```bash
python main.py
```

The CLI is useful for debugging and for interacting with the backend without the web interface.

---

## 💬 Example Usage

### Upload

Upload:

```text
Annual_Report.pdf
Research_Paper.pdf
Company_Profile.pdf
```

### Ask

```text
What was the company's revenue in 2024?
```

or:

```text
What are the key differences between the two reports?
```

or:

```text
Which document discusses the company's expansion into Europe?
```

The system searches the indexed document collection rather than requiring the user to specify which PDF should be searched.

---

## 🔎 Why Hybrid Retrieval?

A single retrieval strategy does not perform equally well for every query.

### Semantic vector search

Semantic retrieval is useful when the wording of the question differs from the wording in the source.

For example:

```text
Query:
How many vehicles did the company ship?

Document:
The manufacturer delivered 1.2 million vehicles.
```

The wording differs, but the underlying meaning is related.

### BM25

BM25 is useful when exact terms matter.

For example:

```text
ISO 27001:2022
```

or:

```text
Model XJ-500
```

Exact lexical matches can be highly valuable for these cases.

### RRF

Reciprocal Rank Fusion combines the two ranked lists so that chunks appearing strongly in either retrieval strategy can contribute to the final ranking.

The resulting design is:

```text
Semantic Retrieval
        +
Keyword Retrieval
        ↓
   RRF Fusion
        ↓
  Better Context
```

---

## 📌 Source Grounding

Each retrieved chunk retains metadata such as:

```json
{
  "document_name": "Annual_Report.pdf",
  "page_number": 14,
  "chunk_index": 7
}
```

This allows the application to associate generated answers with the original document and page.

The Streamlit interface exposes these sources so users can inspect the retrieved passages instead of treating the generated response as an unsupported black box.

---

## ⚠️ Current Limitations

This project intentionally focuses on text-based PDF documents and a relatively lightweight local architecture.

Current limitations include:

- Scanned/image-only PDFs are not handled through OCR.
- BM25 indexing is maintained in memory for the running application rather than through a dedicated search service.
- ChromaDB is local and is not a production distributed vector database.
- The system is designed primarily as a portfolio/learning application rather than a production multi-user service.
- The application depends on a Gemini API key for embeddings and generation.
- Document identity is currently based on filenames rather than content hashing.
- The current generation pipeline is not designed to provide guaranteed factual correctness beyond the retrieved context.

These are deliberate scope decisions for the current version.

---

## 🔐 Security Notes

Never commit secrets to the repository.

The following should remain local and excluded by `.gitignore`:

```text
.env
.venv/
chroma_db/
```

If the project is deployed, configure the Gemini API key using the hosting platform's secret/environment-variable mechanism rather than committing the key to source control.

---

## 🧪 Testing Checklist

Before presenting or deploying the application, verify:

```text
[ ] A text-based PDF uploads successfully
[ ] PDF text is extracted
[ ] Chunks are created
[ ] Gemini embeddings are generated
[ ] Document is added to ChromaDB
[ ] Indexed document appears in the UI
[ ] A basic question returns an answer
[ ] Sources show the expected PDF/page
[ ] A second PDF can be indexed
[ ] Queries can retrieve information from multiple documents
[ ] Duplicate filenames are handled
[ ] Clear Chat does not delete indexed documents
[ ] .env is not committed
[ ] chroma_db/ is not committed
```

---

## 🌐 Deployment

The application is designed to be deployable as a Streamlit web application.

A lightweight deployment path is:

```text
GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
app.py
       ↓
Public Web Application
```

For deployment:

1. Push the project to GitHub.
2. Create a Streamlit Community Cloud app from the repository.
3. Select `app.py` as the application entry point.
4. Configure `GEMINI_API_KEY` as a deployment secret.
5. Do not upload `.env` or `chroma_db/` as part of the deployment source.

> **Note:** local persistent storage behavior differs from a production database. A production-scale version would use managed persistence and a more robust document/index lifecycle.

---

## 📈 Future Improvements

Possible future directions include:

- Better document ingestion for complex PDFs
- OCR for scanned documents
- More advanced chunking strategies
- Better metadata filtering
- Persistent BM25 indexing
- Retrieval evaluation and benchmarking
- Query rewriting for conversational follow-up questions
- User authentication and document isolation
- Managed/vector database infrastructure
- FastAPI backend
- React-based frontend
- Cloud deployment with production-grade storage
- Streaming model responses
- Document previews and deeper source navigation

These are intentionally outside the current MVP/portfolio scope.

---

## 🎯 Project Goals

DocuMind was developed with a simple goal:

> **Make document-based information easier to query while keeping generated answers grounded in the source material.**

The project focuses on understanding and implementing the core RAG pipeline rather than hiding the architecture behind a large framework.

---

## 📚 Concepts Demonstrated

This project demonstrates practical experience with:

- Retrieval-Augmented Generation (RAG)
- Text preprocessing
- Chunking strategies
- Embeddings
- Vector similarity search
- Vector databases
- Keyword retrieval
- BM25
- Reciprocal Rank Fusion
- Prompt construction
- Source metadata
- LLM application development
- Streamlit application development
- Multi-document retrieval
- Python environment and dependency management

---

## 📷 Screenshots

Add screenshots of the application here after the final UI polish.

Suggested screenshots:

1. Main DocuMind interface
2. Multiple indexed documents
3. Example question and answer
4. Expanded source citations
5. Multi-document comparison question

Example:

```md
![DocuMind interface](screenshots/main-ui.png)
```

---

## 🧑‍💻 Author

**Arush Gupta**

B.Tech Computer Science Engineering — AI/ML

GitHub: [YOUR_GITHUB_PROFILE]

LinkedIn: [YOUR_LINKEDIN_PROFILE]

---

## 📄 License

This project is intended as a learning and portfolio project.

If this repository is intended to be open source, add a license file such as `LICENSE` and update this section accordingly.

---

## ⭐ Acknowledgements

Built using open-source Python libraries and Google's Gemini API.

Core technologies:

- Python
- Streamlit
- PyMuPDF
- ChromaDB
- BM25
- Google Gemini

---

## 💡 Project Summary

```text
DocuMind
│
├── Multi-document PDF ingestion
├── Gemini embeddings
├── ChromaDB semantic search
├── BM25 keyword retrieval
├── Reciprocal Rank Fusion
├── Gemini grounded generation
├── Source-aware responses
└── Streamlit web interface
```

**Build. Retrieve. Understand.**
