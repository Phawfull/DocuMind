# 📄 Document Question Answering System (RAG) using Gemini & ChromaDB

A Retrieval-Augmented Generation (RAG) based Document Question Answering system built in Python using the Gemini API and ChromaDB.

This project allows users to ingest PDF documents, convert them into vector embeddings, store them in a vector database, retrieve semantically similar content, and generate accurate answers using Google's Gemini model.

---

## 🚀 Features

- Extract text from PDF documents
- Clean and preprocess extracted text
- Split documents into overlapping chunks
- Generate embeddings using Gemini Embedding API
- Store embeddings in ChromaDB
- Perform semantic similarity search
- Generate context-aware answers using Gemini
- Modular project structure

---

## 🏗️ Project Structure

```
document_qa_poc/
│
├── ingestion.py      # PDF extraction, chunking, embeddings, storage
├── retrieval.py      # Semantic search using ChromaDB
├── generation.py     # Gemini answer generation
├── main.py           # Entry point
│
├── chroma_db/        # Persistent vector database
├── .env              # Gemini API Key (ignored)
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

- Python 3.14
- Google Gemini API
- ChromaDB
- PyMuPDF (fitz)
- python-dotenv

---

## 📚 How It Works

### 1. Document Ingestion

The PDF is read using PyMuPDF.

```
PDF
    ↓
Extract Text
```

---

### 2. Text Chunking

The extracted text is split into smaller overlapping chunks.

```
Text
    ↓
Chunk 1
Chunk 2
Chunk 3
...
```

Chunk overlap preserves context between adjacent chunks.

---

### 3. Embedding Generation

Each chunk is converted into a dense vector using Gemini's embedding model.

```
Chunk
    ↓
Gemini Embedding API
    ↓
Vector Embedding
```

Model used:

```
models/text-embedding-004
```

---

### 4. Storage

Each chunk is stored inside ChromaDB along with:

- Text
- Embedding
- Page Number
- Chunk Index

```
Chunk
      ↓
Embedding
      ↓
ChromaDB Collection
```

---

### 5. Retrieval

When a user asks a question:

```
Question
      ↓
Embedding
      ↓
Vector Search
      ↓
Top Relevant Chunks
```

Semantic similarity is computed using cosine similarity.

---

### 6. Answer Generation

The retrieved chunks are supplied as context to Gemini.

```
Retrieved Context
         +
User Question
         ↓
Gemini
         ↓
Final Answer
```

---

## 🔄 RAG Pipeline

```
                PDF
                 │
                 ▼
          Extract Text
                 │
                 ▼
          Chunk Document
                 │
                 ▼
      Generate Embeddings
                 │
                 ▼
        Store in ChromaDB
                 │
────────────────────────────────────
                 │
          User Question
                 │
                 ▼
      Generate Query Embedding
                 │
                 ▼
      Search ChromaDB
                 │
                 ▼
      Retrieve Top Chunks
                 │
                 ▼
        Gemini Generation
                 │
                 ▼
            Final Answer
```

---

## 🧠 Concepts Used

- Artificial Intelligence
- Large Language Models (LLMs)
- Retrieval Augmented Generation (RAG)
- Embeddings
- Semantic Search
- Vector Databases
- Cosine Similarity
- Prompt Engineering
- Context Injection

---

## 🛠️ Installation

Clone the repository

```bash
git clone https://github.com/Phawfull/document_qa_poc.git
cd document_qa_poc
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```
GEMINI_API_KEY=YOUR_API_KEY
```

---

## ▶️ Running

```bash
python main.py
```

---

## 📦 Dependencies

- google-generativeai
- chromadb
- pymupdf
- python-dotenv

Install everything using

```bash
pip install -r requirements.txt
```

---

## 📖 Learning Objectives

This project was built to understand the fundamentals of Retrieval-Augmented Generation without relying on high-level frameworks such as LangChain.

Key learning outcomes include:

- PDF Processing
- Text Chunking
- Embedding Generation
- Vector Databases
- Semantic Retrieval
- LLM Integration
- RAG Pipeline Design

---

## 🔮 Future Improvements

- Streamlit Web Interface
- Multiple PDF Support
- Source Citation
- Chat History
- Batch Embedding
- Hybrid Search
- Metadata Filtering
- OCR Support for Scanned PDFs

---

## 👨‍💻 Author

**Arush Gupta**

B.Tech CSE (AI & ML)

SRM Institute of Science and Technology

GitHub: https://github.com/Phawfull
