# 📚 Document Question Answering System using Retrieval-Augmented Generation (RAG)

A Retrieval-Augmented Generation (RAG) based Document Question Answering system built using **Python**, **Google Gemini**, and **ChromaDB**.

The application enables users to upload PDF documents, extract and process their contents, generate semantic embeddings, store them in a vector database, and ask natural language questions whose answers are generated using Google's Gemini model based on retrieved document context.

---

# Table of Contents

- Project Overview
- Features
- Architecture
- Project Structure
- Technology Stack
- Workflow
- Configuration
- Installation
- Running the Project
- Error Handling
- Performance Considerations
- Limitations
- Future Enhancements
- License

---

# Project Overview

Traditional Large Language Models rely only on their pre-trained knowledge and cannot answer questions about private or domain-specific documents.

This project solves that problem using **Retrieval-Augmented Generation (RAG)**.

Instead of sending an entire document to the LLM, the system:

1. Extracts text from the uploaded PDF.
2. Splits it into meaningful chunks.
3. Converts each chunk into vector embeddings.
4. Stores embeddings inside ChromaDB.
5. Retrieves only the most relevant chunks for a user's query.
6. Uses Gemini to generate an answer grounded in the retrieved context.

This significantly reduces hallucinations while improving accuracy and reducing token usage.

---

# Features

- PDF text extraction using PyMuPDF
- Automatic text cleaning and preprocessing
- Configurable chunking with overlap
- Embedding generation using Gemini Embedding API
- Persistent vector storage using ChromaDB
- Semantic similarity search using cosine similarity
- Context-aware answer generation using Gemini
- Modular project architecture
- Error handling for invalid inputs
- Environment variable based API key management

---

# Architecture

```
                    User
                      │
                      ▼
                  main.py
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   ingestion.py             retrieval.py
          │                       │
          ▼                       ▼
    ChromaDB              Query Embedding
          │                       │
          └───────────┬───────────┘
                      ▼
               generation.py
                      │
                      ▼
               Gemini 2.5 Flash
                      │
                      ▼
                 Final Answer
```

---

# Project Structure

```
document_qa_poc/
│
├── chroma_db/              # Persistent vector database
│
├── ingestion.py            # PDF extraction, chunking, embeddings
├── retrieval.py            # Semantic retrieval
├── generation.py           # Answer generation
├── main.py                 # Application entry point
│
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

---

# Technology Stack

| Technology | Purpose |
|------------|---------|
| Python | Core Programming Language |
| Google Gemini | Embeddings & Answer Generation |
| ChromaDB | Vector Database |
| PyMuPDF | PDF Text Extraction |
| python-dotenv | Environment Variable Management |

---

# Workflow

## 1. Document Ingestion

The uploaded PDF is opened using **PyMuPDF**, and text is extracted page by page.

```
PDF
    │
    ▼
Extract Text
```

---

## 2. Text Chunking

Large documents are split into smaller overlapping chunks.

```
Text
    │
    ▼
Chunk 1
Chunk 2
Chunk 3
```

Chunk overlap ensures contextual continuity between adjacent chunks.

---

## 3. Embedding Generation

Each chunk is converted into a dense semantic vector.

```
Chunk
    │
    ▼
Gemini Embedding Model
    │
    ▼
Vector Embedding
```

Embedding Model:

```
models/text-embedding-004
```

---

## 4. Vector Storage

Each chunk is stored inside ChromaDB along with:

- Text
- Embedding
- Page Number
- Chunk Index

```
Chunk
    │
    ▼
Embedding
    │
    ▼
ChromaDB
```

---

## 5. Retrieval

When a user submits a question:

```
Question
      │
      ▼
Embedding
      │
      ▼
Cosine Similarity Search
      │
      ▼
Top-K Relevant Chunks
```

---

## 6. Answer Generation

Retrieved chunks are injected into the prompt before sending the request to Gemini.

```
Retrieved Context
          +
User Question
          │
          ▼
Gemini
          │
          ▼
Final Answer
```

---

# Complete RAG Pipeline

```
                PDF
                 │
                 ▼
          Extract Text
                 │
                 ▼
          Clean Text
                 │
                 ▼
         Chunk Document
                 │
                 ▼
     Generate Embeddings
                 │
                 ▼
      Store in ChromaDB
────────────────────────────────────
                 │
          User Question
                 │
                 ▼
     Generate Query Embedding
                 │
                 ▼
     Cosine Similarity Search
                 │
                 ▼
      Retrieve Top-K Chunks
                 │
                 ▼
      Prompt Construction
                 │
                 ▼
      Gemini Answer Generation
                 │
                 ▼
            Final Response
```

---

# Configuration

The retrieval behaviour can be customized by modifying the following parameters.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `CHUNK_SIZE` | Maximum size of each text chunk | `500` |
| `CHUNK_OVERLAP` | Overlap between adjacent chunks | `50` |
| `TOP_K` | Number of retrieved chunks | `3` |

## Recommended Configuration

| PDF Size | Chunk Size | Overlap | Top-K |
|-----------|-----------:|--------:|------:|
| 1–20 Pages | 300–400 | 30–50 | 2 |
| 20–100 Pages | 500 | 50 | 3 |
| 100–300 Pages | 700 | 70 | 4 |
| 300+ Pages | 800–1000 | 100 | 5 |

Choosing an appropriate chunk size and retrieval count improves retrieval accuracy while minimizing token usage and response latency.

---

# Installation

Clone the repository

```bash
git clone https://github.com/Phawfull/document_qa_poc.git

cd document_qa_poc
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```text
GEMINI_API_KEY=YOUR_API_KEY
```

---

# Running

```bash
python main.py
```

---

# Error Handling

The application validates several failure scenarios before processing.

- Invalid PDF path
- Unsupported file type
- Empty user questions
- Missing API key
- Gemini API errors
- ChromaDB retrieval failures

Meaningful error messages are displayed whenever an operation cannot be completed.

---

# Performance Considerations

- Embeddings are generated only once during ingestion.
- ChromaDB stores embeddings persistently.
- Only the Top-K most relevant chunks are retrieved.
- Cosine similarity enables efficient semantic search.
- Reduced prompt size lowers token consumption and improves response time.

---

# Concepts Used

- Artificial Intelligence
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Embeddings
- Semantic Search
- Vector Databases
- Cosine Similarity
- Prompt Engineering
- Context Injection

---

# Current Limitations

- Supports only text-based PDF documents.
- Images and scanned PDFs are not processed.
- No OCR support.
- Single-document retrieval.
- No conversational memory.

---

# Future Enhancements

- Multi-document retrieval
- OCR support for scanned PDFs
- Gemini Vision integration
- Hybrid Search (Vector + BM25)
- Metadata filtering
- FastAPI REST API
- Docker deployment
- Conversation history
- Source citations with page numbers
- Web interface

---

# Security

- API keys should never be committed to version control.
- Store secrets using a `.env` file.
- Ensure `.env` is included in `.gitignore`.

---

# License

This project was developed as part of an internship learning project.
