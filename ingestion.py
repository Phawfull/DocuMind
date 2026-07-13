import os
import fitz
import chromadb
from google import genai
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
EMBEDDING_MODEL = "gemini-embedding-001"
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

def extract_pdf(file_path):
    pages=[]
    pdf_document = fitz.open(file_path)
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        text = page.get_text()
        text = " ".join(text.split())
        if text.strip():
            pages.append((text, page_number + 1))
    pdf_document.close()
    return pages

def chunk_text(pages, chunk_size=500, overlap=50):
    chunks = []
    for page_text, page_number in pages:
        words = page_text.split()
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk = " ".join(chunk_words)
            chunks.append({
                "text": chunk,
                "page_number": page_number,
                "chunk_index": len(chunks)
            })
            start = end - overlap
    return chunks


def create_embeddings(chunks):
    embedded_chunks = []
    for chunk in chunks:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=chunk["text"]
        )
        embedded_chunks.append({
            "text": chunk["text"],
            "embedding": response.embeddings[0].values,
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"]
        })
    return embedded_chunks


def store_embeddings(embedded_chunks):
    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for chunk in embedded_chunks:
        ids.append("chunk_" + str(chunk["chunk_index"]))
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])
        metadatas.append({
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"]
        })
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )


def process_pdf(file_path):
    print("Processing PDF...")
    pages = extract_pdf(file_path)
    print("Chunking text...")
    chunks = chunk_text(pages)
    print("Creating embeddings...")
    embedded_chunks = create_embeddings(chunks)
    print("Storing embeddings in ChromaDB...")
    store_embeddings(embedded_chunks)
    print("Document processed successfully!")