import os
import logging
import fitz
import chromadb
from google import genai
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
client = genai.Client(api_key=API_KEY)
EMBEDDING_MODEL = "gemini-embedding-001"
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

def extract_pdf(file_path: str) -> list[tuple[str, int]]:
    """
       Extracts text from each page of a PDF document.
       Args:
           file_path (str): Path to the PDF file.
       Returns:
           list: A list of tuples containing the extracted page text
           and corresponding page number.
       """
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
    """
        Splits extracted PDF text into overlapping chunks for embedding.
        Args:
            pages (list): List of tuples containing page text and page number.
            chunk_size (int): Maximum number of words per chunk.
            overlap (int): Number of overlapping words between consecutive chunks.
        Returns:
            list: A list of dictionaries containing chunk text and metadata.
        """
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


def create_embeddings(chunks, document_name):
    """
        Splits extracted PDF text into overlapping chunks for embedding.
        Args:
            pages (list): List of tuples containing page text and page number.
            chunk_size (int): Maximum number of words per chunk.
            overlap (int): Number of overlapping words between consecutive chunks.
        Returns:
            list: A list of dictionaries containing chunk text and metadata.
        """
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
            "chunk_index": chunk["chunk_index"],
            "document_name": document_name
        })
    return embedded_chunks


def store_embeddings(embedded_chunks):
    """
        Stores text embeddings and metadata in the ChromaDB vector database.
        Args:
            embedded_chunks (list): List containing embeddings, text,
            and metadata for each chunk.
        Returns:
            None
        """
    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for chunk in embedded_chunks:
        ids.append(
            chunk["document_name"] + "_" +
            str(chunk["chunk_index"])
        )
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])
        metadatas.append({
            "document_name": chunk["document_name"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"]
        })
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )


def process_pdf(file_path: str) -> None:
    """
        Executes the complete document ingestion pipeline.
        The pipeline performs:
        1. PDF text extraction
        2. Text chunking
        3. Embedding generation
        4. Storage in ChromaDB
        Args:
            file_path (str): Path to the PDF document.
        Returns:
            None
        """
    logging.info("Processing PDF...")
    pages = extract_pdf(file_path)
    print("Chunking text...")
    chunks = chunk_text(pages)

    print("Creating embeddings...")

    document_name = os.path.basename(file_path)

    embedded_chunks = create_embeddings(
        chunks,
        document_name
    )
    print("Storing embeddings in ChromaDB...")
    store_embeddings(embedded_chunks)
    logging.info("PDF processed successfully.")