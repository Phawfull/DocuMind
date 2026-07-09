import os
import fitz
import google.generativeai as genai
import chromadb
from dotenv import load_dotenv

load_dotenv()                                    # reads the .env file
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

EMBEDDING_MODEL = "models/text-embedding-004"


chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}     #tells chromadb to use cosine similarity for vector search
)

def extract_pdf(file_path)
    pages=[]
    pdf_document = fitz.open(file_path)
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        text = page.get_text()   #get_text is a PyMuPDF method that extracts text from a page
        text = " ".join(text.split())      #.join removes extra whitespace and newlines from the text
        if text.strip():            #strip removes spaces from start and end.. if page is blank, IF will skip it
            pages.append((text, page_number + 1))

    pdf_document.close()
    return pages


def chunk_text(pages, chunk_size=500, overlap=50):
    chunks = []

    for page_text, page_number in pages: #pages list has something in the format of [(text, page_number), (text, page_number)]
        words = page_text.split()

        start = 0
        while start < len(words):
            end = start + chunk_size

            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)    #joins multiple strings together

            chunks.append({
                "text": chunk_text,
                "page_number": page_number,
                "chunk_index": len(chunks)
            })

            start = end - overlap

    return chunks


def create_embeddings(chunks):
    embedded_chunks = []

    for chunk in chunks:
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=chunk["text"],
            task_type="retrieval_document"
        )

        embedded_chunks.append({
            "text": chunk["text"],
            "embedding": response["embedding"],
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
        ids.append("chunk_" + str(chunk["chunk_index"]))   #gives sm like chunk_0, chunk_1, chunk_2
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])
        metadatas.append(chunk["metadata"])
        collection..add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )


def process_pdf(file_path):
    print("Processing PDF: ")
    pages = extract_pdf(file_path)
    print("Chunking Text... ")
    chunks = chunk_text(pages)
    print("Creating Embeddings... ")
    embedded_chunks = create_embeddings(chunks)
    print("Storing Embeddings in ChromaDB... ")
    store_embeddings(embedded_chunks)
    print("Document Processed and Stored Successfully")

