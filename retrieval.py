import os
import chromadb
import googlel.generativeai as genai
from dotenv import load_dotenv

from ingestion import EMBEDDING_MODEL

load_dotenv()

API_KEY = os.getenv("API_KEY")
genai = genai.GenerativeAI(API_KEY)


EMBEDDING_MODEL = "models/text-embedding-004"
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}

def embed_query(question):
    embedding_response = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=question,
        task_type="retrieval_query"
        )
    return embedding_response["embedding"]   #rather than returning the whole dict, it returns value of embedding key.

def search_document(query_embedding, top_k=3)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results
