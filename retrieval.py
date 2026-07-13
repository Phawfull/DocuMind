import os
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
EMBEDDING_MODEL = "text-embedding-004"
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)
def embed_query(question):
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question
    )
    return response.embeddings[0].values

def search_document(query_embedding, top_k=3):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0]