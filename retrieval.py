import os
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
def embed_query(question: str) -> list[float]:
    """
       Converts a user question into a vector embedding.
       Args:
           question (str): User's input question.
       Returns:
           list: Embedding vector representing the query.
       """
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question
    )
    return response.embeddings[0].values

def search_document(query_embedding, top_k=3):
    """
        Retrieves the most relevant document chunks from ChromaDB.
        Args:
            query_embedding (list): Embedding vector of the user's query.
            top_k (int): Number of relevant chunks to retrieve.
        Returns:
            dict: Search results returned by ChromaDB.
        """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0]