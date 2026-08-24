import re
from rank_bm25 import BM25Okapi

from retrieval import embed_query, search_document, collection

RRF_K = 60

_cached_bm25_indexes = {}

def invalidate_bm25_cache(session_id=None):
    """Clear the cached BM25 index for a session."""
    cache_key = session_id or "__all__"
    _cached_bm25_indexes.pop(cache_key, None)


def _chunk_key(chunk: dict) -> tuple:
    """Uniquely identify a chunk across vector and BM25 results."""
    return (
        chunk["document_name"],
        chunk["page_number"],
        chunk["chunk_index"],
    )

def tokenize(text: str) -> list[str]:
    """
    Normalize text into lowercase word/number tokens for BM25.
    """
    return re.findall(r"\b\w+\b", text.lower())
def build_bm25_index(session_id=None):
    """
    Read all chunks from ChromaDB and build a BM25 index in memory.
    Returns the BM25 index and the ordered list of chunk records used to
    map BM25 result positions back to metadata.
    """
    cache_key = session_id or "__all__"

    if cache_key in _cached_bm25_indexes:
        return _cached_bm25_indexes[cache_key]

    query_kwargs = {
        "include": ["documents", "metadatas"]
    }

    if session_id is not None:
        query_kwargs["where"] = {
            "session_id": session_id
        }

    data = collection.get(**query_kwargs)

    chunks = []
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    for text, metadata in zip(documents, metadatas):
        chunks.append({
            "text": text,
            "document_name": metadata["document_name"],
            "page_number": metadata["page_number"],
            "chunk_index": metadata["chunk_index"],
        })

    tokenized_corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]
    if tokenized_corpus:
        bm25_index = BM25Okapi(tokenized_corpus)
    else:
        bm25_index = None

    _cached_bm25_indexes[cache_key] = (
        bm25_index,
        chunks
    )

    return bm25_index, chunks


def _bm25_search(query: str, bm25_index: BM25Okapi, chunks: list, top_k: int) -> list:
    """Run keyword search and return structured chunks ranked by BM25 score."""
    if bm25_index is None or not chunks:
        return []

    tokenized_query = tokenize(query)
    scores = bm25_index.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results = []
    for index in ranked_indices:
        if scores[index] <= 0:
            continue
        chunk = chunks[index]
        results.append({
            "text": chunk["text"],
            "document_name": chunk["document_name"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "distance": None,
        })

    return results


def _reciprocal_rank_fusion(
    vector_results: list,
    bm25_results: list,
    top_k: int,
) -> list:
    """
    Combine ranked lists from vector search and BM25 using Reciprocal Rank Fusion.

    RRF score for a chunk = sum of 1 / (RRF_K + rank) across each list it appears in,
    where rank is 1-based position in that list.
    """
    rrf_scores = {}
    chunk_by_key = {}
    distance_by_key = {}

    for rank, chunk in enumerate(vector_results, start=1):
        key = _chunk_key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (RRF_K + rank)
        chunk_by_key[key] = chunk
        distance_by_key[key] = chunk.get("distance")

    for rank, chunk in enumerate(bm25_results, start=1):
        key = _chunk_key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (RRF_K + rank)
        if key not in chunk_by_key:
            chunk_by_key[key] = chunk

    ranked_keys = sorted(
        rrf_scores.keys(),
        key=lambda key: rrf_scores[key],
        reverse=True,
    )[:top_k]

    fused_results = []
    for key in ranked_keys:
        chunk = chunk_by_key[key]
        fused_results.append({
            "text": chunk["text"],
            "document_name": chunk["document_name"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "distance": distance_by_key.get(key),
        })

    return fused_results


def hybrid_search(
    query: str,
    top_k: int = 3,
    session_id=None
) -> list:
    """
    Hybrid retrieval: vector search + BM25 keyword search, fused with RRF.
    """
    candidate_k = max(top_k * 5, 20)

    query_embedding = embed_query(query)

    vector_results = search_document(
        query_embedding,
        top_k=candidate_k,
        session_id=session_id
    )

    bm25_index, chunks = build_bm25_index(
        session_id=session_id
    )

    bm25_results = _bm25_search(
        query,
        bm25_index,
        chunks,
        top_k=candidate_k
    )

    candidates = _reciprocal_rank_fusion(
        vector_results,
        bm25_results,
        top_k=candidate_k
    )

    return candidates[:top_k]
