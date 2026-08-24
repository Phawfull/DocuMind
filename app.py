import hashlib
import logging
import os
import tempfile
import uuid

import streamlit as st

from generation import generate_answer
from hybrid_retrieval import (
    hybrid_search,
    invalidate_bm25_cache,
)
from ingestion import collection, process_pdf


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


APP_NAME = "DocuMind"
APP_TAGLINE = "Ask questions across your PDFs using hybrid RAG."

MAX_QUESTIONS_PER_SESSION = 10
MAX_UPLOADS_PER_SESSION = 5
MAX_FILE_SIZE_MB = 20
PASSAGE_PREVIEW_LENGTH = 280

GITHUB_URL = "https://github.com/Phawfull/DocuMind"


def initialize_session() -> None:
    """Initialize Streamlit session state used by the application."""

    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "processed_uploads" not in st.session_state:
        st.session_state.processed_uploads = set()

    if "questions_used" not in st.session_state:
        st.session_state.questions_used = 0

    if "uploads_used" not in st.session_state:
        st.session_state.uploads_used = 0


def get_session_id() -> str:
    """Return the current browser session identifier."""
    return st.session_state.session_id


def get_indexed_documents() -> list[str]:
    """Return documents belonging only to the current session."""

    data = collection.get(
        where={"session_id": get_session_id()},
        include=["metadatas"],
    )

    metadatas = data.get("metadatas") or []

    names = {
        metadata["document_name"]
        for metadata in metadatas
        if metadata and metadata.get("document_name")
    }

    return sorted(names)


def document_exists(document_name: str) -> bool:
    """Check whether a document already exists in the current session."""

    results = collection.get(
        where={
            "$and": [
                {"session_id": get_session_id()},
                {"document_name": document_name},
            ]
        },
        limit=1,
    )

    return bool(results.get("ids"))


def unique_sources(chunks: list[dict]) -> list[dict]:
    """Return unique document/page source entries."""

    seen = set()
    sources = []

    for chunk in chunks:
        key = (
            chunk.get("document_name"),
            chunk.get("page_number"),
        )

        if key in seen:
            continue

        seen.add(key)
        sources.append(chunk)

    return sources


def upload_fingerprint(uploaded_file) -> str:
    """Create a stable fingerprint for the uploaded file."""

    content = uploaded_file.getvalue()

    return hashlib.sha256(content).hexdigest()


def process_uploaded_file(uploaded_file) -> str:
    """Save an uploaded PDF temporarily and send it to the existing ingestion pipeline."""

    original_name = os.path.basename(uploaded_file.name)

    file_size_mb = uploaded_file.size / (1024 * 1024)

    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"{original_name} is larger than the "
            f"{MAX_FILE_SIZE_MB} MB demo limit."
        )

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, original_name)

    try:
        with open(temp_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        process_pdf(
            temp_path,
            session_id=get_session_id(),
        )

        invalidate_bm25_cache(
            get_session_id()
        )

        return original_name

        return original_name

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if os.path.isdir(temp_dir):
            os.rmdir(temp_dir)


def delete_session_documents() -> None:
    """Delete all documents belonging to the current session."""

    collection.delete(
        where={"session_id": get_session_id()}
    )

    st.session_state.processed_uploads.clear()
    st.session_state.messages = []


def render_usage() -> None:
    """Display current demo usage limits."""

    questions_remaining = max(
        MAX_QUESTIONS_PER_SESSION - st.session_state.questions_used,
        0,
    )

    uploads_remaining = max(
        MAX_UPLOADS_PER_SESSION - st.session_state.uploads_used,
        0,
    )

    st.markdown("### Demo Usage")

    st.caption(
        f"Questions: "
        f"{st.session_state.questions_used}/"
        f"{MAX_QUESTIONS_PER_SESSION}"
    )

    st.progress(
        min(
            st.session_state.questions_used
            / MAX_QUESTIONS_PER_SESSION,
            1.0,
        )
    )

    st.caption(
        f"PDF uploads: "
        f"{st.session_state.uploads_used}/"
        f"{MAX_UPLOADS_PER_SESSION}"
    )

    st.caption(
        f"{questions_remaining} question"
        f"{'' if questions_remaining == 1 else 's'} remaining."
    )

    st.caption(
        f"{uploads_remaining} upload"
        f"{'' if uploads_remaining == 1 else 's'} remaining."
    )


def render_sidebar() -> list[str]:
    """Render the application sidebar."""

    with st.sidebar:
        st.title("📚 DocuMind")

        st.caption(
            "A multi-document RAG demo using "
            "semantic search, BM25, and Gemini."
        )

        st.link_button(
            "View on GitHub",
            GITHUB_URL,
            use_container_width=True,
        )

        st.divider()

        st.subheader("Upload Documents")

        uploaded_files = st.file_uploader(
            "Drag and drop PDFs here",
            type=["pdf"],
            accept_multiple_files=True,
            help=(
                f"Maximum {MAX_FILE_SIZE_MB} MB per PDF. "
                "Documents are scoped to your current session."
            ),
        )

        if uploaded_files:
            indexed_names = set(get_indexed_documents())

            for uploaded_file in uploaded_files:

                upload_key = upload_fingerprint(uploaded_file)

                if upload_key in st.session_state.processed_uploads:
                    continue

                if st.session_state.uploads_used >= MAX_UPLOADS_PER_SESSION:
                    st.warning(
                        "Demo upload limit reached for this session."
                    )
                    break

                document_name = os.path.basename(uploaded_file.name)

                if document_name in indexed_names:
                    st.info(
                        f"{document_name} is already indexed "
                        "in this session."
                    )

                    st.session_state.processed_uploads.add(
                        upload_key
                    )

                    continue

                try:
                    with st.status(
                        f"Processing {document_name}...",
                        expanded=True,
                    ) as status:

                        st.write("Extracting document text...")
                        st.write("Creating semantic embeddings...")
                        st.write("Indexing document...")

                        process_uploaded_file(
                            uploaded_file
                        )

                        status.update(
                            label=(
                                f"{document_name} "
                                "indexed successfully."
                            ),
                            state="complete",
                        )

                    indexed_names.add(document_name)

                    st.session_state.processed_uploads.add(
                        upload_key
                    )

                    st.session_state.uploads_used += 1

                    st.toast(
                        f"{document_name} is ready.",
                        icon="✅",
                    )

                except Exception:
                    logging.exception(
                        "Failed to process uploaded PDF: %s",
                        document_name,
                    )

                    st.error(
                        f"Could not process "
                        f"{document_name}. "
                        "Please try another PDF."
                    )

        st.divider()

        documents = get_indexed_documents()

        st.subheader("Indexed Documents")

        if documents:
            st.caption(
                f"{len(documents)} "
                f"document{'s' if len(documents) != 1 else ''} "
                "indexed"
            )

            for name in documents:
                with st.container(border=True):
                    st.markdown(f"📄 **{name}**")
                    st.caption("✓ Ready to query")

        else:
            st.caption(
                "No documents indexed yet."
            )

        st.divider()

        render_usage()

        st.divider()

        if st.button(
            "Clear Chat",
            use_container_width=True,
        ):
            st.session_state.messages = []
            st.rerun()

        if st.button(
            "Clear My Documents",
            use_container_width=True,
        ):
            if documents:
                delete_session_documents()

                st.success(
                    "Your documents were removed "
                    "from this session."
                )

                st.rerun()

            else:
                st.info(
                    "There are no documents to remove."
                )

        st.divider()

        st.caption(
            "🔒 Documents are isolated to "
            "your current app session."
        )

    return documents


def render_landing() -> None:
    st.title("📚 DocuMind")

    st.subheader(
        "Ask questions across your PDFs using hybrid RAG."
    )

    st.write(
        "Upload one or more PDF documents and ask questions across them. "
        "DocuMind combines semantic vector search with BM25 keyword retrieval "
        "and Reciprocal Rank Fusion before generating a grounded answer with Gemini."
    )

    st.markdown("### How it works")

    st.info(
        "📄 PDFs  →  🧩 Chunk + Embed  →  🔎 Vector + BM25  →  🔗 RRF  →  🤖 Gemini"
    )

    st.markdown("### Built with")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("LLM", "Gemini")

    with col2:
        st.metric("Vector DB", "ChromaDB")

    with col3:
        st.metric("Retrieval", "BM25 + RRF")

    with col4:
        st.metric("UI", "Streamlit")

    st.caption(
        "Upload a PDF from the sidebar to get started."
    )

def render_passage(text: str) -> None:
    """Render a source passage with a compact preview."""

    st.caption("Relevant passage:")

    if not text:
        return

    if len(text) <= PASSAGE_PREVIEW_LENGTH:
        st.write(text)
        return

    preview = text[:PASSAGE_PREVIEW_LENGTH].rstrip() + "…"

    st.write(preview)

    with st.expander("Show full passage"):
        st.write(text)


def render_sources(chunks: list[dict]) -> None:
    """Render source document/page information."""

    sources = unique_sources(chunks)

    if not sources:
        return

    with st.expander(
        f"Sources ({len(sources)})"
    ):

        for chunk in sources:
            st.markdown(
                f"📄 **{chunk['document_name']}**  "
                f"· Page {chunk['page_number']}"
            )

            render_passage(
                chunk.get("text", "")
            )

            st.markdown("")


def render_chat(documents: list[str]) -> None:
    """Render the chat interface."""

    if not documents and not st.session_state.messages:
        render_landing()
        return

    st.title("Chat with your documents")

    st.caption(
        "Ask questions across all documents "
        "indexed in this session."
    )

    st.divider()

    if not st.session_state.messages:
        st.caption(
            "Try asking: "
            "\"What are the main findings in these documents?\""
        )

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if message["role"] == "assistant":
                render_sources(
                    message.get("sources", [])
                )

    questions_remaining = (
        MAX_QUESTIONS_PER_SESSION
        - st.session_state.questions_used
    )

    if questions_remaining <= 0:

        st.warning(
            "You've reached the demo question limit "
            "for this session."
        )

        st.caption(
            "Thanks for trying DocuMind."
        )

        return

    question = st.chat_input(
        "Ask a question about your documents"
    )

    if not question:
        return

    question = question.strip()

    if not question:
        return

    st.session_state.questions_used += 1

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        context = []

        try:

            with st.spinner(
                "Searching your documents..."
            ):
                context = hybrid_search(
                    question,
                    session_id=get_session_id(),
                )

            with st.spinner(
                "Generating answer..."
            ):
                answer = generate_answer(
                    question,
                    context,
                )

        except Exception:

            logging.exception(
                "Failed to answer question"
            )

            answer = (
                "Something went wrong while answering "
                "your question. Please try again."
            )

            st.error(answer)

        st.markdown(answer)

        render_sources(context)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": context,
            }
        )


def main() -> None:
    st.set_page_config(
        page_title="DocuMind",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_session()

    documents = render_sidebar()

    render_chat(documents)


if __name__ == "__main__":
    main()