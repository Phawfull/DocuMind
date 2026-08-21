import logging
import os
import tempfile

import streamlit as st

from generation import generate_answer
from hybrid_retrieval import hybrid_search
from ingestion import collection, process_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

PASSAGE_PREVIEW_LENGTH = 280


def get_indexed_documents() -> list[str]:
    data = collection.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []
    names = {
        metadata["document_name"]
        for metadata in metadatas
        if metadata and metadata.get("document_name")
    }
    return sorted(names)


def unique_sources(chunks: list) -> list[dict]:
    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk.get("document_name"), chunk.get("page_number"))
        if key in seen:
            continue
        seen.add(key)
        sources.append(chunk)
    return sources


def process_uploaded_file(uploaded_file) -> str:
    """Save an upload with its original filename, then index it with process_pdf()."""
    original_name = os.path.basename(uploaded_file.name)
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, original_name)
    try:
        with open(temp_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        process_pdf(temp_path)
        return original_name
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.isdir(temp_dir):
            os.rmdir(temp_dir)


def render_sidebar() -> list[str]:
    if "processed_uploads" not in st.session_state:
        st.session_state.processed_uploads = set()

    with st.sidebar:
        st.title("📚 DocuMind")
        st.divider()
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Drag and drop PDFs here",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            indexed_names = set(get_indexed_documents())
            for uploaded_file in uploaded_files:
                upload_key = f"{uploaded_file.name}_{uploaded_file.size}"
                if upload_key in st.session_state.processed_uploads:
                    continue

                document_name = os.path.basename(uploaded_file.name)
                if document_name in indexed_names:
                    st.info(f"{document_name} is already indexed.")
                    st.session_state.processed_uploads.add(upload_key)
                    continue

                try:
                    with st.status(
                        f"Processing {document_name}...",
                        expanded=True,
                    ) as status:
                        process_uploaded_file(uploaded_file)
                        st.write("✓ Extracting document")
                        st.write("✓ Creating embeddings")
                        st.write("✓ Indexing document")
                        status.update(
                            label=f"{document_name} indexed successfully.",
                            state="complete",
                        )
                    indexed_names.add(document_name)
                    st.session_state.processed_uploads.add(upload_key)
                except Exception:
                    logging.exception(
                        "Failed to process uploaded PDF: %s", document_name
                    )
                    st.error(
                        "Something went wrong while processing this document."
                    )

        documents = get_indexed_documents()
        st.subheader("Indexed Documents")
        if documents:
            st.caption(f"{len(documents)} document{'s' if len(documents) != 1 else ''}")
            for name in documents:
                st.markdown(f"**📄 {name}**")
                st.caption("✓ Indexed")
        else:
            st.caption("No documents indexed yet.")

        st.divider()
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return documents


def render_landing() -> None:
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.write("")
        st.write("")
        st.markdown("## 📚 DocuMind")
        st.write("Chat with your documents using AI.")
        st.caption("Upload one or more PDF documents from the sidebar to get started.")


def render_passage(text: str) -> None:
    st.caption("Relevant passage:")
    if not text:
        return
    if len(text) <= PASSAGE_PREVIEW_LENGTH:
        st.write(text)
        return
    st.write(text[:PASSAGE_PREVIEW_LENGTH].rstrip() + "…")
    with st.expander("Show full passage"):
        st.write(text)


def render_sources(chunks: list) -> None:
    sources = unique_sources(chunks)
    if not sources:
        return

    with st.expander("Sources"):
        for chunk in sources:
            st.markdown(
                f"📄 **{chunk['document_name']}** · Page {chunk['page_number']}"
            )
            render_passage(chunk.get("text", ""))
            st.markdown("")


def render_chat(documents: list[str]) -> None:
    st.title("Chat with your documents")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.caption("Ask a question about your documents.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    question = st.chat_input("Ask a question about your documents")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not documents:
        warning = "Please upload at least one PDF before asking a question."
        st.session_state.messages.append({
            "role": "assistant",
            "content": warning,
            "sources": [],
        })
        with st.chat_message("assistant"):
            st.markdown(warning)
        return

    with st.chat_message("assistant"):
        context = []
        try:
            with st.spinner("Searching your documents..."):
                context = hybrid_search(question)
            with st.spinner("Generating answer..."):
                answer = generate_answer(question, context)
        except Exception:
            logging.exception("Failed to answer question")
            answer = (
                "Something went wrong while answering your question. "
                "Please try again."
            )
            st.error(answer)

        st.markdown(answer)
        render_sources(context)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": context,
        })


def main() -> None:
    st.set_page_config(page_title="DocuMind", page_icon="📚", layout="wide")
    documents = render_sidebar()
    if not documents:
        render_landing()
        return
    render_chat(documents)


if __name__ == "__main__":
    main()
