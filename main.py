from ingestion import process_pdf
from retrieval import embed_query, search_document
from generation import generate_answer
import os

def main():
    """
        Runs the Document Question Answering application.
        Allows the user to:
        1. Upload a PDF document.
        2. Process and store embeddings.
        3. Ask questions about the document.
        4. Receive AI-generated answers based on retrieved context.
        Returns:
            None
        """
    while True:
        pdf_path = input("Enter PDF Path: ")
        if not os.path.exists(pdf_path):
            print("Error: File does not exist. Please try again.\n")
            continue
        if not pdf_path.lower().endswith(".pdf"):
            print("Error: Please provide a PDF file.\n")
            continue
        try:
            process_pdf(pdf_path)
            break
        except Exception as e:
            print(f"Error processing PDF: {e}")
            print("Please try another file.\n")
    print("\nDocument is ready for questions!\n")

    while True:
        question = input("Ask a question (or type 'exit'): ")
        if question.lower() == "exit":
            print("Exiting...")
            break
        query_embedding = embed_query(question)
        context = search_document(query_embedding)
        answer = generate_answer(question, context)
        print("\nAnswer:")
        print(answer)
        print()
if __name__== "__main__":
    main()