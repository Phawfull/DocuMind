from ingestion import process_pdf, collection
from hybrid_retrieval import hybrid_search
from generation import generate_answer
import os
import logging


def document_exists(document_name: str) -> bool:
    results = collection.get(
        where={"document_name": document_name},
        limit=1
    )
    return len(results["ids"]) > 0


def add_document():
    pdf_path = input("Enter PDF path: ")
    if not os.path.exists(pdf_path):
        print("Error: File does not exist.\n")
        return
    if not pdf_path.lower().endswith(".pdf"):
        print("Error: Please provide a PDF file.\n")
        return

    document_name = os.path.basename(pdf_path)
    if document_exists(document_name):
        print("Document already exists in the database.\n")
        return

    try:
        process_pdf(pdf_path)
        print("Document added successfully.\n")
    except Exception as e:
        logging.error(f"Error processing PDF: {e}")
        print("Error processing document. Please try again.\n")


def ask_questions():
    results = collection.get(limit=1)
    if not results["ids"]:
        print("No documents in the database. Add a document first.\n")
        return

    print("\nAsk questions about your documents (type 'back' to return to menu).\n")
    while True:
        question = input("Ask a question: ")
        if question.lower() == "back":
            break
        context = hybrid_search(question)
        answer = generate_answer(question, context)
        print("\nAnswer:")
        print(answer)
        print()


def main():
    print("\nDocument Q&A System\n")
    while True:
        print("1. Add document")
        print("2. Ask question")
        print("3. Exit")
        choice = input("\nSelect an option: ")

        if choice == "1":
            add_document()
        elif choice == "2":
            ask_questions()
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid option. Please try again.\n")


if __name__ == "__main__":
    main()
