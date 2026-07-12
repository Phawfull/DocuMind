from ingestion import ingest_document
from retrieval import embed_query, search_document
from generation import generate_answer


def main():
    pdf_path = input("Enter PDF path: ")
    ingest_document(pdf_path)
    print("Document ingested successfully!\n")
    while True:
        question = input("Ask a question (or type 'exit'): ")
        if question.lower() == "exit":
            break

        else:

            query_embedding = embed_query(question)
            context = search_document(query_embedding)
            answer = generate_answer(question, context)

        print("\nAnswer:")
        print(answer)
        print()


if __name__ == "__main__":
    main()