from ingestion import process_pdf
from retrieval import embed_query, search_document
from generation import generate_answer


def main():
    pdf_path = input("Enter PDF path: ").strip().strip('"')
    process_pdf(pdf_path)
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