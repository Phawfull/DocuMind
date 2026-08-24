import os
from google import genai
from dotenv import load_dotenv
import time
from google.genai import errors

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)


def generate_answer(question, context):
    """
    Generates an answer using Gemini based only on the retrieved context.
    Includes source information for each retrieved chunk.
    """

    context_parts = []

    for chunk in context:
        context_parts.append(
            f"[Source: {chunk['document_name']} — "
            f"Page {chunk['page_number']}, "
            f"Chunk {chunk['chunk_index']}]\n"
            f"{chunk['text']}"
        )

    context_text = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the information provided in the context below.

For every important claim in your answer, mention the page number
where the information was found.

If the answer is not present in the context, reply exactly:
"I could not find the answer in the provided document."

Context:
{context_text}

Question:
{question}
"""

    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            break

        except errors.ServerError as e:
            if attempt == max_attempts - 1:
                raise

            wait_time = 2 ** attempt
            time.sleep(wait_time)

    return response.text