import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)


def generate_answer(question, context):
    """
        Generates an answer using Gemini based only on the retrieved context.
        Args:
            question (str): User's question.
            context (list): List of retrieved document chunks.
        Returns:
            str: Generated answer from the language model.
        """
    context_text = "\n\n".join(context)
    prompt = f"""
You are a helpful AI assistant.
Answer ONLY using the information provided in the context below.
If the answer is not present in the context, reply exactly:
"I could not find the answer in the provided document."
Context:
{context_text}
Question:
{question}
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text