import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="You are a helpful assistant. Answer only using the provided context. If the answer is not present, say so."
)
def generate_answer(question, context):

    context_text = "\n\n".join(context)

    prompt = f"""
You are answering questions based ONLY on the provided context.

If the answer is not present in the context, reply:
"I could not find the answer in the provided document."

Context:
{context_text}

Question:
{question}
"""

    response = model.generate_content(prompt)

    return response.text
