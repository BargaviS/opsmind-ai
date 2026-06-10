from google import genai
from google.genai import types
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger("opsmind.llm")

SYSTEM_PROMPT = """You are OpsMind AI, an intelligent assistant that answers questions
based strictly on the provided document context.

Rules:
- Answer only from the context provided. Do not use outside knowledge.
- If the context does not contain enough information, say: I do not have enough information in the uploaded documents to answer this.
- Be concise, accurate, and professional.
- Cite source names when referencing specific information.
"""


class LLMService:
    def __init__(self):
        settings = get_settings()
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def answer(self, query: str, context: str) -> str:
        user_message = f"Context from uploaded documents:\n\n{context}\n\nQuestion: {query}"
        logger.info(f"Sending query to Gemini: '{query}'")
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
            ),
        )
        logger.info("Gemini response received")
        return response.text
