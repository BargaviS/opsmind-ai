from groq import Groq
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger("opsmind.llm")

SYSTEM_PROMPT = """You are HR Policy Bot, an intelligent assistant that answers questions
based strictly on the provided HR policy document context.

Rules:
- Answer only from the context provided. Do not use outside knowledge.
- If the context does not contain enough information, say: I don't have enough information in the uploaded HR documents to answer this.
- Be concise, accurate, and professional.
- Cite source names when referencing specific information.
"""


class LLMService:
    def __init__(self):
        settings = get_settings()
        if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    def answer(self, query: str, context: str) -> str:
        user_message = f"Context from HR documents:\n\n{context}\n\nQuestion: {query}"
        logger.info(f"Sending query to Groq: '{query}'")
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            max_tokens=1024,
        )
        logger.info("Groq response received")
        return response.choices[0].message.content
