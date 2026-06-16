from groq import Groq
from typing import List, Dict
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger("opsmind.agent")

SYSTEM_PROMPT = """You are HR Policy Bot, a smart assistant that answers questions about HR policies and company documents.
Rules:
- Answer strictly from the provided document context.
- If context is insufficient, say: I don't have enough information in the uploaded HR documents to answer this.
- Be concise, professional, and cite sources.
- If the user greets you, respond warmly and explain what you can help with.
"""

class AgentService:
    def __init__(self):
        settings = get_settings()
        if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set.")
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    def chat(self, query: str, context: str, history: List[Dict[str, str]]) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in history:
            messages.append({
                "role": turn["role"] if turn["role"] == "user" else "assistant",
                "content": turn["content"]
            })
        user_message = (
            f"HR Document context:\n\n{context}\n\nQuestion: {query}"
        ) if context else query
        messages.append({"role": "user", "content": user_message})
        logger.info(f"Agent query: '{query}' | history_turns={len(history)}")
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1024,
        )
        logger.info("Groq agent response received")
        return response.choices[0].message.content
