from typing import List, Dict, Any
import asyncio
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ConversationManager:
    MAX_TOKENS = 4000  # Adjust based on Gemini model limits (e.g., ~4096 tokens)

    @staticmethod
    def _truncate_text(text: str, max_length: int = 1000) -> str:
        """Truncate text to a maximum length, preserving sentence boundaries."""
        if len(text) <= max_length:
            return text
        truncated = text[:max_length].rsplit(".", 1)[0] + "."
        return truncated

    @staticmethod
    def _build_context(retrieved_chunks: List[Dict[str, Any]], max_length: int = 2000) -> str:
        """Build context from chunks, respecting token limits."""
        context = ""
        for chunk in retrieved_chunks:
            chunk_text = chunk["text"]
            if len(context) + len(chunk_text) + 2 > max_length:
                remaining_space = max_length - len(context) - 2
                context += "\n\n" + ConversationManager._truncate_text(chunk_text, remaining_space)
                break
            context += "\n\n" + chunk_text
        logger.debug(f"Built context (length={len(context)}): {context[:200]}...")
        return context.strip()

    @staticmethod
    def _build_history(conversation_history: List[Dict[str, str]], max_length: int = 1000) -> str:
        """Format conversation history, prioritizing the last assistant response and recent messages."""
        if not conversation_history:
            return ""
        
        # Get the last assistant response (if any)
        last_response = next((msg["content"] for msg in reversed(conversation_history) if msg["role"] == "assistant"), "")
        history_lines = []
        
        if last_response:
            history_lines.append(f"Assistant (Previous Response): {last_response}")
        
        # Add last 4 user/assistant pairs (excluding the last assistant response if already added)
        recent_messages = [msg for msg in conversation_history[-5:] if msg["role"] != "assistant" or msg["content"] != last_response]
        history_lines.extend(f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_messages[-4:])
        
        history_text = "\n".join(history_lines)
        if len(history_text) > max_length:
            history_text = ConversationManager._truncate_text(history_text, max_length)
        logger.debug(f"Built history (length={len(history_text)}): {history_text[:200]}...")
        return history_text

    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _generate_response(gemini_model, prompt: str) -> str:
        """Generate response with retry logic for transient failures."""
        try:
            response = await asyncio.to_thread(gemini_model.generate_content, prompt)
            response_text = response.text.strip() if response.text else ""
            logger.info(f"Gemini response (length={len(response_text)}): {response_text[:200]}...")
            if not response_text:
                raise ValueError("Empty response from Gemini API.")
            return response_text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            if "429" in str(e):
                raise Exception("Quota exceeded. Please try again later or upgrade your plan.")
            raise

    @staticmethod
    async def generate_answer(gemini_model, question: str, retrieved_chunks: List[Dict[str, Any]], 
                            conversation_history: List[Dict[str, str]]) -> str:
        """Generate an answer based on context and history."""
        if not retrieved_chunks:
            logger.info("No chunks retrieved for answer generation.")
            return "The answer is not available in the provided books."

        context = ConversationManager._build_context(retrieved_chunks, max_length=2000)
        history_text = ConversationManager._build_history(conversation_history, max_length=1000)

        prompt = f"""
        You are a knowledgeable and concise assistant. Answer the question based on the provided context and conversation history.
        - If the question is about the conversation history or your previous response, answer based on the conversation history.
        - For questions about the books, try to provide a relevant answer based on the context, even if it's not a direct match.
        - Use clear, professional language and format your response in markdown where appropriate.
        - If you cannot find any relevant information in the context or history, respond with: "The answer is not available in the provided books."
        - If unsure or partially confident, indicate your confidence level (e.g., "Based on limited context...").

        **Conversation History**:
        {history_text}

        **Context from Books**:
        {context}

        **Question**:
        {question}

        **Answer**:
        """

        total_length = len(prompt)
        if total_length * 0.25 > ConversationManager.MAX_TOKENS:
            logger.warning("Prompt exceeds token limit, truncating context.")
            context = ConversationManager._truncate_text(context, max_length=1000)
            prompt = f"""
            You are a knowledgeable and concise assistant. Answer the question based on the provided context and conversation history.
            - If the question is about the conversation history or your previous response, answer based on the conversation history.
            - For questions about the books, try to provide a relevant answer based on the context, even if it's not a direct match.
            - Use clear, professional language and format your response in markdown where appropriate.
            - If you cannot find any relevant information in the context or history, respond with: "The answer is not available in the provided books."

            **Conversation History**:
            {history_text}

            **Context from Books**:
            {context}

            **Question**:
            {question}

            **Answer**:
            """

        logger.debug(f"Sending prompt to Gemini API (length={len(prompt)}): {prompt[:500]}...")
        try:
            answer = await ConversationManager._generate_response(gemini_model, prompt)
            return answer
        except ValueError:
            return "No answer generated from the provided context. The API returned an empty response."
        except Exception as e:
            if "quota" in str(e).lower():
                return "API quota exceeded. Please try again later or upgrade your plan."
            logger.error(f"Answer generation failed: {e}")
            raise Exception(f"Failed to generate answer: {str(e)}")