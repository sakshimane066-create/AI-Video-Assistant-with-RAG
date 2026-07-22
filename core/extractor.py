# Action items, decisions, questions

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os


def get_llm():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set in environment / .env")
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0.2
    )


def build_chain(system_prompt: str):
    """Build a simple LLM chain with a system prompt."""
    llm = get_llm()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    """Extract action items from transcript."""
    if not transcript or not transcript.strip():
        return "No transcript available."

    try:
        chain = build_chain(
            "You are an expert meeting analyst. From the meeting transcript, "
            "extract all action items. For each provide:\n"
            "- Task description\n"
            "- Owner (who is responsible)\n"
            "- Deadline (if mentioned, else write 'Not specified')\n\n"
            "Format as a numbered list. If none found say 'No action items found.'"
        )
        # ← Only send first 4000 chars to stay within token limits
        return chain.invoke(transcript[:4000])

    except RuntimeError as e:
        return f"Configuration error: {e}"
    except Exception as e:
        print(f"⚠️ Action items extraction failed: {e}")
        return "Could not extract action items."


def extract_key_decisions(transcript: str) -> str:
    """Extract key decisions from transcript."""
    if not transcript or not transcript.strip():
        return "No transcript available."

    try:
        chain = build_chain(
            "You are an expert meeting analyst. From the meeting transcript, "
            "extract all key decisions made during the meeting. For each provide:\n"
            "- Decision made\n"
            "- Context (why the decision was made, if mentioned)\n\n"
            "Format as a numbered list. If none found say 'No key decisions found.'"
        )
        return chain.invoke(transcript[:4000])

    except RuntimeError as e:
        return f"Configuration error: {e}"
    except Exception as e:
        print(f"⚠️ Key decisions extraction failed: {e}")
        return "Could not extract key decisions."


def extract_questions(transcript: str) -> str:
    """Extract unresolved questions from transcript."""
    if not transcript or not transcript.strip():
        return "No transcript available."

    try:
        chain = build_chain(
            "From the meeting transcript, extract all unresolved questions "
            "or topics needing follow-up. For each provide:\n"
            "- The question or topic\n"
            "- Who raised it (if mentioned)\n\n"
            "Format as a numbered list. If none found say 'No open questions found.'"
        )
        return chain.invoke(transcript[:4000])

    except RuntimeError as e:
        return f"Configuration error: {e}"
    except Exception as e:
        print(f"⚠️ Questions extraction failed: {e}")
        return "Could not extract open questions."


def extract_all(transcript: str) -> dict:
    """Run all extractions at once and return as a dict."""
    if not transcript or not transcript.strip():
        return {
            "action_items": "No transcript available.",
            "key_decisions": "No transcript available.",
            "questions": "No transcript available.",
        }

    print("Extracting action items, decisions, and questions...")
    return {
        "action_items": extract_action_items(transcript),
        "key_decisions": extract_key_decisions(transcript),
        "questions": extract_questions(transcript),
    }