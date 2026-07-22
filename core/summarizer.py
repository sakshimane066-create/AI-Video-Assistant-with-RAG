from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os


def get_llm():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set in environment / .env")
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0.3
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    """Summarize full transcript using map-reduce approach."""
    if not transcript or not transcript.strip():
        return "No transcript available to summarize."

    try:
        llm = get_llm()

        # ── Step 1: Map — summarize each chunk individually ──────────
        map_prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize this portion of a meeting transcript concisely."),
            ("human", "{text}"),
        ])
        map_chain = map_prompt | llm | StrOutputParser()

        chunks = split_transcript(transcript)
        if not chunks:
            return "Transcript too short to summarize."

        print(f"Summarizing {len(chunks)} chunk(s)...")

        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            try:
                print(f"  → Summarizing chunk {i + 1}/{len(chunks)}...")
                summary = map_chain.invoke({"text": chunk})
                chunk_summaries.append(summary)
            except Exception as e:
                print(f"⚠️ Skipping chunk {i + 1} due to error: {e}")

        if not chunk_summaries:
            return "Summarization failed for all chunks."

        combined = "\n\n".join(chunk_summaries)

        # ── Step 2: Reduce — combine all summaries into one ──────────
        combined_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert meeting summarizer. Combine these partial summaries "
                "into one final professional meeting summary with clear sections:\n"
                "- Key Discussion Points\n"
                "- Decisions Made\n"
                "- Action Items\n"
                "- Next Steps\n"
                "Use bullet points for clarity.",
            ),
            ("human", "{text}"),
        ])

        combined_chain = (
            RunnablePassthrough()
            | RunnableLambda(lambda x: {"text": x})
            | combined_prompt
            | llm
            | StrOutputParser()
        )

        return combined_chain.invoke(combined)

    except RuntimeError as e:
        return f"Configuration error: {e}"
    except Exception as e:
        return f"Summarization failed: {e}"


def generate_title(transcript: str) -> str:
    """Generate a short professional title from the transcript."""
    if not transcript or not transcript.strip():
        return "Untitled Meeting"

    try:
        llm = get_llm()

        title_chain = (
            RunnablePassthrough()
            | RunnableLambda(lambda x: {"text": x})
            | ChatPromptTemplate.from_messages([
                (
                    "system",
                    "Based on the meeting transcript, generate a short professional meeting title "
                    "(max 8 words). Only return the title, nothing else.",
                ),
                ("human", "{text}"),
            ])
            | llm
            | StrOutputParser()
        )

        # ← Only send first 2000 chars to save tokens
        title = title_chain.invoke(transcript[:2000])
        return title.strip().strip('"').strip("'")  # ← Clean any quotes

    except RuntimeError as e:
        return f"Configuration error: {e}"
    except Exception as e:
        print(f"⚠️ Title generation failed: {e}")
        return "Meeting Summary"