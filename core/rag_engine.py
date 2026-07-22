import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever


def get_llm():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set in environment / .env")
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0.3,
    )


def format_docs(docs):
    """Format retrieved docs into a single string."""
    if not docs:
        return "No relevant context found."
    return "\n\n".join([doc.page_content for doc in docs])


RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
    ),
    ("human", "{question}"),
])


def _build_chain(retriever):
    """Build the LCEL RAG chain from a retriever."""
    llm = get_llm()
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain


def build_rag_chain(transcript: str):
    """Build a RAG chain from a fresh transcript."""
    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty — cannot build RAG chain.")

    try:
        print("Building RAG chain from transcript...")
        vector_store = build_vector_store(transcript)
        retriever = get_retriever(vector_store, k=4)
        chain = _build_chain(retriever)
        print("RAG chain ready.")
        return chain

    except Exception as e:
        raise RuntimeError(f"Failed to build RAG chain: {e}")


def load_rag_chain():
    """Load a RAG chain from existing vector store on disk."""
    try:
        print("Loading RAG chain from existing vector store...")
        vector_store = load_vector_store()
        retriever = get_retriever(vector_store, k=4)  # ← Fixed: was missing argument
        chain = _build_chain(retriever)
        print("RAG chain loaded.")
        return chain

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Vector store not found: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load RAG chain: {e}")


def ask_question(rag_chain, question: str) -> str:
    """Ask a question using the RAG chain."""
    if not question or not question.strip():
        return "Please enter a valid question."

    try:
        print(f"Question: {question}")
        answer = rag_chain.invoke(question)
        print(f"Answer: {answer}")
        return answer

    except Exception as e:
        print(f"⚠️ RAG chain error: {e}")
        return f"Sorry, I encountered an error answering your question: {e}"