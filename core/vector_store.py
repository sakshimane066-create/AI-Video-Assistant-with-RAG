import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # ← Fixed import
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}  # ← Better similarity scores
    )


def build_vector_store(transcript: str) -> Chroma:
    """Build a new vector store from transcript text."""
    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty — cannot build vector store.")

    print("Building vector store...")

    # ── Clean up old vector store before building new one ────────────
    if os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print("Cleared old vector store.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(transcript)

    if not chunks:
        raise ValueError("Transcript too short to split into chunks.")

    docs = [
        Document(
            page_content=chunk,
            metadata={"chunk_index": i, "total_chunks": len(chunks)}
        )
        for i, chunk in enumerate(chunks)
    ]

    print(f"Embedding {len(docs)} chunks...")

    try:
        embeddings = get_embeddings()
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR
        )
        print("Vector store built successfully.")
        return vector_store

    except Exception as e:
        raise RuntimeError(f"Failed to build vector store: {e}")


def load_vector_store() -> Chroma:
    """Load existing vector store from disk."""
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            "Vector store not found. Please analyse a video first."
        )

    try:
        embeddings = get_embeddings()
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR
        )
        print("Vector store loaded successfully.")
        return vector_store

    except Exception as e:
        raise RuntimeError(f"Failed to load vector store: {e}")


def vector_store_exists() -> bool:
    """Check if a vector store already exists on disk."""
    return os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0


def get_retriever(vector_store: Chroma, k: int = 4):
    """Get a similarity retriever from the vector store."""
    try:
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create retriever: {e}")