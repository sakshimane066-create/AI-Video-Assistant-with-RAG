# AI Video Assistant with RAG

An AI-powered assistant that transcribes videos, summarizes their content, and answers questions using Retrieval-Augmented Generation (RAG) over the transcript.

## Features

- 🎙️ Audio extraction and transcription from video files
- 📝 Automatic content summarization
- 🔍 RAG-based Q&A grounded in the video transcript
- 🗂️ Vector store for semantic search over embeddings

## Tech Stack

- Python
- Vector database for embeddings/similarity search
- LLM-based summarization and retrieval pipeline

## Project Structure

```
AI-Video-Assistant-with-RAG/
├── core/
│   ├── extractor.py       # Extracts audio from video
│   ├── transcriber.py     # Speech-to-text
│   ├── summarizer.py      # Content summarization
│   ├── rag_engine.py      # RAG-based Q&A pipeline
│   └── vector_store.py    # Embeddings & vector search
├── utils/
│   └── audio_processor.py
├── app.py                 # Application entry point
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/sakshimane066-create/AI-Video-Assistant-with-RAG.git
cd AI-Video-Assistant-with-RAG
pip install -r requirements.txt
python app.py
```

## License

MIT License
