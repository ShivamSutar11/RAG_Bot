# RAG_Bot

A local RAG (Retrieval-Augmented Generation) assistant using Qwen2.5-0.5B, FAISS, BM25, and a Cross-Encoder.

## Architecture
- **Backend:** FastAPI, Python (FAISS, BM25, Langchain, SentenceTransformers, Transformers)
- **Frontend:** React, Vite, Tailwind CSS

## Prerequisites
- Node.js & npm
- Python 3.9+
- Ollama (running locally with `qwen2.5:0.5b-instruct` or equivalent, though this project uses a local transformers model)

## Setup & Startup

### 1. Start the Backend
```bash
# Activate your virtual environment
source venv/bin/activate

# Install requirements (if not already installed)
pip install fastapi uvicorn python-multipart transformers sentence-transformers langchain-community faiss-cpu rank_bm25

# Start the FastAPI server
cd backend
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
uvicorn main:app --host 127.0.0.1 --port 8000
```

### 2. Start the Frontend
```bash
# In a new terminal
cd frontend

# Install dependencies (first time only)
npm install

# Start the Vite dev server
npm run dev
```

### 3. Usage
- Open your browser to the URL provided by Vite (usually `http://localhost:5173`).
- Upload a PDF, TXT, or DOCX document using the sidebar.
- Select the document.
- Start asking questions!
