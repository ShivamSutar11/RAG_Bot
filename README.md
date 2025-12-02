📘 Local RAG Chatbot (Qwen 0.5B + MiniLM + FAISS)

A lightweight, document-restricted RAG chatbot that runs fully locally.
Built using Qwen 0.5B, MiniLM embeddings, FAISS vector search, and LangChain utilities.

Fast, clean, and surprisingly smooth — even on CPU/MPS machines.

🚀 Features
🔒 Document-Restricted RAG

The bot answers only from your uploaded documents (PDFs/TXTs).
If the answer isn’t found, it simply says:

“I cannot find that information in the context.”

🤖 Strict Extractive Mode

No hallucinations.
No unwanted explanations.
No extra paragraphs.

⚡ Lightweight Local Pipeline

Qwen 0.5B — small, fast, responsive

MiniLM-L6-v2 — compact but accurate embeddings

FAISS — high-speed vector search

LangChain loaders + splitter for PDF/TXT ingestion

🖥️ Runs Fully Offline

No APIs. No cloud. Everything happens on your machine.

🛠️ Architecture Overview

Load documents (PDF/Text) using LangChain loaders

Split content using RecursiveCharacterTextSplitter

Embed chunks with MiniLM

Store + index them in FAISS

Embed user query

Retrieve top-k relevant chunks

Feed strict context + system prompt into Qwen

Generate extractive answer

📁 Project Structure
RAG_Bot/
│── docs/                 # Your PDFs or text files
│── rag_engine.py         # RAG pipeline logic
│── ui_app.py             # Gradio UI
│── rag_env/              # Virtual environment (optional)
│── README.md

▶️ Getting Started
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Add documents

Place PDFs or text files inside:

/docs

3️⃣ Run the app
python ui_app.py


Then open the provided local URL in your browser.

🧩 Technologies Used
Component	Choice	Why
LLM	Qwen2.5-0.5B-Instruct	Fast, lightweight, great quality for small size
Embeddings	MiniLM-L6-v2	Accurate, tiny, fast for local RAG
Vector Store	FAISS	Super fast similarity search
Framework	LangChain	Easy document loading + vector DB integration
UI	Gradio	Quick, smooth local interface
😅 Challenges I Faced

System prompt chaos:
The model kept echoing long chunks and explaining concepts I never asked for.

LangChain retrieval quirks:
Even with FAISS and splitting, retrieval sometimes returned odd chunks until fine-tuned.

Needed strict RAG enforcement:
Without rules, the model behaved like it was writing a thesis.

Choosing the right model:
Qwen 0.5B ended up much more stable and responsive than other small LLMs (Flan-T5, Gemma-2B, etc.).

But now the bot listens.
Until it decides not to. 😌🤖

📌 Notes

This chatbot is strictly document-bound — no outside knowledge.

Works fully offline once the models are downloaded.

Perfect for study notes, research papers, or private knowledge bases.

⭐ Future Improvements

Add multi-document RAG ranking

Add citation highlights

Add chat history memory

Add GPU acceleration for embeddings

TO Run app : http://127.0.0.1:7860
