# rag_engine.py

import os
import requests
import json

# Local .env parser
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ[k] = v

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, PyPDFDirectoryLoader
from langchain.embeddings.base import Embeddings
from sentence_transformers.cross_encoder import CrossEncoder
import os
import torch
import numpy as np
from rank_bm25 import BM25Okapi
# 1) LOAD QWEN 0.5B MODEL
# ------------------------------------------------

print("⚡ Loading Qwen2.5-0.5B-Instruct model...")

MODEL_NAME = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    trust_remote_code=True
)

# Prevent model from echoing system/user tags
model.generation_config.stop_strings = ["<|im_end|>", "<|im_start|>"]

print("✅ Model loaded.")

# ------------------------------------------------
# 2) LOAD EMBEDDINGS & RERANKER
# ------------------------------------------------

print("🔍 Loading MiniLM embeddings...")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ------------------------------------------------
# 3) LOAD DOCUMENTS
# ------------------------------------------------

def load_documents():
    docs = []

    if not os.path.exists("./docs"):
        raise Exception("❗ Create a folder named 'docs' and put PDFs/TXTs inside it.")

    for f in os.listdir("./docs"):
        path = os.path.join("./docs", f)

        if f.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())

        elif f.endswith(".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())

    print(f"📄 Loaded {len(docs)} documents.")
    return docs


# ------------------------------------------------
# 4) BUILD FAISS INDEX
# ------------------------------------------------

from rank_bm25 import BM25Okapi
import numpy as np

bm25_index = None
docs_global = []

def build_faiss():
    global bm25_index, docs_global
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    docs_global = chunks

    texts = [c.page_content for c in chunks]

    class MiniEmb(Embeddings):
        def embed_documents(self, docs):
            return embed_model.encode(docs, convert_to_tensor=False)

        def embed_query(self, text):
            return embed_model.encode([text], convert_to_tensor=False)[0]

    db = FAISS.from_texts(texts, MiniEmb())
    
    tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
    bm25_index = BM25Okapi(tokenized_corpus)
    print("✅ FAISS & BM25 indexes built.")
    return db


db = build_faiss()


# ------------------------------------------------
# 5) CLEAN OUTPUT — ***THIS FIXES YOUR ISSUE***
# ------------------------------------------------

def clean_output(text, question):
    # Extract only the assistant part
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]

    remove_words = ["<|im_end|>", "<|im_start|>", "system",
                    "user", "assistant"]

    for w in remove_words:
        text = text.replace(w, "")

    # Remove repeated question
    text = text.replace(f"Question: {question}", "")

    return text.strip()


# ------------------------------------------------
# 6) RAG ANSWERING FUNCTION
# ------------------------------------------------



LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "hf")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

def rag_answer(question, k=5, max_new_tokens=200, temperature=0.0, do_sample=False):

    # Dense Search
    query_emb = embed_model.encode([question], convert_to_tensor=False)[0]
    dense_docs = db.similarity_search_by_vector(query_emb, k=20)
    
    # Sparse Search (BM25)
    tokenized_query = question.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    top_n = np.argsort(bm25_scores)[::-1][:20]
    sparse_docs = [docs_global[i] for i in top_n]
    
    # Deduplicate candidate pool
    unique_docs = {}
    for d in dense_docs + sparse_docs:
        unique_docs[d.page_content] = d
        
    candidates = list(unique_docs.values())
    
    # Cross-Encoder Reranking
    cross_inp = [[question, doc.page_content] for doc in candidates]
    cross_scores = reranker_model.predict(cross_inp)
    
    for idx in range(len(cross_scores)):
        candidates[idx].metadata['cross_score'] = cross_scores[idx]
        
    # Sort by cross-encoder score descending
    candidates = sorted(candidates, key=lambda x: x.metadata['cross_score'], reverse=True)
    best_docs = candidates[:k]
    
    # Lost in the middle ordering
    reordered_docs = []
    for i, doc in enumerate(best_docs):
        if i % 2 == 0:
            reordered_docs.insert(0, doc)
        else:
            reordered_docs.append(doc)
            
    global last_retrieved_docs
    last_retrieved_docs = reordered_docs

    context = "\n\n".join([d.page_content for d in reordered_docs])

    # Deterministic Refusal Check: If question asks about a specific proper noun or number not in the context, refuse.
    words = question.split()
    if len(words) > 1:
        # Check capitalized words or words with numbers (ignoring the first word which is usually capitalized)
        salient_terms = [w.strip("?.,\"'") for w in words[1:] if (not w.islower() or any(c.isdigit() for c in w)) and len(w.strip("?.,\"'")) > 1]
        context_lower = context.lower()
        for term in salient_terms:
            if term.lower() not in context_lower:
                return "I cannot find enough information in the uploaded document to answer this question."

    system_prompt = (
        "You are an expert technical AI assistant.\n"
        "Your task is to answer the user's question accurately and comprehensively using ONLY the provided context.\n"
        "Instructions:\n"
        "1. Write your answer in full, detailed sentences.\n"
        "2. Include all relevant concepts, numbers, and mathematical definitions from the context to fully satisfy the question.\n"
        "3. When asked about metrics (like BLEU score or layers), identify the exact number for the specific model or language requested.\n"
        "4. If the context does not contain enough information to answer, reply exactly: 'I cannot find enough information in the uploaded document to answer this question.'\n"
        "Do NOT invent facts or use outside knowledge.\n"
    )

    if LLM_PROVIDER == "ollama":
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            "options": {
                "temperature": temperature,
                "num_predict": max_new_tokens
            },
            "stream": False
        }
        try:
            resp = requests.post("http://127.0.0.1:11434/api/chat", json=payload)
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]
        except Exception as e:
            raw = f"Error calling Ollama: {str(e)}"
    else:
        prompt = (
            "<|im_start|>system\n"
            f"{system_prompt}"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            tokenizer=tokenizer,
        )
        raw = tokenizer.decode(output[0], skip_special_tokens=False)

    cleaned = clean_output(raw, question)
    return cleaned


# ------------------------------------------------
# 7) EXPOSE FOR UI
# ------------------------------------------------
def answer(query):
    return rag_answer(query)
