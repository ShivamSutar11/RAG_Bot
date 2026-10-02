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

class MiniEmb(Embeddings):
    def embed_documents(self, docs):
        return embed_model.encode(docs, convert_to_tensor=False)
    def embed_query(self, text):
        return embed_model.encode([text], convert_to_tensor=False)[0]

def build_index_for_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    src_counts = {}
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        if src not in src_counts:
            src_counts[src] = 0
        c.metadata["chunk_idx"] = src_counts[src]
        src_counts[src] += 1
        
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    
    db = FAISS.from_texts(texts, MiniEmb(), metadatas=metadatas)
    tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return db, bm25, chunks

def build_faiss():
    global bm25_index, docs_global
    docs = load_documents()
    if not docs: return None
    db, bm25_index, docs_global = build_index_for_docs(docs)
    print("✅ Global FAISS & BM25 indexes built.")
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

def rag_answer(question, k=5, max_new_tokens=200, temperature=0.0, do_sample=False, custom_db=None, custom_bm25=None, custom_docs=None):

    # Dense Search
    query_emb = embed_model.encode([question], convert_to_tensor=False)[0]
    dense_docs = (custom_db or db).similarity_search_by_vector(query_emb, k=20)
    
    # Sparse Search (BM25)
    tokenized_query = question.lower().split()
    bm25_scores = (custom_bm25 or bm25_index).get_scores(tokenized_query)
    top_n = np.argsort(bm25_scores)[::-1][:20]
    sparse_docs = [(custom_docs or docs_global)[i] for i in top_n]
    
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
        
    # PHASE 2: Score fusion (RRF) + Preserve lexical
    # Compute dense ranks
    dense_ranks = {doc.page_content: i for i, doc in enumerate(dense_docs)}
    # Compute sparse ranks
    sparse_ranks = {doc.page_content: i for i, doc in enumerate(sparse_docs)}
    
    # Sort candidates by cross_score for cross ranks
    candidates_by_cross = sorted(candidates, key=lambda x: x.metadata['cross_score'], reverse=True)
    cross_ranks = {doc.page_content: i for i, doc in enumerate(candidates_by_cross)}
    
    # RRF parameters
    K = 60
    
    for doc in candidates:
        r_dense = dense_ranks.get(doc.page_content, 100)
        r_sparse = sparse_ranks.get(doc.page_content, 100)
        r_cross = cross_ranks.get(doc.page_content, 100)
        
        # Fuse scores
        doc.metadata['fused_score'] = (1.0 / (K + r_dense)) + (1.0 / (K + r_sparse)) + (1.0 / (K + r_cross))
        
    # Sort by fused score
    candidates = sorted(candidates, key=lambda x: x.metadata['fused_score'], reverse=True)
    
    # Preserve Strategy: Ensure the #1 BM25 candidate makes it into the top k
    best_docs = []
    top_bm25 = sparse_docs[0] if sparse_docs else None
    if top_bm25:
        best_docs.append(top_bm25)
        
    for doc in candidates:
        if doc not in best_docs:
            best_docs.append(doc)
        if len(best_docs) == k:
            break
    
    # 1. Context Expansion Strategy
    def looks_incomplete(text):
        t = text.strip()
        if not t: return False
        if t[0].islower(): return True # Starts mid-sentence
        if not t[-1] in ".!?\"'": return True # Ends mid-sentence or heading
        if t.count('|') > 3: return True # Table fragment
        if "=" in t and t.endswith("="): return True # Formula fragment
        return False

    expanded_docs = []
    chunk_map = { (d.metadata.get("source"), d.metadata.get("chunk_idx")): d for d in (custom_docs or docs_global) if "chunk_idx" in d.metadata }

    for i, doc in enumerate(best_docs):
        expanded_docs.append(doc)
        src = doc.metadata.get("source")
        c_idx = doc.metadata.get("chunk_idx")
        if src is None or c_idx is None:
            continue
            
        # Unconditional neighbor expansion for the absolute top chunk
        expand_prev = (i == 0) or looks_incomplete(doc.page_content[:150])
        expand_next = (i == 0) or looks_incomplete(doc.page_content[-150:])
        
        if expand_prev and (src, c_idx - 1) in chunk_map:
            expanded_docs.append(chunk_map[(src, c_idx - 1)])
        if expand_next and (src, c_idx + 1) in chunk_map:
            expanded_docs.append(chunk_map[(src, c_idx + 1)])

    # Deduplicate expanded pool
    unique_expanded = {}
    for d in expanded_docs:
        unique_expanded[d.page_content] = d
    final_docs = list(unique_expanded.values())
    
    # Sort chronologically to restore document reading order
    final_docs = sorted(final_docs, key=lambda x: (x.metadata.get("source", ""), x.metadata.get("chunk_idx", 0)))
    
    # Merge overlapping/redundant text
    merged_blocks = []
    current_block = None
    
    def merge_strings_with_overlap(s1, s2):
        max_overlap = min(len(s1), len(s2), 500)
        for i in range(max_overlap, 0, -1):
            if s1.endswith(s2[:i]):
                return s1 + s2[i:]
        return s1 + "\n\n" + s2

    for d in final_docs:
        if current_block is None:
            current_block = {"source": d.metadata.get("source"), "idxs": [d.metadata.get("chunk_idx")], "text": d.page_content}
        else:
            if d.metadata.get("source") == current_block["source"] and d.metadata.get("chunk_idx") == current_block["idxs"][-1] + 1:
                current_block["text"] = merge_strings_with_overlap(current_block["text"], d.page_content)
                current_block["idxs"].append(d.metadata.get("chunk_idx"))
            else:
                merged_blocks.append(current_block)
                current_block = {"source": d.metadata.get("source"), "idxs": [d.metadata.get("chunk_idx")], "text": d.page_content}
    if current_block:
        merged_blocks.append(current_block)
        
    class DummyDoc:
        def __init__(self, content):
            self.page_content = content
            self.metadata = {}

    # Pack into dummy docs for compatibility
    reordered_docs = [DummyDoc(b["text"]) for b in merged_blocks]

    global last_retrieved_docs
    last_retrieved_docs = reordered_docs

    context = "\n\n".join([d.page_content for d in reordered_docs])

    # Deterministic Refusal Check: Relaxed Match Count to avoid OOD false positives
    words = question.split()
    if len(words) > 1:
        salient_terms = [w for w in words[1:] if (not w.islower() or any(c.isdigit() for c in w))]
        
        if salient_terms:
            context_norm = context.lower().replace("√", "sqrt").replace("\\sqrt", "sqrt")
            context_alpha = "".join(c for c in context_norm if c.isalnum())
            
            match_count = 0
            for term in salient_terms:
                term_norm = term.lower().replace("√", "sqrt").replace("\\sqrt", "sqrt")
                term_alpha = "".join(c for c in term_norm if c.isalnum())
                
                if len(term_alpha) > 1 and (term_alpha in context_alpha or term.lower().strip("?.,\"'") in context.lower()):
                    match_count += 1
            
            if match_count == 0:
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
def answer(query, custom_db=None, custom_bm25=None, custom_docs=None):
    return rag_answer(query, custom_db=custom_db, custom_bm25=custom_bm25, custom_docs=custom_docs)
