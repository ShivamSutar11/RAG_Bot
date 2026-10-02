import sys, os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from langchain_community.document_loaders import PyPDFLoader
from rag_engine import build_index_for_docs, answer, rag_answer

docs = PyPDFLoader("test_cfo.pdf").load()
db, bm25, chunks = build_index_for_docs(docs)

# Temporarily remove the deterministic check in the script
import rag_engine
original_rag_answer = rag_engine.rag_answer

ans = original_rag_answer("What was NVIDIA's Q4 FY26 revenue?", custom_db=db, custom_bm25=bm25, custom_docs=chunks)
print(f"Bypassed Gate Final Answer: {ans}")
