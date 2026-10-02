import sys, os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from rag_engine import answer, build_index_for_docs, rag_answer
from langchain_community.document_loaders import TextLoader

with open("doc_a.txt", "w") as f:
    f.write("Apples are usually red or green. The ID is 999.")
with open("doc_b.txt", "w") as f:
    f.write("Bananas are long and yellow. The ID is 888.")

docs_a = TextLoader("doc_a.txt").load()
docs_b = TextLoader("doc_b.txt").load()
for d in docs_a: d.metadata["source"] = "doc_a.txt"
for d in docs_b: d.metadata["source"] = "doc_b.txt"

db_a, bm25_a, chunks_a = build_index_for_docs(docs_a)
db_b, bm25_b, chunks_b = build_index_for_docs(docs_b)

ans_leak = rag_answer("What is the ID for Apples?", custom_db=db_b, custom_bm25=bm25_b, custom_docs=chunks_b)
print(f"ANS_LEAK (expected Refusal because ID 999 and Apples are not in doc_b): {ans_leak}")
