import sys, os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from rag_engine import answer, build_index_for_docs
from langchain_community.document_loaders import TextLoader

with open("doc_a.txt", "w") as f:
    f.write("Apples are usually red or green. They grow on trees.")
with open("doc_b.txt", "w") as f:
    f.write("Bananas are long and yellow. They grow in bunches.")

docs_a = TextLoader("doc_a.txt").load()
docs_b = TextLoader("doc_b.txt").load()
for d in docs_a: d.metadata["source"] = "doc_a.txt"
for d in docs_b: d.metadata["source"] = "doc_b.txt"

db_a, bm25_a, chunks_a = build_index_for_docs(docs_a)
db_b, bm25_b, chunks_b = build_index_for_docs(docs_b)

ans_a = answer("What color are apples?", custom_db=db_a, custom_bm25=bm25_a, custom_docs=chunks_a)
ans_b = answer("What color are bananas?", custom_db=db_b, custom_bm25=bm25_b, custom_docs=chunks_b)

print(f"ANS_A (expected Red/Green): {ans_a}")
print(f"ANS_B (expected Yellow): {ans_b}")

# Check leakage
ans_leak = answer("What color are apples?", custom_db=db_b, custom_bm25=bm25_b, custom_docs=chunks_b)
print(f"ANS_LEAK (expected Refusal): {ans_leak}")

