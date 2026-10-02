import sys, os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from langchain_community.document_loaders import PyPDFLoader
from rag_engine import build_index_for_docs, answer

docs = PyPDFLoader("test_cfo.pdf").load()
db, bm25, chunks = build_index_for_docs(docs)

ans1 = answer("What was NVIDIA's Q4 FY26 revenue?", custom_db=db, custom_bm25=bm25, custom_docs=chunks)
print(f"Ans 1 (NVIDIA's Q4 FY26 revenue): {ans1}\n")

ans2 = answer("What was Q3 FY26 revenue?", custom_db=db, custom_bm25=bm25, custom_docs=chunks)
print(f"Ans 2 (Q3 FY26 revenue): {ans2}\n")

ans3 = answer("What was Q4 FY25 revenue?", custom_db=db, custom_bm25=bm25, custom_docs=chunks)
print(f"Ans 3 (Q4 FY25 revenue): {ans3}\n")

ans4 = answer("What was Q4 FY26 GAAP net income?", custom_db=db, custom_bm25=bm25, custom_docs=chunks)
print(f"Ans 4 (Q4 FY26 GAAP net income): {ans4}\n")

ans5 = answer("What was Q4 FY26 non-GAAP net income?", custom_db=db, custom_bm25=bm25, custom_docs=chunks)
print(f"Ans 5 (Q4 FY26 non-GAAP net income): {ans5}\n")

ans_ood = answer("Who won the 2018 World Cup?", custom_db=db, custom_bm25=bm25, custom_docs=chunks)
print(f"Ans OOD: {ans_ood}\n")
