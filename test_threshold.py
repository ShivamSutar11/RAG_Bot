import json
import sys
import os
import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import rag_engine

with open("evals/eval_dataset.json", "r") as f:
    dataset = json.load(f)

for item in dataset:
    question = item["question"]
    is_ood = not item["is_answerable"]
    
    query_emb = rag_engine.embed_model.encode([question], convert_to_tensor=False)[0]
    dense_docs = rag_engine.db.similarity_search_by_vector(query_emb, k=20)
    
    tokenized_query = question.lower().split()
    bm25_scores = rag_engine.bm25_index.get_scores(tokenized_query)
    top_n = np.argsort(bm25_scores)[::-1][:20]
    sparse_docs = [rag_engine.docs_global[i] for i in top_n]
    
    unique_docs = {}
    for d in dense_docs + sparse_docs:
        unique_docs[d.page_content] = d
        
    candidates = list(unique_docs.values())
    cross_inp = [[question, doc.page_content] for doc in candidates]
    cross_scores = rag_engine.reranker_model.predict(cross_inp)
    
    max_score = max(cross_scores) if len(cross_scores) > 0 else -999
    
    print(f"[{'OOD' if is_ood else 'ANS'}] Max Score: {max_score:6.2f} | Q: {question}")
