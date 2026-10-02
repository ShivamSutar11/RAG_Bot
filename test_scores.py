import json
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import rag_engine

with open("evals/eval_dataset.json", "r") as f:
    dataset = json.load(f)

for item in dataset:
    question = item["question"]
    
    # Run the retrieval manually to see the scores
    query_emb = rag_engine.embed_model.encode([question], convert_to_tensor=False)[0]
    dense_docs = rag_engine.db.similarity_search_by_vector(query_emb, k=20)
    
    tokenized_query = question.lower().split()
    bm25_scores = rag_engine.bm25_index.get_scores(tokenized_query)
    import numpy as np
    top_n = np.argsort(bm25_scores)[::-1][:20]
    sparse_docs = [rag_engine.docs_global[i] for i in top_n]
    
    unique_docs = {}
    for d in dense_docs + sparse_docs:
        unique_docs[d.page_content] = d
        
    candidates = list(unique_docs.values())
    cross_inp = [[question, doc.page_content] for doc in candidates]
    cross_scores = rag_engine.reranker_model.predict(cross_inp)
    
    max_score = max(cross_scores)
    is_ood = not item["is_answerable"]
    print(f"[{'OOD' if is_ood else 'ANS'}] Max Cross-Encoder Score: {max_score:.4f} | Q: {question}")

