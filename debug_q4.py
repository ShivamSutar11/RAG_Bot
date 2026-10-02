import sys, os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from langchain_community.document_loaders import PyPDFLoader
from rag_engine import build_index_for_docs, rag_answer, answer

print("Loading test_cfo.pdf...")
docs = PyPDFLoader("test_cfo.pdf").load()

# Find page with "Q4 FY26" and "Revenue"
target_page_text = ""
for i, d in enumerate(docs):
    if "Revenue" in d.page_content and "Q4 FY26" in d.page_content:
        target_page_text = d.page_content
        print(f"\n--- 1. Extracted Text from Page {i} ---")
        print(target_page_text)
        break

db, bm25, chunks = build_index_for_docs(docs)

print(f"\n--- 2. Chunk(s) created from that page ---")
for i, c in enumerate(chunks):
    if "Revenue" in c.page_content and "Q4 FY26" in c.page_content:
        print(f"Chunk {i}:\n{c.page_content}\n")

# To get internal retrieval info, we need to inspect the inner workings of rag_answer.
# I'll just temporarily monkeypatch or reimplement the core logic here to capture candidates.
question = "What was NVIDIA's Q4 FY26 revenue?"
tokenized_query = question.lower().split()
import numpy as np
bm25_scores = bm25.get_scores(tokenized_query)
top_n = np.argsort(bm25_scores)[::-1][:20]
sparse_docs = [chunks[i] for i in top_n]

print("\n--- 3. BM25 Candidates (Top 3) ---")
for i, d in enumerate(sparse_docs[:3]):
    print(f"BM25 {i}:\n{d.page_content}\n")

from rag_engine import embed_model
query_emb = embed_model.encode([question], convert_to_tensor=False)[0]
dense_docs = db.similarity_search_by_vector(query_emb, k=20)

print("\n--- 4. Dense Retrieval Candidates (Top 3) ---")
for i, d in enumerate(dense_docs[:3]):
    print(f"Dense {i}:\n{d.page_content}\n")

unique_docs = {}
for d in dense_docs + sparse_docs:
    unique_docs[d.page_content] = d
candidates = list(unique_docs.values())

from rag_engine import reranker_model
cross_inp = [[question, d.page_content] for d in candidates]
cross_scores = reranker_model.predict(cross_inp)

for idx in range(len(cross_scores)):
    candidates[idx].metadata['cross_score'] = cross_scores[idx]

print("\n--- 5. Reranker Scores (Top 5) ---")
candidates_by_cross = sorted(candidates, key=lambda x: x.metadata['cross_score'], reverse=True)
for i, d in enumerate(candidates_by_cross[:5]):
    print(f"Reranker Score {d.metadata['cross_score']:.4f}:\n{d.page_content}\n")

# Run the actual RAG to get the final context, raw output, and post-processed
print("\n--- 6. Final Retrieved Context ---")
# I can just grab it from rag_answer by looking at the last retrieved docs or patching the final string.
import rag_engine
original_rag_answer = rag_engine.rag_answer

def my_rag_answer(q, k, max_new_tokens, temperature, do_sample, custom_db, custom_bm25, custom_docs):
    from rag_engine import last_retrieved_docs
    # Execute normal
    ret = original_rag_answer(q, k, max_new_tokens, temperature, do_sample, custom_db, custom_bm25, custom_docs)
    context = "\n\n".join([d.page_content for d in rag_engine.last_retrieved_docs])
    print(context)
    return ret

rag_engine.rag_answer = my_rag_answer

print("\n--- 7 & 8. Raw LLM and Post-processed ---")
final_ans = answer(question, custom_db=db, custom_bm25=bm25, custom_docs=chunks)
print(f"\nFinal Answer: {final_ans}")

