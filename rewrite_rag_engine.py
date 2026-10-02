with open("rag_engine.py", "r") as f:
    content = f.read()

# Replace build_faiss with refactored version
import re

old_build_faiss = r"""def build_faiss\(\):.*?print\("✅ FAISS & BM25 indexes built\."\)\n    return db"""
new_build_faiss = """class MiniEmb(Embeddings):
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
    return db"""

content = re.sub(old_build_faiss, new_build_faiss, content, flags=re.DOTALL)

# Refactor rag_answer
content = content.replace("def rag_answer(question, k=5, max_new_tokens=200, temperature=0.0, do_sample=False):",
                          "def rag_answer(question, k=5, max_new_tokens=200, temperature=0.0, do_sample=False, custom_db=None, custom_bm25=None, custom_docs=None):")

content = content.replace("dense_docs = db.similarity_search_by_vector(query_emb, k=20)",
                          "dense_docs = (custom_db or db).similarity_search_by_vector(query_emb, k=20)")

content = content.replace("bm25_scores = bm25_index.get_scores(tokenized_query)",
                          "bm25_scores = (custom_bm25 or bm25_index).get_scores(tokenized_query)")

content = content.replace("sparse_docs = [docs_global[i] for i in top_n]",
                          "sparse_docs = [(custom_docs or docs_global)[i] for i in top_n]")

content = content.replace("chunk_map = { (d.metadata.get(\"source\"), d.metadata.get(\"chunk_idx\")): d for d in docs_global if \"chunk_idx\" in d.metadata }",
                          "chunk_map = { (d.metadata.get(\"source\"), d.metadata.get(\"chunk_idx\")): d for d in (custom_docs or docs_global) if \"chunk_idx\" in d.metadata }")

content = content.replace("def answer(query):", "def answer(query, custom_db=None, custom_bm25=None, custom_docs=None):")
content = content.replace("return rag_answer(query)", "return rag_answer(query, custom_db=custom_db, custom_bm25=custom_bm25, custom_docs=custom_docs)")

with open("rag_engine.py", "w") as f:
    f.write(content)
print("rag_engine patched successfully.")
