with open("rag_engine.py", "r") as f:
    code = f.read()

target_build_faiss = """def build_faiss():
    global bm25_index, docs_global
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)"""

replacement_build_faiss = """def build_index_for_docs(docs):
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

    class MiniEmb(Embeddings):
        def embed_documents(self, docs):
            return embed_model.encode(docs, convert_to_tensor=False)
        def embed_query(self, text):
            return embed_model.encode([text], convert_to_tensor=False)[0]

    metadatas = [c.metadata for c in chunks]
    db = FAISS.from_texts(texts, MiniEmb(), metadatas=metadatas)
    
    tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return db, bm25, chunks

def build_faiss():
    global bm25_index, docs_global, db
    docs = load_documents()
    if not docs:
        print("No documents found in ./docs.")
        return None
    db, bm25_index, docs_global = build_index_for_docs(docs)
    print("✅ Global FAISS & BM25 indexes built.")
    return db

# We need to temporarily mock out the old build_faiss logic inside the file to prevent duplication.
# Actually let's just replace the whole build_faiss block!"""

# Since replacing a large chunk with regex is tricky, let's write a python script that does it using AST or just rewrite rag_engine.py carefully.
