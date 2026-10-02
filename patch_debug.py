with open("backend/main.py", "r") as f:
    code = f.read()

target = """    try:
        ans = answer(request.question, """

replacement = """    try:
        # RETRIEVAL DEBUG
        print("=== RETRIEVAL DEBUG ===")
        print(f"Selected document_id: {request.document_id}")
        print(f"Selected filename: {doc_data['filename']}")
        print(f"Index object used (FAISS): {doc_data['db']}")
        print(f"Index object used (BM25): {doc_data['bm25']}")
        
        # Test what chunks are retrieved
        from rag_engine import embed_model, bm25_index
        import numpy as np
        
        # This is just for printing the debug requested by user before answering
        q_emb = embed_model.encode([request.question], convert_to_tensor=False)[0]
        d_docs = doc_data['db'].similarity_search_by_vector(q_emb, k=5)
        
        t_q = request.question.lower().split()
        b_scores = doc_data['bm25'].get_scores(t_q)
        top_n = np.argsort(b_scores)[::-1][:5]
        s_docs = [doc_data['chunks'][i] for i in top_n]
        
        all_d = d_docs + s_docs
        print("Retrieved chunk metadata:")
        for d in all_d:
            print(f" - {d.metadata}")
            print(f"   Source filename: {d.metadata.get('source')}")
        print("=======================")

        ans = answer(request.question, """

if target in code:
    code = code.replace(target, replacement)
    with open("backend/main.py", "w") as f:
        f.write(code)
    print("Patched debug prints.")
else:
    print("Target not found.")
