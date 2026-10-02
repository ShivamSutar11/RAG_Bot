import re

with open("rag_engine.py", "r") as f:
    code = f.read()

target = """    # Sort by cross-encoder score descending
    candidates = sorted(candidates, key=lambda x: x.metadata['cross_score'], reverse=True)
    best_docs = candidates[:k]"""

replacement = """    # PHASE 2: Score fusion (RRF) + Preserve lexical
    # Compute dense ranks
    dense_ranks = {doc.page_content: i for i, doc in enumerate(dense_docs)}
    # Compute sparse ranks
    sparse_ranks = {doc.page_content: i for i, doc in enumerate(sparse_docs)}
    
    # Sort candidates by cross_score for cross ranks
    candidates_by_cross = sorted(candidates, key=lambda x: x.metadata['cross_score'], reverse=True)
    cross_ranks = {doc.page_content: i for i, doc in enumerate(candidates_by_cross)}
    
    # RRF parameters
    K = 60
    
    for doc in candidates:
        r_dense = dense_ranks.get(doc.page_content, 100)
        r_sparse = sparse_ranks.get(doc.page_content, 100)
        r_cross = cross_ranks.get(doc.page_content, 100)
        
        # Fuse scores
        doc.metadata['fused_score'] = (1.0 / (K + r_dense)) + (1.0 / (K + r_sparse)) + (1.0 / (K + r_cross))
        
    # Sort by fused score
    candidates = sorted(candidates, key=lambda x: x.metadata['fused_score'], reverse=True)
    
    # Preserve Strategy: Ensure the #1 BM25 candidate makes it into the top k
    best_docs = []
    top_bm25 = sparse_docs[0] if sparse_docs else None
    if top_bm25:
        best_docs.append(top_bm25)
        
    for doc in candidates:
        if doc not in best_docs:
            best_docs.append(doc)
        if len(best_docs) == k:
            break"""

if target not in code:
    print("WARNING: Phase 2 target not found")
else:
    code = code.replace(target, replacement)
    with open("rag_engine.py", "w") as f:
        f.write(code)
    print("Phase 2 patch applied.")
