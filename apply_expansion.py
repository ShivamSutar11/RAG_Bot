import re

with open("rag_engine.py", "r") as f:
    code = f.read()

target1 = """    docs_global = chunks

    texts = [c.page_content for c in chunks]"""
    
replacement1 = """    src_counts = {}
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        if src not in src_counts:
            src_counts[src] = 0
        c.metadata["chunk_idx"] = src_counts[src]
        src_counts[src] += 1
    docs_global = chunks

    texts = [c.page_content for c in chunks]"""

code = code.replace(target1, replacement1)

target2 = """    # Sort by cross-encoder score descending
    candidates = sorted(candidates, key=lambda x: x.metadata['cross_score'], reverse=True)
    best_docs = candidates[:k]
    
    # Lost in the middle ordering
    reordered_docs = []
    for i, doc in enumerate(best_docs):
        if i % 2 == 0:
            reordered_docs.insert(0, doc)
        else:
            reordered_docs.append(doc)
            
    global last_retrieved_docs
    last_retrieved_docs = reordered_docs

    context = "\\n\\n".join([d.page_content for d in reordered_docs])"""

replacement2 = """    # Sort by cross-encoder score descending
    candidates = sorted(candidates, key=lambda x: x.metadata['cross_score'], reverse=True)
    best_docs = candidates[:k]
    
    # 1. Context Expansion Strategy
    def looks_incomplete(text):
        t = text.strip()
        if not t: return False
        if t[0].islower(): return True # Starts mid-sentence
        if not t[-1] in ".!?\\"'": return True # Ends mid-sentence or heading
        if t.count('|') > 3: return True # Table fragment
        if "=" in t and t.endswith("="): return True # Formula fragment
        return False

    expanded_docs = []
    chunk_map = { (d.metadata.get("source"), d.metadata.get("chunk_idx")): d for d in docs_global if "chunk_idx" in d.metadata }

    for i, doc in enumerate(best_docs):
        expanded_docs.append(doc)
        src = doc.metadata.get("source")
        c_idx = doc.metadata.get("chunk_idx")
        if src is None or c_idx is None:
            continue
            
        # Unconditional neighbor expansion for the absolute top chunk
        expand_prev = (i == 0) or looks_incomplete(doc.page_content[:150])
        expand_next = (i == 0) or looks_incomplete(doc.page_content[-150:])
        
        if expand_prev and (src, c_idx - 1) in chunk_map:
            expanded_docs.append(chunk_map[(src, c_idx - 1)])
        if expand_next and (src, c_idx + 1) in chunk_map:
            expanded_docs.append(chunk_map[(src, c_idx + 1)])

    # Deduplicate expanded pool
    unique_expanded = {}
    for d in expanded_docs:
        unique_expanded[d.page_content] = d
    final_docs = list(unique_expanded.values())
    
    # Sort chronologically to restore document reading order
    final_docs = sorted(final_docs, key=lambda x: (x.metadata.get("source", ""), x.metadata.get("chunk_idx", 0)))
    
    # Merge overlapping/redundant text
    merged_blocks = []
    current_block = None
    
    def merge_strings_with_overlap(s1, s2):
        max_overlap = min(len(s1), len(s2), 500)
        for i in range(max_overlap, 0, -1):
            if s1.endswith(s2[:i]):
                return s1 + s2[i:]
        return s1 + "\\n\\n" + s2

    for d in final_docs:
        if current_block is None:
            current_block = {"source": d.metadata.get("source"), "idxs": [d.metadata.get("chunk_idx")], "text": d.page_content}
        else:
            if d.metadata.get("source") == current_block["source"] and d.metadata.get("chunk_idx") == current_block["idxs"][-1] + 1:
                current_block["text"] = merge_strings_with_overlap(current_block["text"], d.page_content)
                current_block["idxs"].append(d.metadata.get("chunk_idx"))
            else:
                merged_blocks.append(current_block)
                current_block = {"source": d.metadata.get("source"), "idxs": [d.metadata.get("chunk_idx")], "text": d.page_content}
    if current_block:
        merged_blocks.append(current_block)
        
    class DummyDoc:
        def __init__(self, content):
            self.page_content = content
            self.metadata = {}

    # Pack into dummy docs for compatibility
    reordered_docs = [DummyDoc(b["text"]) for b in merged_blocks]

    global last_retrieved_docs
    last_retrieved_docs = reordered_docs

    context = "\\n\\n".join([d.page_content for d in reordered_docs])"""

if target1 not in code:
    print("Warning: target1 not found!")
if target2 not in code:
    print("Warning: target2 not found!")

code = code.replace(target1, replacement1)
code = code.replace(target2, replacement2)

with open("rag_engine.py", "w") as f:
    f.write(code)
print("Patch applied.")
