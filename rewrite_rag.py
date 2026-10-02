import re

with open("backend/rag_engine.py", "r") as f:
    code = f.read()

# 1. Update the hard safety check to return a tuple
code = code.replace(
    'return "I cannot find enough information in the uploaded document to answer this question."',
    'return "I cannot find enough information in the uploaded document to answer this question.", []'
)

# 2. Find the deterministic gate and replace it entirely
gate_pattern = re.compile(
    r"# Deterministic Refusal Check.*?if match_count == 0:\s+return \"I cannot find enough information in the uploaded document to answer this question.\", \[\]",
    re.DOTALL
)

new_gate = """# Improved Grounding Gate
    import string
    stopwords = {"what", "is", "the", "function", "of", "how", "why", "when", "who", "where", "are", "to", "in", "on", "at", "for", "a", "an", "and", "or", "but", "with", "by", "about", "between", "difference", "can", "could", "should", "would", "do", "does", "did", "from", "be", "have", "has", "had", "they", "their", "there", "were", "this", "that", "it"}
    
    clean_q = question.lower().translate(str.maketrans('', '', string.punctuation))
    words = clean_q.split()
    salient_terms = [w for w in words if w not in stopwords and len(w) > 2 and not w.isnumeric()]
    
    max_cross_score = max([d.metadata.get("cross_score", -999) for d in best_docs]) if best_docs else -999
    
    context_norm = context.lower()
    match_count = 0
    for term in salient_terms:
        if term in context_norm:
            match_count += 1
            
    if (salient_terms and match_count == 0) or max_cross_score < -5.0:
        return "I cannot find enough information in the uploaded document to answer this question.", []
"""

code = gate_pattern.sub(new_gate, code)

# 3. Update the final return in rag_answer
code = code.replace('return cleaned\n', 'return cleaned, best_docs\n')

# 4. Update the answer() wrapper
answer_target = """def answer(query, custom_db=None, custom_bm25=None, custom_docs=None):
    return rag_answer(query, custom_db=custom_db, custom_bm25=custom_bm25, custom_docs=custom_docs)"""

answer_replacement = """def answer(query, custom_db=None, custom_bm25=None, custom_docs=None, return_docs=False):
    ans, docs = rag_answer(query, custom_db=custom_db, custom_bm25=custom_bm25, custom_docs=custom_docs)
    if return_docs:
        return ans, docs
    return ans"""

code = code.replace(answer_target, answer_replacement)

with open("backend/rag_engine.py", "w") as f:
    f.write(code)

print("rag_engine.py rewritten")
