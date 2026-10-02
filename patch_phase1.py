import re

with open("rag_engine.py", "r") as f:
    code = f.read()

target = """    # Deterministic Refusal Check: If question asks about a specific proper noun or number not in the context, refuse.
    words = question.split()
    if len(words) > 1:
        # Check capitalized words or words with numbers (ignoring the first word which is usually capitalized)
        salient_terms = [w.strip("?.,\\"'") for w in words[1:] if (not w.islower() or any(c.isdigit() for c in w)) and len(w.strip("?.,\\"'")) > 1]
        context_lower = context.lower()
        for term in salient_terms:
            if term.lower() not in context_lower:
                return "I cannot find enough information in the uploaded document to answer this question.\""""

replacement = """    # Deterministic Refusal Check: Token-based matching to tolerate formatting
    words = question.split()
    if len(words) > 1:
        salient_terms = [w for w in words[1:] if (not w.islower() or any(c.isdigit() for c in w))]
        
        def extract_tokens(text):
            import re
            return [t for t in re.split(r'[^a-zA-Z0-9]+', text.lower()) if len(t) > 1]
            
        context_norm = context.lower().replace("√", "sqrt").replace("\\\\sqrt", "sqrt").replace("_", "")
        context_tokens = set(extract_tokens(context_norm))
        
        for term in salient_terms:
            term_norm = term.lower().replace("√", "sqrt").replace("\\\\sqrt", "sqrt").replace("_", "")
            term_tokens = extract_tokens(term_norm)
            
            # If the term tokenizes successfully, ensure at least one of its meaningful tokens is in the context
            # Actually, to be strict, we require ALL of its sub-tokens to be present in context
            missing_tokens = [wt for wt in term_tokens if wt not in context_tokens]
            
            # If tokens are missing, fall back to basic substring just in case tokenization broke it
            if missing_tokens and term.lower().strip("?.,\\"'") not in context.lower():
                return "I cannot find enough information in the uploaded document to answer this question.\""""

if target not in code:
    print("WARNING: Phase 1 target not found")
else:
    code = code.replace(target, replacement)
    with open("rag_engine.py", "w") as f:
        f.write(code)
    print("Phase 1 patch applied.")
