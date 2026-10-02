with open("rag_engine.py", "r") as f:
    lines = f.readlines()

out = []
skip = False
for line in lines:
    if "Deterministic Refusal Check:" in line:
        skip = True
        out.append("""    # Deterministic Refusal Check: Relaxed Match Count to avoid OOD false positives
    words = question.split()
    if len(words) > 1:
        salient_terms = [w for w in words[1:] if (not w.islower() or any(c.isdigit() for c in w))]
        
        if salient_terms:
            context_norm = context.lower().replace("√", "sqrt").replace("\\\\sqrt", "sqrt")
            context_alpha = "".join(c for c in context_norm if c.isalnum())
            
            match_count = 0
            for term in salient_terms:
                term_norm = term.lower().replace("√", "sqrt").replace("\\\\sqrt", "sqrt")
                term_alpha = "".join(c for c in term_norm if c.isalnum())
                
                if len(term_alpha) > 1 and (term_alpha in context_alpha or term.lower().strip("?.,\\"'") in context.lower()):
                    match_count += 1
            
            if match_count == 0:
                return "I cannot find enough information in the uploaded document to answer this question."
""")
        continue
    
    if skip and "system_prompt = (" in line:
        skip = False
        
    if not skip:
        out.append(line)

with open("rag_engine.py", "w") as f:
    f.writelines(out)
print("rag_engine.py updated with robust grounding gate.")
