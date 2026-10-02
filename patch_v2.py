with open("rag_engine.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if "Deterministic Refusal Check:" in line:
        skip = True
        new_lines.append("""    # Deterministic Refusal Check: Alphanumeric substring matching to tolerate formatting
    words = question.split()
    if len(words) > 1:
        salient_terms = [w for w in words[1:] if (not w.islower() or any(c.isdigit() for c in w))]
        
        # Normalize context into a single alphanumeric string for robust matching
        context_norm = context.lower().replace("√", "sqrt").replace("\\\\sqrt", "sqrt")
        context_alpha = "".join(c for c in context_norm if c.isalnum())
        
        for term in salient_terms:
            term_norm = term.lower().replace("√", "sqrt").replace("\\\\sqrt", "sqrt")
            term_alpha = "".join(c for c in term_norm if c.isalnum())
            
            if len(term_alpha) > 1 and term_alpha not in context_alpha:
                # Fallback to pure substring in case of weird concatenation
                if term.lower().strip("?.,\\"'") not in context.lower():
                    return "I cannot find enough information in the uploaded document to answer this question."
""")
        continue
    
    if skip and "system_prompt = (" in line:
        skip = False
        
    if not skip:
        new_lines.append(line)

with open("rag_engine.py", "w") as f:
    f.writelines(new_lines)
