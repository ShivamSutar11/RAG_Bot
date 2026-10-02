with open("rag_engine.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if "Deterministic Refusal Check:" in line:
        skip = True
        new_lines.append("    # Relevance / evidence check\n")
        new_lines.append("    if len(best_docs) == 0 or best_docs[0].metadata.get('cross_score', -999) < 0.0:\n")
        new_lines.append('        return "I cannot find enough information in the uploaded document to answer this question."\n')
        continue
        
    if skip:
        if "system_prompt = (" in line:
            skip = False
            new_lines.append(line)
        continue
        
    new_lines.append(line)

with open("rag_engine.py", "w") as f:
    f.writelines(new_lines)
