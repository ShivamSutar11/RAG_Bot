with open("rag_engine.py", "r") as f:
    code = f.read()

target = "best_docs = candidates[:k]"
replacement = "best_docs = candidates[:k]\n    print(f'MAX CROSS SCORE: {candidates[0].metadata.get(\"cross_score\", -99)} for Q: {question}')"

code = code.replace(target, replacement)
with open("rag_engine.py", "w") as f:
    f.write(code)
