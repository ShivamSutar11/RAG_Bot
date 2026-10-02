with open("rag_engine.py", "r") as f:
    code = f.read()

target = "candidates = sorted(candidates, key=lambda x: x.metadata['cross_score'], reverse=True)"
replacement = "candidates = sorted(candidates, key=lambda x: x.metadata['cross_score'], reverse=True)\n    max_score = candidates[0].metadata['cross_score']\n    print(f'MAX CROSS SCORE: {max_score:.4f} for Q: {question}')"

code = code.replace(target, replacement)
with open("rag_engine.py", "w") as f:
    f.write(code)
