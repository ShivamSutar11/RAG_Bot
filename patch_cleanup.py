with open("rag_engine.py", "r") as f:
    code = f.read()

code = code.replace('print(f"DEBUG REFUSAL -> term: {term}, missing_tokens: {missing_tokens}")\n', '')
code = code.replace('print(f"CONTEXT EXTRACT: {context}")\n', '')

with open("rag_engine.py", "w") as f:
    f.write(code)
