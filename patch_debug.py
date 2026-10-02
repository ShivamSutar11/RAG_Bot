import re

with open("rag_engine.py", "r") as f:
    code = f.read()

target = """            if missing_tokens and term.lower().strip("?.,\\"'") not in context.lower():
                return "I cannot find enough information in the uploaded document to answer this question.\""""
replacement = """            if missing_tokens and term.lower().strip("?.,\\"'") not in context.lower():
                print(f"DEBUG REFUSAL -> term: {term}, missing_tokens: {missing_tokens}")
                return "I cannot find enough information in the uploaded document to answer this question.\""""

code = code.replace(target, replacement)
with open("rag_engine.py", "w") as f:
    f.write(code)
