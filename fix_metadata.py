with open("rag_engine.py", "r") as f:
    code = f.read()

target = "db = FAISS.from_texts(texts, MiniEmb())"
replacement = "metadatas = [c.metadata for c in chunks]\n    db = FAISS.from_texts(texts, MiniEmb(), metadatas=metadatas)"

code = code.replace(target, replacement)

with open("rag_engine.py", "w") as f:
    f.write(code)
print("Metadata fixed in FAISS.")
