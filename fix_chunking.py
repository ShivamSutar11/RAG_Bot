import sys

with open("rag_engine.py", "r") as f:
    text = f.read()

text = text.replace('RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)', 'RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)')

with open("rag_engine.py", "w") as f:
    f.write(text)
