import sys

with open("rag_engine.py", "r") as f:
    text = f.read()

text = text.replace('embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")', 'embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")\nreranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")')

with open("rag_engine.py", "w") as f:
    f.write(text)
