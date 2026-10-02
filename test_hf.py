from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import AnswerRelevancy

embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
m = AnswerRelevancy(embeddings=embeddings)
print("Modern HF embeddings work!")
