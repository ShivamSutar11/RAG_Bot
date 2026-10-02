import os
from ragas import evaluate
from ragas.metrics.collections import faithfulness
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset

data = {
    "user_input": ["What is Paris?"],
    "response": ["Paris is the capital of France."],
    "retrieved_contexts": [["Paris is a city in France."]],
    "reference": ["Paris is the capital city of France."]
}
dataset = Dataset.from_dict(data)

llm = ChatOllama(model="llama3.1:latest", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

res = evaluate(dataset, metrics=[faithfulness], llm=llm, embeddings=embeddings)
print(res)
