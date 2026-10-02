from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

llm = ChatOllama(model="llama3.1:latest", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

metric = Faithfulness(llm=llm)

data = {
    "question": ["What is Paris?"],
    "answer": ["Paris is the capital of France."],
    "contexts": [["Paris is a city in France."]],
    "ground_truth": ["Paris is the capital city of France."]
}
dataset = Dataset.from_dict(data)

res = evaluate(dataset, metrics=[metric])
print("RES:", res)
