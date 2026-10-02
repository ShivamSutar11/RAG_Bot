import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import Faithfulness
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings

ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
llm = llm_factory("llama3.1:latest", client=ollama_client)

metrics = [Faithfulness(llm=llm)]

data = {
    "user_input": ["What is Paris?"],
    "response": ["Paris is the capital of France."],
    "retrieved_contexts": [["Paris is a city in France."]],
    "reference": ["Paris is the capital city of France."]
}
dataset = Dataset.from_dict(data)

try:
    res = evaluate(dataset, metrics=metrics)
    print("RES:", res)
except Exception as e:
    print("ERROR:", e)
