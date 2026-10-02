import json
import sys
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import rag_engine

with open("evals/eval_dataset.json", "r") as f:
    dataset = json.load(f)

for item in dataset:
    q = item["question"]
    # We can just intercept the candidates before generation
    # But wait, we can just call rag_engine.rag_answer, it has the print statement
    try:
        rag_engine.rag_answer(q)
    except Exception as e:
        print(f"Error for {q}: {e}")
