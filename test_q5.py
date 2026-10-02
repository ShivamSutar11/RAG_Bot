import sys
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import rag_engine

q5 = "Why is the dot product scaled by 1/sqrt(d_k)?"
print("Testing Q5...")
ans = rag_engine.rag_answer(q5)
print(f"Q5 Answer: {ans}")

q_ood = "Who won the 2018 World Cup?"
print("Testing OOD...")
ans_ood = rag_engine.rag_answer(q_ood)
print(f"OOD Answer: {ans_ood}")
