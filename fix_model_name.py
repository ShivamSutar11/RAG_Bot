import sys
with open("rag_engine.py", "r") as f:
    text = f.read()

text = text.replace('MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"', 'MODEL_NAME = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")')

with open("rag_engine.py", "w") as f:
    f.write(text)
