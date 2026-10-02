import json

with open("evals/results.json", "r") as f:
    results = json.load(f)

for i, item in enumerate(results):
    q = item["question"].replace("\n", " ")
    is_ood = item.get("refusal_pass") is not None
    print(f"[{i:02d}] Q: {q[:60]}... | OOD: {is_ood}")
