import json

with open("evals/results.json", "r") as f:
    results = json.load(f)
with open("evals/eval_dataset.json", "r") as f:
    dataset = json.load(f)

for item, d_item in zip(results, dataset):
    if not d_item["is_answerable"]:
        print(f"--- Q: {item['question']} ---")
        print(f"Answer: {item['answer']}")
        print(f"Refusal Pass: {item.get('refusal_pass')}")
