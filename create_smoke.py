import json

with open("evals/results.json", "r") as f:
    results = json.load(f)

with open("evals/eval_dataset.json", "r") as f:
    dataset_orig = json.load(f)

# Select IDs: 1 (ans), 7 (ans), 13 (ans), 15 (num), 26 (ood)
# Wait, let's just pick indices 0, 6, 12, 14, 25 (0-indexed)
indices = [0, 6, 12, 14, 25]

smoke_results = [results[i] for i in indices]
smoke_dataset = [dataset_orig[i] for i in indices]

with open("evals/results_smoke.json", "w") as f:
    json.dump(smoke_results, f, indent=2)

with open("evals/eval_dataset_smoke.json", "w") as f:
    json.dump(smoke_dataset, f, indent=2)

print("Created smoke test datasets with 5 items.")
