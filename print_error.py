import json

with open("evals/results.json", "r") as f:
    results = json.load(f)

print(results[0]["error"])
