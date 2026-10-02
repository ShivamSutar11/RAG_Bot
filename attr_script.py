import json

with open("evals/results.json", "r") as f:
    results = json.load(f)

with open("evals/eval_dataset.json", "r") as f:
    dataset = json.load(f)

for item, d_item in zip(results, dataset):
    if d_item["is_answerable"]:
        if item["correctness"] < 2 or item["completeness"] < 2:
            print(f"--- Q{item['id']}: {item['question']} ---")
            print(f"Keywords expected: {d_item['keywords']}")
            print(f"Correctness: {item['correctness']} | Completeness: {item['completeness']} | Retrieval Score: {item['retrieval_score']}")
            print(f"Model Answer: {item['answer']}")
            
            # Print a snippet of the retrieved context
            context_text = "\n".join([c["content"] for c in item["retrieval_debug"]])
            print(f"Context Snippet (first 300 chars): {context_text[:300]}...")
            
            # Check if keywords are actually present in the full context
            found = []
            missing = []
            for kw in d_item["keywords"]:
                if kw.lower() in context_text.lower():
                    found.append(kw)
                else:
                    missing.append(kw)
            print(f"Evidence in context -> Found: {found}, Missing: {missing}")
            print("\n")
