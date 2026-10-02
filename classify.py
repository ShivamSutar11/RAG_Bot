import json

with open("evals/results.json", "r") as f:
    results = json.load(f)
with open("evals/eval_dataset.json", "r") as f:
    dataset = json.load(f)

categories = {
    "Retrieval failure": 0,
    "Partial retrieval": 0,
    "Generation failure": 0,
    "Table / numeric extraction failure": 0,
    "Answer-length / truncation failure": 0,
    "Other": 0
}

report = ""

for item, d_item in zip(results, dataset):
    if d_item["is_answerable"]:
        if item["correctness"] < 2 or item["completeness"] < 2:
            retrieval_score = item["retrieval_score"]
            q_lower = item["question"].lower()
            is_numeric = any(w in q_lower for w in ["how many", "score", "value", "dimensionality"])
            ans_len = len(item["answer"].split())
            
            cat = "Other"
            if retrieval_score == 0:
                cat = "Retrieval failure"
            elif retrieval_score == 1:
                cat = "Partial retrieval"
            elif retrieval_score == 2:
                if is_numeric and item["correctness"] < 2:
                    cat = "Table / numeric extraction failure"
                elif ans_len < 8 and item["completeness"] < 2:
                    cat = "Answer-length / truncation failure"
                else:
                    cat = "Generation failure"
                    
            categories[cat] += 1
            
            report += f"**Q{item['id']}**: {item['question']}\n"
            report += f"- Correctness: {item['correctness']} | Completeness: {item['completeness']}\n"
            report += f"- Required evidence present: {'Yes (Full)' if retrieval_score == 2 else 'Partial' if retrieval_score == 1 else 'No'}\n"
            context_snippet = "\\n".join([c["content"] for c in item["retrieval_debug"]])[:150].replace("\n", " ")
            report += f"- Retrieved evidence snippet: {context_snippet}...\n"
            report += f"- Model answer: {item['answer']}\n"
            report += f"- Failure category: **{cat}**\n\n"

total_failed = sum(categories.values())

summary = "### Failure Attribution Summary\n\n"
summary += "| Failure Type | Count | Percentage |\n"
summary += "|---|---|---|\n"
for k, v in categories.items():
    if v > 0:
        pct = (v / total_failed) * 100
        summary += f"| {k} | {v} | {pct:.1f}% |\n"

print(report)
print(summary)
