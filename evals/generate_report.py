import json

def main():
    print("Loading custom evaluation results from evals/results.json...")
    with open("evals/results.json", "r") as f:
        results = json.load(f)

    custom_metrics = {
        "correctness_score_sum": 0,
        "completeness_score_sum": 0,
        "faithfulness_score_sum": 0,
        "hit_rate_score_sum": 0,
        "refusal_acc": 0,
        "total": len(results),
        "total_ood": 0,
        "total_latency": 0
    }

    for item in results:
        custom_metrics["correctness_score_sum"] += item["correctness"]
        custom_metrics["completeness_score_sum"] += item["completeness"]
        custom_metrics["faithfulness_score_sum"] += item["faithfulness"]
        custom_metrics["hit_rate_score_sum"] += item["retrieval_score"]
        custom_metrics["total_latency"] += item.get("latency", 0)
        
        if item.get("refusal_pass") is not None:
            custom_metrics["total_ood"] += 1
            if item["refusal_pass"]:
                custom_metrics["refusal_acc"] += 1
                
    custom_correctness = custom_metrics["correctness_score_sum"] / custom_metrics["total"]
    custom_completeness = custom_metrics["completeness_score_sum"] / custom_metrics["total"]
    custom_faithfulness = custom_metrics["faithfulness_score_sum"] / custom_metrics["total"]
    custom_hit_rate = custom_metrics["hit_rate_score_sum"] / custom_metrics["total"]
    custom_ood_acc = custom_metrics["refusal_acc"] / max(1, custom_metrics["total_ood"])
    avg_latency = custom_metrics["total_latency"] / custom_metrics["total"]
    
    report = "### RAG Evaluation Summary\n\n"
    report += "*(Note: RAGAS evaluation was skipped due to local LLM JSON-parsing incompatibilities. Showing Custom Framework metrics.)*\n\n"
    report += "| Metric | Score |\n"
    report += "|---|---|\n"
    report += f"| **Correctness** | {custom_correctness:.2f} / 2.0 |\n"
    report += f"| **Completeness** | {custom_completeness:.2f} / 2.0 |\n"
    report += f"| **Faithfulness** | {custom_faithfulness:.2f} / 2.0 |\n"
    report += f"| **Retrieval Hit Rate** | {custom_hit_rate:.2f} / 2.0 |\n"
    report += f"| **OOD Refusal Accuracy** | {custom_ood_acc*100:.0f}% |\n"
    report += f"| **Average Latency** | {avg_latency:.2f} seconds |\n"
    
    with open("evals/final_report.md", "w") as f:
        f.write(report)
        
    print("Report generated at evals/final_report.md")

if __name__ == "__main__":
    main()
