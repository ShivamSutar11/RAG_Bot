import json
import time
import sys
import os
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import rag_engine

# No need to mock search anymore, rag_engine saves it globally.

def compute_overlap(text, context):
    words = re.findall(r'\w+', text.lower())
    if not words: return 0.0
    context_words = set(re.findall(r'\w+', context.lower()))
    matches = sum(1 for w in words if w in context_words)
    return matches / len(words)

def evaluate():
    with open("evals/eval_dataset.json", "r") as f:
        dataset = json.load(f)

    results = []
    metrics = {
        "total": len(dataset),
        "answerable_count": sum(1 for d in dataset if d["is_answerable"]),
        "ood_count": sum(1 for d in dataset if not d["is_answerable"]),
        "retrieval_score_sum": 0,
        "correctness_score_sum": 0,
        "completeness_score_sum": 0,
        "faithfulness_score_sum": 0,
        "refusal_pass": 0,
        "total_latency": 0,
        "failed_questions": []
    }

    for item in dataset:
        print(f"Evaluating Q{item['id']}: {item['question']}")
        start_time = time.time()
        
        try:
            answer = rag_engine.answer(item["question"])
            error = None
        except Exception as e:
            answer = ""
            error = str(e)
            print(f"ERROR: {error}")
            
        latency = time.time() - start_time
        metrics["total_latency"] += latency
        
        context_text = "\n".join([d.page_content for d in rag_engine.last_retrieved_docs])
        
        # Scoring
        retrieval_score = 0
        correctness = 0
        completeness = 0
        faithfulness = 0
        refusal_pass = False
        
        if item["is_answerable"]:
            # Retrieval
            matched_kws = sum(1 for kw in item["keywords"] if kw.lower() in context_text.lower())
            if matched_kws == len(item["keywords"]): retrieval_score = 2
            elif matched_kws > 0: retrieval_score = 1
            
            # Correctness
            matched_ans_kws = sum(1 for kw in item["keywords"] if kw.lower() in answer.lower())
            if matched_ans_kws == len(item["keywords"]): correctness = 2
            elif matched_ans_kws > 0: correctness = 1
            
            # Completeness (Penalty for extremely short answers like "Scaled Dot-Product Attention")
            ans_words = len(re.findall(r'\w+', answer))
            if ans_words < 8 and len(item["keywords"]) > 1:
                completeness = 0
            elif correctness == 2:
                completeness = 2
            elif correctness == 1:
                completeness = 1
                
            # Faithfulness
            overlap = compute_overlap(answer, context_text)
            if overlap > 0.8: faithfulness = 2
            elif overlap > 0.4: faithfulness = 1
            else: faithfulness = 0
            
            # Record if it failed badly
            if correctness == 0 or completeness == 0:
                metrics["failed_questions"].append({
                    "id": item["id"],
                    "question": item["question"],
                    "reason": "Missing keywords" if correctness == 0 else "Incomplete/short answer",
                    "answer": answer
                })
                
        else:
            # OOD
            if "cannot find enough information" in answer.lower():
                refusal_pass = True
                correctness = 2
                completeness = 2
                faithfulness = 2
                metrics["refusal_pass"] += 1
            else:
                metrics["failed_questions"].append({
                    "id": item["id"],
                    "question": item["question"],
                    "reason": "Failed to refuse",
                    "answer": answer
                })

        metrics["retrieval_score_sum"] += retrieval_score
        metrics["correctness_score_sum"] += correctness
        metrics["completeness_score_sum"] += completeness
        metrics["faithfulness_score_sum"] += faithfulness

        retrieval_debug = []
        for d in rag_engine.last_retrieved_docs:
            retrieval_debug.append({
                "score": 0.0,
                "source": d.metadata.get("source", ""),
                "page": d.metadata.get("page", ""),
                "content": d.page_content
            })

        results.append({
            "id": item["id"],
            "question": item["question"],
            "answer": answer,
            "retrieval_score": retrieval_score,
            "correctness": correctness,
            "completeness": completeness,
            "faithfulness": faithfulness,
            "refusal_pass": refusal_pass,
            "latency": latency,
            "error": error,
            "retrieval_debug": retrieval_debug
        })

    with open("evals/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate Report
    avg_retrieval = metrics["retrieval_score_sum"] / metrics["answerable_count"]
    avg_correct = metrics["correctness_score_sum"] / metrics["total"]
    avg_complete = metrics["completeness_score_sum"] / metrics["total"]
    avg_faith = metrics["faithfulness_score_sum"] / metrics["total"]
    avg_latency = metrics["total_latency"] / metrics["total"]
    refusal_acc = (metrics["refusal_pass"] / metrics["ood_count"]) * 100

    report = f"# Baseline RAG Evaluation Report\n\n"
    report += f"**Total Questions:** {metrics['total']}\n"
    report += f"**Answerable Accuracy (Avg Correctness):** {avg_correct:.2f} / 2.0\n"
    report += f"**Refusal Accuracy:** {refusal_acc:.1f}%\n"
    report += f"**Retrieval Hit Rate (Avg Score):** {avg_retrieval:.2f} / 2.0\n"
    report += f"**Average Completeness Score:** {avg_complete:.2f} / 2.0\n"
    report += f"**Average Faithfulness Score:** {avg_faith:.2f} / 2.0\n"
    report += f"**Average Latency per Query:** {avg_latency:.2f} seconds\n\n"
    
    report += "## Failed Questions\n"
    if metrics["failed_questions"]:
        for fq in metrics["failed_questions"]:
            report += f"- **Q{fq['id']}**: {fq['question']}\n"
            report += f"  - **Reason**: {fq['reason']}\n"
            report += f"  - **Answer Given**: {fq['answer']}\n\n"
    else:
        report += "No failed questions!\n"

    with open("evals/report.md", "w") as f:
        f.write(report)

if __name__ == "__main__":
    evaluate()
