import json
import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, AnswerCorrectness
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

import warnings
warnings.filterwarnings("ignore")

def main():
    print("Loading custom evaluation results from evals/results.json...")
    with open("evals/results_smoke.json", "r") as f:
        results = json.load(f)
        
    with open("evals/eval_dataset_smoke.json", "r") as f:
        dataset_orig = json.load(f)
    
    expected_map = {item["id"]: item["expected_answer"] for item in dataset_orig}

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    custom_metrics = {
        "correctness_score_sum": 0,
        "completeness_score_sum": 0,
        "faithfulness_score_sum": 0,
        "hit_rate_score_sum": 0,
        "refusal_acc": 0,
        "total": len(results),
        "total_ood": 0
    }

    for item in results:
        data["question"].append(item["question"])
        data["answer"].append(item["answer"])
        data["contexts"].append([ctx["content"] for ctx in item["retrieval_debug"]])
        data["ground_truth"].append(expected_map.get(item["id"], ""))
        
        custom_metrics["correctness_score_sum"] += item["correctness"]
        custom_metrics["completeness_score_sum"] += item["completeness"]
        custom_metrics["faithfulness_score_sum"] += item["faithfulness"]
        custom_metrics["hit_rate_score_sum"] += item["retrieval_score"]
        if item.get("refusal_pass") is not None:
            custom_metrics["total_ood"] += 1
            if item["refusal_pass"]:
                custom_metrics["refusal_acc"] += 1
                
    dataset = Dataset.from_dict(data)
    
    print("Initializing evaluator LLM and Embeddings...")
    llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0, format="json")
    
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Wrap in LangchainEmbeddingsWrapper to add embed_text method required by Ragas 0.2.2!
    embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
    
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
        AnswerCorrectness(llm=llm, embeddings=embeddings)
    ]
    
    print("Running RAGAS evaluation on 30 questions...")
    start_time = time.time()
    run_config = RunConfig(timeout=300, max_workers=1)
    res = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings, run_config=run_config)
    end_time = time.time()
    
    eval_time = end_time - start_time
    print(f"RAGAS evaluation completed in {eval_time:.2f} seconds.")
    
    res_dict = {k: v for k, v in res.items()}

    out = {
        "evaluator_model": "qwen2.5:7b-instruct",
        "embeddings": "sentence-transformers/all-MiniLM-L6-v2",
        "runtime_seconds": eval_time,
        "scores": res_dict
    }
    
    with open("evals/ragas_results_smoke.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
        
    print(res)
    print("Generating report...")
    
    custom_correctness = custom_metrics["correctness_score_sum"] / custom_metrics["total"]
    custom_completeness = custom_metrics["completeness_score_sum"] / custom_metrics["total"]
    custom_faithfulness = custom_metrics["faithfulness_score_sum"] / custom_metrics["total"]
    custom_hit_rate = custom_metrics["hit_rate_score_sum"] / custom_metrics["total"]
    custom_ood_acc = custom_metrics["refusal_acc"] / max(1, custom_metrics["total_ood"])
    
    report = "### RAG Evaluation Summary\n\n"
    report += "| Evaluation | Metric | Score |\n"
    report += "|---|---|---|\n"
    report += f"| Custom | Correctness | {custom_correctness:.2f} / 2.0 |\n"
    report += f"| Custom | Completeness | {custom_completeness:.2f} / 2.0 |\n"
    report += f"| Custom | Faithfulness | {custom_faithfulness:.2f} / 2.0 |\n"
    report += f"| Custom | Retrieval Hit Rate | {custom_hit_rate:.2f} / 2.0 |\n"
    report += f"| Custom | OOD Refusal | {custom_ood_acc*100:.0f}% |\n"
    
    for metric_name, score in res_dict.items():
        report += f"| RAGAS | {metric_name} | {score:.4f} |\n"
        
    with open("evals/ragas_report_smoke.md", "w") as f:
        f.write(report)
        
    print("Saved to evals/ragas_report.md")

if __name__ == "__main__":
    main()
