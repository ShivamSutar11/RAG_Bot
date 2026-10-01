# Baseline RAG Evaluation Report

**Total Questions:** 30
**Answerable Accuracy (Avg Correctness):** 1.33 / 2.0
**Refusal Accuracy:** 100.0%
**Retrieval Hit Rate (Avg Score):** 1.52 / 2.0
**Average Completeness Score:** 1.33 / 2.0
**Average Faithfulness Score:** 1.63 / 2.0
**Average Latency per Query:** 13.58 seconds

## Failed Questions
- **Q5**: Why is the dot product scaled by 1/sqrt(d_k)?
  - **Reason**: Missing keywords
  - **Answer Given**: I cannot find enough information in the uploaded document to answer this question.

- **Q8**: How many attention heads are used in the base Transformer model?
  - **Reason**: Missing keywords
  - **Answer Given**: I cannot find enough information in the uploaded document to answer this question. The context mentions varying the number of attention heads in experiments but does not specify the exact number used in the base Transformer model.

- **Q15**: What is the BLEU score of the big model on English-to-French translation?
  - **Reason**: Missing keywords
  - **Answer Given**: The BLEU score of the big model on the English-to-French translation task is 41.8.

- **Q21**: What mathematical operation helps stabilize the gradients in the attention calculation?
  - **Reason**: Missing keywords
  - **Answer Given**: To stabilize the gradients in the attention calculation, we scale the dot products by 1/√dk. This is done to counteract the effect of dot products growing large in magnitude, which can push the softmax function into regions where it has extremely small gradients.

