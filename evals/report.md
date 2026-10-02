# Baseline RAG Evaluation Report

**Total Questions:** 30
**Answerable Accuracy (Avg Correctness):** 1.33 / 2.0
**Refusal Accuracy:** 100.0%
**Retrieval Hit Rate (Avg Score):** 1.68 / 2.0
**Average Completeness Score:** 1.33 / 2.0
**Average Faithfulness Score:** 1.73 / 2.0
**Average Latency per Query:** 3.55 seconds

## Failed Questions
- **Q5**: Why is the dot product scaled by 1/sqrt(d_k)?
  - **Reason**: Missing keywords
  - **Answer Given**: The dot product is scaled by 1/sqrt(d_k) because it represents the similarity between two vectors, and in the context of multi-head attention, the key idea is to combine multiple representations from different parts of the input. By scaling the dot product by √(d_k), we ensure that the attention mechanism is computationally efficient while still capturing the important relationships between the inputs. This helps in maintaining the overall performance of the model during training.

- **Q14**: What is the BLEU score of the base model on English-to-German translation?
  - **Reason**: Missing keywords
  - **Answer Given**: The BLEU score of the base model on the English-to-German translation task is 28.4.

- **Q16**: How many training steps were used for the base model?
  - **Reason**: Missing keywords
  - **Answer Given**: The base model was trained for a total of 12 hours.

- **Q19**: Does the model use convolutional layers?
  - **Reason**: Missing keywords
  - **Answer Given**: Yes, the model uses convolutional layers inside the encoder and decoder.

- **Q23**: Describe the dropout strategy used in the architecture.
  - **Reason**: Missing keywords
  - **Answer Given**: The dropout strategy used in the Transformer architecture is Pdrop = 0.1.

