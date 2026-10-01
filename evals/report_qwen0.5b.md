# Baseline RAG Evaluation Report

**Total Questions:** 30
**Answerable Accuracy (Avg Correctness):** 1.23 / 2.0
**Refusal Accuracy:** 100.0%
**Retrieval Hit Rate (Avg Score):** 1.52 / 2.0
**Average Completeness Score:** 1.23 / 2.0
**Average Faithfulness Score:** 1.63 / 2.0
**Average Latency per Query:** 3.03 seconds

## Failed Questions
- **Q5**: Why is the dot product scaled by 1/sqrt(d_k)?
  - **Reason**: Missing keywords
  - **Answer Given**: I cannot find enough information in the uploaded document to answer this question.

- **Q7**: What is Multi-Head Attention?
  - **Reason**: Missing keywords
  - **Answer Given**: Multi-Head Attention is a type of attention mechanism used in neural machine translation models like the Transformer. It allows the model to jointly attend to information from different representation subspaces at different positions.

- **Q8**: How many attention heads are used in the base Transformer model?
  - **Reason**: Missing keywords
  - **Answer Given**: In the base Transformer model, there is one attention head used.

- **Q11**: How does the Transformer handle the order of the sequence?
  - **Reason**: Missing keywords
  - **Answer Given**: The Transformer handles the order of the sequence through its unique approach involving multi-head attention and positional encoding. Specifically:

1. **Multi-Head Attention**: This mechanism allows the transformer to attend to multiple positions simultaneously, enabling it to process sequences of varying lengths. It works by combining the outputs of several heads into a single vector, which then attends to the entire sequence.

2. **Positional Encoding**: This step adds information about the relative or absolute position of each token within the sequence. By doing so, the transformer can better understand the structure of the sequence and how tokens interact with one another.

Together, these components enable the transformer to effectively manage the order of the sequence, making it suitable for tasks like text processing, natural language understanding, and machine translation.

- **Q13**: What regularization techniques were used during training?
  - **Reason**: Missing keywords
  - **Answer Given**: During training, three types of regularization were employed:

1. Competitive models: This type of regularization involves employing multiple models that compete against each other during training. In this case, the authors used three different types of models: 
   - A single model obtained by averaging the last 5 checkpoints, which were written at 10-minute intervals.
   - An average of the last 20 checkpoints.
   - A beam search with a beam size of 4 and length penalty α = 0.6.

2. Base models: These were averaged over the last 5 checkpoints, resulting in a single model that was trained on the entire dataset. 

3. Big models: These were averaged over the last 20 checkpoints, leading to a single model that was trained on the entire dataset.

- **Q14**: What is the BLEU score of the base model on English-to-German translation?
  - **Reason**: Missing keywords
  - **Answer Given**: The BLEU score of the base model on the English-to-German translation task is 28.4.

- **Q16**: How many training steps were used for the base model?
  - **Reason**: Missing keywords
  - **Answer Given**: The base model was trained for a total of 12 hours.

- **Q19**: Does the model use convolutional layers?
  - **Reason**: Missing keywords
  - **Answer Given**: Yes, the model uses convolutional layers. According to the context, "To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of sizer in the input sequence centered around the respective output position."

