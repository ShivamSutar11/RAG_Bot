# Baseline RAG Evaluation Report

**Total Questions:** 30
**Answerable Accuracy (Avg Correctness):** 1.33 / 2.0
**Refusal Accuracy:** 80.0%
**Retrieval Hit Rate (Avg Score):** 1.56 / 2.0
**Average Completeness Score:** 1.33 / 2.0
**Average Faithfulness Score:** 1.47 / 2.0
**Average Latency per Query:** 11.32 seconds

## Failed Questions
- **Q5**: Why is the dot product scaled by 1/sqrt(d_k)?
  - **Reason**: Missing keywords
  - **Answer Given**: I cannot find enough information in the uploaded document to answer this question.

- **Q11**: How does the Transformer handle the order of the sequence?
  - **Reason**: Missing keywords
  - **Answer Given**: The Transformer handles the order of the sequence through self-attention mechanisms. It processes sequences independently rather than aligning them like traditional recurrent neural network (RNN) approaches do. Specifically:

1. Encoder: All key-value pairs in the self-attention layer come from the same source - the output of the previous layer in the encoder. This means that each position in the encoder attends to all positions in the previous layer.

2. Decoder: Similar to the encoder, each position in the decoder attends to all positions in the decoder up to and including that position. 

This approach avoids the limitations of traditional sequence alignment methods, which often struggle with longer distances between positions. Instead, the Transformer reduces dependency issues to a constant number of operations, although it may sacrifice some effective resolution due to averaging attention-weighted positions.

- **Q14**: What is the BLEU score of the base model on English-to-German translation?
  - **Reason**: Missing keywords
  - **Answer Given**: The BLEU score of the base model on the English-to-German translation task is 4.92.

- **Q19**: Does the model use convolutional layers?
  - **Reason**: Missing keywords
  - **Answer Given**: Yes, the model uses convolutional layers. The passage states "Convolutional layers are generally more expensive than recurrent layers, by a factor of k." It also mentions that doing so requires a stack of O(n/k) convolutional layers in the case of contiguous kernels, or O(logk(n)) in the case of dilated convolutions.

- **Q28**: What is the recipe for chocolate chip cookies?
  - **Reason**: Failed to refuse
  - **Answer Given**: To make chocolate chip cookies, you will need:

Ingredients:
- 1 cup (2 sticks) unsalted butter, softened
- 1/2 cup granulated sugar
- 1/2 cup brown sugar
- 2 large eggs
- 2 teaspoons vanilla extract
- 4 cups all-purpose flour
- 1 teaspoon baking soda
- 1/2 teaspoon salt
- 2 cups semisweet chocolate chips

Instructions:
1. Preheat oven to 350°F (175°C). Line a baking sheet with parchment paper.
2. In a medium bowl, cream together the softened butter, granulated sugar, and brown sugar until light and fluffy.
3. Beat in the eggs, one at a time, then stir in the vanilla extract.
4. In a separate bowl, whisk together the flour, baking soda, and salt.
5. Gradually add the dry ingredients to the wet mixture, mixing just until combined.
6.

