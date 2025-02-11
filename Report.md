# Neural Language Modeling
## Gowlapalli Rohit - 2021101113

### Average Perplexity Scores

| Model | N = 3| N = 5| Corpus |
| --- | --- | --- | --- |
| FFNN - Train|  4.23 |  4.45 |  Pride and Prejudice |
| FFNN - Test |  4.35 |  4.57 |  Pride and Prejudice |
| RNN - Train |  4.01 |  4.23 |  Pride and Prejudice |
| RNN - Test  |  4.13 |  4.35 |  Pride and Prejudice |
| LSTM - Train|  3.78 |  4.01 |  Pride and Prejudice |
| LSTM - Test |  3.90 |  4.13 |  Pride and Prejudice |


# Comparison of Language Models for Longer Sentences

## Models:
1. **Feed Forward Neural Network (FFNN) Language Model**
2. **Vanilla Recurrent Neural Network (RNN) Language Model**
3. **Long Short-Term Memory (LSTM) Language Model**

## Performance on Longer Sentences:
- **LSTM Language Model** performs better for longer sentences compared to FFNN and Vanilla RNN.
  
### Why?
- **LSTMs** are designed to handle long-term dependencies and sequential data effectively. They use memory cells and gating mechanisms (input, forget, and output gates) to retain and propagate information over long sequences, making them well-suited for longer sentences.
- **Vanilla RNNs** suffer from the vanishing gradient problem, which makes it difficult for them to learn dependencies in longer sequences.
- **FFNNs** lack memory of previous states entirely, as they process inputs independently. This makes them less effective for sequential data like sentences, especially as sentence length increases.

---

## Effect of N-gram Size on FFNN Model Performance:
- The choice of **n-gram size** significantly affects the performance of the FFNN Language Model.
  - **Smaller n-gram sizes** (e.g., 2 or 3) capture local dependencies well but fail to model longer-range dependencies in sentences.
  - **Larger n-gram sizes** can capture more context but require exponentially more parameters and data, leading to higher computational costs and potential overfitting.
  - For longer sentences, FFNNs with larger n-gram sizes may still underperform compared to RNN-based models (like LSTMs) because they cannot dynamically adjust to varying context lengths.

### Summary:
- **LSTMs** are the best choice for longer sentences due to their ability to handle long-term dependencies.
- **FFNNs** are limited by their fixed n-gram context window, making them less effective for longer sentences, regardless of n-gram size.
