# Human-like Language Generation
Practical implementation of [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) from scratch.


## Overview

This project is a practical demonstration and implementation of the methods discussed in [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) from scratch, *Nucleus Sampling* being the main contribution of the paper.
The primary goal is to implement these methods to gain a deeper understanding of the underlying ideas.


## Decoding Methods

### Greedy Decoding
The first intuitive generation method that is also the least efficient.

The probability of generating the next token $\( y_t \)$ given the context of the previously generated tokens $\( y_1, y_2, \ldots, y_{t-1} \)$ is as follows (greedy way):

$$\ P(y_t | y_1, y_2, \ldots, y_{t-1}) = \arg\max_{y_t} P(y_t | y_1, y_2, \ldots, y_{t-1}) \$$

In other words, at each decoding step $\( t \)$, the model selects the token $\( y_t \)$ that maximizes the conditional probability given the context of the previously generated tokens.

This process is repeated iteratively until the model generates an end-of-sequence token or reaches a predefined maximum sequence length.

### Temperature Sampling

Temperature sampling involves adjusting the softmax function during the generation of sequences in natural language processing (NLP) tasks. The softmax function, which converts logits (unscaled log probabilities) into a probability distribution, is defined as:

$$ P(y_i) = \frac{e^{\frac{\text{logit}(y_i)}{T}}}{\sum_{j}e^{\frac{\text{logit}(y_j)}{T}}} \$$

where:
- $P(y_i)$ is the probability of selecting token $\(y_i\)$,
- $\\text{logit}(y_i)\$ is the logit (pre-softmax) value associated with token $\(y_i\)$,
- \(T\) is the temperature parameter.

The temperature parameter \(T\) controls the level of exploration in the sampling process. A higher \(T\) increases the diversity of the generated outputs, while a lower \(T\) focuses the sampling on more probable tokens. The temperature-scaled logits are exponentiated and normalized to obtain the final probability distribution used for sampling. Adjusting the temperature allows practitioners to fine-tune the balance between exploration and exploitation during sequence generation in NLP models.


### Top-K Sampling
 At each time step, the top k possible next tokens are sampled according to their relative probabilities, then we rescale the probabilities of these top k tokens and sample from them. The authors of the paper actually describe this method as inefficient and leading to too much repetition and non-flexibility regarding the context.

1. **Compute Probabilities:**
   The model computes probabilities for each word in the vocabulary based on the context: $P(w_t |contex)$.

2. **Sort Probabilities:**
   Sort the probabilities in descending order.

3. **Select Top-k Words:**
   Choose the top-k words with the highest probabilities.

4. **Normalize Probabilities:**
   Normalize the selected probabilities to create a distribution over the top-k words:
   $$P_{\text{top-k}}(w_t | \text{context}) = \frac{\text{top\_k\_words}}{\sum_{i=1}^{k} \text{top\_k\_words}_i}$$

5. **Sample from Distribution:**
   Sample a word from this distribution to obtain the next predicted word: $w_{t+1} \sim P_{\text{top-k}}(w_t | \text{context})$.


### Nucleus Sampling

## Results

## Experimental Setting

### Model Used

The experiments are conducted using a *GPT-2 large* model with approximately 774 M parameters. The model is pre-trained on a vast corpus of English text in a self-supervised fashion. The choice of this model is deliberate – a conscious decision to avoid overly "fancy" models, such as the latest chat models, in order to observe the impact of the generating methods without the influence of pre-existing biases towards "human-like" language generation.

### Prompt and Task

By providing the model with a lot of context, I hope to discriminate better *meaningful* generations from *senseless* ones.
Since GPT2 was trained on Wikipedia, it certainly has the required knowledge to tell us something interesting, the whole point is now to evaluate how impactful the decoding method is in generating a meaningful text. I fixed the $max\_new\_token$ hyperparameter to 100, note that the generation stops when it reaches the upper limit or when the model generates the End of Sentence token.
The generation process is constrained by the `max_new_token` hyperparameter set to 100, ensuring that the generation stops either when the upper limit is reached or when the model generates the End of Sentence token.

## Usage

