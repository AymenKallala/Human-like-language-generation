# Human-like Language Generation
Practical implementation of [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) from scratch.


## Overview

This project is a practical demonstration and implementation of the methods discussed in [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) from scratch, *Nucleus Sampling* being the main contribution of the paper.
The primary goal is to implement these methods to gain a deeper understanding of the underlying ideas.


## Decoding Methods

### Greedy Decoding
The first intuitive generation method that is also the least efficient.

The probability of generating the next token $\( y_t \)$ given the context of the previously generated tokens $\( y_1, y_2, \ldots, y_{t-1} \)$ is as follows (greedy way):

$\ P(y_t | y_1, y_2, \ldots, y_{t-1}) = \arg\max_{y_t} P(y_t | y_1, y_2, \ldots, y_{t-1}) \$

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

### Nucleus Sampling

## Results

## Experimental Setting

### Model Used

The experiments are conducted using a *GPT-2 large* model with approximately 774 M parameters. The model is pre-trained on a vast corpus of English text in a self-supervised fashion. The choice of this model is deliberate – a conscious decision to avoid overly "fancy" models, such as the latest chat models, in order to observe the impact of the generating methods without the influence of pre-existing biases towards "human-like" language generation.

### Prompt and Task

The experiments focus on a generative task using a single open-ended prompt: "The capital of France." This choice is made to minimize bias and evaluate the impact of decoding methods in generating meaningful text. Since GPT-2 has been trained on Wikipedia, it is expected to possess the necessary knowledge to provide interesting responses. The generation process is constrained by the `max_new_token` hyperparameter set to 100, ensuring that the generation stops either when the upper limit is reached or when the model generates the End of Sentence token.


## Usage

