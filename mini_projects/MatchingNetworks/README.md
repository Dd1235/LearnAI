# Notes on the paper [Matching Networks](https://proceedings.neurips.cc/paper_files/paper/2016/file/90e1357833654983612fb05e3ec9148c-Paper.pdf)

This paper is from 2017. (This feels crazy, it is technically 8 years old as of now but just the fact that you go from learning things from 500 years back to maybe a 100 years back for just the recent decade)

This paper was published before attention is all you need!!!! (it is also 2017)

## Abstract

Framework learns a networks that maps a small labelled support set, and an unlabeleed examples to its labels.

## Results

Key insights:
    - one shot learning is easier if you train network to do one shot learning. If you train it with say 50 images per bird, but only one photo per bird species while testing it wont workout.
    - parametric models have a fixed set of parameters like weights, one trained they don't change unless retrained
    - **doubt**: what about Siamese? Embedding function is parametric but the infernence log is non-parametric like
    - non parametric methods allow us to store and refer back to training data directly eg. KNN, store examples, on new input, look up the most similar past examples
    - Matching networks use non-parametric memory(the support set)
    - use an attention mechanism to compare the new input with the examples

## Some terms:

- say I have 10 images of fruits, each for each fruit say an apple, I have 3 examples, ie 3 images of apples, 3 images of bananas etc. And I have a query. Then this is 10 way 3 shot.

## Methodology


A human child looks at one picture of a giraffe, and would be able to recognize a giraffe, while our best deep learning systems needs many many examples. Hence one shot learning is a very "biologically" inspired idea like Reinforcement Learning.

In low-data regime, techniques like Data Augmentation (apply some transformation to data, like flip, rotate, etc), and Regularization (dropout, L2 weight penalty, early stopping) still lead to overfitting. 

Furthermore, learning is slow, and on large datasets, requiring many weights updates using SGD. Acc to authors, this is because of parametric aspect.

Matching Nets: a neural network that uses attention, and memory that enable rapid learning.
([] PS need to re read that attention is all you need paper because I have forgotten about this -_-)

---

### Interlude: Attention (yes I shall copy Three Easy Pieces Style)

Attention is a mechanism that lets a model **assign different weights** to different parts of the input when makning a decision or prediction.

- A query Q
- a set of Keys K
- a set of Values V

Attention(Q,K,V) = softmax(QK^T / root(dk))V
Scaled dot-product attention

Given this query, how similary is it to each key, then use hte similarity to weight the values. The output is a **wegihted sum of values** based on how relevant each key is to the query.

eg, imagine the keys as index of topics in the textbook, and values as the content of each topic. The query could be your question, and attention, compared the question to each topic, assins more weights to topics that match the question,and returns a summary of the most relevant pages/values.

1. Additive attention
2. Dot Product attention
3. Self attention: used in transformer. Here EVERY token loos at every other token and assigns weights to each one based on relevance, creating a new contextual embedding. So the word will geta  different meaning depending on the context. There are dozens of transformer layers each refining hte attention and understanding more deeply. When generating a response, it doesnt retrieve static answers, but generate one token at a time. For each token, apply attention over fully query + existing history, predict teh most likely token.
(Oh god, SO INCREDIBLY sohpisticated)

So Recommender Systems: attention over past user actions
Speech Recognition/ASR : attention over audio frames

Thoughts:
- Softmax over large matrices wasn't practical until better GPU's
- early models didn't make good embeddings for inputs

---

Training procedure: test and train condition must match
- show a few examples per class
- switch the task from minibatch to minibatch

### Model

-  Mapping \( \mathbf{S} \to c_s \)
- training strategy tailed for one-shot learning from the support set S

In Matching Networks, the **support set** \( \mathbf{S} = \{(x_i, y_i)\}_{i=1}^k \) is a set of labeled examples.

We define a mapping:


\( \mathbf{S} \mapsto c_s(x) = \sum_{i=1}^k a(x, x_i) \cdot y_i \)


Where:
- \( x \) is the **query input**
- \( a(x, x_i) \) is the **attention weight** (similarity between query and each support example)
- \( y_i \) is a one-hot label vector
- \( c_s(x) \) is the predicted label distribution for \( x \)


Imagine you are a student writing an open book exam. You have handwritten a bunch of stuff in your notes.

Now you see a question, and, in your head you assign a score  to each topic in your notes based on similarity (a(x,xi)) (consider a hypothetical situation wherein your notes consist of nicely formatted topic content pairs and not some illegible gibberish). 

Then you combine all the content from those topics weighted by how simiar they are, resulting a reasoned guess c_s(x)

Set-to-set framework 
Architecture model P(B|A) where they are both sets.

So in siamese:

Train it on some images of cats, and dogs, and learn to compare them. 
So during testing, I give an image of a cat and a dog (support set), and a query images of a dog, and model tells me that it is more similar to the second image.

In Matching Networks and Meta Learning:
- each training setp is a mini-task, Sample: a support set, a query set, for each query, compare against the small labelled supprt set, and predict the label of the query using attention-weighted label matching.

Testing: same format, examples and query. so here it is like you are give a mini notebook(different from the notebook you got in training), from my training, I remember a procedure on how to go through a notebook and retrieve an answer in en exam.

So the weights are computed per query via attention, in non-parametric models.



Equation (1): Matching Network Prediction

\[
P(\hat{y} \mid \hat{x}, \mathbf{S}) = \sum_{i=1}^{k} a(\hat{x}, x_i) \cdot y_i \tag{1}
\]

Where:

- \( x_i, y_i \) are the inputs and corresponding **label distributions** from the support set  
  \[
  \mathbf{S} = \{(x_i, y_i)\}_{i=1}^k
  \]
- \( a(\hat{x}, x_i) \) is an **attention mechanism**, described below.

---

What does Equation (1) do?

It computes the **prediction for a query input** \( \hat{x} \) by taking a **weighted sum** of the labels \( y_i \) from the support set, weighted by how similar each support item \( x_i \) is to the query.

This makes the output a **linear combination of known labels** — i.e., a "soft" classification based on similarity.

---

Interpreting \( a \) as a kernel

If the attention function \( a \) acts like a **kernel** over the input space \( X \times X \), then:

- Equation (1) behaves like a **Kernel Density Estimator (KDE)**.
- If \( a(\hat{x}, x_i) = 0 \) for all but the closest \( b \) support points (i.e., query attends only to its \( k - b \) nearest neighbors), then (1) acts like a **\( k \)-nearest neighbor classifier**.

This makes the Matching Network framework flexible enough to **subsumes both KDE and k-NN** methods, depending on how attention \( a \) is designed.

---

Associative Memory View

Another interpretation of (1):

- \( a \) behaves like an attention-based **key-value lookup**.
- \( x_i \): key  
- \( y_i \): value

You can think of this like a **hash table** or a **memory**:  
- Given a query \( \hat{x} \), the network “points to” the most similar keys \( x_i \) in the support set, and retrieves their values (labels).

This is similar to how **associative memory** works.

> ➤ The final classifier function \( c_{\mathbf{S}}(\hat{x}) \) is **not fixed** — it's recomputed for every new support set, making it **highly adaptable** to new tasks.

---

- Cosine distance for attention mechanism 

Look at the equations in 2.1 of the paper

- We are not just using a feedfoward layer that would look at the support set once
- We want multi-step iterative attention over memory
- This comes from ideas in **Memory Networks**
    - use a controller like LSTM vs GRU
    - Update beliefs abouts a query by repeatdly attending to teh suppotr set
    - LSTM combines last belief h_k-1, new evidence r_k-1 and original query f(query) to produce new belief h_k-1. Do this K times.
    - essentially you have your notes of past questions and answers. You are given a query/question to answer. You don't just over the book once, you read it again and again. And remember. this is where LSTM is helping you, remembering. so the controller devides how to combine current query + evidence. Updates refines the query embedding across attention steps.
- If you have 5 old qnas, looking at each in isolation to match the query is not good. You need to read all as a group. so you embed the query and the support items(eg from a CNN). Now we want to refine the query embedding in K reads. 
- Your input is the current version of the query, last query state, last read from the support. You update new hidden state and new memory cell.
- eq 3: add skip connection. Its a residual connection like in ResNets
- weight each support embedding using an attention score a, and sum to get context vector r_k-1.
- then compute attention weights, **softmax over cosine similarity** between refined query hk, support embedding g(xi). 

---
### Interlude: LSTM

LSTM Internals Refresher

Each LSTM unit tracks **two types of memory**:

| Name | Meaning |
|------|---------|
| \( c_t \) | **Cell state** (long-term memory) |
| \( h_t \) | **Hidden state** (short-term/output memory) |

They are updated using 3 gates:

| Gate | Purpose |
|------|---------|
| Forget gate \( f_t \) | What to **discard** from long-term memory |
| Input gate \( i_t \) | What **new information** to **store** |
| Output gate \( o_t \) | What to **expose** as current output |

---

Equations (standard LSTM, per timestep \( t \)):

Let’s say \( x_t \) is the input at time \( t \):

\[
\begin{align*}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad &\text{(forget gate)} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad &\text{(input gate)} \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad &\text{(output gate)} \\
\tilde{c}_t &= \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) \quad &\text{(candidate memory)} \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad &\text{(new cell state)} \\
h_t &= o_t \odot \tanh(c_t) \quad &\text{(new hidden state)}
\end{align*}
\]

---

In Matching Networks (Section 2.1.2), where does this come in?

When the paper writes:

\[
\hat{h}_k, c_k = \text{LSTM}(f'(\hat{x}), [h_{k-1}, r_{k-1}], c_{k-1})
\]

It’s doing this:

- Input:  
  - The query embedding \( f'(\hat{x}) \)  
  - The previous hidden state \( h_{k-1} \)  
  - The last attention readout from memory \( r_{k-1} \)

- Passes it through **a full LSTM cell**, with all the gates above.

So:
- The **forget gate** decides:  
  > “Should I discard anything from the previous belief?”

- The **input gate** decides:  
  > “Should I update my memory with the new attention vector \( r_{k-1} \)?”

- The **output gate** decides:  
  > “What should I output now as my updated query state?”

---

Why it fits perfectly here

Matching Networks use LSTM:
- Not to model a time sequence like video or text
- But to model **multi-step reasoning** about a query
- Where **attention guides what to look at**, and the LSTM decides **how to incorporate that**

---
Why not use just GRU?

GRUs are simpler (no separate cell state), but LSTMs are:
- More powerful for long-term dependencies
- More flexible in how much they retain or overwrite

Which can be useful when you're **thinking in multiple stages** over a task.

---

# More papers to checkout  (unrelated to this project)

ASR paper

## combining with RL 

https://arxiv.org/abs/1611.02779
https://arxiv.org/abs/1703.03400
https://arxiv.org/abs/1903.08254
