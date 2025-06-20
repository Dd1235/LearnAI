# Attempt at Understanding Attention and Transformers/Notes

Plan:
[] understand properly including the math
[] Pytorch implementation
[] Write a "write up" explaining it in my own words

# This will be formatted later!

A transformer is a neural network
Go through a bunch of attention blocks and multi layer perceptron blobs.
Sample from the distribution.

"Embedding" a word : word -> vector.
E(woman) - E(man) = E(queen) - E(king) = E(neice) - E(nephew) = E(father) - E(mother)

E(hitler) + E(italy) - E(germany) = E(mussolini)
e(sushi) + e(germnay) - e(japan) = e(bratwurst)

so before attention -> Embedding
input = embedding + positional information


Embedding matrix: each column for each token
Total weights/parameters = embedding dimension * vocab size

They don't just represent individual words, that encodes information about the word. Like the word "King"'s embedding will be changed as you learn hes a King from the Chola dynasty vs when he is a King from a chess game.

We want ot "empower" a word to store context efficiently.

Context size: initially when you create an "array" out of your sentence, you just have the embedding of the word from the embedding matrix. Flows through networks, each vector evolves based on the context. You infer a lot about what a word means from surroundgin stuff, sometimes from a sentence many many sentences away.  "Context size" : limits how much text you can incorporate into making hte prediction.

Directed output: Probability distribution over the tokens that can occur next.

why use only the last embedding to make prediction? you use each one of those layers to make the next one.

Unembedding matrix: one tow for each word, as many elements as the embedding dimension. 

Softmax function: you want a sequence of numbers to act as a probability distribution, each value has to be between 0 and 1 and all have to add up to one.
But default outputs are not like this.
Softmax function: exp(x) / sum(exp(x))
Turn an arbitray vector into a probability distribution.

Temperature: add to the denominator of the exponents.
When T is larger you get more weight to the lower values.T = 0, all the weights goes to the maximum value.
T = 0, always goes to the max value, higher temperatoure, can choose more interesting, less prob ones.

Inputs: Logits
Outputs: Probabilities

You need to make sure the final word should have the full context. Imagine ..........., hence the murder was ___. You need the full context.

Single *head* of attention. Also encode the position of the word.
Series of computations -> more refined set of embeddings.

Query: another vector, but much smaller dimension

Wq * embedding = Q (transform the embedding to a lower dimension)

Query: embedding space -> query key space

Embeddings of ... "attend to" the embedding of ...

Embeddings of fluffly and blue attend to hte embedding of creature
Grid gives you a score , take a weighted sum in a column, compute a softmax along a column to normalize these values.

Attention pattern: Attention(Q,K,V) formula
divide by square root of dimension fo the key query space and apply softmax on each

Never allow later words influence the earlier ones.
Can't just set to zero so set all these to -inf before applying softmax - *masking*

Sparse attention mechanism
Multiply value matrix with embeddings to get valeu vectors, multply each value vector by the corresponding weights adn update the embedding.

# value params = # query params + # key params
esp in terms of multi head 

break into two matrices.
constraining hte value map to be a low rank transformation

Slef-attention head vs Cross attention head

cross attention: two types of data. Text in one language to one to another
Key and Query on different data
These remind me of on policy and off policy methods in RL

They crashed the car, there are a lot of implicaitons on the structure of the car.
Wizerd.. hogwards...hermoine, .... harry
vs 
sussex...queen...prince...harry

Since attention head, a full attention block has multi headed attention, each having their own Wk, Wq, Wv

The value map is factored into two matrices, value up and value down.
In papers, all the values up matrices from all the heads, output matrix.
Data flowing through a transformer goies through many copies of attention block and a multi layer perceptron.

d_query*d_embed*n_heads*n_layers

100s of billions of parameters
# 175 billion parameters

How do LLMs store facts


Vectors like in a high dimensinoal space.
embibe a much richer meaning

What is inside the MLP?

Each individual vectors -> series of operations. Linear ReLU, Linear, This all happens in parallel.

Multiply the vector by a very big matrix (Linear) and add a bias.
Pass the large intermediate vector to a non Linear function, eg ReLU
Mimics the behavious of the and gate.
Gelu.

Neurons of a transformer, combination of a linear step and a term wise non linear function.


Essentially, take a linear combination of the columns of the matrix and add the bias term. The columns are the weights of the neuron. 

Johnson-Lindestraus Lemma

independent ideas -> almost perpendicular, so model performance scales very well with size.

# Interlude?

simpsons paradox

Treatment A vs Treatment B
People who got treatment B survived more than people who got treatment A.
But patients with smaller kidney stones did better with treatment A.
Patients with larger kidney stones did better with treatment A.
How come?
not properly randomized. Moajority of patients with larger stones were given treatment A, and majority of patients with smaller stones were given treatment B.

cross entropy, one distribution to another.
it doesn't commute
measure of represetning how much two distributions differ.
-> KL divergence

# Prereq

---
#  Attention Mechanisms – Quick Reference

## Input

Let input tokens be:

$$
X = [x_1, x_2, ..., x_T], \quad x_i \in \mathbb{R}^d
$$

## Linear Projections

Project input into **Query, Key, Value** vectors using learned matrices:

$$
Q = X W_q \quad (T \times d_q) \\
K = X W_k \quad (T \times d_k) \\
V = X W_v \quad (T \times d_v)
$$

Where:

* $W_q \in \mathbb{R}^{d \times d_q}$
* $W_k \in \mathbb{R}^{d \times d_k}$
* $W_v \in \mathbb{R}^{d \times d_v}$

## Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{Q K^T}{\sqrt{d_k}} \right) V
$$

* Input: $Q, K \in \mathbb{R}^{T \times d_k}, V \in \mathbb{R}^{T \times d_v}$
* Output: $\mathbb{R}^{T \times d_v}$

## Multi-Head Attention (MHA)

This was introduced in AIAYN.
Instead of Q,K,V, you have say h heads, so h weight matrices per Q,K,V, and you instead get h outputs from the scaled dot product attention that are then concatenated. This is also a better approach, and captures more of the subspace of the input.

Split Q, K, V into $h$ heads of size $d_k = d/h$:

$$
\text{head}_i = \text{Attention}(Q_i, K_i, V_i) \quad \in \mathbb{R}^{T \times d/h}
$$

$$
\text{MHA}(X) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W_o \quad \in \mathbb{R}^{T \times d}
$$

* $W_o \in \mathbb{R}^{d \times d}$ projects back to original dimension.

## Self vs Cross Attention

| Type            | Q from | K, V from | Used In                      |
| --------------- | ------ | --------- | ---------------------------- |
| Self-attention  | X      | X         | Encoder/Decoder              |
| Cross-attention | Y      | X         | Decoder attending to Encoder |

## What Q, K, V Represent

| Component | Meaning                                | Role in Attention                         |
| --------- | -------------------------------------- | ----------------------------------------- |
| Q         | What this token wants to know          | Matches against K to get relevance scores |
| K         | What this token offers                 | Controls *who gets selected*              |
| V         | What this token provides when selected | Final information passed on               |

---

Position embeddings are calculated using sine and cosine functions based on even and odd indices of the token position. This allows the model to learn relative positions of tokens in a sequence.

Each encoder layer has MHA and FFN. Each sublayer is in a residual block.

Input: PE + TE

x + MHA(x) -> layer normalization -> position wise FFN
Output passed tot eh next encoder layer and hte adjacent decoder layer. Each decoder has a multi head self attention, and multi head cross attention.

Masked Self-Attention -> don't attend to the future tokens.
Cross-Attention: K,  from adjacent encoder, Q from preceding MH self attention sub layer.

Adam optimizer with warmup stage (4000 steps). LR increases for first 4000 steps and then decreases gradually.


For every word you got an embedding vector. Which will be multiplied to the three matrices. Same for every input. 


Now say you got a sentence "The bank of the river was flooded", you wanna make sure that "bank" seems to be motivated by the words "river" and "flooded", so what you do is you take the query vector for "bank" (you got this by multiplying the embedding vector bank with Wq), now multiply with the key vector of every other word in the sentence, so that is a dot product giving you a score. Now the score will be higher for "river" and less for the CLS token, and "the" etc. Now bank is gonna have an attention score for every word in the sentence, so we know which words to focus on most. 

Value vector learns/knows what each word means. The final MatMul tells you that the "bank" is not a financial institution but a river bank, because of the attention scores. So you get a context aware embedding of "bank".

X - Txd
Wq - dxd
Wk - dxd
Wv - dxd
Q - Txd
K - Txd
V - Txd

take the 1xd query row, corresponding to "bank"
[.....(bank)][k1 k2 .. kT] = [score_for_word1, .. . score_for_wordT] - 1xT
K^T - dxT
QK^T - 1xT 

[score1, .. scoreT] x V - 1xT x TxD = 1xD

for its like for each input dimension, you get weighted sum of all the words weighted by their score.


consider the first column of V, this has one value for each of the T words, that get weighted by the scores.

Query is like asking a question. It projects the embedding space to a much smaller space, something like an embedding space of 12,288 to 128, where a noun might ask, "Hey are there any adjectives near me?", and the Key matrix as potentionally answering these questions. If the keys of "river" and "flooded" attend to the query of the "bank", then these dot products will be high. Now for each query, you get a score for each key produced by each word, so like a score for each word. So you add the  value vectors of each word weighted by the scores, and you get a new vector for "bank" that is context aware, and has information about the words "river" and "flooded".

