---
# Bayesian Classification: Salmon vs Catfish Example 🐟

We want to classify a fish from a lake as **salmon** or **catfish**, based on its observed features.
---

## 1. Setup

- **Classes:**
  $ w_1, \ldots, w_n $ are the possible classes.
  In our case:
  $
  w_1 = \text{salmon}, \quad w_2 = \text{catfish}
  $

- **Observed data:**
  ( x \in \mathbb{R}^d ) is the observed feature vector (e.g., length, weight, color intensity, fin shape).

---

## 2. Bayes’ Theorem for Classification

$
P(w_j \mid x) = \frac{p(x \mid w_j) P(w_j)}{p(x)}
$

- **Prior ( P(w_j) ):**
  The probability of class ( w_j ) before seeing the data.
  Example:
  $
  P(\text{salmon}) = 0.9, \quad P(\text{catfish}) = 0.1
  $
  (perhaps because salmon are much more common in the lake).

- **Likelihood ( p(x \mid w_j) ):**
  The probability of seeing features ( x ), given class ( w_j ).
  Typically modeled using a **multivariate Gaussian**:
  $
  p(x \mid w_j) = \mathcal{N}(x; \mu_j, \Sigma_j)
  $

- **Evidence ( p(x) ):**
  The probability of the observed data under all classes:
  $
  p(x) = \sum_{j} p(x \mid w_j) P(w_j)
  $

- **Posterior ( P(w_j \mid x) ):**
  The probability of the fish being class ( w*j ), \_after* observing its features.

---

## 3. Decision Rule

We assign the input vector ( x ) to the class with the highest posterior probability:

$
\hat{w} = \arg\max_j P(w_j \mid x)
$

This is called the **Maximum A Posteriori (MAP)** classifier.

---

## 4. Extra ML Insights

- If we assume **equal priors** (( P(w_j) = \frac{1}{n} )), then this reduces to the **Maximum Likelihood (ML)** classifier:
  $
  \hat{w} = \arg\max_j p(x \mid w_j)
  $

- If we assume **Gaussian likelihoods with equal covariance** (( \Sigma_j = \Sigma )), the decision boundary becomes **linear** (Linear Discriminant Analysis, LDA).

- If covariance matrices differ, the decision boundary becomes **quadratic** (Quadratic Discriminant Analysis, QDA).

- This is the foundation of **Naïve Bayes classifiers**, which assume independence among features:
  $
  p(x \mid w_j) = \prod_{k=1}^d p(x_k \mid w_j)
  $

✨ **Summary:**
We use Bayes’ theorem to combine **prior knowledge** (how common each fish is) with **likelihoods** (feature distribution given class) to compute a **posterior probability**. The MAP rule then chooses the class with the highest posterior — giving us a principled way to classify new observations.

---

# Gaussian Likelihoods and LDA

## 1. Likelihoods with Gaussians

When we say:

$
p(x \mid w_j) = \mathcal{N}(x; \mu_j, \Sigma_j)
$

it means:

- Each class ( w_j ) (salmon, catfish) has its **own Gaussian distribution** of features.
- That Gaussian is described by:

  - **Mean vector** ( \mu_j ) (average features of salmon, or catfish).
  - **Covariance matrix** ( \Sigma_j ) (how features vary together, e.g. length and weight correlation).

👉 Example:

- Salmon might cluster around **(length=30cm, weight=4kg)** with some spread.
- Catfish might cluster around **(length=40cm, weight=5kg)** with its own spread.

So in feature space, each class is an **elliptical blob** (multivariate Gaussian).

---

## 2. Equal Covariance Assumption

If we assume **equal covariance matrices across classes**:

$
\Sigma_1 = \Sigma_2 = \cdots = \Sigma
$

that means:

- All classes (salmon and catfish) have **similar shape and spread** of their data distribution.
- The only difference is **where the blobs are centered** (their means).

👉 Geometrically:

- Both salmon and catfish ellipses have the same size/orientation, just shifted in space.

---

## 3. Why Does That Matter?

- The **decision boundary** (the surface separating salmon vs catfish) comes from comparing:

$
\log P(w_1 \mid x) \quad \text{vs.} \quad \log P(w_2 \mid x)
$

- With **equal covariance**, the quadratic terms cancel out.
- So the boundary is **linear** (a straight line or hyperplane).

This classifier is called **Linear Discriminant Analysis (LDA)**.

---

## 4. If Covariances Differ

- If ( \Sigma_1 \neq \Sigma_2 ), then the quadratic terms remain.
- Decision boundaries are **curved (ellipses, parabolas, etc.)**.
- That classifier is **Quadratic Discriminant Analysis (QDA)**.

---

## 5. Visual Intuition

Imagine feature space is 2D:

- **Equal covariance (LDA):**
  The boundary between salmon and catfish is a **straight line** that best separates their blobs.

- **Different covariance (QDA):**
  The boundary bends to account for different spreads — like a **curved boundary** wrapping around one class.

---

## 6. Connection to Practice

- **LDA is simpler:** just means + one shared covariance estimate.

- Works well when data for different classes has similar spread.

- Example: Spam vs Ham emails, where word frequencies are roughly Gaussian with similar variance.

- **QDA is more flexible but needs more data** (since you estimate a covariance matrix for each class).

- Can overfit if you don’t have enough samples.

---

✨ **Summary in Plain Words**

- Each class is modeled as a Gaussian blob.
- If you assume they all have the **same spread/shape** (equal covariance), the dividing boundary is **straight → LDA**.
- If they have **different spreads/shapes**, the dividing boundary is **curved → QDA**.

---
