# Loss


The loss function we use naturally arises from the MLE for the parameters.
(fill in the theory here or paste pictures)

MLE:
Go through Probability Course's chapter
L(parameters | data) = P(data | parameters)


For Linear Regression
- You can model it as y = mx + e, where the error follows guassian distribution. It is essentially guassian noise. So the MLE leads to the squared loss. Laplace noise leads to MAE. Posissin Noise will lead to poisson log loss.

For Logistic Regression
- You model is as a Bernoulli distribution, so the MLE leads to the cross entropy loss.
- y = sigmoid(wx + b)
- P(Y=y | X=x) = y * P(Y=1 | X=x) + (1-y) * P(Y=0 | X=x)
- So the likelihood function directly leads to the log loss / BCE loss
- gives you the probability that y = 1 given x (conditional probability)

- Another reason is wanting a convex function where we do not get stuck in local minima.
- For logistic regression, consider y = 0 and y = 1 and plot both the loss functions. Notice that if you use MSE vs log loss, we penalize the misclassifications much more. The gradient is steeper for log loss. For regression we want smaller gradient, y = 10 yhat = 8 is okay. It should not be penalized too much. But for classificaiton y = 1, yhat = 0 is a big problem. So we want to penalize it more. Log loss does that.
- Also for regression cannot use something like log loss as y lie in range (-inf, inf) while y = {0,1} 


- also we use RMSE and R2 for accuracy of Linear Regression
- we use confusion matrix, roc, auc for Logistic regression metric

- Linear Regression carees about numerical closeness. 
- Logistic, the correctness of the class label.

These things can also be seen from a KL divergence perspective.

Bayesian definition: Probability is viewed as the as a subjective degree of belief or confidence

there is a 70% probability it rains tomorrow. 0.7 represents the measure of confidence rather than a frequency

Sampling: generative outcomes from a probability distribution

[watch](https://www.youtube.com/watch?v=KHVR587oW8I&t=153s)


entropy- measure of uncertainty/surprise in a probability distribution. When probabilities multiply, "surprise" adds up.


H = -Σ P(x) log(P(x))

Internal model (our belief about the distribution) is used to assign probabilities and compute surprise.

Even a die is not a uniform distribution. There are microscopic imperfections in its sturcture and air resistance. There is a tiny probability that it lands on a side or that it disintegrates in thin air.

So we approximate it with our beliefs and assign probabilities to each outcome.

What if the internal model differs from the true distribution? We can use KL divergence to measure the difference between the two distributions.

Cross-Entropy : average surprise you will get by observing a random variable governed by distribution P while believing in its model Q

H(P,Q) = -Σ P(x) log(Q(x))

P: outcome probabilities, and Q for the surprisal term
ps : how often the state s is observed
qs: how surprised you will be to see it.

P = Q: H(P,Q) = H(P)
For any model, the cross-entropy can never be lower than the entropy of the underlying generating distribution.

H(P,Q) >= H(P)

This is because the model Q can never be more accurate than the true distribution P. The model can only approximate the true distribution, so the cross-entropy will always be greater than or equal to the entropy of the true distribution.

P and Q do not commute, so H(P,Q) != H(Q,P)

KL divergence:

Isolate the surprise from models inaccuracy and not the inherent uncertainty of the distribution.

H(P,Q) - H(P) = D_KL(P || Q)
D_KL(P || Q) = Σ P(x) log(P(x)/Q(x))

Measures the surprise of using model Q when underlying distribution is P.

Suppose you want to make an image of a cat. You cannot sample from all the cat images in the world. So you want to bulid a model that approximate with a computationally tractable model.

Optimize the parameters of the neural network iwth gradient descend to approixmate the distribution of training data.

minimize D_KL(P_data ||  Q_model)

Want to find Q that minimize D_KL(P_data || Q_model)
H(P) is a constant, so we can ignore it. The parameters of a neural network do not vary H(P). So we can just minimize the cross entropy H(P,Q).


For linear regression, y ~ yhat + e, where e ~ N(0, sigma^2)
so y ~ N(yhat, sigma^2)