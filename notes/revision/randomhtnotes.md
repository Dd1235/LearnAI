Reading along [this](https://cloud.google.com/blog/products/ai-machine-learning/hyperparameter-tuning-cloud-machine-learning-engine-using-bayesian-optimization)

- Grid search suffers from the curse of dimensionality while random search does not make use of prior knowledge.

- Guassian process bandits: forms of Bayesian optimization
- Idea is to commpute a posterior distribution over the objective function.
- Comes from Guassian process regression (GPR) and describes the predictive distribution of the new target point y_t+1 at a test point x_t+1 given the data observed so far up to time t, denoted by D_t.
- Dt = {(x_1, y_1), (x_2, y_2), ..., (x_t, y_t)}
- The predicted value y+t+1 follows a normal distribution with mean $ \mu_t(x_{t+1}) $ and variance $ \sigma^2_t(x_{t+1}) + \sigma^2_noise $.
- The firts term in variance is the model uncertainty, and second term the intrinsic noise from the observations assumed to be Gaussian iid.

Given data \( D_t = \{(x_1, y_1), \dots, (x_t, y_t)\} \), the predictive distribution of the next observation \( y_{t+1} \) at a new input \( x_{t+1} \) under a Gaussian Process is:

\[
P(y_{t+1} \mid D_t, x_{t+1}) = \mathcal{N}\left(\mu_t(x_{t+1}), \sigma_t^2(x_{t+1}) + \sigma_{\text{noise}}^2\right)
\]

- \( \mu_t(x_{t+1}) \): predicted mean from the GP posterior
- \( \sigma_t^2(x_{t+1}) \): predictive variance (model uncertainty)
- \( \sigma_{\text{noise}}^2 \): observation noise variance (independent of the model)

This shows that the total uncertainty in prediction combines both the model's uncertainty and inherent data noise.

- acquisition functions provides a single measure of how useful it would be to try any given point

UCB:


- $ \alpha(x) = \mu_t(x) + \beta \sigma_t(x) $
- By varying beta we can encourage the algorithm to explore or exploit more.

- MPI (Maximum Probability of Improvement) is a common acquisition function.

- there are some more, look at the article.


# Optuna


