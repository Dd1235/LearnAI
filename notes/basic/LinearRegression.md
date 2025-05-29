# Linear Regression

## Simple Linear Regression

- For predicting a quantitive response Y on the basis of a single predictor variable X
$$
Y = \beta_0 + \beta_1 X + \epsilon$$
$$
- regressing Y on X
- $\beta_0$ is the intercept
- $\beta_1$ is the slope
- $\epsilon$ is the error term

- coefficents/parameters
- use training data to produce estimates for the model coefficients, can predict using
$$
\hat{Y} = \hat{\beta}_0 + \hat{\beta}_1 X
$$
- n pairs of observations
- minimize the least squares criterion
- e_i = Y_i - \hat{Y}_i is the ith residual
- RSS = $\sum_{i=1}^n e_i^2$ is the residual sum of squares
- We use calculus to show that the least squares estimates of the coefficients are given by
$$
\hat{\beta}_1 = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\sum_{i=1}^n (X_i - \bar{X})^2}$$
+- The least squares estimate of the intercept is given by
+$$
+\hat{\beta}_0 = \bar{Y} - \hat{\beta}_1 \bar{X}$$
$$

- epsilon is a mean-zero random error term

- we can use either the normal equantion or the gradient descent algorithm to estimate the coefficients
- gradient descent is an iterative optimization algorithm used to minimize the cost function
- we use mse loss because
- the cost function is convex, so gradient descent will converge to the global minimum
- it is differentiable
- penalizes outliers more than MAE
- gradient descent is more efficient for large datasets
- normal equation is more efficient for small datasets
- but we need to select the learning rate and number of iterations for gradient descent
- with normal equation it is O(mn^2+ n^3) for n features due to matrix inverison
- with gradient descent,it is O(kmn) 
