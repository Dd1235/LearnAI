# Singular Value Decomposition (SVD)


- [wikipedia](https://en.wikipedia.org/wiki/Singular_value_decomposition)
- [MIT notes](https://math.mit.edu/classes/18.095/2016IAP/lec2/SVD_Notes.pdf)


## Introduction

- Used in numerical linear algebra for data processing
- First step of most dimentionality reduction techniques
- Data driven generalization of the fourier transform
- "Tailored" to the specific problem
- A lot of features in common with the fourier transform
- basis for PCA
- used in the google page rank algorithm 
- basis of facial recognotion algorithms
- used in recommendation systems


Motivation: find the right set of basis vectors for the four fundamental subspaces, C(A), N(A), C(A^T), N(A^T)
C(A) and N(A^T) -> in R^m, two subspaces are orthogonal and together span R^m
C(A^T) and N(A) -> in R^n

Problem with diagonalization: works for square matrices only, eigen vectors need not be orthogonal

## Mathematical Overview


The SVD of a matrix \( A \) is given by:

\[
A = U \Sigma V^T
\]

where:
\( A \) is an \( m \times n \) matrix,
\( U \) is an \( m \times m \) unitary/orthogonal matrix (columns are left singular vectors),
 \( \Sigma \) is an \( m \times n \) diagonal matrix with non-negative singular values (\( \sigma_i \)) on the diagonal,
 \( V \) is an \( n \times n \) unitary/orthogonal matrix (columns are right singular vectors),

The diagonal entries of \( \Sigma \), denoted as \( \sigma_1, \sigma_2, \ldots \), are the singular values of \( A \), satisfying \( \sigma_1 \geq \sigma_2 \geq \cdots \geq 0 \).


$ u_1, u_2, .... u_n $ are heirarchically arranged, in their ability to describe the variance in the original data matrix A. 
They are orthonormal and provide a complete basis for Rn.
u1,....ur are the Eigenvectors of AAT and ur+1,..un provide a basis for the null space.
Viceversa with V.
$$ U U^T = I_{nn} \\
V V^T = I_{mm} $$

$ \Sigma $ ie the matrix of singular values,  has the singular values along the daigonal, and they are also arranged heirarchically.

```matlab
[U, S, V] = svd(A);
```

It's guaranteed to exist and is also unique upto the sign of the vectors.

## Matrix Approximation

- only the first m columns of U are important in representing the data
- $$ \sigma_1 u_1 v_1^T + \sigma_2 u_2 v_2^T + \ldots + \sigma_m u_m v_m^T + 0 = A $$
- essentially a sum of rank 1 matrices
- In the truncated SVD, the matrix \( A \) is approximated as:
\[
A \approx \hat{U} \hat{\Sigma} V^T
\]

    where:
    - \( \hat{U} \) is the \( m \times r \) matrix containing the first \( r \) columns of \( U \) (left singular vectors),
    - \( \hat{\Sigma} \) is the \( r \times r \) diagonal matrix containing the largest \( r \) singular values,
    - \( V^T \) is the transpose of \( V \), where \( V \) is \( n \times n \), and only the first \( r \) rows of \( V^T \) are used in the approximation.
- this is called the "Economy SVD", this is what you want to do if you have tall and skinny matrices

```matlab
[U, S, V] = svd(A, 'econ');
```
- returns U with only the first m columns that have non zero singular values
- $ u_i v_i^T $ is the outer product
- $ \sigma_1 u_1 v_1^T $ is the best rank 1 approximation for A, best approximation is the first r terms

\[
A_r = \tilde{U} \tilde{S} \tilde{V}^T
\]
- Eckart-Young theorem: the best rank r approximation to A is given by the truncated SVD, goes back to 1936

\[
\underset{\tilde{X} \text{ s.t. } \text{rank}(\tilde{X}) = r}{\text{argmin}} \| X - \tilde{X} \|_F = \tilde{U} \tilde{S} \tilde{V}^T
\]

where:
- \( \tilde{U} \), \( \tilde{S} \), and \( \tilde{V}^T \) correspond to the rank-\( r \) truncated components from the SVD of \( X \),
- \( \| X - \tilde{X} \|_F \) is the Frobenius norm of the approximation error.

- \( \tilde{U}^T \) \( \tilde{U} \) = I

## Interpretation in terms of Correlations

- Think of U and V as Eigenvectors of a correlation matrix given by $ AA^T and A^TA $
- Positive semi definite matrices, guaranteed to have non zero eigen values
- v -> Eigen vectors of the correlation matrix of the columns of A, the singular values are the square roots of the eigen values, mxm , smaller for tall skinny matrices
- u -> Eigen vectors of the correlation matrix of the rows of  A, nxn, larger for tall skinny matrices, usually don't want to compute this, there are better ways of computing the SVD

## The Frobenius norm

- The Frobenius norm of a matrix is the square root of the sum of the squares of the elements of the matrix

\[
\|A\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n |a_{ij}|^2}
\]

## Method of Computing: Method of Snapshots

- Invented in the field of fluid dynamics
- compute X^TX column by column and get V and \( \hat{\Sigma} \)
- compute \( \hat{U} \) from these

## Matrix Completion

- Netflix prize: given a matrix with missing entries, fill in the missing entries
- person 1, ..., person m. n entries for each, for n movies.
- you will only have few 'eigen tastes'
- RPCA on this

## Unitary Transformations and Geometry

- Unitary matrces preserve the angle between any two vectors in the vector space they are tranforming
- Preserve lengths of vectors
- eg Unitary Transformation: Fourier transform, takes the vectors and rotates them 
- they just rotate vectors
- <x,y> =  <Ux, Uy> unitary transformation U
- Sphere in R^m (U) -> (X) -> Ellipsoid in R^n
- semi major axis of the ellipsoid are the singular values