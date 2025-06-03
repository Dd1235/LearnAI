# Notes on Fast Fourier Transform (FFT)

To do: 
- [x] Go through gilbrt strang's lecture on FFT

## Introduction

- Vandermonde matrix: we never actually compute it, but we use it to derive the FFT algorithm
- DFT calculation: O(n^2)
- FFT : O(nlogn)
- trivia: Guass actually discovered FFT and used it as a mental math trick, didn't consider it worthy to publish(welp no computers back in the day), but it was Cooley and Tukey who published it 100s of years later
- we use it to solve PDEs, denoising data, analysis of data, audio and iamge compression
- wavelet transform is build on top of FFT

## The algorithm

### FFT Matrix Representation

For the Fast Fourier Transform (FFT), the matrix can be divided into **even** and **odd** components(say n = 1024):

$$
\mathbf{f} =
\begin{bmatrix}
I_{512} & D_{512} \\
I_{512} & -D_{512}
\end{bmatrix}
\begin{bmatrix}
F_{512} & 0 \\
0 & F_{512}
\end{bmatrix}
\begin{bmatrix}
f_{\text{even}} \\
f_{\text{odd}}
\end{bmatrix}
$$

Where:
- \( I_{512} \) is the identity matrix.
- \( D_{512} \) is a diagonal matrix with twiddle factors \( \omega_n^k \).
- \( F_{512} \) represents the DFT of half the size.

---

Recursively applying the above formula, we can get the FFT algorithm.

## Some math 
- Innerproduct for complex numbers: y hermitian x
$ y^H x$
- Symmetric matrix: A = A^H (Hermitian matrices)
- Perpendicular q1, q2, ... qn
$$
q_i^H q_j =
\begin{cases}
0 & \text{if } i \neq j, \\
1 & \text{if } i = j.
\end{cases}
$$
- Unitary(<-orthogonal)matrix: Q^H Q= I
- $F_n$ has orthogonal columns
- It isn't orthonormal, they don't have length 1, just divide by length of each column to make it orthonormal, now Q^H Q = I
- ${w_n}^2 = {w_\frac{n}{2}}$
- Breakdown $F_n$ into this "fix up matrix" made of Identity and Diagonal matrices muliplied by matrix with $F_{\frac{n}{2}}$ and Os, multiplied to this permutation matrix