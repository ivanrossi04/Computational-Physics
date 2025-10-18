## Study of the Convergence of Numerical Integration with Different Methods

Date: **October 17, 2025**

We want to verify the order of accuracy of the following integration methods:
- naive rectangles
- rectangles
- trapezoidal
- Simpson

We have seen in class that the (absolute) error of these methods varies with $N$ (number of samples) as:

$$
	{Err} = a \cdot N^{-n}
$$

where $n$ is the order of accuracy of the method. Taking the logarithm of both sides, we find:

$$
\log(\text{Err}) = \log(a) - n \log(N)
$$

Therefore, if we plot the variation of the error as a function of $N$ on a log-log scale, we expect linear trends whose slope is given by the order of accuracy $n$.

We want to create a simple Python program that computes:

$$
\int_0^1 e^x dx
$$

and calculates the error with respect to the exact result $e - 1$.

For $N$, the values $10, 10^2, 10^3, \ldots, 10^7$ will be used.

To speed up the calculation (which will take at most a few seconds), we will use numpy slicing functions and the `np.sum` method to sum all the elements of a vector.