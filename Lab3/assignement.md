## Integral equation solving: Bisection and Newton–Raphson methods

Date: **October 24, 2025**

We want to solve the equation:

$$
\int_0^x f(y)\,dy = C
$$

where $f$ and $C$ are given. The integral will be evaluated numerically using the Simpson method.

Methods:

1. Bisection
2. Newton–Raphson (the user chooses the initial guess)

Test functions:

- $f(y) = \sin(y)$
- $f(y) = e^y$
- $f(y) = e^{y^2}$

Build a program that prompts the user to choose a function between the three choices, then computes the root of the function based on the additional input required for the two methods (for method 1: {$a$, $b$, $\epsilon$}, for method 2: {$x_0$})

Verification case (for $\sin$): choose $C = 1$, search in $[0.5, 2]$, and verify that the solution is $x = \tfrac{\pi}{2}$. Target accuracy: $10^{-12}$.