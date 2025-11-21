## Schrodinger's Equation

Date: **November 21, 2025**

Consider the time-independent Schrödinger equation in one dimension:

\[-\frac{\hbar^2}{2m}\nabla^2\psi(x) + V(x)\psi(x) = E\psi(x)\]

where V can be an infinite potential well:

\[ V_1(x) = \begin{cases} +\infty & \text{for } x < -a/2 \\ +\infty & \text{for } x > a/2 \\ 0 & \text{otherwise} \end{cases} \]

or a harmonic potential:

\[ V_2(x) = \begin{cases} +\infty & \text{for } x < -a/2 \\ +\infty & \text{for } x > a/2 \\ kx^2 & \text{otherwise} \end{cases} \]

Create a program that calculates the (first) eigenfunctions and (first) eigenvalues. We will choose ψ(x) ∈ ℝ.

For simplicity, use Hartree atomic units: ħ = 1, m = 1, e = 1. With this choice, the unit of energy is the Hartree (1 Hr = 27.2114079 eV) and the unit of length is the Bohr radius (1 Bohr = 0.529 · 10⁻¹⁰ m).

First, verify the correctness of the implementation by checking that the eigenenergies for the infinite potential well V₁ are equal to:

\[ E_n = \frac{\hbar^2\pi^2n^2}{8ma^2} \]

Use the `scipy.linalg.eigh_tridiagonal` function to solve the obtained eigenvalue problem. Try the program by using $a = 10$ Bohr and $N = 1001$ points, then check the first 5 eigenvalues.

For large n, the agreement with the analytical value worsens. Why?

Next, we will plot the first normalized wave functions ψ. To normalize them, we will first calculate their squared modulus using Simpson's method.

Then, we will try to calculate the eigenenergies and eigenstates with the potential V₂ using the same parameters as before and k = 1 a.u., and compare them with the analytical solutions for the harmonic oscillator:

\[ E_n = \hbar\sqrt{\frac{k}{m}}\left(n - \frac{1}{2}\right) \]
