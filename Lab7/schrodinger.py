# !V1

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def main():

    # Insert the problem data -----------------------------

    # define potential limits
    a = -5 # Bohr
    b = -a # Bohr

    # Number of steps (must be odd to apply simpson's rule for integration)
    N = 1001
    step = 2 * b / (N - 1)

    # Number of eigenvalues to compute
    eigen_computed = 20

    # Define the potential function V(x) (infinite potential well)
    def V(x):
        if abs(x) <= b : return 0
        else: return np.inf

    # Compute the theoretical energy values for the first 10 states
    energy_values = np.array([ (np.pi**2 * n**2) / (8 * b**2) for n in range(1, eigen_computed + 1)])
    print("Theoretical energy values \n", energy_values)
    
    # -----------------------------------------------------

    x = np.linspace(a, b, N)

    # Initialize the Solving Matrix M (could just use two np.arrays since it is tridiagonal)
    M = np.zeros((N-2, N-2))

    # Populate the Solving Matrix M with the needed values from the approximation method (finite differences)
    for k in range(N - 2):
        if(k > 0): M[k][k - 1] = -1/(2 * step**2)       # Filling lower tridiagonal cells 
        M[k][k] = 1 / (step**2) + V(x[k + 1])           # Filling diagonal cells
        if(k < N - 3): M[k][k + 1] = -1/(2 * step**2)   # Filling upper tridiagonal cells

    print("Solving Matrix M \n", M)

    # Solve the eigenvalue problem using a tridiagonal matrix solver
    eigenval, eigenvec = sp.linalg.eigh_tridiagonal(np.diag(M), np.diag(M, k = 1), select='i', select_range=(0, eigen_computed - 1))
    print("Computed energy_values \n", eigenval[0:eigen_computed])
    
    
    plt.figure()
    # only plotting the first three eigen functions for visual clarity, it is possible to visualize the others in the same way
    for phi_n in eigenvec.T[0:3]:
        phi_n_sq = phi_n**2

        # Compute the normalizing factor using Simpson's rule
        normalizing_factor = phi_n_sq[0]**2 * step / 3 + np.sum(phi_n_sq[1:-1:2] * step * 4 / 3) + np.sum(phi_n_sq[2:-1:2] * step * 2 / 3) + phi_n_sq[-1] * step / 3
        
        # Plot the normalized eigenfunction
        plt.plot(x[1:-1], phi_n / np.sqrt(normalizing_factor))
    
    plt.legend()
    plt.title("First three normalized eigenfunctions")
    plt.xlabel("X")
    plt.ylabel("phi(x)")
    plt.show(block=True)

    plt.figure()
    plt.plot(range(1, eigen_computed + 1), energy_values - eigenval[0:eigen_computed], 'o', label='E_true - E_computed')
    plt.legend()
    plt.title("Error between theoretical and computed energy values")
    plt.xlabel("State")
    plt.ylabel("Energy Error")
    plt.show(block=True)

    return 0

if __name__ == "__main__":
    main()
