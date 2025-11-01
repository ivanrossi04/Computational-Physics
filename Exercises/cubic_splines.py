# !V1

'''
Cubic splines interpolation implementation.
'''

import numpy as np
import matplotlib.pyplot as plot

def main():
    
    # ! Define the function to solve and the initial parameters
    f = lambda x: np.sin(x)
    a = 0
    b = 2 * np.pi
    n = 4 # number of grid points
    n_points_per_spline = 100 # plotting resolution
    # ! -------------------------------------------------------

    # grid points are equally spaced
    x = np.linspace(a, b, n)
    y = f(x)
    h = (b - a) / (n - 1)

    A = np.zeros((n - 2, n - 2))
    z = np.zeros(n - 2)

    # Set up the linear system
    for i in range(n - 2):
        # Fill the matrix A as a tridiagonal matrix
        A[i][i] = 2 * h
        if i < n - 3:
            A[i][i + 1] = h
            A[i + 1][i] = h

        # Fill the vector z
        # z[i] corresponds to the intermediate value at x[i + 1]
        z[i] = 6 *(y[i + 2] - 2 * y[i + 1] + y[i]) / h

    # Compute the second derivatives at the intermediate grid points and apply natural spline boundary conditions (y''(x[0]) = y''(x[n-1]) = 0)
    d2p_dx = np.concatenate([[0], np.linalg.solve(A, z), [0]])

    # Coefficients of the cubic splines
    alpha = d2p_dx[1:] / (6 * h)
    beta = -d2p_dx[:-1] / (6 * h)
    gamma = -d2p_dx[1:] * h / 6 + y[1:] / h
    eta = d2p_dx[:-1] * h / 6 - y[:-1] / h

    # Function to evaluate the piecewise cubic spline at a given point
    def p(x_0):
        # find the interval for the given point and extract the index i
        i = 0
        while x_0 > x[i + 1]: i += 1

        return alpha[i] * (x_0 - x[i]) ** 3 + beta[i] * (x_0 - x[i + 1]) ** 3 + gamma[i] * (x_0 - x[i]) + eta[i] * (x_0 - x[i + 1])

    # Compute the plotting points
    x_plot = np.linspace(a, b, n_points_per_spline * (n - 1))
    y = f(x_plot)

    y_approx = []
    for x_0 in x_plot:
        y_approx.append(p(x_0))

    # Plot and comparison of the results
    plot.plot(x_plot, y, label="Original function")
    plot.plot(x_plot, y_approx, label="Polynomial approximation")
    plot.legend()
    plot.title("Cubic splines, N = " + str(n))
    plot.xlabel("x")
    plot.ylabel("y")
    plot.grid()
    plot.show(block = True)

    return

if __name__ == "__main__":
    main()