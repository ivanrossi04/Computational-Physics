import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

def derivative_function(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, h: float, method: str = 'central'):
    '''
    Compute the numerical derivative of a function f at points x using finite difference methods.
    Parameters:
        f: The function to differentiate. Must accept and return numpy arrays.
        x: The points at which to compute the derivative.
        h: The spacing between points in x.
        method: The finite difference method to use ('right', 'left', 'central').
    Returns:
        An array of the derivative values at points x.
    '''

    y = f(x)
    df_dx = np.array([])
    
    if method == 'right':
        df_dx = (y[1:] - y[:-1]) / h
        df_dx = np.append(df_dx, 0)
    elif method == 'left':
        df_dx = (y[1:] - y[:-1]) / h
        df_dx= np.insert(df_dx, 0, 0)
    elif method == 'central':
        df_dx = (y[2:] - y[:-2]) / (2 * h)
        df_dx = np.insert(df_dx, 0, 0)
        df_dx = np.append(df_dx, 0)
    return df_dx

def main():

    # Take the interval and number of points as input
    x_min = float(input('Enter X minimum: '))
    x_max = float(input('Enter X maximum: '))
    n = int(input('Enter the number of points: '))

    x = np.linspace(x_min, x_max, n) # note h = x_max/(n-1) since we have n elements and x_max is included
    
    # !define manually the function and compute values for its exact derivative
    f = lambda x: np.exp(-x**2)
    df = -2*x*np.exp(-x**2)
    # --------------------------------------    

    h = (x_max - x_min) / (n - 1)
    
    # compute the derivative approximations
    df_dx = derivative_function(f, x, h, method='right')
    df_sx = derivative_function(f, x, h, method='left')
    df_best = derivative_function(f, x, h, method='central')

    plt.rcParams.update({'font.size': 14})
    # Plot the functions
    plt.figure()
    plt.plot(x, f(x),  ls='--', label="F(x)")
    plt.plot(x, df,  ls='-', label="dF/dx exact")
    plt.plot(x[:-1], df_dx[:-1], ls='-', label="Right approximation")
    plt.plot(x[1:], df_sx[1:], ls='-', label="Left approximation")
    plt.plot(x[1:-1], df_best[1:-1], ls='-', label="Central approximation")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show(block=True)

    # Plot the errors
    plt.figure()
    plt.plot(x[:-1], abs(df - df_dx)[:-1], ls='-', label="Right approximation error")
    plt.plot(x[1:], abs(df - df_sx)[1:], ls='-', label="Left approximation error")
    plt.plot(x[1:-1], abs(df - df_best)[1:-1], ls='-', label="Central approximation error")
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.xlabel("X")
    plt.ylabel("Error")
    plt.yscale('log')
    plt.title("Error of Derivative Approximations")
    plt.show(block=True)

if __name__ == "__main__":
    main()
