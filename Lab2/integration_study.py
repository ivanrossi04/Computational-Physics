# !V1

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import time

def integrate_naive(function: Callable[[np.ndarray], np.ndarray], x_min: float, x_max: float, N: int) -> float:
    '''Naive rectangle integration method
    This method approximates the integral by summing the areas of rectangles
    formed by the function values at the left endpoints of each subinterval.
    '''

    h = (x_max - x_min) / (N - 1)

    # Evaluate function at the left endpoints
    f = function(np.linspace(x_min, x_max, N))[0:N-1]

    return np.sum(f * h)

def integrate_rectangle(function: Callable[[np.ndarray], np.ndarray], x_min: float, x_max: float, N: int) -> float:
    '''Rectangle integration method using midpoints
    This method improves the naive rectangle method by using the midpoint of each interval
    to evaluate the function, leading to a better approximation of the integral.
    '''

    h = (x_max - x_min) / (N - 1)

    # Evaluate function at the midpoints
    f_half = function(np.linspace(x_min, x_max, N) + 1/2 * h)[0:N-1]

    return np.sum(f_half * h)

def integrate_trapezoid(function: Callable[[np.ndarray], np.ndarray], x_min: float, x_max: float, N: int) -> float:
    '''Trapezoid integration method
    This method approximates the area under the curve as a series of trapezoids, making it more accurate than the rectangle method.
    '''

    h = (x_max - x_min) / (N - 1)

    # Evaluate function at all points
    f = function(np.linspace(x_min, x_max, N))

    return (f[0] * h / 2 + np.sum(f[1:N-1] * h) + f[N-1] * h / 2)

def integrate_simpson(function: Callable[[np.ndarray], np.ndarray], x_min: float, x_max: float, N: int) -> float:
    '''
    Simpson integration method: 
    This method is based on the idea of approximating the integrand by a second-degree polynomial
    and integrating this polynomial exactly.
    '''

    # The simpson method requires an odd number of intervals
    if(not(N % 2)): raise Exception('The number N of intervals should be odd, instead got N = ' + str(N))

    h = (x_max - x_min) / (N - 1)

    # Evaluate function at all points
    f = function(np.linspace(x_min, x_max, N))

    return (f[0] * h / 3 + np.sum(f[1:N-1:2] * h * 4 / 3) + np.sum(f[2:N-1:2] * h * 2 / 3) + f[N-1] * h / 3)

def main():
    npoints = np.power(10, np.arange(1, 8))

    rett_naif = []
    rett = []
    trap = []
    simp = []

    # Define the function and true value of the integral
    f = np.exp
    x_min = 0
    x_max = 1
    true_value = np.exp(1) - 1
   
    # Compute the errors for each method
    start_time = time.perf_counter()
    
    for N in npoints:
        rett_naif.append(abs(true_value - integrate_naive(f, x_min, x_max, N)))
        rett.append(abs(true_value - integrate_rectangle(f, x_min, x_max, N)))
        trap.append(abs(true_value - integrate_trapezoid(f, x_min, x_max, N)))
        simp.append(abs(true_value - integrate_simpson(f, x_min, x_max, N + 1)))

    end_time = time.perf_counter()
    duration = end_time - start_time

    print("Computation time for every integration: ", duration)
    
    plt.rcParams.update({'font.size': 16}) # cambiamo il font
    
    plt.figure()
    plt.plot(npoints,rett_naif,  ls='-', label="Naive Rectangles")
    plt.plot(npoints,rett,  ls='-', label="Rectangles")
    plt.plot(npoints,trap,  ls='-', label="Trapezoids")
    plt.plot(npoints,simp,  ls='-', label="Simpson")

    plt.xscale('log')
    plt.yscale('log')

    plt.legend()
    plt.xlabel("N",fontsize=16)
    plt.ylabel("Errore",fontsize=16)
    plt.show(block=True)

if __name__ == "__main__":
    main()
