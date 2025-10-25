# !V1
import numpy as np
import math
from typing import Callable

def integrate_simpson(function: Callable[[np.ndarray], np.ndarray], x_min: float, x_max: float, N: int) -> float:
    '''
    Simpson integration method: 
    This method is based on the idea of approximating the integrand by a second-degree polynomial
    and integrating this polynomial exactly.

    Parameters:
        function (Callable[[np.ndarray], np.ndarray]): Function to integrate. Must accept and return numpy arrays.
        x_min (float): Lower limit of integration.
        x_max (float): Upper limit of integration.
        N (int): Number of points.

    Returns:
        float: Approximation of the integral.

    Raises:
        Exception: If N is even (The simpson method requires an odd number of intervals).
    '''

    # The simpson method requires an odd number of intervals
    if(not(N % 2)): raise Exception('The number N of intervals should be odd, instead got N = ' + str(N))

    h = (x_max - x_min) / (N - 1)

    # Evaluate function at all points
    f = function(np.linspace(x_min, x_max, N))

    return (f[0] * h / 3 + np.sum(f[1:N-1:2] * h * 4 / 3) + np.sum(f[2:N-1:2] * h * 2 / 3) + f[N-1] * h / 3)

def solve_bisection(f: Callable[[float], float], a: float, b: float, epsilon: float) -> float:
    '''
    This function solves finds the zero of a given f: R -> R with the bisection method.
    The function must change sign between [a, b] for a zero to be noticed in that interval.
    Given an initial interval [a, b], the midpoint 'c' between them is found and f(c) is computed.
    Based on where the change of sign happens, the midpoint becomes one of the extremes of the next interval.
    The algorithm stops when a zero is found or when the width of the interval is less than epsilon.

    Parameters:
    - f (Callable[[float], float]): The function for which we are trying to find a root.
    - a (float): The lower bound of the interval.
    - b (float): The upper bound of the interval.
    - epsilon (float): The tolerance for stopping the algorithm.

    Returns:
    - float: The approximate root of the function within the specified interval.

    '''

    answer = math.nan
    f_a = f(a)
    f_b = f(b)

    # Bisection method (iterative)
    if(math.copysign(1, f_a) * math.copysign(1, f_b) < 0):
        while b - a > epsilon:
            c = (a + b) / 2
            answer = c
            f_c = f(c)
            
            if math.copysign(1, f_a) * math.copysign(1, f_c) < 0:
                b = c
                f_b = f_c
            elif math.copysign(1, f_c) * math.copysign(1, f_b) < 0:
                a = c
                f_a = f_c
            else:
                break # answer is zero in this case
    elif f(b) == 0: answer = b
    elif f(a) == 0: answer = a

    return answer

def solve_newton_rhapson(f: Callable[[float], float], df_dx: Callable[[float], float], x_trial: float, epsilon: float) -> float:
    '''
    This function finds the zero of a given f: R -> R with the Newton-Rhapson method.
    Given an initial guess x_trial, the function's derivative is passed as the df_dx.
    A new approximation x_sol is then computed using the formula:
    x_sol = x_trial - f(x_trial) / f'(x_trial)
    The process is repeated until the difference between successive approximations is less than epsilon.

    Parameters:
    - f (Callable[[float], float]): The function for which we are trying to find a root.
    - df_dx (Callable[[float], float]): The derivative of the function.
    - x_trial (float): Initial guess for the root.
    - epsilon (float): The tolerance for stopping the algorithm.

    Returns: 
    - float: The approximate root of the function.
    '''

    x_sol = 0.0

    while True:
        x_sol = x_trial - f(x_trial)/df_dx(x_trial)

        if(abs(x_trial - x_sol) < epsilon): break
        
        x_trial = x_sol 

    return x_sol

def main():
    itype = int(input('Choose a function: \n0.Sin \n1.Exp \n2.Exp^2\n'))
    
    # choosing the function based on input
    df_dx = lambda x: np.sin(x)
    if itype == 1:
        df_dx = lambda x: np.exp(x)
    elif itype == 2:
        df_dx = lambda x: np.exp(x**2)
    elif itype > 2: print('Invalid function input, defaulting to sin(x)\n')
    
    N = 10001
    f = lambda x: integrate_simpson(df_dx, 0.0, x, N) - res 

    res = float(input('Input the value searched for the integral: '))
    a = float(input('Input x_min: '))
    b = float(input('Input x_max: '))
    eps = float(input('Input the required precision: '))
     
    print('Solution with the bisection method: ', solve_bisection(f, a, b, eps))

    x_0 = float(input('Input the first guess for the Newton-Raphson method: '))
    print('Solution with the Newton-Raphson method: ', solve_newton_rhapson(f, df_dx, x_0, eps))

if __name__ == "__main__":
    main()
