# !V1

'''
Simple equations solver using the bisection method(iterative)
Problems with this method can arise when there are multiple roots in the interval [a, b] 
or when the function does not change sign over the interval. e.g. f(x) = x**2

Choose the interval [a, b] appropriately.
'''

import math
from typing import Callable

def solve_bisection(f: Callable[[float], float], a: float, b: float, epsilon: float) -> float:
    '''
    This function solves finds the zero of a given f: R -> R with the bisection method.
    The function must change sign between [a, b] for a zero to be noticed in that interval.
    Given an initial interval [a, b], the midpoint 'c' between them is found and f(c) is computed.
    Based on where the change of sign happens, the midpoint becomes one of the extremes of the next interval.
    The algorithm stops when a zero is found or when the width of the interval is less than epsilon.

    Parameters:
        f (Callable[[float], float]): The function for which we are trying to find a root.
        a (float): The lower bound of the interval.
        b (float): The upper bound of the interval.
        epsilon (float): The tolerance for stopping the algorithm.

    Returns:
        float: The approximate root of the function within the specified interval.

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

def solve_bisection_recursive(f: Callable[[float], float], a: float, b: float, epsilon: float) -> float:
    answer = math.nan

    if b - a > epsilon:
        f_a = f(a)
        f_b = f(b)

        # Bisection method (recursive)
        if(math.copysign(1, f_a) * math.copysign(1, f_b) < 0):
            c = (a + b) / 2
            f_c = f(c)

            if math.copysign(1, f_a) * math.copysign(1, f_c) < 0: answer = solve_bisection_recursive(f, a, c, epsilon)
            elif math.copysign(1, f_c) * math.copysign(1, f_b) < 0: answer = solve_bisection_recursive(f, c, b, epsilon)
            else: answer = c
        elif f_b == 0: answer = a
        elif f_a == 0: answer = b
    else: answer = b
    
    return answer

def solve_newton_rhapson(f: Callable[[float], float], x_trial: float, epsilon: float) -> float:
    '''
    This function finds the zero of a given f: R -> R with the Newton-Rhapson method.
    Given an initial guess x_trial, the function's derivative at that point is approximated using finite differences (midpoint method).
    A new approximation x_sol is then computed using the formula:
    x_sol = x_trial - f(x_trial) / f'(x_trial)
    The process is repeated until the difference between successive approximations is less than epsilon.

    Parameters:
        f (Callable[[float], float]): The function for which we are trying to find a root.
        x_trial (float): Initial guess for the root.
        epsilon (float): The tolerance for stopping the algorithm.

    Returns:
        float: The approximate root of the function
    '''

    x_sol = 0.0

    while True:
        df_dx = (f(x_trial + epsilon) - f(x_trial - epsilon)) / (2 * epsilon) 
        x_sol = x_trial - f(x_trial)/df_dx

        if(abs(x_trial - x_sol) < epsilon): break
        
        x_trial = x_sol 

    return x_sol

def solve_newton_rhapson_recursive(f: Callable[[float], float], x_trial: float, epsilon: float) -> float:
    df_dx = (f(x_trial + epsilon) - f(x_trial - epsilon)) / (2 * epsilon) 
    x_sol = x_trial - f(x_trial) / df_dx

    if(abs(x_trial - x_sol) > epsilon): return solve_newton_rhapson_recursive(f, x_sol, epsilon)
    return x_sol

def main():

    # ! Define the function to solve and the initial parameters
    f = lambda x: math.sin(x-1)
    a = 0
    b = 2
    epsilon = 1e-15
    # ! -------------------------------------------------------

    print("The root is approximately at: ", solve_newton_rhapson(f, a, epsilon))

if __name__ == "__main__":
    main()
