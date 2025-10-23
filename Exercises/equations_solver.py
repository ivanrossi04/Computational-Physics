# !V1

'''
Simple equations solver using the bisection method(iterative)
Problems with this method can arise when there are multiple roots in the interval [a, b] 
or when the function does not change sign over the interval. e.g. f(x) = x**2

Choose the interval [a, b] appropriately.
'''

import math
from typing import Callable

# Auxiliary function to determine the sign of a number
def sign(x: float) -> int:
    '''
    Determine the sign of a number
    
    Parameters:
        x : float: The number to check

    Returns:
    int : 1 if x > 0, -1 if x < 0, 0 if x == 0
    '''

    return (x > 0) - (x < 0)

def solve_bisection(f: Callable[[float], float], a: float, b: float, epsilon: float) -> float:
    answer = math.nan
    f_a = f(a)
    f_b = f(b)

    # Bisection method (iterative)
    if(sign(f_a) * sign(f_b) < 0):
        while b - a > epsilon:
            c = (a + b) / 2
            answer = c
            f_c = f(c)
            if sign(f_a) * sign(f_c) < 0:
                b = c
                f_b = f_c
            elif sign(f_c) * sign(f_b) < 0:
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
        if(sign(f_a) * sign(f_b) < 0):
            c = (a + b) / 2
            f_c = f(c)

            if sign(f_a) * sign(f_c) < 0: answer = solve_bisection_recursive(f, a, c, epsilon)
            elif sign(answer) * sign(f_b) < 0: answer = solve_bisection_recursive(f, c, b, epsilon)
            else: answer = c
        elif f_b == 0: answer = a
        elif f_a == 0: answer = b
    else: answer = b
    
    return answer

h = 1e-5

def solve_newton_rhapson(f: Callable[[float], float], x_trial: float, epsilon: float) -> float:
    x_sol = 0.0

    while True:
        df_dx = (f(x_trial + h) - f(x_trial - h)) / (2 * h) 
        x_sol = x_trial - f(x_trial)/df_dx

        if(abs(x_trial - x_sol) < epsilon): break
        
        x_trial = x_sol 

    return x_sol

def solve_newton_rhapson_recursive(f: Callable[[float], float], x_trial: float, epsilon: float) -> float:
    df_dx = (f(x_trial + h) - f(x_trial - h)) / (2 * h) 
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
