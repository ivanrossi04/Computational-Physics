# !V1

'''
Simple equations solver using the bisection method(iterative)
Problems with this method can arise when there are multiple roots in the interval [a, b] 
or when the function does not change sign over the interval. e.g. f(x) = x**2

Choose the interval [a, b] appropriately.
'''

import math

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

def main():

    # ! Define the function to solve and the initial parameters
    f = lambda x: math.sin(x)
    a = -1
    b = 1
    epsilon = 1e-10
    # ! -------------------------------------------------------

    answer = None
    f_a = f(a)
    f_b = f(b)

    # Bisection method (iterative)
    if(sign(f_a) * sign(f_b) < 0):
        while b - a > epsilon:
            c = (a + b) / 2
            answer = f(c)

            # Debug statement
            # print("Current interval: [", a, ", ", c, ", ", b, "] -> f(c) = ", answer)

            if sign(f_a) * sign(answer) < 0:
                b = c
                f_b = answer
            elif sign(answer) * sign(f_b) < 0:
                a = c
                f_a = answer
            else:
                break # answer is zero in this case
    elif f(b) == 0: answer = f(b)
    elif f(a) == 0: answer = f(a)

    print("The root is approximately at: ", answer)

if __name__ == "__main__":
    main()