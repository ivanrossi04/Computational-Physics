# !V1

'''
How to run this script:
- set function in the main() function
- run the script
- input the interval, number of points and derivative order when prompted
'''

import math
import numpy as np
import matplotlib.pyplot as plt

def main():

    x_min = float(input('Input the minimum value of x:'))
    x_max = float(input('Input the maximum value of x:'))
    n = int(input('Input the number of points (precision of the approximation):'))
    deriv_n = int(input('Input the order of the derivative: '))

    # compute the function values at the determined intervals
    x = np.linspace(x_min,x_max,n)
    
    # ! define manually the function
    f = np.sin
    # f = lambda x: np.exp(-x**2)
    # ! ----------------------------

    y = f(x)

    # compute the interval delta
    h = (x_max - x_min) / (n - 1)

    # init the derivative function value list
    df = np.zeros(n, dtype=np.float64)

    # dimension of the A matrix
    # an additional row and column is computed if deriv_n is odd
    a_m_dim = deriv_n + 1 + (deriv_n % 2)

    # compute the offsets and factorials needed for the Taylor expansion (for efficiency)
    steps = [(deriv_n/2 - k) * h for k in range(a_m_dim)] # starts from the rightmost point and goes to the left
    factorials = [math.factorial(k) for k in range(a_m_dim)]

    for i in range(n): # for every x_i values
        f_i = np.zeros(a_m_dim)
        
        # populate the a_matrix with the needed values from the Taylor approximation
        a_m = np.ones((a_m_dim, a_m_dim), dtype=np.float64)
        for j in range(a_m_dim):
            # compute the f_i needed for the derivative approximation
            f_i[j] = f(x[i] + steps[j])

            for k in range(1, a_m_dim):
                a_m[j][k] = (steps[j] ** k) / factorials[k]
        
        # solve the linear system to find the n-th derivative
        df[i] = np.linalg.solve(a_m, f_i)[deriv_n]

    # plotting the function
    plt.rcParams.update({'font.size': 14})
    plt.figure()
    plt.plot(x, f(x),  ls='-', label="F(x)")
    plt.plot(x, df,  ls='-', label= str(deriv_n) + " derivative approx.")

    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show(block=True)

if __name__ == "__main__":
    main()
