# !V1

import numpy as np
import math
import matplotlib.pyplot as plt

# TODO: the program only works for sin(x), make it general
# TODO: the program only works for even n derivatives, make it general

def main():
    x_max = float(input('Input the maximum value of x:'))
    n=int(input('Input the number of points (precision of the approximation):'))
    deriv_n = int(input('Input the order of the derivative: '))

    # compute the function values at the determined intervals
    x=np.linspace(0,x_max,n)
    f=np.sin(x)

    # compute the interval delta
    h = x_max / (n - 1)

    # init the derivative function value list
    df = np.zeros(n, dtype=np.float64)

    for i in range(n): # for every x_i values
        f_i = np.zeros(deriv_n + 1)
        
        # populate the a_matrix with the needed values from the Taylor approximation
        a_m = np.ones((deriv_n + 1,deriv_n + 1),dtype=np.float64)
        for j in range(deriv_n + 1):            
            # compute the f_i needed for the derivative approximation
            f_i[j] = np.sin(x[i] + (deriv_n/2 - j) * h)

            for k in range(1, deriv_n + 1):
                a_m[j][k] = ((deriv_n/2 - j) * h) ** k / math.factorial(k)
        
        # print('F_', x[i] , ': ', f_i, '\n')
        # print(a_m)
        
        # invert the a matrix and solve the linear system to find the n-th derivative
        a_m_inv = np.linalg.inv(a_m)
        df[i] = np.matmul(a_m_inv, f_i)[deriv_n] # only storing the n-th derivative

    # plotting the function
    fig, ax = plt.subplots()

    plt.rcParams.update({'font.size': 14})
    ax.set_xlim(0,x_max)
    ax.set_ylim(-1,1)

    plt.plot(x,f,  ls='-', label="Sin")
    plt.plot(x,df,  ls='-', label= str(deriv_n) + " derivative approx.")

    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show(block=True)

if __name__ == "__main__":
    main()
