'''
!V1 -----------------------------------------------------

Insert the precicion of the taylor series as input
Compute the taylor series of a value (or a set of values)
Plot the values and compare them to the defaul np.sin()

The more distance from 0, the more terms are needed for a good approximation
---------------------------------------------------------
'''

import math
import matplotlib.pyplot as plot

def maclaurin_sin(x: float, n: int) -> float:
    ''' compute the approximation of the sin function with the Maclaurin '''
    
    sin_x = 0
    for i in range(int(n/2)):
        sin_x += ((-1) ** i) * (x ** (2 * i + 1)) / math.factorial(2 * i + 1)

    return sin_x

def main():
    
    # input of the needed values
    value = float(input("Enter the argument of the sin function: "))
    precision = int(input("Enter the degree of precision of the approximation: "))
    

    '''
    # Single value computation test
    approx_val = maclaurin_sin(value, precision)
    exact_val = math.sin(value)
    print(exact_val, approx_val)
    '''

    # Whole period computation
    interval = math.pi / 100 # 100 is an arbitrary value that allows for a smooth curve

    values = []
    approx_sin = []
    exact_sin = []
    error = []
    
    i = value - math.pi # this operation makes 'value' the middle point in the plot
    while (i < value + math.pi):
        values.append(i)
        approx_sin.append(maclaurin_sin(i, precision))
        exact_sin.append(math.sin(i))
        error.append(exact_sin[-1] - approx_sin[-1])

        i += interval

    # Plot and comparison of the results
    plot.plot(values, approx_sin, label="Maclaurin Approximation")
    plot.plot(values, exact_sin, label="Exact sin(x)")
    plot.legend()
    plot.title("Maclaurin Series Approximation of sin(x)")
    plot.xlabel("x")
    plot.ylabel("sin(x)")
    plot.grid()
    plot.show(block = True)

    plot.plot(values, error, label="Error")
    plot.legend()
    plot.title("Error of the Maclaurin Approximation")
    plot.xlabel("x")
    plot.ylabel("Sin(x) - Approximation")
    plot.grid()
    plot.show(block = True)

if __name__ == "__main__":
    main()
