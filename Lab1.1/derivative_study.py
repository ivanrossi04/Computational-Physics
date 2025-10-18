import numpy as np
import matplotlib.pyplot as plt

# TODO: the program only works for sin(x), make it general

def main():
    # this program creates a list of n equally spaced numbers
    # between 0 and x_max

    x_max = float(input('Enter X maximum: '))
    n = int(input('Enter the number of points: '))

    x = np.linspace(0, x_max, n) # note h = x_max/(n-1) since we have n elements and x_max is included
    f = np.sin(x)

    h = x_max / (n - 1)

    df = np.zeros((len(x)))
    for i in range(len(f)):
        df[i] = np.cos(x[i]) # exact derivative

    df_dx = np.zeros((len(x)))
    for i in range(len(f) - 1):
        df_dx[i] = (f[i+1] - f[i]) / h # right derivative formula

    df_sx = np.zeros((len(x)))
    for i in range(1, len(f)):
        df_sx[i] = (f[i] - f[i - 1]) / h # left derivative formula

    df_best = np.zeros((len(x)))
    for i in range(1, len(f) - 1):
        df_best[i] = (f[i + 1] - f[i - 1]) / (2 * h) # central derivative formula

    # Define a plot
    plt.rcParams.update({'font.size': 14})

    # Plot the functions
    plt.figure()
    plt.plot(x, f,  ls='--', label="Sin")
    plt.plot(x, df,  ls='-', label="Cos")
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
