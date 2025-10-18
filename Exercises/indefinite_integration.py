# !V1
# Indefinite integration and plotting using the trapezoids method.

import numpy as np
import matplotlib.pyplot as plt

def main():

    # ! Define the function to integrate and the integration parameters
    f = lambda x: np.sin(x) / x
    x_min = -10
    x_max = 10.0
    c = 0.0  # Constant of integration
    N = 1000
    # ! ---------------------------------------------------------------

    x = np.linspace(x_min, x_max, N)
    y = f(x)

    h = (x_max - x_min) / (N - 1)

    integration = [c]
    for i in range(0, N - 1):
        integration.append(integration[-1] + h / 2 * (y[i] + y[i + 1]))

    plt.figure()
    plt.plot(x, y, ls = '--', label="function")
    plt.plot(x[1:], integration[1:], ls = '-', label="integral function")
    plt.title("Indefinite Integration using the trapezoids method")
    plt.legend()
    plt.xlabel("X",fontsize=16)
    plt.ylabel("Y",fontsize=16)
    plt.show(block=True)

if __name__ == "__main__":
    main()
