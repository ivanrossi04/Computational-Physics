# !V1

import numpy as np
import scipy as sp # only needed for the theoretical comparison

def main():
    # Enter the computation parameters
    dim = int(input("Enter the dimension of the integration space: "))
    iterations = int(input("Enter the number of iterations of the integration: "))

    seed = None
    try:
        seed = int(input("Enter the seed for the random number generation (press enter to skip):"))
    except:
        print("Custom seeding was skipped")    

    # Compute the theoretical value of the volume of the m-dimensional sphere
    theoretical_value = np.pi ** (dim / 2) / sp.special.gamma(dim / 2 + 1)
    print("\nThe theoretical value is: ", theoretical_value)

    # Initialize the random number (if none is inserted the current time is used)
    rng = np.random.default_rng(seed)

    integral = 0
    for _ in range(iterations):
        x = np.array(rng.random(dim, dtype=np.float64))
        if np.dot(x, x) < 1:
            integral += 1
    
    # Divide by the number of iteration and 2^m
    integral = integral / iterations * (2 ** dim)

    # The uncertainty formula is not generic, but simplified based on the assumption that [f^2(x) = f(x)]
    # TODO: check uncertainty formula
    uncertainty = np.sqrt(integral * (1 - integral / 2 ** dim) / 2**dim / iterations)

    print("The estimated integral is: ", integral, " ± ", uncertainty)
    
    return

if __name__ == "__main__":
    main()
