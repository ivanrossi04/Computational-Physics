# Computation of Pi using the method of inscribed polygons

import numpy as np

def main():

    n = int(input("Number of iterations: "))

    l0_good = np.sqrt(2.0)
    l0_bad = np.sqrt(2.0)

    print("Iteration:  Pi (good)\t|  Pi (bad)")
    for i in range(n):
        # Good method: complete formula for the side of the polygon
        '''
        This method calculates the side length of the inscribed polygon using the Pythagorean theorem,
        which maintains accuracy even as the number of sides increases.
        '''
        l1_good = np.sqrt((l0_good/2.0)**2+(1.0-np.sqrt(1.0-(l0_good/2.0)**2))**2)
        l0_good = l1_good
        pi_good = l1_good * 4 * 2.0**(i)

        # Bad method: simplified formula for the side of the polygon
        '''
        This method is numerically unstable for large i because it involves
        subtracting two nearly equal numbers (1 - sqrt(1 - (l0_bad/2)^2)).
        As i increases, l0_bad becomes very small, making (l0_bad/2)^2 also very small.
        As a result of round-off errors in floating-point arithmetic, the expression
        sqrt(1 - (l0_bad/2)^2) approaches 1, leading to the calculation of pi_bad approaching 0.
        '''
        l1_bad = np.sqrt(2-2*np.sqrt(1-(l0_bad/2)**2))
        l0_bad = l1_bad
        pi_bad = l1_bad * 4 * 2.0**(i)

        print(f"{i:2d}:  {pi_good:.16f}\t| {pi_bad:.16f}")

if __name__ == "__main__":
    main()