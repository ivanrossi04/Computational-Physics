import numpy as np
from numba import jit
import time

@jit(nopython=True)
def matrix_multiply(a,b,c):
    for i in range(a.shape[0]):
        for j in range (a.shape[0]):
            for k in range (a.shape[0]):
                c[i,j]+=a[i,k]*b[k,j]


def main():    
    n = int(input("Insert the dimension of the matrices:\n"))

    ma = np.zeros((n,n),dtype=np.float32)
    mb = np.zeros((n,n),dtype=np.float32)
    mc = np.zeros((n,n),dtype=np.float32)

    print("Multiplying ...")

    start_time = time.perf_counter()
    matrix_multiply(ma,mb,mc)
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time

    print("Elapsed time: ", elapsed_time, "seconds")
    print(mc[0,0], " ", mc[1,0], " ", mc[2,0])

if __name__ == "__main__":
    main()
