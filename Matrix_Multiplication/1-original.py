import numpy as np
import time

def main():

    n = int(input("Insert the dimension of the matrices:\n"))

    ma = np.ones((n,n),dtype=np.float32)
    mb = np.ones((n,n),dtype=np.float32)
    mc = np.zeros((n,n),dtype=np.float32)

    print("Multiplying ...")

    start_time = time.perf_counter()
    for i in range(n):
        for j in range (n):
            for k in range (n):
                mc[i,j]+=ma[i,k]*mb[k,j]

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print("Elapsed time: ", elapsed_time, "seconds")
    print(mc[0,0], " ", mc[1,0], " ", mc[2,0])

if __name__ == "__main__":
    main()