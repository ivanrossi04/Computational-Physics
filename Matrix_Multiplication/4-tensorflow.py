import numpy as np
import time
import tensorflow as tf

def main():

    n = int(input("Insert the dimension of the matrices:\n"))

    ma = np.ones((n,n),dtype=np.float32)
    mb = np.ones((n,n),dtype=np.float32)

    print("Converting to tensorflow...")
    ta = tf.convert_to_tensor(ma)
    tb = tf.convert_to_tensor(mb)

    print("Multiplying ...")
    start_time = time.perf_counter()
    tc = tf.matmul(ta,tb)
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time, "seconds")

    print("Converting to numpy...")
    mc = tc.numpy()

    print(mc[0,0], " ", mc[1,0], " ", mc[2,0])
    

if __name__ == "__main__":
    main()
