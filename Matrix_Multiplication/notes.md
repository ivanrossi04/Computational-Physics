# Matrix Multiplication

### [The problem](1-original.py)
The computation of floating-point matrix operations may seem like a trivial task that doesn't require special attention. Square matrix multiplication in particular is straightforward: given two N × N matrices A and B, their product is the N × N matrix C whose entries are given by

$$C_{ij} = \sum_{k=0}^{N-1} A_{ik}\,B_{kj}\,$$

This (textbook) implementation is typically written as three nested loops and has time complexity $\mathcal{O}(N^3)$ and space complexity $\mathcal{O}(N^2)$. That cost is not problematic for small N, but it becomes a major concern for large-scale physics simulations or machine-learning workloads where matrix operations are fundamental.

_Note: All the operations are done with float32 numbers_

### [Compilation latency](2-original-numba.py)
One first possible optimization for an interpreted language like Python is to compile the critical code; this is done with libraries like Numba and its JIT (Just-In-Time) compilation module.

Interpreters such as CPython add latency for a few reasons: bytecode must be dispatched by the interpreter loop, dynamic typing requires runtime type checks and dispatch, and Python's object model increases memory and indirection overhead for tight numeric loops. Tools like Numba mitigate these costs by compiling typed numerical code to native machine code and using native numeric representations inside the compiled region.

### [Decomposition of the computation](3-matmul.py)
Another problem that causes latency is that the faster memories (registers and caches) are limited and cannot hold the entire matrices. This means the CPU often waits for data to be loaded from slower memory levels.

Algorithms like GEMM (General Matrix Multiplication) subdivide the matrices into smaller tiles or blocks that fit in cache or registers and therefore increase data reuse.

These algorithms are the basis of the BLAS (Basic Linear Algebra Subprograms) specification. High-performance implementations exist for many languages and platforms, for example: Fortran ([LAPACK](https://www.netlib.org/lapack/)), C ([OpenBLAS](http://www.openmathlib.org/OpenBLAS/)), C++ ([std::linalg](https://www.en.cppreference.com/w/cpp/numeric/linalg.html)), and Python ([NumPy](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html), which typically dispatches to a BLAS implementation).

### [Usage of GPU](4-tensorflow.py)
GPUs accelerate matrix multiplication because of their many parallel cores, high memory bandwidth and optimized libraries (e.g., cuBLAS) that extract very high throughput. However transfers over PCIe can hide those gains for small or frequently copied data. For this reason it is better to keep the needed data on the GPU, batch operations, overlap transfers with computation, and use pinned/unified memory to reduce latency.