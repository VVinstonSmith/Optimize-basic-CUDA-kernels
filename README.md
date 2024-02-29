#  optimize basic kernels for GPU
This is a series of GPU optimization topics. Here we introduce several basic CUDA kernel optimizations, including: Reduce, GEMM, GEMV, SPMV, Softmax, etc. All the following performance data are run in CUDA environment.

If you have any questions, you can directly contact: 1148399054@qq.com

## 1. Reduce
Six optimization methods were used to optimize Reduce operator.

## 2. GEMM (General Matrix-Matrix Multiplication)
Three optimization methods were used to optimize GEMM operator.

## 3. GEMV (General Matrix-Vector Multiplication)
The core of sgemv kernel optimization lies in the design of blocks and threads, and it is necessary to avoid the situation of thread idle as much as possible.

## 4.SPMV (Sparse Matrix-Vector Multiplication)
A CUDA implemention of SPMV.

## 5. Softmax
Three optimization methods were used to optimize Softmax operator.

## 6. Laplace3d
Two optimization methods were used to optimize laplace3d operator. We further compare its performance with the CPU implemention.
