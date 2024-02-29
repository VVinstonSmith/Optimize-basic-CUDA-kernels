
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "freshman.h"
#include <cublas_v2.h>
#include "kernels.cu"

int main(int argc, char**argv){
    if (argc != 3) {
        printf("usage: ./main [M] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t N = atoi(argv[2]);
    
    // v0, v1
    const int THREAD_Y_PER_BLOCK = 8;
    int BLOCK_X_PER_GRID = (M + THREAD_Y_PER_BLOCK - 1) / THREAD_Y_PER_BLOCK;
    dim3 dimBlock(32, THREAD_Y_PER_BLOCK);
    dim3 dimGrid(BLOCK_X_PER_GRID, 1);

    // v2
    int BLOCK_X_PER_GRID_16 = (M + 2 * THREAD_Y_PER_BLOCK - 1) / (2 * THREAD_Y_PER_BLOCK);
    dim3 dimGrid_16(BLOCK_X_PER_GRID_16, 1);

    // v3
    const int THREAD_X_PER_BLOCK_V3 = 32;
    const int THREAD_Y_PER_BLOCK_V3 = 8;
    int BLOCK_X_PER_GRID_V3 = (M + THREAD_Y_PER_BLOCK_V3 - 1) / THREAD_Y_PER_BLOCK_V3;
    dim3 dimBlock_v3(THREAD_X_PER_BLOCK_V3, THREAD_Y_PER_BLOCK_V3);
    dim3 dimGrid_v3(BLOCK_X_PER_GRID_V3, 1);

    size_t bytes_A = M * N * sizeof(float);
    size_t bytes_x = N * sizeof(float);
    size_t bytes_y = M * sizeof(float);

    auto h_A = (float*)malloc(bytes_A);
    auto h_x = (float*)malloc(bytes_x);
    auto h_y = (float*)malloc(bytes_y);
    auto h_y1 = (float*)malloc(bytes_y);

    float* d_A, * d_x, * d_y;
    CHECK(cudaMalloc(&d_A, bytes_A));
    CHECK(cudaMalloc(&d_x, bytes_x));
    CHECK(cudaMalloc(&d_y, bytes_y));

    for (int i = 0; i < M * N; i++) {
        h_A[i] = (float)i / 13;
    }
    for (int i = 0; i < N; i++) {
        h_x[i] = (float)i / 13;
    }

    memset(h_y, 0, M * sizeof(float));
    memset(h_y1, 0, M * sizeof(float));

    int nIter = 1000;
    CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, bytes_y, cudaMemcpyHostToDevice));

    float msecTotal = 0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < nIter; i++) {
         //Sgemv_v0 << <dimGrid, dimBlock >> > (d_A, d_x, d_y, M, N);
         //Sgemv_v1 << <dimGrid, dimBlock >> > (d_A, d_x, d_y, M, N);
         //Sgemv_v2 << <dimGrid_16, dimBlock >> > (d_A, d_x, d_y, M, N);

        Sgemv_v3<THREAD_X_PER_BLOCK_V3, THREAD_Y_PER_BLOCK_V3> << <dimGrid_v3, dimBlock_v3 >> > (d_A, d_x, d_y, M, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);

    CHECK(cudaMemcpy(h_y, d_y, bytes_y, cudaMemcpyDeviceToHost));

    double flopsPerMatMul = 2.0 * M * N;
    double msecPerMatMul = msecTotal / nIter;
    double gigaFlops = (flopsPerMatMul * 1.0e-9f) / (msecPerMatMul / 1000.0f);
    printf("My gemv performance= %.2f GFlops, Time= %.3f msec, size= %.0f Ops\n", gigaFlops, msecPerMatMul, flopsPerMatMul);

    /*--------------------------cublas--------------------------*/
    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    float alpha = 1.0, beta = 0.0;
    CHECK(cudaMemcpy(d_y, h_y1, bytes_y, cudaMemcpyHostToDevice));
    
    CHECK(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++) {
        cublasSgemv(blas_handle, CUBLAS_OP_T,
            N, M, &alpha,
            d_A, N, d_x, 1, &beta, d_y, 1);
    }
    CHECK(cudaEventRecord(stop));
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);

    CHECK(cudaMemcpy(h_y1, d_y, bytes_y, cudaMemcpyDeviceToHost));
    
    msecPerMatMul = msecTotal / nIter;
    gigaFlops = (flopsPerMatMul * 1.0e-9f) / (msecPerMatMul / 1000.0f);
    printf("cublas  performance= %.2f GFlops, Time= %.3f msec, size= %.0f Ops\n", gigaFlops, msecPerMatMul, flopsPerMatMul);

    /*-------------------------------------------------------------------------------------------------------*/
    /*int result = checkResult(h_y, h_y1, M, 1.e-3);
    if (result != 0) {
        printf("check fail!\n");
        printf("num of errors: %d\n", result);
    }
    else
        printf("check success!\n");
    */

    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < M; i++) {
        double abs_err = fabs(h_y[i] - h_y1[i]);
        double dot_length = M;
        double abs_val = fabs(h_y[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                i, h_y[i], h_y1[i], eps);
            correct = false;
            break;
        }
    }
    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");

    /*for (int i = 0; i < 10; i++) {
        printf("%f~%f \n", h_y[i], h_y1[i]);
    }*/

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    free(h_A);
    free(h_x);
    free(h_y);
    free(h_y1);

    return 0;
}
