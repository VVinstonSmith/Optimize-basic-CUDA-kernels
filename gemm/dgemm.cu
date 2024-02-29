
#include "cuda_runtime.h"
#include <cublas_v2.h>
#include "device_launch_parameters.h"
#include <iostream>
#include "assert.h"
#include "freshman.h"

//#define GEMM_SCALAR
//#define GEMM_FLOAT4
#define GEMM_DBUF

#ifdef GEMM_SCALAR
	#include "dense.cu"
#endif
#ifdef GEMM_FLOAT4
	#include "dense_float4.cu"
#endif
#ifdef GEMM_DBUF
	#include "dense_dbuf.cu"
#endif

int main(int argc, char** argv) {
	//InitCUDA();
	if (argc != 4) {
		printf("usage: ./main [M] [K] [N]\n");
		exit(0);
	} 
	size_t M = atoi(argv[1]);
	size_t K = atoi(argv[2]);
	size_t N = atoi(argv[3]);
	assert(M % 8 == 0);
	assert(N % 8 == 0);
	assert(K % 8 == 0);

	const int BLOCK_SIZE_M = 128;
	const int BLOCK_SIZE_K = 8;
	const int BLOCK_SIZE_N = 128;
	const int THREAD_SIZE_X = 8;
	const int THREAD_SIZE_Y = 8;
	float alpha = 1.0, beta = 0.0;
	
	size_t size_A = M * K;
	size_t size_B = K * N;
	size_t size_C = M * N;
	size_t bytes_A = size_A * sizeof(float);
	size_t bytes_B = size_B * sizeof(float);
	size_t bytes_C = size_C * sizeof(float);

	auto h_A = (float*)malloc(bytes_A);
	auto h_B = (float*)malloc(bytes_B);
	auto h_C = (float*)malloc(bytes_C);
	auto h_C1 = (float*)malloc(bytes_C);

	float* d_A, * d_B, * d_C;
	CHECK(cudaMalloc((void**)&d_A, bytes_A));
	CHECK(cudaMalloc((void**)&d_B, bytes_B));
	CHECK(cudaMalloc((void**)&d_C, bytes_C));

	// generate A
	for (int i = 0; i < M * K; i++) {
		h_A[i] = i / 13;
	}
	// generate B
	for (int i = 0; i < K * N; i++) {
		h_B[i] = i % 13;
	}
	for (int i = 0; i < M * N; i++) {
		h_C[i] = h_C1[i] = 0.0;
	}

	/*---------------------------------------------gemm_myself---------------------------------------------*/
	CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));
	float msecTotal = 0.0;
	int nIter = 100;

	dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
	dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
	printf("dimGrid(%d, %d)\n", dimGrid.x, dimGrid.y);
	printf("dimBlock(%d, %d)\n", dimBlock.x, dimBlock.y);
	CHECK(cudaEventRecord(start));
	for (int run = 0; run < nIter; run++) {
#ifdef GEMM_SCALAR
		gemm_scalar<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X>
			<< <dimGrid, dimBlock >> > (d_A, d_B, d_C, M, K, N, alpha, beta);
#endif
#ifdef GEMM_FLOAT4
		gemm_float4<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X>
			<< <dimGrid, dimBlock >> > (d_A, d_B, d_C, M, K, N);
#endif
#ifdef GEMM_DBUF
		gemm_dbuf<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X>
			<< <dimGrid, dimBlock >> > (d_A, d_B, d_C, M, K, N);
#endif
	}
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	CHECK(cudaEventElapsedTime(&msecTotal, start, stop));

	CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

	double flopsPerMatMul = 2.0 * M * N * K;
	double msecPerMatMul = msecTotal / nIter;
	double gigaFlops = (flopsPerMatMul * 1.0e-9f) / (msecPerMatMul / 1000.0f);
	printf("My gemm performance= %.2f GFlops, Time= %.3f msec, size= %.0f Ops\n", gigaFlops, msecPerMatMul, flopsPerMatMul);
	
	/*---------------------------------------------gemm_cublas---------------------------------------------*/
	cublasHandle_t blas_handle;
	cublasCreate(&blas_handle);
	CHECK(cudaMemcpy(d_C, h_C1, bytes_C, cudaMemcpyHostToDevice));

	CHECK(cudaEventRecord(start));
	for (int run = 0; run < nIter; run++) {
		cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
			N, M, K, &alpha,
			d_B, N, d_A, K, &beta, d_C, N);
	}
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	CHECK(cudaEventElapsedTime(&msecTotal, start, stop));

	CHECK(cudaMemcpy(h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost));

	msecPerMatMul = msecTotal / nIter;
	gigaFlops = (flopsPerMatMul * 1.0e-9f) / (msecPerMatMul / 1000.0f);
	printf("cublas  performance= %.2f GFlops, Time= %.3f msec, size= %.0f Ops\n", gigaFlops, msecPerMatMul, flopsPerMatMul);			

	cublasDestroy(blas_handle);
	/*-------------------------------------------------------------------------------------------------------*/
	int result = checkResult(h_C, h_C1, M * N, 1.e-3);
	if (result != 0) {
		printf("check fail!\n");
		printf("num of errors: %d\n", result);
	}
	else
		printf("check success!\n");

	/*for (int i = 0; i < N; i++) {
		printf("%f ", h_C[i]);
	} printf("\n");
	for (int i = 0; i < N; i++) {
		printf("%f ", h_C1[i]);
	} printf("\n");*/

	CHECK(cudaFree(d_A));
	CHECK(cudaFree(d_B));
	CHECK(cudaFree(d_C));
	free(h_A);
	free(h_B);
	free(h_C);
	free(h_C1);
}

