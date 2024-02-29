#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#define THREAD_SPLIT

#define OFFSET(row, col, ld) ((row)*(ld) + (col))

template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_K, const int  BLOCK_SIZE_N,
	const int THREAD_SIZE_Y, const int THREAD_SIZE_X>
__global__ void gemm_scalar(
	float* __restrict__ A,
	float* __restrict__ B,
	float* __restrict__ C,
	const int M, const int K, const int N, float alpha, float beta) {

	__shared__ float As[BLOCK_SIZE_M][BLOCK_SIZE_K];
	__shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
	const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
	const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

	int tid = ty * THREAD_X_PER_BLOCK + tx;

	float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = { 0 };

	const int A_TILE_ROW_START = tid / BLOCK_SIZE_K;
	const int B_TILE_ROW_START = tid / BLOCK_SIZE_N;

	const int A_TILE_COL = tid % BLOCK_SIZE_K;
	const int B_TILE_COL = tid % BLOCK_SIZE_N;

	const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;
	const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_N;

	for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {
#pragma unroll
		for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
			const int row = (by * BLOCK_SIZE_M) + A_TILE_ROW_START + i;
			const int col = tile_idx + A_TILE_COL;
			if (bx == blockDim.x - 1 || by == blockDim.y - 1) {
				if (row >= M || col >= K)
					As[A_TILE_ROW_START + i][A_TILE_COL] = 0.0;
				else
					As[A_TILE_ROW_START + i][A_TILE_COL] = A[OFFSET(row, col, K)];
			}
			else {
				As[A_TILE_ROW_START + i][A_TILE_COL] = A[OFFSET(row, col, K)];
			}
		}

#pragma unroll
		for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
			const int row = tile_idx + B_TILE_ROW_START + i;
			const int col = (bx * BLOCK_SIZE_N) + B_TILE_COL;
			if (bx == blockDim.x - 1 || by == blockDim.y - 1) {
				if (row >= K || col >= N)
					Bs[B_TILE_ROW_START + i][B_TILE_COL] = 0.0;
				else
					Bs[B_TILE_ROW_START + i][B_TILE_COL] = B[OFFSET(row, col, N)];
			}
			else {
				Bs[B_TILE_ROW_START + i][B_TILE_COL] = B[OFFSET(row, col, N)];
			}
		}
		__syncthreads();

		// compute C_local
#pragma unroll
		for (int k = 0; k < BLOCK_SIZE_K; k++) {
#pragma unroll
			for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++) {
#pragma unroll
				for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x++) {
#ifdef THREAD_SPLIT
					int row = thread_y * THREAD_Y_PER_BLOCK + ty;
					int col = thread_x * THREAD_X_PER_BLOCK + tx;
#else
					int row = THREAD_SIZE_Y * ty + thread_y;
					int col = THREAD_SIZE_X * tx + thread_x;
#endif
					accum[thread_y][thread_x] += As[row][k] * Bs[k][col];
				}
			}
		}
		__syncthreads();
	
	}

	// store back to C
#pragma unroll
	for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++) {
#pragma unroll
		for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x++) {
#ifdef THREAD_SPLIT
			int row = (by * BLOCK_SIZE_M) + thread_y * THREAD_Y_PER_BLOCK + ty;
			int col = (bx * BLOCK_SIZE_N) + thread_x * THREAD_X_PER_BLOCK + tx;
#else
			int row = (by * BLOCK_SIZE_M) + ty * THREAD_SIZE_Y + thread_y;
			int col = (bx * BLOCK_SIZE_N) + tx * THREAD_SIZE_X + thread_x;
#endif
			if (bx == gridDim.x - 1 || by == gridDim.y - 1) {
				if(row<M && col<N)
					C[OFFSET(row, col, N)] = accum[thread_y][thread_x];
			}
			else {
				C[OFFSET(row, col, N)] = accum[thread_y][thread_x];
			}
		}
	}

}