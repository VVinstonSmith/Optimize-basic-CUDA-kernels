#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define THREAD_SPLIT

#define OFFSET(row, col, ld) ((row)*(ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template<
	const int BLOCK_SIZE_M, const int BLOCK_SIZE_K, const int BLOCK_SIZE_N,
	const int THREAD_SIZE_Y, const int THREAD_SIZE_X>
__global__ void gemm_float4(
	float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, 
	const int M, const int K, const int N) {

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

	float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = { 0.0 };
	float frag_a[THREAD_SIZE_Y];
	float frag_b[THREAD_SIZE_X];

	const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
	const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

	const int A_TIlE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
	const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

	const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
	const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

	const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
	const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

	for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {

		// load A from GM to SM
		#pragma unroll
		for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
			int row = (by * BLOCK_SIZE_M) + i + A_TIlE_ROW_START;
			int col = tile_idx + A_TILE_COL;
			FETCH_FLOAT4(As[i + A_TIlE_ROW_START][A_TILE_COL]) = FETCH_FLOAT4(A[OFFSET(row, col, K)]);
		}

		// load B from GM to SM
		#pragma unroll
		for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
			int row = tile_idx + i + B_TILE_ROW_START;
			int col = (bx * BLOCK_SIZE_N) + B_TILE_COL;
			FETCH_FLOAT4(Bs[i + B_TILE_ROW_START][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(row, col, K)]);
		}
		__syncthreads();

		// compute C
		#pragma unroll
		for (int k = 0; k < BLOCK_SIZE_K; k++) {
			#pragma unroll
			for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++) {
			#ifdef THREAD_SPLIT
				int row = thread_y * THREAD_Y_PER_BLOCK + ty;
			#else
				int row = ty * THREAD_SIZE_Y + thread_y;
			#endif
				frag_a[thread_y] = As[row][k];
			}

		#ifdef THREAD_SPLIT
			#pragma unroll
			for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x++) {
				int col = thread_x * THREAD_X_PER_BLOCK + tx;
				frag_b[thread_x] = Bs[k][col];
			}
		#else
			/*#pragma unroll
			for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x++) {
				int col = tx * THREAD_SIZE_X + thread_x;
				frag_b[thread_x] = Bs[k][col];
			}*/
			#pragma unroll
			for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
				int col = tx * THREAD_SIZE_X + thread_x;
				FETCH_FLOAT4(frag_b[thread_x]) = FETCH_FLOAT4(Bs[k][col]);
			}
		#endif

			#pragma unroll
			for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++) {
				#pragma unroll
				for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x++) {
					accum[thread_y][thread_x] += frag_a[thread_y] * frag_b[thread_x];
				}
			}

		}

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
				C[OFFSET(row, col, N)] = accum[thread_y][thread_x];
			}
		}

	}

}



