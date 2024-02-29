#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template<
	const int BLOCK_SIZE_M, const int BLOCK_SIZE_K, const int BLOCK_SIZE_N,
	const int THREAD_SIZE_Y, const int THREAD_SIZE_X>
__global__ void gemm_dbuf(
	float* __restrict__ A, float* __restrict__ B, float* __restrict__ C,
	const int M, const int K, const int N) {
	
	__shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
	__shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
	const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
	const int THREAD_NUM_PER_BLOCK = THREAD_Y_PER_BLOCK * THREAD_X_PER_BLOCK;

	const int tid = ty * THREAD_X_PER_BLOCK + tx;

	float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = { 0.0 };
	float frag_a[2][THREAD_SIZE_Y];
	float frag_b[2][THREAD_SIZE_X];

	const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
	const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

	const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
	const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;
	
	const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
	const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;
	
	const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
	const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

	const int ldg_num_a = BLOCK_SIZE_K * BLOCK_SIZE_M / THREAD_NUM_PER_BLOCK;
	const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / THREAD_NUM_PER_BLOCK;

	float ldg_a_reg[ldg_num_a];
	float ldg_b_reg[ldg_num_b];
	
	A = A + by * BLOCK_SIZE_M * K;
	B = B + bx * BLOCK_SIZE_N;
	
	
	// tile_idx = 0 µÚÒ»¸öK_BLOCK
	#pragma unroll
	for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
		int ldg_index = i / A_TILE_ROW_STRIDE * 4;
		FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(i + A_TILE_ROW_START, A_TILE_COL, K)]);
		As[0][A_TILE_COL][i + A_TILE_ROW_START] = ldg_a_reg[ldg_index];
		As[0][A_TILE_COL + 1][i + A_TILE_ROW_START] = ldg_a_reg[ldg_index + 1];
		As[0][A_TILE_COL + 2][i + A_TILE_ROW_START] = ldg_a_reg[ldg_index + 2];
		As[0][A_TILE_COL + 3][i + A_TILE_ROW_START] = ldg_a_reg[ldg_index + 3];
	}
	#pragma unroll
	for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
		//int ldg_index = i / B_TILE_ROW_STRIDE * 4;
		FETCH_FLOAT4(Bs[0][i+B_TILE_ROW_START][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(i + B_TILE_ROW_START, B_TILE_COL, N)]);
	}

	__syncthreads();

	#pragma unroll
	for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
		FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[0][0][ty * THREAD_SIZE_Y + thread_y]);
	}
	#pragma unroll
	for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
		FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[0][0][tx * THREAD_SIZE_X + thread_x]);
	}

	int write_stage_idx = 1;
	int tile_idx = 0;
	
	do {

		tile_idx += BLOCK_SIZE_K;

		if (tile_idx < K) {
			#pragma unroll
			for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
				int ldg_index = i / A_TILE_ROW_STRIDE * 4;
				FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(i + A_TILE_ROW_START, A_TILE_COL + tile_idx, K)]);
			}
			#pragma unroll
			for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
				int ldg_index = i / B_TILE_ROW_STRIDE * 4;
				FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(i + B_TILE_ROW_START + tile_idx, B_TILE_COL, N)]);
			}
		}
	
		int load_stage_idx = write_stage_idx ^ 1;
		
		#pragma unroll
		for (int k = 0; k < BLOCK_SIZE_K - 1; k++) {
			// frag_a[k % 2] <- As[load_stage_idx]
			#pragma unroll
			for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
				FETCH_FLOAT4(frag_a[(k + 1) % 2][thread_y]) = FETCH_FLOAT4(As[load_stage_idx][k + 1][ty * THREAD_SIZE_Y + thread_y]);
			}
			// frag_a[k % 2] <- As[load_stage_idx]
			#pragma unroll
			for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
				FETCH_FLOAT4(frag_b[(k + 1) % 2][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx][k + 1][tx * THREAD_SIZE_X + thread_x]);
			}
			#pragma unroll
			for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++) {
				#pragma unroll
				for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x++) {
					accum[thread_y][thread_x] += frag_a[k % 2][thread_y] * frag_b[k % 2][thread_x];
				}
			}
		}

		if(tile_idx < K){
			#pragma unroll
			for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
				int ldg_index = i / A_TILE_ROW_STRIDE * 4;
				As[write_stage_idx][A_TILE_COL][i + A_TILE_ROW_START] = ldg_a_reg[ldg_index];
				As[write_stage_idx][A_TILE_COL + 1][i + A_TILE_ROW_START] = ldg_a_reg[ldg_index + 1];
				As[write_stage_idx][A_TILE_COL + 2][i + A_TILE_ROW_START] = ldg_a_reg[ldg_index + 2];
				As[write_stage_idx][A_TILE_COL + 3][i + A_TILE_ROW_START] = ldg_a_reg[ldg_index + 3];
			}
			#pragma unroll
			for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
				int ldg_index = i / B_TILE_ROW_STRIDE * 4;
				FETCH_FLOAT4(Bs[write_stage_idx][i + B_TILE_ROW_START][B_TILE_COL]) = \
					FETCH_FLOAT4(ldg_b_reg[ldg_index]);
			}
			__syncthreads();

			write_stage_idx ^= 1;
		
			
		}
		#pragma unroll
		for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
			FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[load_stage_idx ^ 1][0][ty * THREAD_SIZE_Y + thread_y]);
		}
		#pragma unroll
		for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
			FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx ^ 1][0][tx * THREAD_SIZE_X + thread_x]);
		}
		#pragma unroll
		for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++) {
			#pragma unroll
			for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x++) {
				accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
			}
		}

	} while (tile_idx < K);
		
	#pragma unroll
	for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++) {
		#pragma unroll
		for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
			int row = by * BLOCK_SIZE_M + ty * THREAD_SIZE_Y + thread_y;
			int col = bx * BLOCK_SIZE_N + tx * THREAD_SIZE_X + thread_x;
			FETCH_FLOAT4(C[OFFSET(row, col, N)]) = FETCH_FLOAT4(accum[thread_y][thread_x]);
		}
	}
}





