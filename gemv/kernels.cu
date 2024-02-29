#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define OFFSET(row, col, ld) ((row)*(ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])


template <int warpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
	if (warpSize >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16);
	if (warpSize >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);
	if (warpSize >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);
	if (warpSize >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);
	if (warpSize >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);
	return sum;
}

// for N=32
__global__ void Sgemv_v0(float* A, float* x, float* y, const int M, const int N) {
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	const int warpSize = 32;
	int row_idx = bx * blockDim.y + ty;

	if (row_idx < M) {
		float sum = 0.0;
		#pragma unroll
		for (int k = tx; k < N; k += warpSize) {
			sum += A[OFFSET(row_idx, k, N)] * x[k];
		}
		sum = warpReduceSum<warpSize>(sum);
		if (tx == 0)
			y[row_idx] = sum;
	}
}

// for N >= 128(4*32)
__global__ void Sgemv_v1(float* A, float* x, float* y, const int M, const int N) {
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	const int warpSize = 32;
	int row_idx = bx * blockDim.y + ty;

	if (row_idx < M) {
		float sum = 0.0;
		for (int k = tx*4; k < N; k += warpSize * 4) {
			// float4只能用于向量化访存，而不支持运算
			float4 dA = FETCH_FLOAT4(A[OFFSET(row_idx, k, N)]);
			float4 dx = FETCH_FLOAT4(x[k]);
			dA.x *= dx.x;
			dA.y *= dx.y;
			dA.z *= dx.z;
			dA.w *= dx.w;
			sum += dA.x + dA.y + dA.z + dA.w;
		}
		sum = warpReduceSum<warpSize>(sum);
		if (tx == 0)
			y[row_idx] = sum;
	}
}

__global__ void Sgemv_v2(float* A, float* x, float* y, const int M, const int N) {
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	const int warpSize = 32;
	// one warp for 2 rows
	int row_idx = (bx * blockDim.y + ty) * 2;

	if (row_idx < M) {
		float sum[2] = {0.0};
		for (int k = tx; k < (N << 1); k += warpSize) {
			int sub_row_idx = k / N;
			sum[sub_row_idx] += A[OFFSET(row_idx + sub_row_idx, k % N, N)] * x[k % N];
		}
		sum[0] = warpReduceSum<warpSize>(sum[0]);
		sum[1] = warpReduceSum<warpSize>(sum[1]);
		if (tx == 0) {
			y[row_idx] = sum[0];
			y[row_idx + 1] = sum[1];
		}
	}
}

// blockDim.x个线程处理一行
template<const int BLOCK_DIM_X, const int BLOCK_DIM_Y>
__global__ void Sgemv_v3(float* A, float* x, float* y, const int M, const int N) {
	__shared__ float sdata[BLOCK_DIM_Y][BLOCK_DIM_X];
	int bx = blockIdx.x;
	int tx = threadIdx.x; // blockDim.x == BLOCK_DIM_X
	int ty = threadIdx.y;
	const int warpSize = 32;
	int row_idx = bx * blockDim.y + ty;

	//if (row_idx >= M && tx==0) {
	//	printf("row_idx=%d\n", row_idx);
	//}

	if (row_idx < M) {
		float sum = 0.0;
		for (int k = tx; k < N; k += BLOCK_DIM_X) {
			sum += A[OFFSET(row_idx, k, N)] * x[k];
		}
		sdata[ty][tx] = sum;
		__syncthreads();

		for (int s = BLOCK_DIM_X >> 1; s >= warpSize; s >>= 1) {
			if (tx < s) {
				sdata[ty][tx] += sdata[ty][tx + s];
			}
			__syncthreads();
		}
		if (tx < warpSize)
			sum = warpReduceSum<warpSize>(sdata[ty][tx]);

		if (tx == 0)
			y[row_idx] = sum;
	}
}

