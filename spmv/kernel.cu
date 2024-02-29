
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template<const int warpSize>
__device__ float warpReduce(float sum) {
	if (warpSize >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16);
	if (warpSize >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);
	if (warpSize >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);
	if (warpSize >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);
	if (warpSize >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);
	return sum;
}

template<int THREADS_PER_ROW>
__global__ void spmv_csr_kernel(int n_rows,
	const int* row_offset, const int* col_index,
	const float* A_vals, const float* x, float* y) {

	const int tid = threadIdx.x;
	const int id = blockDim.x * blockIdx.x + tid;
	const int row_idx = id / THREADS_PER_ROW;
	const int lane_id = id % THREADS_PER_ROW;

	if (row_idx < n_rows) {
		const int start = row_offset[row_idx];
		const int end = row_offset[row_idx + 1];
		float sum = 0.0;
		for (int i = lane_id; i < end - start; i += THREADS_PER_ROW) {
			sum += x[col_index[start+i]] * A_vals[start + i];
		}
		sum = warpReduce<THREADS_PER_ROW>(sum);

		if (lane_id == 0) {
			y[row_idx] = sum;
			//printf("id=%d y[%d]=%f\n", id, row_idx, sum);
		}
	}
}
