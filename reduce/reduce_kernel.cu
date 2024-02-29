#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/* baseline */
template<const int THREAD_PER_BLOCK>
__global__ void reduce_v0(float* d_in, float* d_out) {
	__shared__ float sdata[THREAD_PER_BLOCK];
	
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	sdata[tid] = d_in[gid];
	__syncthreads();

	for (int s = 1; s < THREAD_PER_BLOCK; s <<= 1) {
		if (tid % (s<<1) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		d_out[blockIdx.x] = sdata[0];
		//printf("%d, %d\n", THREAD_PER_BLOCK, blockDim.x);

	}
}

/* no divergence_branch */
template<const int THREAD_PER_BLOCK>
__global__ void reduce_v1(float* d_in, float* d_out) {
	__shared__ float sdata[THREAD_PER_BLOCK];

	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	sdata[tid] = d_in[gid];
	__syncthreads();

	int index = tid;
	for (int s = 1; s < THREAD_PER_BLOCK; s <<= 1) {
		index <<= 1;
		if(index < blockDim.x){
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}

	if (tid == 0)
		d_out[blockIdx.x] = sdata[0];
}

/* no bank conflict */
template<const int THREAD_PER_BLOCK>
__global__ void reduce_v2(float* d_in, float* d_out) {
	__shared__ float sdata[THREAD_PER_BLOCK];

	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	sdata[tid] = d_in[gid];
	__syncthreads();

	for(int s=THREAD_PER_BLOCK>>1; s>0; s>>=1){
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();	
	}

	if (tid == 0)
		d_out[blockIdx.x] = sdata[0];
}

/* no idle */
template<const int THREAD_PER_BLOCK>
__global__ void reduce_v3(float* d_in, float* d_out) {
	__shared__ float sdata[THREAD_PER_BLOCK];

	int gid = blockIdx.x * (THREAD_PER_BLOCK*2) + threadIdx.x;
	int tid = threadIdx.x;
	sdata[tid] = d_in[gid] + d_in[gid + THREAD_PER_BLOCK];
	__syncthreads();

	for (int s = THREAD_PER_BLOCK >> 1; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0)
		d_out[blockIdx.x] = sdata[0];
}

/* warp_reduce */
__device__ __inline__ void warpReduce(volatile float* cache, int tid) {
	cache[tid] += cache[tid + 32];
	cache[tid] += cache[tid + 16];
	cache[tid] += cache[tid + 8];
	cache[tid] += cache[tid + 4];
	cache[tid] += cache[tid + 2];
	cache[tid] += cache[tid + 1];
}

template<const int THREAD_PER_BLOCK>
__global__ void reduce_v4(float* d_in, float* d_out) {
	__shared__ float sdata[THREAD_PER_BLOCK];

	int gid = blockIdx.x * (2 * THREAD_PER_BLOCK) + threadIdx.x;
	int tid = threadIdx.x;
	sdata[tid] = d_in[gid] + d_in[gid + THREAD_PER_BLOCK];
	__syncthreads();

	for (int s = THREAD_PER_BLOCK>>1; s > 32; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid < 32)
		warpReduce(sdata, tid);

	if (tid == 0)
		d_out[blockIdx.x] = sdata[0];
}

/* total unroll */
template<const int THREAD_PER_BLOCK>
__global__ void reduce_v5(float* d_in, float* d_out) {
	__shared__ float sdata[THREAD_PER_BLOCK];

	int gid = blockIdx.x * (THREAD_PER_BLOCK * 2) + threadIdx.x;
	int tid = threadIdx.x;
	sdata[tid] = d_in[gid] + d_in[gid + THREAD_PER_BLOCK];
	__syncthreads();

	if (THREAD_PER_BLOCK >= 1024) {
		if (tid < 512) {
			sdata[tid] += sdata[tid + 512];
		}
		__syncthreads();
	}
	if (THREAD_PER_BLOCK >= 512) {
		if (tid < 256) {
			sdata[tid] += sdata[tid + 256];
		}
		__syncthreads();
	}
	if (THREAD_PER_BLOCK >= 256) {
		if (tid < 128) {
			sdata[tid] += sdata[tid + 128];
		}
		__syncthreads();
	}
	if (THREAD_PER_BLOCK >= 128) {
		if (tid < 64) {
			sdata[tid] += sdata[tid + 64];
		}
		__syncthreads();
	}
	if (tid < 32)
		warpReduce(sdata, tid);
	if (tid == 0)
		d_out[blockIdx.x] = sdata[0];
}


/*--------------------------------multi add--------------------------------*/
template<const int THREAD_PER_BLOCK, const int TILE_PER_THREAD>
__global__ void reduce_v6(float* d_in, float* d_out) {
	__shared__ float sdata[THREAD_PER_BLOCK];

	int tile_size = gridDim.x * (THREAD_PER_BLOCK * 2);
	int gid = blockIdx.x * (2 * THREAD_PER_BLOCK) + threadIdx.x;
	int tid = threadIdx.x;

	sdata[tid] = 0.0;
	for (int tile_idx = 0; tile_idx < TILE_PER_THREAD; tile_idx++) {
		int offset = tile_idx * tile_size;
		sdata[tid] += d_in[offset + gid] + d_in[offset + gid + THREAD_PER_BLOCK];
	}
	__syncthreads();

	for (int s = THREAD_PER_BLOCK >> 1; s > 32; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid < 32)
		warpReduce(sdata, tid);

	if (tid == 0)
		d_out[blockIdx.x] = sdata[0];
}

/*--------------------------------shuffle--------------------------------*/
__device__ __forceinline__ float warpReduceSum(float sum) {
	sum += __shfl_down_sync(0xffffffff, sum, 16);
	sum += __shfl_down_sync(0xffffffff, sum, 8);
	sum += __shfl_down_sync(0xffffffff, sum, 4);
	sum += __shfl_down_sync(0xffffffff, sum, 2);
	sum += __shfl_down_sync(0xffffffff, sum, 1);

}


