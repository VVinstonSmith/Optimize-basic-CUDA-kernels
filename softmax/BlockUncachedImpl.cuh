#ifndef BLOCK_UNCACHED_IMPL_CUH
#define BLOCK_UNCACHED_IMPL_CUH

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include "Operators.cuh"

// ^V^
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size, Algorithm algorithm>
__global__ void SoftmaxBlockUncachedImpl(LOAD load, STORE store, const int64_t rows, const int64_t cols) {
    const int num_packs = cols / pack_size;
    const int tid = threadIdx.x;
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        ComputeType thread_max = -Inf<ComputeType>();
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
            ComputeType pack[pack_size];
            load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
            for (int i = 0; i < pack_size; i++) {
                thread_max = max(thread_max, pack[i]);
            }
        }
        const ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max);

        ComputeType thread_sum = 0;
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
            ComputeType pack[pack_size];
            load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
            for (int i = 0; i < pack_size; i++) {
                thread_sum += Exp(pack[i] - row_max);
            }
        }
        const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);

        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
            ComputeType pack[pack_size];
            load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
            for (int i = 0; i < pack_size; i++) {
                if (algorithm == Algorithm::kSoftmax) {
                    pack[i] = Div(Exp(pack[i] - row_max), row_sum);
                }
                else if (algorithm == Algorithm::kLogSoftmax) {
                    pack[i] = (pack[i] - row_max) - Log(row_sum);
                }
                else {
                    __trap();
                }
            }
            store.template store<pack_size>(pack, row, pack_id * pack_size);
        }
    }
}

// ^V^
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
inline cudaError_t LaunchSoftmaxBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store,
    const int64_t rows, const int64_t cols) {
    // 每个 Block 使用 1024 个线程
    constexpr int block_size = 64;
    // waves 需要满足32组
    constexpr int waves = 32;
    // 根据 BlockSize 以及硬件参数计算 Block 数量
    int grid_dim_x;
    {
        cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
        if (err != cudaSuccess) { return err; }
    }
    // 启动第三者实现的 cuda kernel
    SoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, pack_size, block_size, algorithm>
        << <grid_dim_x, block_size, 0, stream >> > (load, store, rows, cols);
    return cudaPeekAtLastError();
}

// ^V^
template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
inline cudaError_t DispatchSoftmaxBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store,
    const int64_t rows, const int64_t cols) {
    if (cols % 2 == 0) {
        return LaunchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, 2, algorithm>(
            stream, load, store, rows, cols);
    }
    else {
        return LaunchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, 1, algorithm>(
            stream, load, store, rows, cols);
    }
}


#endif