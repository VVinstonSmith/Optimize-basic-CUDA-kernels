#ifndef BLOCK_SMEM_IMPL_CUH
#define BLOCK_SMEM_IMPL_CUH

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include "Operators.cuh"

// ^V^
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size, Algorithm algorithm>
__global__ void SoftmaxBlockSMemImpl(LOAD load, STORE store, int64_t rows, int64_t cols) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];//��̬�����ڴ棬��double�Ĵ�С����
    auto* buf = reinterpret_cast<ComputeType*>(shared_buf);
    const int num_packs = cols / pack_size;
    const int tid = threadIdx.x;
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        ComputeType thread_max = -Inf<ComputeType>();
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
            ComputeType pack[pack_size];
            load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
            for (int i = 0; i < pack_size; i++) {
                buf[i * num_packs + pack_id] = pack[i];
                thread_max = max(thread_max, pack[i]);
            }
        }
        ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max);

        ComputeType thread_sum = 0;
        for (int col = tid; col < cols; col += block_size) {
            if (algorithm == Algorithm::kSoftmax) {
                ComputeType exp_x = Exp(buf[col] - row_max);
                buf[col] = exp_x;
                thread_sum += exp_x;
            }
            else if (algorithm == Algorithm::kLogSoftmax) {
                ComputeType x = buf[col] - row_max;
                buf[col] = x;
                thread_sum + Exp(x);
            }
        }
        ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);

        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
            ComputeType pack[pack_size];
#pragma unroll
            for (int i = 0; i < pack_size; i++) {
                if (algorithm == Algorithm::kSoftmax) {
                    pack[i] = Div(buf[i * num_packs + pack_id], row_sum);
                }
                else if (algorithm == Algorithm::kLogSoftmax) {
                    pack[i] = buf[i * num_packs + pack_id] - Log(row_sum);
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
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size,
    Algorithm algorithm>
inline cudaError_t LaunchSoftmaxBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store, int smem,
    const int64_t rows, const int64_t cols) {
    constexpr int waves = 32;
    int grid_dim_x;
    {
        cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
        if (err != cudaSuccess) { return err; }
    }
    SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size, algorithm>
        << <grid_dim_x, block_size, smem, stream >> > (load, store, rows, cols);
    return cudaPeekAtLastError();
}

// ^V^
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
inline cudaError_t TryDispatchSoftmaxBlockSMemImplBlockSize(cudaStream_t stream, LOAD load,
    STORE store, int64_t rows, int64_t cols, bool* success) {
    constexpr int block_size_conf_1 = 128;
    constexpr int block_size_conf_2 = 256;
    constexpr int block_size_conf_3 = 512;
    constexpr int block_size_conf_4 = 1024;
    const size_t smem = cols * sizeof(ComputeType); // һ��һ��block�������ڴ�Ĵ�С
    // block_sizeȡ��Сֵ��һ��SM���ж��ٸ�block����ִ��
    int max_active_blocks_conf_1;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_conf_1,
            SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1, algorithm>,
            block_size_conf_1, smem);
        if (err != cudaSuccess) return err;
    }
    if (max_active_blocks_conf_1 <= 0) {
        *success = false;
        return cudaSuccess;
    }

    // ����ʹ���ܲ���ִ�е�block��࣬����ٿ�����block_size�����ܴ�
    int max_active_blocks_conf_4;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_conf_4,
            SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_4, algorithm>,
            block_size_conf_4, smem);
        if (err != cudaSuccess) return err;
    }
    if (max_active_blocks_conf_4 == max_active_blocks_conf_1) {
        *success = true;
        return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType,
                    pack_size, block_size_conf_4, algorithm>(stream, load, store, smem, rows, cols);
    }

    int max_active_blocks_conf_3;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_conf_3,
            SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_3, algorithm>,
            block_size_conf_3, smem);
        if (err != cudaSuccess) return err;
    }
    if (max_active_blocks_conf_3 == max_active_blocks_conf_1) {
        *success = true;
        return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType,
            pack_size, block_size_conf_3, algorithm>(stream, load, store, smem, rows, cols);
    }

    int max_active_blocks_conf_2;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_conf_2,
            SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_2, algorithm>,
            block_size_conf_2, smem);
        if (err != cudaSuccess) return err;
    }
    if (max_active_blocks_conf_2 == max_active_blocks_conf_1) {
        *success = true;
        return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType,
            pack_size, block_size_conf_2, algorithm>(stream, load, store, smem, rows, cols);
    }

    *success = true;
    return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size,
        block_size_conf_1, algorithm>(stream, load, store, smem, rows, cols);
}


// ^V^ ����������ż����pack_size
template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
inline cudaError_t TryDispatchSoftmaxBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store,
    const int64_t rows, const int64_t cols, bool* success) {
    if (cols % 2 == 0) {
        return TryDispatchSoftmaxBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 2, algorithm>(
            stream, load, store, rows, cols, success);
    }
    else {
        return TryDispatchSoftmaxBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 1, algorithm>(
            stream, load, store, rows, cols, success);
    }
}


#endif