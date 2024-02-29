#ifndef WARP_IMPL_CUH
#define WARP_IMPL_CUH

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Operators.cuh"


// ^V^ 一个warp处理一行
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
    int thread_group_width, int rows_per_access, bool padding, Algorithm algorithm>
__global__ void SoftmaxWarpImpl(LOAD load, STORE store, int64_t rows, int64_t cols) {
    constexpr int num_packs = cols_per_thread / pack_size; // num_packs在这里表示一个线程处理的pack数
    ComputeType buf[rows_per_access][cols_per_thread];
    const int64_t num_global_thread_group = gridDim.x * blockDim.y;
    const int64_t global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane_id = threadIdx.x;
    const int64_t step = num_global_thread_group * rows_per_access;

    for (int64_t row = global_thread_group_id * rows_per_access; row < rows; row += step) {

        ComputeType thread_max[rows_per_access];
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; row_id++) {
            thread_max[row_id] = -Inf<ComputeType>();
            ComputeType* row_buf = buf[row_id];
#pragma unroll
            for (int pack_id = 0; pack_id < num_packs; pack_id++) {
                int64_t col = (pack_id * thread_group_width + lane_id) * pack_size;
                int pack_offset = pack_id * pack_size;
                if (!padding || col < cols) {
                    load.template load<pack_size>(row_buf + pack_offset, row + row_id, col); // 从全局内存load到寄存器
#pragma unroll
                    for (int i = 0; i < pack_size; i++) {
                        thread_max[row_id] = max(thread_max[row_id], row_buf[pack_offset + i]);
                    }
                }
                else {
#pragma unroll
                    for (int i = 0; i < pack_size; i++) {
                        row_buf[pack_offset + i] = -Inf<ComputeType>();
                    }
                }
            }
        }
        ComputeType warp_max[rows_per_access];
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; row_id++) {
            warp_max[row_id] = WarpAllReduce<MaxOp, ComputeType, thread_group_width>(thread_max[row_id]);
        }

        ComputeType thread_sum[rows_per_access];
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; row_id++) {
            thread_sum[row_id] = 0;
            ComputeType* row_buf = buf[row_id];
#pragma unroll
            for (int col = 0; col < cols_per_thread; col++) {
                if (algorithm == Algorithm::kSoftmax) {
                    row_buf[col] = Exp(row_buf[col] - warp_max[row_id]);
                    thread_sum[row_id] += row_buf[col];
                }
                else if (algorithm == Algorithm::kLogSoftmax) {
                    row_buf[col] -= warp_max[row_id];
                    thread_sum[row_id] += Exp(row_buf[col]);
                }
                else {
                    __trap();
                }
            }
        }
        ComputeType warp_sum[rows_per_access];
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; row_id++) {
            warp_sum[row_id] = WarpAllReduce<SumOp, ComputeType, thread_group_width>(thread_sum[row_id]);
        }

        for (int row_id = 0; row_id < rows_per_access; row_id++) {
            ComputeType* row_buf = buf[row_id];
            for (int col = 0; col < cols_per_thread; col++) {
                if (algorithm == Algorithm::kSoftmax) {
                    row_buf[col] = Div(row_buf[col], warp_sum[row_id]);
                }
                else if (algorithm == Algorithm::kLogSoftmax) {
                    row_buf[col] -= Log(warp_sum[row_id]);
                }
                else {
                    __trap();
                }
            }

            for (int pack_id = 0; pack_id < num_packs; pack_id++) {
                int64_t col = (pack_id * thread_group_width + lane_id) * pack_size;
                if (!padding || col < cols) {
                    store.template store<pack_size>(row_buf + pack_id * pack_size, row + row_id, col);
                }
            }
        }
    }
}

// ^V^
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
    int thread_group_width, int rows_per_access, bool padding, Algorithm algorithm>
inline cudaError_t LaunchSoftmaxWarpImpl(cudaStream_t stream, LOAD load, STORE store, int64_t rows, int64_t cols) {
    constexpr int block_size = 128;
    constexpr int waves = 32; // 批数量
    constexpr int thread_groups_per_block = block_size / thread_group_width;
    dim3 block_dim(thread_group_width, thread_groups_per_block);
    const int64_t max_num_blocks = (rows / rows_per_access + block_dim.y - 1) / block_dim.y;
    int grid_dim_x;
    {
        cudaError_t err = GetNumBlocks(block_size, max_num_blocks, waves, &grid_dim_x);
        if (err != cudaSuccess) return err;
    }
    SoftmaxWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread, 
        thread_group_width, rows_per_access, padding, algorithm>
        << <grid_dim_x, block_dim, 0, stream >> > (load, store, rows, cols);
    return cudaPeekAtLastError();
}

// ^V^
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
    int thread_group_width, int rows_per_access, Algorithm algorithm>
inline cudaError_t DispatchSoftmaxWarpImplPadding(cudaStream_t stream, LOAD load, STORE store,
    const int64_t rows, const int64_t cols) {
    // 如果每个线程处理的元素个数 * 处理元素的线程组的宽度(warp_size)和cols相等，就不需要padding
    if (cols == cols_per_thread * thread_group_width) {
        return LaunchSoftmaxWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread,
            thread_group_width, rows_per_access, false, algorithm>(
                stream, load, store, rows, cols);
    }
    else {
        return LaunchSoftmaxWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread,
            thread_group_width, rows_per_access, true, algorithm>(
                stream, load, store, rows, cols);
    }
}

// ^V^
// 确定cols_per_thread、thread_group_width和rows_per_access
// 若cols<=kWarpSize*pack_size，则cols_per_thread=pack_size，求出thread_group_width满足cols<=thread_group_width*pack_size
// 若cols>kWarpSize*pack_size，则cols_per_thread>pack_size，thread_group_width=kWarpSize，求出cols_per_thread满足cols<=kWarpSize*cols_per_thread
// 若rows%2==0, 则rows_per_access=2, 否则rows_per_access=1
// pack_size==1的情况
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
typename std::enable_if<pack_size == 1, cudaError_t>::type DispatchSoftmaxWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, int64_t rows, int64_t cols) {
    if (cols <= 0) return cudaErrorInvalidValue;
#define DEFINE_ONE_ELIF(thread_group_width)                                                         \
    else if(cols <= (thread_group_width)*pack_size){                                                \
        if (rows % 2 == 0) {                                                                        \
            return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType,                         \
                        pack_size, pack_size, thread_group_width, 2, algorithm>(                    \
                        stream, load, store, rows, cols);                                           \
        } else {                                                                                    \
            return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType,                         \
                        pack_size, pack_size, thread_group_width, 1, algorithm>(                    \
                        stream, load, store, rows, cols);                                           \
        }                                                                                           \
    }
    DEFINE_ONE_ELIF(1)
        DEFINE_ONE_ELIF(2)
        DEFINE_ONE_ELIF(4)
        DEFINE_ONE_ELIF(8)
        DEFINE_ONE_ELIF(16)
        DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(cols_per_thread)                                                            \
    else if (cols <= (cols_per_thread)*kWarpSize) {                                                 \
        if (rows % 2 == 0) {                                                                        \
            return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType,                         \
                        pack_size, cols_per_thread, kWarpSize, 2, algorithm>(                       \
                        stream, load, store, rows, cols);                                           \
        } else {                                                                                    \
            return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType,                         \
                        pack_size, cols_per_thread, kWarpSize, 1, algorithm>(                       \
                        stream, load, store, rows, cols);                                           \
        }                                                                                           \
    }
    DEFINE_ONE_ELIF(2)
        DEFINE_ONE_ELIF(3)
        DEFINE_ONE_ELIF(4)
        DEFINE_ONE_ELIF(5)
        DEFINE_ONE_ELIF(6)
        DEFINE_ONE_ELIF(7)
        DEFINE_ONE_ELIF(8)
        DEFINE_ONE_ELIF(9)
        DEFINE_ONE_ELIF(10)
        DEFINE_ONE_ELIF(11)
        DEFINE_ONE_ELIF(12)
        DEFINE_ONE_ELIF(13)
        DEFINE_ONE_ELIF(14)
        DEFINE_ONE_ELIF(15)
        DEFINE_ONE_ELIF(16)
        DEFINE_ONE_ELIF(17)
        DEFINE_ONE_ELIF(18)
        DEFINE_ONE_ELIF(19)
        DEFINE_ONE_ELIF(20)
        DEFINE_ONE_ELIF(21)
        DEFINE_ONE_ELIF(22)
        DEFINE_ONE_ELIF(23)
        DEFINE_ONE_ELIF(24)
        DEFINE_ONE_ELIF(25)
        DEFINE_ONE_ELIF(26)
        DEFINE_ONE_ELIF(27)
        DEFINE_ONE_ELIF(28)
        DEFINE_ONE_ELIF(29)
        DEFINE_ONE_ELIF(30)
        DEFINE_ONE_ELIF(31)
        DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
        return cudaErrorInvalidValue;
}
// pack_size==2的情况
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
typename std::enable_if<pack_size == 2, cudaError_t>::type DispatchSoftmaxWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, int64_t rows, int64_t cols) {
    if (cols <= 0) return cudaErrorInvalidValue;
#define DEFINE_ONE_ELIF(thread_group_width)                                                         \
    else if (cols <= (thread_group_width) * pack_size) {                                            \
        if (rows % 2 == 0) {                                                                        \
            return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size,   \
                        thread_group_width, 2, algorithm>(stream, load, store, rows, cols);         \
        } else {                                                                                    \
            return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size,   \
                        thread_group_width, 1, algorithm>(stream, load, store, rows, cols);         \
        }                                                                                           \
    }
    DEFINE_ONE_ELIF(1)
        DEFINE_ONE_ELIF(2)
        DEFINE_ONE_ELIF(4)
        DEFINE_ONE_ELIF(8)
        DEFINE_ONE_ELIF(16)
        DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(cols_per_thread)                                                            \
    else if (cols <= (cols_per_thread) * kWarpSize) {                                               \
        if (rows % 2 == 0) {                                                                        \
            return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size,              \
                    cols_per_thread, kWarpSize, 2, algorithm>(stream, load, store, rows, cols);     \
        } else {                                                                                    \
            return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size,              \
                    cols_per_thread, kWarpSize, 2, algorithm>(stream, load, store, rows, cols);     \
        }                                                                                           \
    }
        DEFINE_ONE_ELIF(4) // 要保证cols_per_thread%pack_size==0 
        DEFINE_ONE_ELIF(6)
        DEFINE_ONE_ELIF(8)
        DEFINE_ONE_ELIF(10)
        DEFINE_ONE_ELIF(12)
        DEFINE_ONE_ELIF(14)
        DEFINE_ONE_ELIF(16)
        DEFINE_ONE_ELIF(18)
        DEFINE_ONE_ELIF(20)
        DEFINE_ONE_ELIF(22)
        DEFINE_ONE_ELIF(24)
        DEFINE_ONE_ELIF(26)
        DEFINE_ONE_ELIF(28)
        DEFINE_ONE_ELIF(30)
        DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
    return cudaErrorInvalidValue;
}



// ^V^ 通过cols是否为偶数决定pack_size=1或2
// 每个Warp处理一行或两行元素时最原始的dispatch接口
template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
inline cudaError_t DispatchSoftmaxWarpImpl(cudaStream_t stream, LOAD load, STORE store,
    const int64_t rows, const int64_t cols) {
    if (cols % 2 == 0) {
        return DispatchSoftmaxWarpImplCols<LOAD, STORE, ComputeType, 2, algorithm>(stream, load,
            store, rows, cols);
    }
    else {
        //printf("pack_size == 1\n");
        return DispatchSoftmaxWarpImplCols<LOAD, STORE, ComputeType, 1, algorithm>(stream, load,
            store, rows, cols);
    }
}




#endif

