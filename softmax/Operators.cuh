#ifndef TEMPLATE_OP_CUH
#define TEMPLATE_OP_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <math_constants.h>

#define OFFSET(row, col, N) ((row)*(N)+(col))

constexpr int kWarpSize = 32;

/*=================================math operators=================================*/
// ^V^
template<typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(T a, T b) {
        return a + b;
    }
};

// ^V^
template<typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(T a, T b) {
        return max(a, b);
    }
};

// ^V^
template<typename T>
__inline__ __device__ T Inf();
template<>
__inline__ __device__ float Inf<float>() {
    return CUDART_INF_F;
}
template<>
__inline__ __device__ double Inf<double>() {
    return CUDART_INF;
}

// ^V^
template<typename T>
__inline__ __device__ T Exp(T val);
template<>
__inline__ __device__ float Exp<float>(float val) {
#ifdef USE_FAST_MATH
    return __exp(val);
#else
    return exp(val);
#endif
}
template<>
__inline__ __device__ double Exp<double>(double val) {
    return exp(val);
}

// ^V^
template<typename T>
__inline__ __device__ T Log(T val);
template<>
__inline__ __device__ float Log<float>(float val) {
#ifdef USE_FAST_MATH
    return __log(val);
#else
    return log(val);
#endif
}
template<>
__inline__ __device__ double Log<double>(double val) {
    return log(val);
}

// ^V^
template<typename T>
__inline__ __device__ T Div(T a, T b);
template<>
__inline__ __device__ float Div(float a, float b) {
#ifdef USE_FAST_MATH
    return __fdevidef(a, b);
#else
    return a / b;
#endif
}
template<>
__inline__ __device__ double Div(double a, double b) {
    return a / b;
}

enum class Algorithm {
    kSoftmax = 0,
    kLogSoftmax = 1,
};
/*=================================math operators=================================*/

/*=================================reduce operators=================================*/
// ^V^ warp_reduce
template<template<typename> class ReduceOp, typename T, int thread_group_width = kWarpSize>
__inline__ __device__ T WarpAllReduce(T val) {
    for (int mask = thread_group_width >> 1; mask > 0; mask >>= 1) {
        val = ReduceOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

// ^V^ block_reduce
template<template<typename> class ReduceOp, typename T, int block_size>
__inline__ __device__ T BlockAllReduce(T val) {
    typedef cub::BlockReduce<T, block_size> BlockReduce; // BlockReduce类, 模板参数：T,block_size
    __shared__ typename BlockReduce::TempStorage temp_storage; // 中间结果的共享缓存
    T result = BlockReduce(temp_storage).Reduce(val, ReduceOp<T>()); // 每个线程得到自己的规约结果，只有threadId0结果有效
    __shared__ T result_broadcast; // 共享计算结果
    if (threadIdx.x == 0)
        result_broadcast = result;
    __syncthreads();
    return result_broadcast;
}
/*=================================reduce operators=================================*/

/*=================================get block paras=================================*/
// ^V^ compute how many blocks to create
inline cudaError_t GetNumBlocks(int64_t block_size, int64_t max_blocks, int64_t waves, int* num_blocks) {
    int dev;
    {
        auto err = cudaGetDevice(&dev);
        if (err != cudaSuccess) return err;
    }
    int n_SM;
    {
        auto err = cudaDeviceGetAttribute(&n_SM, cudaDevAttrMultiProcessorCount, dev);
        if (err != cudaSuccess) return err;
    }
    int max_threads_per_sm;
    {
        auto err = cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
        if (err != cudaSuccess) return err;
    }
    *num_blocks = std::max<int>(1, std::min<int64_t>(n_SM * max_threads_per_sm / block_size * waves, max_blocks));
    return cudaSuccess;
}
/*=================================get block paras=================================*/

/*=================================load/store operators=================================*/
// ^V^
template<typename T, int N>
using PackType = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;

// ^V^
template<typename T, int N>
union Pack {
    __device__ Pack() {}
    PackType<T, N> storage;
    T elem[N];
};

// ^V^
template<typename SRC, typename DST>
struct DirectLoad {
    DirectLoad(SRC* src, int64_t row_size) : src(src), row_size(row_size) {}
    template<int N>
    __device__ void load(DST* dst, int64_t row, int64_t col) {
        Pack<SRC, N> pack;
        const int64_t offset = OFFSET(row, col, row_size) / N;
        pack.storage = *(reinterpret_cast<const PackType<SRC, N>*>(src) + offset);
#pragma unroll
        for (int i = 0; i < N; i++) {
            dst[i] = static_cast<DST>(pack.elem[i]);
        }
    }
    const SRC* src;
    const int64_t row_size;
};

// ^V^
template<typename SRC, typename DST>
struct DirectStore {
    DirectStore(DST* dst, int64_t row_size) : dst(dst), row_size(row_size) {}
    template<int N>
    __device__ void store(SRC* src, int64_t row, int64_t col) {
        Pack<DST, N> pack;
        const int64_t offset = OFFSET(row, col, row_size) / N;
#pragma unroll
        for (int i = 0; i < N; i++) {
            pack.elem[i] = static_cast<DST>(src[i]);
        }
        *(reinterpret_cast<PackType<DST, N>*>(dst) + offset) = pack.storage;
    }
    DST* dst;
    const int64_t row_size;
};
/*=================================load/store operators=================================*/


#endif