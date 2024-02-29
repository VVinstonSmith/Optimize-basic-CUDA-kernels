#include <assert.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <cub/cub.cuh>
#include "Operators.cuh"
#include "WarpImpl.cuh"
#include "BlockSMemImpl.cuh"
#include "BlockUncachedImpl.cuh"

using namespace std;

#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);}

// ^V^
template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
inline cudaError_t DispatchSoftmax(cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols) {
    //const int limit = 1024;
    const int limit = 0;
    if (cols < limit) {
        return DispatchSoftmaxWarpImpl<LOAD, STORE, ComputeType, algorithm>(
            stream, load, store, rows, cols);
    }
    else {
        // new added
        //------------------------------
        return DispatchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, Algorithm::kSoftmax>(
            stream, load, store, rows, cols);
        //------------------------------
        bool dispatch_smem_impl_success; // 是否支持使用共享内存的版本
        {
            cudaError_t err = TryDispatchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, algorithm>(
                stream, load, store, rows, cols, &dispatch_smem_impl_success);
            if (err != cudaSuccess) return err;
        }
        if (!dispatch_smem_impl_success) {
            return DispatchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, algorithm>(
                stream, load, store, rows, cols);
        }
        return cudaSuccess;
    }
}

int main() {
    const int rows = 32 * 64 * 128;
    const int cols = 127;
    const int N = rows * cols;
    using ComputeType = float;

    float* input_host = (float*)malloc(N * sizeof(float));
    float* output_host = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) input_host[i] = 1.0;

    float* input_device, * output_device;
    cudaMalloc((void**)&input_device, N * sizeof(float));
    cudaMalloc((void**)&output_device, N * sizeof(float));
    DirectLoad<float, ComputeType> load(input_device, cols);
    DirectStore<ComputeType, float> store(output_device, cols);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const int nIter = 10;
    float msecTotal;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(input_device, input_host, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    for (int i = 0; i < nIter; i++) {
        DispatchSoftmax<decltype(load), decltype(store), ComputeType, Algorithm::kSoftmax>(
            stream, load, store, rows, cols);
    }
    cudaEventRecord(stop);

    cudaMemcpy(output_host, output_device, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Time= %.3f msec\n", msecTotal / nIter);

    // 1 / 128 = 0.0078125
    for (int i = 0; i < 32; i++) {
        printf("%.5f\n", output_host[i]);
    }
    cudaFree(input_device);
    cudaFree(output_device);
    free(input_host);
    free(output_host);
    return 0;
}

