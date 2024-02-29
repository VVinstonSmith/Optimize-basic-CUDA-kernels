
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "freshman.h"
#define TOL 1.0e-3

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


void laplace3d_cpu(int NX, int NY, int NZ, float* u1, float* u2) {
    float sixth = 1.0f / 6.0f;
    for (int k = 0; k < NZ; k++) {
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                int ind = i + j * NX + k * (NX * NY);

                if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1 || k == 0 || k == NZ - 1) {
                    u2[ind] = u1[ind];          // Dirichlet b.c.'s
                }
                else {
                    u2[ind] = (u1[ind - 1] + u1[ind + 1]
                        + u1[ind - NX] + u1[ind + NX]
                        + u1[ind - NX * NY] + u1[ind + NX * NY]) * sixth;
                }
            }
        }
    }
}

__global__ void laplace3d_gpu1(int NX, int NY, int NZ, float* u1, float* u2) {
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + tid;
    int NXNY = NX * NY;
    if (gid >= NXNY * NZ) return;
    float sixth = 1.0f / 6.0f;

    int iz = gid / NXNY;
    int iy = (gid - iz * NXNY) / NX;
    int ix = gid % NX;

    if (ix == 0 || ix == NX - 1 || iy == 0 || iy == NY - 1 || iz == 0 || iz == NZ - 1) {
        u2[gid] = u1[gid];
    }
    else {
        u2[gid] = (u1[gid - 1] + u1[gid + 1] +
            u1[gid - NX] + u1[gid + NX] +
            u1[gid - NXNY] + u1[gid + NXNY]) * sixth;
    }
}

__device__ inline int xyz2sid(int dx, int dy, int ix, int iy, int iz) {
    return (ix + 1) + (iy + 1)*(dx + 2) + (iz + 1)*(dx + 2)*(dy + 2);
}

__global__ void laplace3d_gpu2(int NX, int NY, int NZ, int dx, int dy, int dz, float* A, float* B) {

    extern __shared__ float smem[]; // (dx+2)*(dy+2)*(dz+2)
    float sixth = 1.0f / 6.0f;
    int total = NX * NY * NZ;
    
    int gridDimX = NX / dx;
    int gridDimY = NY / dy;
    int gridDimZ = NZ / dz;

    int tid = threadIdx.x;

    /* local index */
    int iz_l = tid / (dx * dy);
    int iy_l = (tid - iz_l * (dx * dy)) / dx;
    int ix_l = tid % dx;
    int sid = xyz2sid(dx, dy, ix_l, iy_l, iz_l);

    /* global index */
    // blockIdx.x = bidx + bidy*gridDimX + bidz*gridDimX*gridDimY
    int bidz = blockIdx.x / (gridDimX * gridDimY);
    int bidy = (blockIdx.x - bidz * (gridDimX * gridDimY)) / gridDimX;
    int bidx = blockIdx.x % gridDimX;

    int ix_g = bidx * dx + ix_l;
    int iy_g = bidy * dy + iy_l;
    int iz_g = bidz * dz + iz_l;
    int gid = ix_g + iy_g * NX + iz_g * NX * NY; // 内存单元都在全局数组中的id，而非线程的id

    if (gid < total) {
        smem[sid] = A[gid];
        if (ix_g != 0 && ix_g != NX - 1 && iy_g != 0 && iy_g != NY - 1 && iz_g != 0 && iz_g != NZ - 1) {
            if (ix_l == 0)
                smem[sid - 1] = A[gid - 1];
            if (ix_l == dx - 1)
                smem[sid + 1] = A[gid + 1];

            if (iy_l == 0)
                smem[sid - (dx + 2)] = A[gid - NX];
            if (iy_l == dy - 1)
                smem[sid + (dx + 2)] = A[gid + NX];

            if (iz_l == 0)
                smem[sid - (dx + 2) * (dy + 2)] = A[gid - NX * NY];
            if (iz_l == dz - 1)
                smem[sid + (dx + 2) * (dy + 2)] = A[gid + NX * NY];
        }
    }
    
    __syncthreads();

    if (gid < total && ix_g != 0 && ix_g != NX - 1 && iy_g != 0 && iy_g != NY - 1 && iz_g != 0 && iz_g != NZ - 1) {
        B[gid] = (smem[sid - 1] + smem[sid + 1]
            + smem[sid - (dx + 2)] + smem[sid + (dx + 2)]
            + smem[sid - (dx + 2) * (dy + 2)] + smem[sid + (dx + 2) * (dy + 2)]) * sixth;
        /*if (gid == 262752) {
            printf("gid=%d,sid=%d: x=%f,%f, y=%f,%f, z=%f,%f \n", gid, sid,
                smem[sid - 1], smem[sid + 1],
                smem[sid - (dx + 2)], smem[sid + (dx + 2)],
                smem[sid - (dx + 2) * (dy + 2)], smem[sid + (dx + 2) * (dy + 2)]);
            printf("A[%d] = %f\n\n", gid, A[gid]);
        }*/
    }
}

int main(int argc, char** argv) {
    InitCUDA();

    int NX = 512, NY = 512, NZ = 512, NITER = 10;
    size_t u_size = sizeof(float) * NX * NY * NZ;
    if (argc == 5) {
        NX = atoi(argv[1]);
        NY = atoi(argv[2]);
        NZ = atoi(argv[3]);
        NITER = atoi(argv[4]);
    }
    printf("\n Grid dimensions: %d x %d x %d.\n", NX, NY, NZ);

    int blocksize = 512;
    int gridsize = (NX * NY * NZ + blocksize - 1) / blocksize;
    int dx = 8, dy = 8, dz;
    dz = min(NZ, blocksize / (dx * dy));

    // allocate memory for host arrays
    auto u0_host = (float*)malloc(u_size);
    auto u1_cpu_host = (float*)malloc(u_size); 
    auto u2_cpu_host = (float*)malloc(u_size);
    auto u_gpu_host = (float*)malloc(u_size);

    // initiate host array
    for (int k = 0; k < NZ; k++) {
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                int ind = i + j * NX + k * NX * NY;
                //u1_host[ind] = (float)(rand() & 0xffff) / 1000.0f;
                if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1 || k == 0 || k == NZ - 1)
                    u1_cpu_host[ind] = u0_host[ind] = 1.0f; // boundary
                else
                    u1_cpu_host[ind] = u0_host[ind] = 0.0f;
            }
        }
    }

    // execute laplace3d_cpu
    double cpu_time = cpuSecond();
    for (int i = 0; i < NITER; i++) {
        laplace3d_cpu(NX, NY, NZ, u1_cpu_host, u2_cpu_host);
        auto u_temp = u2_cpu_host;
        u2_cpu_host = u1_cpu_host;
        u1_cpu_host = u_temp;
        /*if (checkResult(u1_cpu_host, u2_cpu_host, NX * NY * NZ, TOL) == 1) {
            printf(" num of iteration is : %d\n", i+1);
            return;
        }*/
    }
    cpu_time = cpuSecond() - cpu_time;

    // allocate device arrays
    float* u1_dev, * u2_dev;
    CHECK(cudaMalloc((void**)&u1_dev, u_size));
    CHECK(cudaMalloc((void**)&u2_dev, u_size));

    // send data from host to device
    CHECK(cudaMemcpy(u1_dev, u0_host, u_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(u2_dev, u0_host, u_size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // execute laplace3d_gpu
    cudaEventRecord(start, 0);
    for (int i = 0; i < NITER; i++) {
        laplace3d_gpu1 << <gridsize, blocksize>> > (NX, NY, NZ, u1_dev, u2_dev);        
        auto u_temp = u2_dev;
        u2_dev = u1_dev;
        u1_dev = u_temp;
    }
    /*for (int i = 0; i < NITER; i++) {
        laplace3d_gpu2 <<<gridsize, blocksize, (dx+2)*(dy+2)*(dz+2)*sizeof(float)>>> (NX, NY, NZ, dx, dy, dz, u1_dev, u2_dev);
        auto u_temp = u2_dev;
        u2_dev = u1_dev;
        u1_dev = u_temp;
    }*/
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    CHECK(cudaMemcpy(u_gpu_host, u1_dev, u_size, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    /*for (int k = 0; k < NZ; k++) { 
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                int ind = i + j * NX + k * (NX * NY);
                std::cout << u_gpu_host[ind] <<", ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }*/
    
    if (checkResult(u1_cpu_host, u_gpu_host, NX * NY * NZ, TOL)==1) 
        printf(" Check result success!\n");
    else
        printf(" Check result fail!\n");
    printf(" num of iteration : %d\n", NITER);
    std::cout << " gpu kernel execution time : " << elapsedTime << "ms" << std::endl;
    std::cout << " cpu execution time : " << cpu_time << "s" << std::endl;
    printf("\n");

    cudaFree(u1_dev);
    cudaFree(u2_dev);
    free(u0_host);
    free(u1_cpu_host);
    free(u2_cpu_host);
    free(u_gpu_host);

    return 0;
}

