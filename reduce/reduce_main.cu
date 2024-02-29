
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "freshman.h"
#include "reduce_kernel.cu"

#define THREAD_PER_BLOCK 512

int main() {
	const int N = 32 * 1024 * 1024;
	auto h_a = (float*)malloc(N * sizeof(float));
	float* d_a;
	CHECK(cudaMalloc((void**)&d_a, N * sizeof(float)));

	// 一个thread处理一个元素, 需要几个block
	int block_num = N / THREAD_PER_BLOCK;
	auto h_out = (float*)malloc(block_num * sizeof(float));
	float* d_out;
	CHECK(cudaMalloc((void**)&d_out, block_num * sizeof(float)));
	auto res = (float*)malloc(block_num * sizeof(float));
	int result;

	// initiate input data
	for (int i = 0; i < N; i++) {
		h_a[i] = i % 13;
		//h_a[i] = 1.0;
	}

	// cpu compute
	for (int i = 0; i < block_num; i++) {
		float cur = 0.0;
		for (int j = 0; j < THREAD_PER_BLOCK; j++) {
			cur += h_a[i * THREAD_PER_BLOCK + j];
		}
		res[i] = cur;
	}

	int nIter = 100;
	float msecTotal;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/*--------------------------------warm_up--------------------------------*//*
	printf("------------------------------------------------\n");
	printf("warm up:\n");
	CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));

	cudaEventRecord(start);
	for (int run = 0; run < nIter; run++) {
		reduce_v0<THREAD_PER_BLOCK> << <block_num, THREAD_PER_BLOCK >> > (d_a, d_out);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);

	CHECK(cudaMemcpy(h_out, d_out, block_num * sizeof(float), cudaMemcpyDeviceToHost));

	result = checkResult(h_out, res, block_num, 1e-9);
	if (result == 0)
		printf("check success!\n");
	else
		printf("check fail!\n");
	printf("Time= %.3f msec\n", msecTotal);*/

	/*--------------------------------baseline--------------------------------*/
	printf("------------------------------------------------\n");
	printf("baseline:\n");
	CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));

	cudaEventRecord(start);
	for (int run = 0; run < nIter; run++) {
		reduce_v0<THREAD_PER_BLOCK> << <block_num, THREAD_PER_BLOCK >> > (d_a, d_out);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);

	CHECK(cudaMemcpy(h_out, d_out, block_num * sizeof(float), cudaMemcpyDeviceToHost));

	result = checkResult(h_out, res, block_num, 1e-9);
	if (result == 0)
		printf("check success!\n");
	else
		printf("check fail!\n");
	printf("Time= %.3f msec\n", msecTotal);

	/*--------------------------------no_divergence_branch--------------------------------*/
	printf("------------------------------------------------\n");
	printf("no divergence branch:\n");
	CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));

	cudaEventRecord(start);
	for (int run = 0; run < nIter; run++) {
		reduce_v1<THREAD_PER_BLOCK> << <block_num, THREAD_PER_BLOCK >> > (d_a, d_out);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);

	CHECK(cudaMemcpy(h_out, d_out, block_num * sizeof(float), cudaMemcpyDeviceToHost));

	result = checkResult(h_out, res, block_num, 1e-6);
	if (result == 0)
		printf("check success!\n");
	else
		printf("check fail!\n");
	printf("Time= %.3f msec\n", msecTotal);

	//printf("%d\n", result);
	//printf("%f = %f\n", h_out[3 * (block_num >> 2)], res[3 * (block_num >> 2)]);

	/*--------------------------------no bank conflict--------------------------------*/
	printf("------------------------------------------------\n");
	printf("no bank conflict:\n");
	CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));

	cudaEventRecord(start);
	for (int run = 0; run < nIter; run++) {
		reduce_v2<THREAD_PER_BLOCK> << <block_num, THREAD_PER_BLOCK >> > (d_a, d_out);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);

	CHECK(cudaMemcpy(h_out, d_out, block_num * sizeof(float), cudaMemcpyDeviceToHost));

	result = checkResult(h_out, res, block_num, 1e-6);
	if (result == 0)
		printf("check success!\n");
	else
		printf("check fail!\n");
	printf("Time= %.3f msec\n", msecTotal);

	/*--------------------------------no idle thread--------------------------------*/
	printf("------------------------------------------------\n");
	printf("halved threads:\n");
	CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));

	cudaEventRecord(start);
	for (int run = 0; run < nIter; run++) {
		reduce_v3<THREAD_PER_BLOCK/2> << <block_num, THREAD_PER_BLOCK/2 >> > (d_a, d_out);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);

	CHECK(cudaMemcpy(h_out, d_out, block_num * sizeof(float), cudaMemcpyDeviceToHost));

	result = checkResult(h_out, res, block_num, 1e-6);
	if (result == 0)
		printf("check success!\n");
	else
		printf("check fail!\n");
	printf("Time= %.3f msec\n", msecTotal);

	/*--------------------------------warp reduce--------------------------------*/
	printf("------------------------------------------------\n");
	printf("warp reduce:\n");
	CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));

	cudaEventRecord(start);
	for (int run = 0; run < nIter; run++) {
		reduce_v4<THREAD_PER_BLOCK / 2> << <block_num, THREAD_PER_BLOCK / 2 >> > (d_a, d_out);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);

	CHECK(cudaMemcpy(h_out, d_out, block_num * sizeof(float), cudaMemcpyDeviceToHost));

	result = checkResult(h_out, res, block_num, 1e-6);
	if (result == 0)
		printf("check success!\n");
	else
		printf("check fail!\n");
	printf("Time= %.3f msec\n", msecTotal);

	/*--------------------------------total unroll--------------------------------*/
	printf("------------------------------------------------\n");
	printf("total unroll:\n");
	CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));

	cudaEventRecord(start);
	for (int run = 0; run < nIter; run++) {
		reduce_v5<THREAD_PER_BLOCK / 2> << <block_num, THREAD_PER_BLOCK / 2 >> > (d_a, d_out);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);

	CHECK(cudaMemcpy(h_out, d_out, block_num * sizeof(float), cudaMemcpyDeviceToHost));

	result = checkResult(h_out, res, block_num, 1e-6);
	if (result == 0)
		printf("check success!\n");
	else
		printf("check fail!\n");
	printf("Time= %.3f msec\n", msecTotal);
	//printf("%f\n", h_out[0] + h_out[block_num / 2]);
	//printf("%f\n", h_out[1] + h_out[block_num / 2+1]);

	/*--------------------------------multi add--------------------------------*/
	printf("------------------------------------------------\n");
	const int TILE_PER_THREAD = 2;
	printf("multi add (TILE=%d):\n", TILE_PER_THREAD);

	// cpu compute
	auto res_v6 = (float*)malloc(block_num / TILE_PER_THREAD * sizeof(float));
	for (int i = 0; i < block_num/TILE_PER_THREAD; i++) {
		float cur = 0.0;
		for (int tile_idx = 0; tile_idx < TILE_PER_THREAD; tile_idx++) {
			int offset = N / TILE_PER_THREAD * tile_idx;
			for (int j = 0; j < THREAD_PER_BLOCK; j++) {
				cur += h_a[offset + i * THREAD_PER_BLOCK + j];
			}
		}
		res_v6[i] = cur;
	}

	CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));

	cudaEventRecord(start);
	for (int run = 0; run < nIter; run++) {
		reduce_v6<THREAD_PER_BLOCK / 2, TILE_PER_THREAD> 
			<< <block_num / TILE_PER_THREAD, THREAD_PER_BLOCK / 2 >> > (d_a, d_out);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);

	CHECK(cudaMemcpy(h_out, d_out, block_num / TILE_PER_THREAD * sizeof(float), cudaMemcpyDeviceToHost));

	result = checkResult(h_out, res_v6, block_num / TILE_PER_THREAD, 1e-6);
	if (result == 0)
		printf("check success!\n");
	else
		printf("check fail!\n");
	printf("Time= %.3f msec\n", msecTotal);

	//printf("%d\n", result);
	//printf("%f, %f\n", h_out[0], res_v6[0]);
	//printf("%f, %f\n", h_out[1], res_v6[1]);
	//printf("%f, %f\n", h_a[0], h_a[N / 2]);

	printf("------------------------------------------------\n");


	cudaFree(d_a);
	cudaFree(d_out);
	free(h_a);
	free(h_out);
	return 0;
}
