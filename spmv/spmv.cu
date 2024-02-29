
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cusparse.h>

#include <iostream>
#include <fstream>
#include "freshman.h"
#include "kernel.cu"

using namespace std;

void readVerEdges(int& n_rows, int& n_cols, int& n_elems, const std::string& file) {
	std::ifstream ifs(file+".mtx");	

	while (ifs.peek() == '%') // .peek()为指向的字符
		ifs.ignore(2048, '\n');
	
	// rows cols nonZeros
	ifs >> n_rows >> n_cols >> n_elems;
	
	ifs.close();
}

void add(int row, int col, float val, int* head, \
		int* next, int* col_map, float* val_map, int& idx) {
	col_map[idx] = col;
	val_map[idx] = val;
	next[idx] = head[row];
	head[row] = idx;
	idx++;
}

void readMtxFile(int* row_offset, int* col_index, float* val, const std::string& file) {
	std::ifstream ifs(file + ".mtx");
	while (ifs.peek() == '%')
		ifs.ignore(2048, '\n');
	
	int n_rows, n_cols, n_elems;
	ifs >> n_rows >> n_cols >> n_elems;
	
	/*初始化中间数组*/
	auto head = (int*)malloc(n_rows * sizeof(int));
	auto next = (int*)malloc(n_elems * sizeof(int));
	auto col_map = (int*)malloc(n_elems * sizeof(int));
	auto val_map = (float*)malloc(n_elems * sizeof(float));
	for (int i = 0; i < n_rows; i++)
		head[i] = -1;
	for (int i = 0; i < n_elems; i++)
		next[i] = -1;

	/*计算中间数组*/
	int row, col;
	double d_val;
	int idx = 0;
	while (ifs >> row >> col >> d_val) {
		row--;
		col--;
		float f_val = static_cast<float>(d_val);
		add(row, col, f_val, head, next, col_map, val_map, idx);
		//if (row == 0 || row == 1 || row == 2 || row == 3)
		//	printf("%d, %d, %f\n", row+1, col+1, f_val);
	}
	//printf("idx finally = %d\n", idx);

	/*计算CSR数组*/
	row_offset[0] = 0;
	int k = 0;
	for (int i = 0; i < n_rows; i++) {
		idx = head[i];
		while (idx != -1) {
			col_index[k] = col_map[idx];
			val[k++] = val_map[idx];
			idx = next[idx];
		}
		row_offset[i + 1] = k;
	}
	
	free(head);
	free(next);
	free(col_map);
	free(val_map);
}


void spmv_csr_cpu(int* row_offset, int* col_index, float* values, float* x, float* y, int n_cols, int n_elems) {
	int i = 0, row = 0;
	while (i < n_elems) {
		y[row] = 0.0;
		while (i < row_offset[row + 1]) {
			//printf("row=%d, i=%d row_offset=%d\n", row, i, row_offset[row + 1]);
			y[row] += values[i] * x[col_index[i]];
			i++;
		}
		//printf("y[%d] = %f\n", row, y[row]);
		row++;
	}
}

int main(int argc, char** argv) {
	const string file = "shyy41";
	int n_rows, n_cols, n_elems;
	readVerEdges(n_rows, n_cols, n_elems, file);
	//printf("n_rows=%d, n_cols=%d, n_elems=%d\n", n_rows, n_cols, n_elems);
	int mean_col_num = (n_elems + n_rows - 1) / n_rows;
	printf("The average col num is: %d\n", mean_col_num);

	auto row_offset = (int*)malloc((n_rows + 1) * sizeof(int));
	auto col_index = (int*)malloc(n_elems * sizeof(int));
	auto values = (float*)malloc(n_elems * sizeof(float));
	readMtxFile(row_offset, col_index, values, file);

	/*printf("row offsets:\n");
	for (int i =  0; i < n_rows + 1; i++)
		printf("%d ", row_offset[i]);
	printf("\n");
	printf("column indices:\n");
	for (int i = 0; i < n_elems; i++)
		printf("%d ", col_index[i]);
	printf("\n");
	printf("values:\n");
	for (int i = 0; i < n_elems; i++)
		printf("%f ", values[i]);
	printf("\n");*/
	
	int nIter = 1000;
	auto x = (float*)malloc(n_cols * sizeof(float));
	auto y = (float*)malloc(n_rows * sizeof(float));
	auto y_cpu = (float*)malloc(n_rows * sizeof(float));
	auto y_cusparse = (float*)malloc(n_rows * sizeof(float));
	for (int i = 0; i < n_cols; i++)
		x[i] = (float)i / 13.0;

	// cpu computes SpMV
	spmv_csr_cpu(row_offset, col_index, values, x, y_cpu, n_cols, n_elems);
	
	// allocate device memory
	int* d_row_offset, * d_col_index;
	float* d_values, * d_x, * d_y, * d_y_cusparse;
	CHECK(cudaMalloc((void**)&d_row_offset, (n_rows + 1) * sizeof(int)));
	CHECK(cudaMalloc((void**)&d_col_index, n_elems * sizeof(int)));
	CHECK(cudaMalloc((void**)&d_values, n_elems * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_x, n_cols * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_y, n_rows * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_y_cusparse, n_rows * sizeof(float)));

	CHECK(cudaMemcpy(d_row_offset, row_offset, (n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_col_index, col_index, n_elems * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_values, values, n_elems * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_x, x, n_cols * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_y, y, n_rows * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_y_cusparse, y_cusparse, n_rows * sizeof(float), cudaMemcpyHostToDevice));

	float msecTotal = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// cusparse spmv
	//---------------------------------------------------------------------------
	// CUSPARSE APIs
	float alpha = 1.0, beta = 0.0;
	cusparseHandle_t handle;
	cusparseSpMatDescr_t matA; // 稀疏矩阵
	cusparseDnVecDescr_t vecX, vecY; // 稠密向量
	void* dBuffer;
	size_t bufferSize;
	
	cusparseCreate(&handle);
	// create sparse Mat A in CSR fromat
	cusparseCreateCsr(&matA, n_rows, n_cols, n_elems,
					  d_row_offset, d_col_index, d_values,
					  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
					  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F); // d_row_offset, d_col_index, d_values
	// create dense vector x and y
	cusparseCreateDnVec(&vecX, n_cols, d_x, CUDA_R_32F);
	cusparseCreateDnVec(&vecY, n_rows, d_y_cusparse, CUDA_R_32F);
	// allocate an external buffer if needed
	cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
							&alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
							CUSPARSE_MV_ALG_DEFAULT, &bufferSize); // buffer_size
	CHECK(cudaMalloc(&dBuffer, bufferSize)); // buffer

	cudaEventRecord(start);
	// execute SpMV
	for (int i = 0; i < nIter; i++) {
		cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
					 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
					 CUSPARSE_MV_ALG_DEFAULT, dBuffer);
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);

	double flopsPerSpMV = 2.0 * n_elems;
	double msecPerSpMV = msecTotal / nIter;
	double gigaFlops = (flopsPerSpMV * 1.e-9f) / (msecPerSpMV * 0.001);
	printf("cusparse SpMV performance= %.2f GFlops, Time= %.3f msec, size= %.0f Ops\n", gigaFlops, msecPerSpMV, flopsPerSpMV);

	CHECK(cudaMemcpy(y_cusparse, d_y_cusparse, n_rows * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK(cudaDeviceSynchronize());

	cusparseDestroySpMat(matA);
	cusparseDestroyDnVec(vecX);
	cusparseDestroyDnVec(vecY);
	cusparseDestroy(handle);

	// cusparse spmv
	//---------------------------------------------------------------------------
	// CUSPARSE APIs
	const int THREADS_PER_BLOCK = 128; // 一个block有几个thread
	
	cudaEventRecord(start);
	for (int i = 0; i < nIter; i++) {
		if (mean_col_num <= 2) {
			const int THREADS_PER_ROW = 2; // 每行由几个thread来处理
			const int ROWS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_ROW; // 一个block处理几行
			const int NUM_BLOCKS = (n_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK; // 总共需要几个block
			spmv_csr_kernel<THREADS_PER_ROW> << <NUM_BLOCKS, THREADS_PER_BLOCK >> > (n_rows,
				d_row_offset, d_col_index, d_values, d_x, d_y);
		}
		else if (mean_col_num <= 4) {
			const int THREADS_PER_ROW = 4;
			const int ROWS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_ROW;
			const int NUM_BLOCKS = (n_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
			spmv_csr_kernel<THREADS_PER_ROW> << <NUM_BLOCKS, THREADS_PER_BLOCK >> > (n_rows,
				d_row_offset, d_col_index, d_values, d_x, d_y);
		}
		else if (mean_col_num <= 8) {
			const int THREADS_PER_ROW = 8;
			const int ROWS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_ROW;
			const int NUM_BLOCKS = (n_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
			spmv_csr_kernel<THREADS_PER_ROW> << <NUM_BLOCKS, THREADS_PER_BLOCK >> > (n_rows,
				d_row_offset, d_col_index, d_values, d_x, d_y);
		}
		else if (mean_col_num <= 16) {
			const int THREADS_PER_ROW = 16;
			const int ROWS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_ROW;
			const int NUM_BLOCKS = (n_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
			spmv_csr_kernel<THREADS_PER_ROW> << <NUM_BLOCKS, THREADS_PER_BLOCK >> > (n_rows,
				d_row_offset, d_col_index, d_values, d_x, d_y);
		}
		else { 
			const int THREADS_PER_ROW = 32;
			const int ROWS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_ROW;
			const int NUM_BLOCKS = (n_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
			spmv_csr_kernel<THREADS_PER_ROW> << <NUM_BLOCKS, THREADS_PER_BLOCK >> > (n_rows,
				d_row_offset, d_col_index, d_values, d_x, d_y);
		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);

	msecPerSpMV = msecTotal / nIter;
	gigaFlops = (flopsPerSpMV * 1.e-9f) / (msecPerSpMV * 0.001);
	printf("my       SpMV performance= %.2f GFlops, Time= %.3f msec, size= %.0f Ops\n", gigaFlops, msecPerSpMV, flopsPerSpMV);

	CHECK(cudaMemcpy(y, d_y, n_rows * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK(cudaDeviceSynchronize());

	//---------------------------------------------------------------------------
	// device result check
	int n_errors = checkResult(y, y_cusparse, n_rows, 1e-3);
	if (n_errors == 0)
		printf("check success!\n");
	else {
		printf("check fail!\n");
		printf("n_errors = %d\n", n_errors);
	}
	//for (int i =0; i < 50; i++) {
	//	printf("i=%d, gpu:%f, cusparse:%f\n", i, y[i], y_cusparse[i]);
	//}

	CHECK(cudaFree(d_row_offset));
	CHECK(cudaFree(d_col_index));
	CHECK(cudaFree(d_values));
	CHECK(cudaFree(d_x));
	CHECK(cudaFree(d_y));
	CHECK(cudaFree(d_y_cusparse));
	free(row_offset);
	free(col_index);
	free(values);
	free(x);
	free(y);
	free(y_cpu);
	free(y_cusparse);

	return 0;
}


