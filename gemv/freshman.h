#ifndef FRESHMAN_H
#define FRESHMAN_H

#define CHECK(call) \
{\
	const cudaError_t error = call;\
	if(error!=cudaSuccess){\
		printf("ERROR:%s:%d,", __FILE__, __LINE__);\
		printf("code:%d,reason:%s\n",\
				 error, cudaGetErrorString(error));\
		exit(1);\
	}\
}

#include <time.h>

#ifdef _WIN32
	#include <windows.h>
#else 
	# include <sys/time.h>
#endif

#ifdef _WIN32
int gettimeofday(struct timeval* tp, void* tzp){
	time_t clock;
	struct tm tm;
	SYSTEMTIME wtm;
	GetLocalTime(&wtm);
	tm.tm_year = wtm.wYear - 1900;
	tm.tm_mon = wtm.wMonth - 1;
	tm.tm_mday = wtm.wDay;
	tm.tm_hour = wtm.wHour;
	tm.tm_min = wtm.wMinute;
	tm.tm_sec = wtm.wSecond;
	tm.tm_isdst = -1;
	clock = mktime(&tm);
	tp->tv_sec = clock;
	tp->tv_usec = wtm.wMilliseconds * 1000;
	return (0);
}
#endif

double cpuSecond() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return (double)tp.tv_sec + (double)tp.tv_usec * 1e-6;
}

void initialData(float* ip, int size) {
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < size; i++) {
		ip[i] = (float)(rand() & 0xffff) / 1000.0f;
	}
}

void initialData_int(int* ip, int size){
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < size; i++){
		ip[i] = int(rand() & 0xff);
	}
}

void printMatrix(float* C, const int nx, const int ny){
	float* ic = C;
	printf("Matrix<%d,%d>:", ny, nx);
	for (int i = 0; i < ny; i++){
		for (int j = 0; j < nx; j++){
			printf("%6f ", C[j]);
		}
		ic += nx;
		printf("\n");
	}
}

void initDevice(int devNum){
	int dev = devNum;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));
}

int checkResult(float* hostRef, float* gpuRef, const int N, double epsilon){
	int num_errs = 0;
	for (int i = 0; i < N; i++){
		if (abs(hostRef[i] - gpuRef[i]) > epsilon){
			num_errs++;
			//printf(" Results don\'t match!\n");
			//printf(" %f(hostRef[%d] )!= %f(gpuRef[%d])\n", hostRef[i], i, gpuRef[i], i);
		}
	}
	//printf(" Check result success!\n");
	return num_errs;
}

void printDeviceProp(const cudaDeviceProp& prop) {
	std::cout << "device_name : " << prop.name << std::endl;
	std::cout << "totalGlobalMem : " << (prop.totalGlobalMem >> 20) << " MB" << std::endl;
	std::cout << "sharedMemPerBlock : " << (prop.sharedMemPerBlock >> 10) << " KB" << std::endl;
	std::cout << "warpSize : " << prop.warpSize << std::endl;
	std::cout << "maxThreadsPerBlock : " << prop.maxBlocksPerMultiProcessor << std::endl;
	std::cout << "maxGridSize x:" << prop.maxGridSize[0] << " y:" << prop.maxGridSize[1] << " z:" << prop.maxGridSize[2] << std::endl;
	std::cout << "maxThreadsDim x:" << prop.maxThreadsDim[0] << " y:" << prop.maxThreadsDim[1] << " z:" << prop.maxThreadsDim[2] << std::endl;
	std::cout << "major : " << prop.major << std::endl;
	std::cout << "clockRate : " << prop.clockRate / 1000 << " MHz" << std::endl;
	std::cout << "multiProcessorCount : " << prop.multiProcessorCount << std::endl;
	printf("----------------------------------------------------------\n");
	printf("Number of multiprocessors:                      %d\n", prop.multiProcessorCount);
	printf("Total amount of constant memory:                %4.2f KB\n",
		prop.totalConstMem / 1024.0);
	printf("Total amount of shared memory per block:        %4.2f KB\n",
		prop.sharedMemPerBlock / 1024.0);
	printf("Total number of registers available per block:  %d\n",
		prop.regsPerBlock);
	printf("Warp size                                       %d\n", prop.warpSize);
	printf("Maximum number of threads per block:            %d\n", prop.maxThreadsPerBlock);
	printf("Maximum number of threads per multiprocessor:   %d\n",
		prop.maxThreadsPerMultiProcessor);
	printf("Maximum number of warps per multiprocessor:     %d\n",
		prop.maxThreadsPerMultiProcessor / 32);
	printf("----------------------------------------------------------\n");
	std::cout << std::endl;
}

// CUDA初始化
bool InitCUDA() {
	int count;
	// 取得支持cuda的装置数目
	CHECK(cudaGetDeviceCount(&count));
	if (count == 0) {
		fprintf(stderr, "There's no device\n");
		return false;
	}
	int i = 0;
	for (; i < count; i++) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			printDeviceProp(prop);
			if (prop.major >= 1) { // 是否支持CUDA 1.x
				break;
			}
		}
	}
	if (i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}
	CHECK(cudaSetDevice(i));
	return true;
}



#endif
