#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<sys/time.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include "mm_kernel.h"





#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << #call << ": " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "CUBLAS error in " << #call << std::endl; \
            exit(1); \
        } \
    } while (0)


// https://github.com/NVIDIA/cuda-samples/blob/6be514679b201c8a0f0cda050bc7c01c8cda32ec/Common/helper_cuda.h
// https://github.com/NVIDIA/cuda-samples/blob/6be514679b201c8a0f0cda050bc7c01c8cda32ec/Samples/deviceQuery/deviceQuery.cpp#L131
int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {0x89,  128},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

int print_gpu_info(){
     int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);  // 获取可用的GPU数量

    if (deviceCount == 0) {
        printf("There is no available device that supports CUDA.\n");
        return -1;
    }

    for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, deviceId);

        printf("Device ID: %d\n", deviceId);

        // SM数量
        printf("  SM Number (SMs): %d\n", deviceProp.multiProcessorCount);

        // 每个SM的共享内存大小
        printf("  Shared memory per block: %ld KB\n", deviceProp.sharedMemPerBlock/ (1024));

        printf("  Shared memory per SM: %ld KB\n", deviceProp.sharedMemPerMultiprocessor / (1024));

        // 每个线程块的最大线程数
        printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);

        printf("  Max threads per SM: %d\n", deviceProp.maxThreadsPerMultiProcessor);

        printf("  Warp size: %d\n", deviceProp.warpSize);


        // 最大线程数维度
        printf("  Max dimension of threads per block: (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);

        printf("  L2 Cache Size: %d KB\n", deviceProp.l2CacheSize / 1024);

        // 设备全局内存大小
        printf("  Global memory: %ld GB\n", deviceProp.totalGlobalMem / (1024 * 1024 * 1024) );

        //计算内存带宽和理论算力
        unsigned int memoryClockRate = deviceProp.memoryClockRate;  // 内存时钟频率 khz
        unsigned int memoryBusWidth = deviceProp.memoryBusWidth;    // 内存位宽 bit
        int memoryBandwidth = 2 * memoryClockRate * (memoryBusWidth / 8) / 1e6;  // 内存带宽 GB/s
        printf("  Memory Bandwidth: %d GB/s\n", memoryBandwidth);

        // 不同的架构每个SM有不同的核心数，例如volta架构每个SM有64个CUDA核心
        // 和线程数不同
        int cudacores_persm = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);

        printf("  CUDA Cores per SM: %d\n", cudacores_persm);
        
        // 计算理论TFLOPS = core_number * core_frequency * 2
        // deviceProp.clockRate 是 khz，所以要除以 1.0e9，得到 THz
        printf("  Theoretical TFLOPS: %f\n", deviceProp.multiProcessorCount* cudacores_persm * 2.0 * deviceProp.clockRate / 1.0e9);


        // 最大块数维度
        printf("\n");

       
    }

     return 0;
}



int get_ms(){
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec*1000 + t.tv_usec/1000;
}

int get_us(){
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec*1000000 + t.tv_usec;
}


void cpu_mm(float *A, float *B, float *C, int M, int N, int K){
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            for(int k=0; k<K; k++){
                C[i*N+j] += A[i*K+k]*B[k*N+j];
            }
        }
    }
}




void check_result(float *C, float *h_C, int M, int N){
    float total_error = 0.0;
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            total_error += abs(C[i*N+j] - h_C[i*N+j]);
        }
    }

    printf("total error: %f\n", total_error/(M*N));
    printf("%f, %f\n", C[0], h_C[0]);
    
}



void benchmark(int M, int N, int K, bool correct_check=false){
    // int M = 1024;
    // int N = 1024;
    // int K = 1024;

    printf("Performance Benchmark: M: %d, N: %d, K: %d\n", M, N, K);

    float *A = (float*)malloc(M*K*sizeof(float));
    float *B = (float*)malloc(K*N*sizeof(float));
    float *C = (float*)malloc(M*N*sizeof(float));

    float *h_C = (float*)malloc(M*N*sizeof(float)); // get results from gpu

    for(int i=0; i<M*K; i++){
        A[i] = (float)rand()/(float)RAND_MAX;
       
    }
   
    for(int i=0; i<K*N; i++){
        B[i] = (float)rand()/(float)RAND_MAX;
    
    }
 
    
    for(int i=0; i<M*N; i++){
        C[i] = 0.0;
        h_C[i] = 0.0;
    }

    //cuda memory

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));
    cudaMemcpy(d_A, A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, M*N*sizeof(float), cudaMemcpyHostToDevice);



    //correctness check
    if(correct_check){
        cpu_mm(A, B, C, M, N, K);
        printf("naive_mm check:\n");
        naive_mm(d_A, d_B, d_C, M, N, K);
        cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
        check_result(C, h_C, M, N);

        // printf("cublas_mm check:\n");
        // cublas_mm(d_A, d_B, d_C, M, N, K);
        // cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
        // check_result(C, h_C, M, N);

        printf("shared_mm0 check:\n");
        shared_mm0(d_A, d_B, d_C, M, N, K);
        cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
        check_result(C, h_C, M, N);

        printf("shared_mm1 check:\n");
        shared_mm1(d_A, d_B, d_C, M, N, K);
        cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
        check_result(C, h_C, M, N);

        printf("shared_mm2 check:\n");
        shared_mm2(d_A, d_B, d_C, M, N, K);
        cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
        check_result(C, h_C, M, N);
    }

    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    

    
   



    //bench performance

    //warm up
    for(int i=0; i<10; i++){
        cublasStatus_t status = cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,  // 不转置 A 和 B
            N, M, K,                  // C 的尺寸 MxN，A 的尺寸 MxK，B 的尺寸 KxN
            &alpha, 
            d_B,          //
            N, 
            d_A,          // 
            K, 
            &beta, 
            d_C,          // 
            N
        );
        if(status != CUBLAS_STATUS_SUCCESS){
            printf("cublas error\n");
        }
    }
    cudaDeviceSynchronize();

    int number = 10;
    int start, end;

    start = get_us();
    for(int i=0; i<number; i++){
        cublasStatus_t status = cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,  // 不转置 A 和 B
            N, M, K,                  // C 的尺寸 MxN，A 的尺寸 MxK，B 的尺寸 KxN
            &alpha, 
            d_B,          //
            N, 
            d_A,          // 
            K, 
            &beta, 
            d_C,          // 
            N
        );
        if(status != CUBLAS_STATUS_SUCCESS){
            printf("cublas error\n");
        }
        cudaDeviceSynchronize();
    }
    
    
    end = get_us();
    float cublas_cost = (float)(end-start)/number;
    float cublas_gflops = 2.0*M*N*K/(cublas_cost*1000.0); // (2M*N*K/10^9) / (cost/10^6) = 2M*N*K/cost*1000
    printf("cublas cost time: %f us, GFLOPS: %f.\n",  cublas_cost, cublas_gflops);
    
    float cost_time;
    float gflops;

    start = get_us();
    for(int i=0; i<number; i++){
        naive_mm(d_A, d_B, d_C, M, N, K);
    }
    end = get_us();
    cost_time = (float)(end-start)/number;
    gflops = 2.0*M*N*K/(cost_time*1000.0);
    printf("naive_mm cost time: %f us, GFLOPS: %f, speedup to cublas: %f.\n",  cost_time, gflops, gflops/cublas_gflops);


    start = get_us();
    for(int i=0; i<number; i++){
        shared_mm0(d_A, d_B, d_C, M, N, K);
    }
    end = get_us();
    cost_time = (float)(end-start)/number;
    gflops = 2.0*M*N*K/(cost_time*1000.0);
    printf("shared_mm0 cost time: %f us, GFLOPS: %f, speedup to cublas: %f.\n",  cost_time, gflops, gflops/cublas_gflops);

    start = get_us();
    for(int i=0; i<number; i++){
        shared_mm1(d_A, d_B, d_C, M, N, K);
    }
    end = get_us();
    cost_time = (float)(end-start)/number;
    gflops = 2.0*M*N*K/(cost_time*1000.0);
    printf("shared_mm1 cost time: %f us, GFLOPS: %f, speedup to cublas: %f.\n",  cost_time, gflops, gflops/cublas_gflops);


    start = get_us();
    for(int i=0; i<number; i++){
        shared_mm2(d_A, d_B, d_C, M, N, K);
    }
    end = get_us();
    cost_time = (float)(end-start)/number;
    gflops = 2.0*M*N*K/(cost_time*1000.0);
    printf("shared_mm2 cost time: %f us, GFLOPS: %f, speedup to cublas: %f.\n",  cost_time, gflops, gflops/cublas_gflops);




    //memory free

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    free(h_C);
}



int main(){

    // print_gpu_info();

    // int M = 2048;
    // int N = 2048;
    // int K = 2048;
    int start = 10;
    int end = 20;
    for (int m = 256 * start; m <= 256 * end; m += 256) {
        benchmark(m, m, m, false);
    }
    
    return 0;


}



