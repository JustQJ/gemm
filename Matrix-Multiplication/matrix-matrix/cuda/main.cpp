#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<sys/time.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include "kernel.h"





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


void cpu_mm(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta){
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            for(int k=0; k<K; k++){
                C[i*N+j] =  alpha*A[i*K+k]*B[k*N+j]+beta*C[i*N+j];
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

    float *d_A, *d_B, *d_C, *cublas_C;
    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));
    cudaMalloc(&cublas_C, M*N*sizeof(float));
    cudaMemcpy(d_A, A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cublas_C, C, M*N*sizeof(float), cudaMemcpyHostToDevice);


    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 2.0f;
    float beta = 1.0f;

    cublasStatus_t status = cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,  // 不转置 A 和 B
            N, M, K,                  // C 的尺寸 MxN，A 的尺寸 MxK，B 的尺寸 KxN
            &alpha, 
            d_B,          //
            N, 
            d_A,          // 
            K, 
            &beta, 
            cublas_C,          // 
            N
        );
    if(status != CUBLAS_STATUS_SUCCESS){
            printf("cublas error\n");
        }
    cudaDeviceSynchronize();
    
    

    int start, end;

    start = get_us();
    status = cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // 不转置 A 和 B
        N, M, K,                  // C 的尺寸 MxN，A 的尺寸 MxK，B 的尺寸 KxN
        &alpha, 
        d_B,          //
        N, 
        d_A,          // 
        K, 
        &beta, 
        cublas_C,          // 
        N
    );
    if(status != CUBLAS_STATUS_SUCCESS){
            printf("cublas error\n");
    }
    cudaDeviceSynchronize();

    end = get_us();
    float cublas_cost = (float)(end-start);
    float cublas_gflops = 2.0*M*N*K/(cublas_cost*1000.0);

    float cost_time;
    float gflops;
    start = get_us();
    custom_sgemm(d_A, d_B, d_C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
    end = get_us();
    cost_time = (float)(end-start);
    gflops = 2.0*M*N*K/(cost_time*1000.0);
    
    
    printf("custom gemm / cublas gemm cost time: %f / %f us, GFLOPS: %f / %f, speedup to cublas: %f.\n",  cost_time, cublas_cost, gflops, cublas_gflops, gflops/cublas_gflops);



    cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    cpu_mm(A, B, C, M, N, K, alpha, beta);
    check_result(C, h_C, M, N);

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

