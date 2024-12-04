#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<sys/time.h>
#include<cuda.h>
#include<cuda_runtime.h>

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

        naive_mm(d_A, d_B, d_C, M, N, K);
        cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
        check_result(C, h_C, M, N);

        cublas_mm(d_A, d_B, d_C, M, N, K);
        cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
        check_result(C, h_C, M, N);
    }
   



    //bench performance

    //warm up
    for(int i=0; i<10; i++){
        cublas_mm(d_A, d_B, d_C, M, N, K);
    }

    int number = 1000;
    int start, end;

    start = get_us();
    for(int i=0; i<number; i++){
        cublas_mm(d_A, d_B, d_C, M, N, K);
    }
    end = get_us();
    float cublas_cost = (float)(end-start)/number;
    float cublas_gflops = 2.0*M*N*K/(cublas_cost*1000.0); // (2M*N*K/10^9) / (cost/10^6) = 2M*N*K/cost*1000
    printf("cublas cost time: %f us, GFLOPS: %f.\n",  cublas_cost, cublas_gflops);
    

    start = get_us();
    for(int i=0; i<number; i++){
        naive_mm(d_A, d_B, d_C, M, N, K);
    }
    end = get_us();
    float naive_cost = (float)(end-start)/number;
    float naive_gflops = 2.0*M*N*K/(naive_cost*1000.0);
    printf("naive_mm cost time: %f us, GFLOPS: %f, speedup to cublas: %f.\n",  naive_cost, naive_gflops, naive_gflops/cublas_gflops);




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
    
    benchmark(1024, 1024, 1024);
    return 0;


}



