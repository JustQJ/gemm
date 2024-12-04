
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include "mm_kernel.h"



void cublas_mm(float *d_A, float *d_B, float *d_C, int M, int N, int K){
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(
        handle,
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
    
    
}



// C = alpha * A * B + beta * C
__global__ void __naive_kernel(
    float *A, 
    float *B,
    float *C, 
    int M, 
    int N, 
    int K,
    float alpha,
    float beta
){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row<M && col<N){
        float sum = 0;
        for(int k=0; k<K; k++){
            sum += A[row*K + k] * B[k*N + col];
        }
        C[row*N + col] = alpha*sum+beta*C[row*N + col];
    }
}



void naive_mm(
    float *A, 
    float *B,
    float *C, 
    int M, 
    int N, 
    int K
){
    float alpha = 1.0;
    float beta = 0.0;
    dim3 block(32, 32);
    dim3 grid((N+block.x-1)/block.x, (M+block.y-1)/block.y);
    __naive_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}






