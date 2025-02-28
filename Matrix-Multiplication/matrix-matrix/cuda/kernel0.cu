#include<cuda_runtime.h>


#define BLOCK_SIZE 32



__global__ void sgemm_kernel(float *d_A, float *d_B, float *d_C, int M, int N, int K, float alpha, float beta){
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    
    if(col<N && row<M){
        float val = 0;
        for(int k=0; k<K; k++){
            val += d_A[row*K+k]*d_B[k*N+col];
        }
        d_C[row*N+col] = alpha*val + beta*d_C[row*N+col];
    }
}



void custom_sgemm(float *d_A, float *d_B, float *d_C, int M, int N, int K, float alpha, float beta){
    dim3 block(BLOCK_SIZE,BLOCK_SIZE);
    dim3 grid((N+BLOCK_SIZE-1)/BLOCK_SIZE, (M+BLOCK_SIZE-1)/BLOCK_SIZE);
    sgemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    
}
