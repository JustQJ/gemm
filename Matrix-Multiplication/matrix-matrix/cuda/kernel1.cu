#include<cuda_runtime.h>


#define Tile_M 32
#define Tile_N 32
#define Tile_K 32

//use shared memory, each thread deal with one element
__global__ void sgemm_kernel(float *d_A, float *d_B, float *d_C, int M, int N, int K, float alpha, float beta){
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.y + threadIdx.y;

    __shared__ float As[Tile_M][Tile_K];
    __shared__ float Bs[Tile_K][Tile_N];

    if(col<N && row<M){
        float val = 0;
        for(int k=0; k<K; k++){
            val += d_A[row*K+k]*d_B[k*N+col];
        }
        d_C[row*N+col] = alpha*val + beta*d_C[row*N+col];
    }
}



void custom_sgemm(float *d_A, float *d_B, float *d_C, int M, int N, int K, float alpha, float beta){
    dim3 block(Tile_M,Tile_K);
    dim3 grid((N+Tile_M-1)/Tile_M, (M+Tile_M-1)/Tile_M);
    sgemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    
}
