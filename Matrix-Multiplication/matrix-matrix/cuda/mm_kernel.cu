
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include "mm_kernel.h"



void cublas_mm(float *d_A, float *d_B, float *d_C, int M, int N, int K){
    

    
    
    
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






__global__ void __shared_kernel0(
    float *A,
    float *B,
    float *C,
    int M,
    int N,
    int K,
    float alpha,
    float beta
){
    const int BM = 32;
    const int BN = 32;
    const int BK = 32;
 

    int col = blockIdx.x * BM + threadIdx.x;
    int row = blockIdx.y * BN + threadIdx.y;
    
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float res = 0.0;

    for(int i=0; i<K; i+=BK){
        //load A and B into shared memory
        if(row<M && i+threadIdx.x<K){
            As[threadIdx.y][threadIdx.x] = A[row*K + i+threadIdx.x];
        }

        if(col<N && i+threadIdx.y<K){
            Bs[threadIdx.y][threadIdx.x] = B[(i+threadIdx.y)*N + col];
        }

        __syncthreads();

        //compute
        for(int j=0; j<BK; j++){
            res += As[threadIdx.y][j] * Bs[j][threadIdx.x];
        }

        __syncthreads();
        

    }

    //store result
    if(row<M && col<N){
        C[row*N + col] = alpha*res + beta*C[row*N + col];
    }


    

}


void shared_mm0(
    float *A, 
    float *B,
    float *C, 
    int M, 
    int N, 
    int K
){
    // 1. 每个thread只处理一个元素，所以一个tile的大小为 32*32
    float alpha = 1.0;
    float beta = 0.0;
    int block_size = 32;
    dim3 block(block_size, block_size);
    dim3 grid((N+block.x-1)/block.x, (M+block.y-1)/block.y);
    __shared_kernel0<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}




//shared memory, each thread deal with 8*8 elements
__global__ void __shared_kernel1(
    float *A,
    float *B,
    float *C,
    int M,
    int N,
    int K,
    float alpha,
    float beta
){
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TX = 8; //each thread
    const int TY = 8;

    const int start_row = BM * blockIdx.y;
    const int start_col = BN * blockIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int tile_x = threadIdx.x * TX;
    const int tile_y = threadIdx.y * TY;
    const int load_a_row = tid >> 1; //每个线程load四个元素，一行8个元素，需要两个线程
    const int load_a_col = (tid & 1)<<2; // %2 == 0? 0:4
    const int load_b_row = tid >> 5; //每个线程load四个元素，一行128个元素，需要32个线程
    const int load_b_col = (tid & 31)<<2; // %32 * 4 
    
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float res[TY][TX] = {0.0};

    for(int i=0; i<K; i+=BK){
        //load A and B into shared memory
        //load A, each thread load 4 float, can directly load 4 float at once with float4 type
        int load_a_idx = (start_row+load_a_row)*K + i+load_a_col; 
        int load_b_idx = (i+load_b_row)*N + start_col+load_b_col;

        ((float4 *)(&As[load_a_row][load_a_col]))[0] = ((float4 *)(&A[load_a_idx]))[0]; //cast to float4 pointer, then load 4 float
        ((float4 *)(&Bs[load_b_row][load_b_col]))[0] = ((float4 *)(&B[load_b_idx]))[0];
        __syncthreads();

        //compute
        #pragma unroll
        for(int j=0; j<BK; j++){
            #pragma unroll
            for(int y=0; y<TY; y++){
                #pragma unroll
                for(int x=0; x<TX; x++){
                    res[y][x] += As[y+tile_y][j] * Bs[j][x+tile_x];
                }
            }
        }

        __syncthreads();
        

    }

    //store result
    #pragma unroll
    for(int y=0; y<TY; y++){
        #pragma unroll
        for(int x=0; x<TX; x+=4){ //each time store four float
            int idx = (start_row+y+tile_y)*N + start_col+x+tile_x;
            
            float tmp[4];
            ((float4*)tmp)[0] = ((float4 *)(&C[idx]))[0];
            tmp[0] = alpha*res[y][x] + beta*tmp[0];
            tmp[1] = alpha*res[y][x+1] + beta*tmp[1];
            tmp[2] = alpha*res[y][x+2] + beta*tmp[2];
            tmp[3] = alpha*res[y][x+3] + beta*tmp[3];
            ((float4 *)(&C[idx]))[0] = ((float4 *)tmp)[0];
        }
    }


    

}

void shared_mm1(
    float *A, 
    float *B,
    float *C, 
    int M, 
    int N, 
    int K
){
    // 1. 每个block处理一个tile，tile的大小为 BM*BN, 这里设置BM=BN=128
    // 2. 每个block的线程数为16*16， 则每个线程处理tile中的8*8个元素
    // 3. 每次在K维度循环，BK=8, 即每次载入A的128*8个元素，载入B的8*128个元素，则每个线程载入4*2个元素
    float alpha = 1.0;
    float beta = 0.0;
    int block_size = 16;
    int BM = 128;
    int BN = 128;
    dim3 block(block_size, block_size);
    dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);
    __shared_kernel1<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}



//shared memory, avoid the bank conflict
//1. As进行转置
//2. 每个线程进行4*4的计算，然后进行4次计算，即把整个block划分成4*4，而不是8*8
__global__ void __shared_kernel2(
    float *A,
    float *B,
    float *C,
    int M,
    int N,
    int K,
    float alpha,
    float beta
){
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    // const int TX = 8; //each thread
    // const int TY = 8;

    const int start_row = BM * blockIdx.y;
    const int start_col = BN * blockIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // const int tile_x = threadIdx.x * TX;
    // const int tile_y = threadIdx.y * TY;
    // const int tidx = threadIdx.x;
    // const int tidy = threadIdx.y;
    const int tile_x = threadIdx.x<<2;
    const int tile_y = threadIdx.y<<2;
    const int load_a_row = tid >> 1; //每个线程load四个元素，一行8个元素，需要两个线程
    const int load_a_col = (tid & 1)<<2; // %2 == 0? 0:4
    const int load_b_row = tid >> 5; //每个线程load四个元素，一行128个元素，需要32个线程
    const int load_b_col = (tid & 31)<<2; // %32 * 4 
    
    __shared__ float As[BK][BM]; //转置
    __shared__ float Bs[BK][BN];

    float res[8][8] = {0.0}; //每次计算4*4的结果，一共四次
    float reg_a[4];
    float reg_b[4]; 
    for(int i=0; i<K; i+=BK){
        //load A and B into shared memory
        //load A, each thread load 4 float, can directly load 4 float at once with float4 type
        int load_a_idx = (start_row+load_a_row)*K + i+load_a_col; 
        int load_b_idx = (i+load_b_row)*N + start_col+load_b_col;
        ((float4 *)(reg_a))[0] = ((float4 *)(&A[load_a_idx]))[0]; //cast to float4 pointer, then load 4 float
        As[load_a_col][load_a_row] = reg_a[0];
        As[load_a_col+1][load_a_row] = reg_a[1];
        As[load_a_col+2][load_a_row] = reg_a[2];
        As[load_a_col+3][load_a_row] = reg_a[3];
        ((float4 *)(&Bs[load_b_row][load_b_col]))[0] = ((float4 *)(&B[load_b_idx]))[0];
        __syncthreads();

        //compute
        // #pragma unroll
        // for(int j=0; j<BK; j++){
        //     #pragma unroll
        //     for(int y=0; y<TY; y++){
        //         #pragma unroll
        //         for(int x=0; x<TX; x++){
        //             res[y][x] += As[y+tile_y][j] * Bs[j][x+tile_x];
        //         }
        //     }
        // }
        #pragma unroll
        for(int j=0; j<BK; j++){
            #pragma unroll
            for(int ii=0; ii<2; ii++){
                #pragma unroll
                for(int jj=0; jj<2; jj++){
                    //load reg_a and reg_b
                    ((float4 *)(reg_a))[0] = ((float4 *)(&As[j][ii*64+tile_y]))[0];
                    ((float4 *)(reg_b))[0] = ((float4 *)(&Bs[j][jj*64+tile_x]))[0];

                    //compute
                    #pragma unroll
                    for(int x=0; x<4; x++){
                        #pragma unroll
                        for(int y=0; y<4; y++){
                            res[ii*4+x][jj*4+y] += reg_a[x] * reg_b[y];
                        }
                    }
                }
            }
            
            
        }

        __syncthreads();
        

    }

    //store result
    #pragma unroll
    for(int ii=0; ii<2; ii++){
        int res_x = ii<<2;
        #pragma unroll
        for(int jj=0; jj<2; jj++){
            int res_y = jj<<2;
            #pragma unroll
            for(int y=0; y<4; y++){
                
                int idx = (start_row+ii*64+tile_y+y)*N + start_col+jj*64+tile_x;
                
                float tmp[4];
                ((float4*)tmp)[0] = ((float4 *)(&C[idx]))[0];
                tmp[0] = alpha*res[res_x+y][res_y] + beta*tmp[0];
                tmp[1] = alpha*res[res_x+y][res_y+1] + beta*tmp[1];
                tmp[2] = alpha*res[res_x+y][res_y+2] + beta*tmp[2];
                tmp[3] = alpha*res[res_x+y][res_y+3] + beta*tmp[3];
                ((float4 *)(&C[idx]))[0] = ((float4 *)tmp)[0];
                
            }
        }
    }


    

}

void shared_mm2(
    float *A, 
    float *B,
    float *C, 
    int M, 
    int N, 
    int K
){
    // 1. 每个block处理一个tile，tile的大小为 BM*BN, 这里设置BM=BN=128
    // 2. 每个block的线程数为16*16， 则每个线程处理tile中的8*8个元素
    // 3. 每次在K维度循环，BK=8, 即每次载入A的128*8个元素，载入B的8*128个元素，则每个线程载入4*2个元素
    float alpha = 1.0;
    float beta = 0.0;
    int block_size = 16;
    int BM = 128;
    int BN = 128;
    dim3 block(block_size, block_size);
    dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);
    __shared_kernel2<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}























