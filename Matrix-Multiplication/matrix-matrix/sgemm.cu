#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <torch/types.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define FLOAT4(val) (reinterpret_cast<float4 *>(&val))[0]

//c[m,n] = a[m,k]*b[k,n]
__global__ void kernel_sgemm_fp32(float *C, float *A, float *B, int M, int N, int K){
    int row = threadIdx.y + blockDim.y*blockIdx.y;
    int col = threadIdx.x + blockDim.x*blockIdx.x;

    if(row<M && col<N){
        float val = 0;
        for(int i=0; i<K; i++){
            val += A[row*K + i]*B[i*N+col];
        }

        C[row*N + col] = val;
    }

}


void sgemm_fp32(torch::Tensor C, torch::Tensor A, torch::Tensor B){
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    dim3 block(WARP_SIZE, WARP_SIZE);
    dim3 grid((N+WARP_SIZE-1)/WARP_SIZE, (M+WARP_SIZE-1)/WARP_SIZE);

    kernel_sgemm_fp32<<<grid, block>>>(reinterpret_cast<float *>(C.data_ptr()), reinterpret_cast<float *>(A.data_ptr()), reinterpret_cast<float *>(B.data_ptr()), M, N, K);
}

//shared memory tile, TM=32, TN=32, TK=32
//using template
template<const int BM=32, const int BN=32, const int BK=32>
__global__ void kernel_sgemm_fp32_tile(float *C, float *A, float *B, int M, int N, int K){
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    int row =  threadIdx.y + blockIdx.y*blockDim.y;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    __shared__ float tile_a[BM][BK];
    __shared__ float tile_b[BK][BN];
    float val = 0;
    for(int k=0; k<K; k+=BK){
        //load data into shared memory
        
        tile_a[tidy][tidx] = A[row*K + (k+tidx)];
        tile_b[tidy][tidx] = B[(k+tidy)*N + col];
        
           
        
        __syncthreads();

        for(int kk=0; kk<BK; kk++){
            val+=tile_a[tidy][kk]*tile_b[kk][tidx];

        }
        __syncthreads();
    }

    if(col<N && row<M){
        C[row*N+col] = val;
    }

}

void sgemm_fp32_tile(torch::Tensor C, torch::Tensor A, torch::Tensor B){
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    dim3 block(WARP_SIZE, WARP_SIZE);
    dim3 grid((N+WARP_SIZE-1)/WARP_SIZE, (M+WARP_SIZE-1)/WARP_SIZE);

    kernel_sgemm_fp32_tile<<<grid, block>>>(reinterpret_cast<float *>(C.data_ptr()), reinterpret_cast<float *>(A.data_ptr()), reinterpret_cast<float *>(B.data_ptr()), M, N, K);
}


//increase the BM, BN to increase computing density, BM and BN is lager, the density is lager
//BM=128, BN=128, BK=8, each thread process 8x8 elements
//block size 16x16
template<const int BM=128, const int BN=128, const int BK=8, const int TM=8, const int TN=8>
__global__ void kernel_sgemm_fp32_tile_8x8(float *C, float *A, float *B, int M, int N, int K){
    int col = blockIdx.x*BN;
    int row =  blockIdx.y*BM;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    int n_elements_per_thread = 4; //each thread load 4 elements 
    int n_threads_per_bk = BK/4;
    int n_threads_per_bn = BN/4;
    int tile_a_r = tid/2;
    int tile_a_c = (tid&1)<<2;
    int tile_b_r = tid/32;
    int tile_b_c = (tid&31)<<2;


    __shared__ float tile_a[BM][BK];
    __shared__ float tile_b[BK][BN];
    float res[TM][TN] = {0};
    
    for(int k=0; k<K; k+=BK){
        //load data into shared memory, using float 4 to load
        FLOAT4(tile_a[tile_a_r][tile_a_c]) = FLOAT4(A[(row+tile_a_r)*K + k+tile_a_c]);
        FLOAT4(tile_b[tile_b_r][tile_b_c]) = FLOAT4(B[(k+tile_b_r)*N + col+tile_b_c]);
        __syncthreads();
        #pragma unroll
        for(int kk=0; kk<BK; kk++){ //outer loop, each elements load once
            #pragma unroll
            for(int i=0; i<TM; i++){
                #pragma unroll
                for(int j=0; j<TN; j++){
                
                    res[i][j]+=tile_a[tidy*TM+i][kk]*tile_b[kk][tidx*TN+j];
                }
                
            }
        }

         __syncthreads();
        
    }

    //store

    for(int i=0; i<TM; i++){
        for(int j=0; j<TN; j+=4){ //each time store with float4
            int store_loc = (row+tidy*TM+i)*N+col+tidx*TN+j;
            FLOAT4(C[store_loc]) = FLOAT4(res[i][j]);
            
        }
    }





    

    // if(col<N && row<M){
    //     C[row*N+col] = val;
    // }

}

void sgemm_fp32_tile_8x8(torch::Tensor C, torch::Tensor A, torch::Tensor B){
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);
    // int BM = 128, BN=128;

    dim3 block(16, 16);
    dim3 grid((N+128-1)/128, (M+128-1)/128);

    kernel_sgemm_fp32_tile_8x8<<<grid, block>>>(reinterpret_cast<float *>(C.data_ptr()), reinterpret_cast<float *>(A.data_ptr()), reinterpret_cast<float *>(B.data_ptr()), M, N, K);
}


//

//avoid memory bank conflict, split the 8x8 into 4 4x4, using vector load to avoid conflict
//reverse as for vector load
template<const int BM=128, const int BN=128, const int BK=8, const int TM=8, const int TN=8, const int OFFSET=0>
__global__ void kernel_sgemm_fp32_tile_8x8_bfc(float *C, float *A, float *B, int M, int N, int K){
    int col = blockIdx.x*BN;
    int row =  blockIdx.y*BM;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    int n_elements_per_thread = 4; //each thread load 4 elements 
    int n_threads_per_bk = BK/4;
    int n_threads_per_bn = BN/4;
    int tile_a_r = tid/n_threads_per_bk;
    int tile_a_c = tid%n_threads_per_bk*4;
    int tile_b_r = tid/n_threads_per_bn;
    int tile_b_c = tid%n_threads_per_bn*4;


    __shared__ float tile_a[BK][BM+OFFSET];
    __shared__ float tile_b[BK][BN+OFFSET];
    float res[TM][TN] = {0};
    float load_a[TM]; //8 is using to load shared memory
    float load_b[TN];
    
    for(int k=0; k<K; k+=BK){
        //load data into shared memory, using float 4 to load
        // FLOAT4(tile_a[tile_a_r][tile_a_c]) = FLOAT4(A[(row+tile_a_r)*K + k+tile_a_c]);
        //directly store with vector load, each warp load 128 bytes and store, so need at least 4 memory access, there still have memory conflict 
        //this conflict is ok due to, each memory access can store 32bytes, as long as bigger than 32bytes for one warp, cannot avoid the conflict
        FLOAT4(tile_b[tile_b_r][tile_b_c]) = FLOAT4(B[(k+tile_b_r)*N + col+tile_b_c]);

        //load a into load_a
        FLOAT4(load_a[0]) = FLOAT4(A[(row+tile_a_r)*K + k+tile_a_c]);
        //then store into shared memory with transpose
        //there has memory conflict for each store, due to two consecutive thread use same banck
        //for example, 0 and 1 using same bank 0 for computing its location
        //for warp0
        //0, 2, 4, ..., 30, using bank 0~15
        //1, 3, 5, ...., 31, also uing bank 0~15
        //arr idx for 0, 2, 4,..., 30: 0, 1,2,...,15  % 32 = 0~15
        //arr idx for 1, 3, 5, ...., 31: 128*4, 128*4+1,..., 128*4+15 % 32= 0~15
        //to avoid the conflict, we can let 1, 3, 5, ...., 31 to 16~31 by setting offset to 4, then
        //offset=4, arr idx for 1, 3, 5, ...., 31: (128+4)*4, (128+4)*4+1,..., (128+4)*4+15 % 32= 16~31
        tile_a[tile_a_c][tile_a_r] = load_a[0];
        tile_a[tile_a_c+1][tile_a_r] = load_a[1];
        tile_a[tile_a_c+2][tile_a_r] = load_a[2];
        tile_a[tile_a_c+3][tile_a_r] = load_a[3];


        __syncthreads();
        //computing 4x4 for 4 times 

        #pragma unroll
        for(int kk=0; kk<BK; kk++){ //outer loop, each elements load once
            //load from shared memory
            FLOAT4(load_a[0]) = FLOAT4(tile_a[kk][tidy*TM/2]);
            FLOAT4(load_b[0]) = FLOAT4(tile_b[kk][tidx*TN/2]);

            FLOAT4(load_a[4]) = FLOAT4(tile_a[kk][tidy*TM/2+BM/2]);
            FLOAT4(load_b[4]) = FLOAT4(tile_b[kk][tidx*TN/2+BN/2]);

            #pragma unroll
            for(int i=0; i<TM; i++){
                #pragma unroll
                for(int j=0; j<TN; j++){
                
                    res[i][j]+=load_a[i]*load_b[j];
                }
                
            }
        }


        __syncthreads();
        
    }

    //store
    #pragma unroll
    for(int i=0; i<TM/2; i++){
        int store_loc = (row+tidy*TM/2+i)*N+col+tidx*TN/2;
        FLOAT4(C[store_loc]) = FLOAT4(res[i][0]); 
    }
    #pragma unroll
    for(int i=0; i<TM/2; i++){
        int store_loc = (row+tidy*TM/2+i)*N+col+tidx*TN/2+BN/2;
        FLOAT4(C[store_loc]) = FLOAT4(res[i][TN/2]); 
    }
    #pragma unroll
    for(int i=0; i<TM/2; i++){
        int store_loc = (row+ BM/2+tidy*TM/2+i)*N+col+tidx*TN/2;
        FLOAT4(C[store_loc]) = FLOAT4(res[i+TM/2][0]); 
    }
    #pragma unroll
    for(int i=0; i<TM/2; i++){
        int store_loc = (row+ BM/2+tidy*TM/2+i)*N+col+tidx*TN/2+BN/2;
        FLOAT4(C[store_loc]) = FLOAT4(res[i+TM/2][TN/2]); 
    }





    

    // if(col<N && row<M){
    //     C[row*N+col] = val;
    // }

}


void sgemm_fp32_tile_8x8_bcf(torch::Tensor C, torch::Tensor A, torch::Tensor B){
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);
    int BM = 128, BN=128;

    dim3 block(16, 16);
    dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);
    //set offset to 4 to avoid some conflict
    kernel_sgemm_fp32_tile_8x8_bfc<<<grid, block>>>(reinterpret_cast<float *>(C.data_ptr()), reinterpret_cast<float *>(A.data_ptr()), reinterpret_cast<float *>(B.data_ptr()), M, N, K);
}

//set offset to 4 to avoid some conflict
void sgemm_fp32_tile_8x8_bcf_offset(torch::Tensor C, torch::Tensor A, torch::Tensor B){
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);
    int BM = 128, BN=128;

    dim3 block(16, 16);
    dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);
    //set offset to 4 to avoid some conflict
    kernel_sgemm_fp32_tile_8x8_bfc<128, 128, 8, 8, 8, 4><<<grid, block>>>(reinterpret_cast<float *>(C.data_ptr()), reinterpret_cast<float *>(A.data_ptr()), reinterpret_cast<float *>(B.data_ptr()), M, N, K);
}


#define STRINGFY(str) #str

#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(sgemm_fp32)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_fp32_tile)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_fp32_tile_8x8)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_fp32_tile_8x8_bcf)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_fp32_tile_8x8_bcf_offset)
}