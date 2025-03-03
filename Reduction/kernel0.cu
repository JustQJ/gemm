#include<cuda_runtime.h>

#define BLOCK_SIZE 256 //each block process 512 elements, not the thread number 


__global__ void __kernel_reduce_sum(float *A, int N, float *out){

    int start_id = blockIdx.x*BLOCK_SIZE+ threadIdx.x;
    int tid = threadIdx.x;
    //add the grid 
    float val = 0;
    for(int i=start_id; i<N; i+=gridDim.x*BLOCK_SIZE){
        val += A[i];
    }
    __shared__ float data[BLOCK_SIZE];
    
    data[tid] = val;
    

    __syncthreads();
    

    //reduce 
    for(int offset=1; offset<blockDim.x; offset*=2){
        if(tid%(2*offset)==0){
            data[tid]+=data[tid+offset];
        }
        __syncthreads();
    }

    if(tid==0)
        out[blockIdx.x] = data[0];



}



void custom_reduce_sum(float *A, int N, float *res, int block_num){
    dim3 block(BLOCK_SIZE);
    dim3 grid(block_num);
   

    __kernel_reduce_sum<<<grid, block>>>(A, N, res);
    
    // cudaDeviceSynchronize();
    // //assure grid.x<=BLOCK_SIZE
    // __kernel_reduce_sum<<<1, block>>>(output, grid.x, res);
    

}