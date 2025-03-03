#include<cuda_runtime.h>



#define BLOCK_SIZE 256

// avoid warp divergence
// n/2->0, n/4->0, n/8->0
__global__ void __kernel_reduce_sum(float *A, int N, float *res){

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

    for(int offset=BLOCK_SIZE/2; offset>0; offset>>=1){
        if(tid<offset){
            data[tid]+=data[tid+offset];
        }
        __syncthreads();
    }
    if(tid==0)
        res[blockIdx.x] = data[0];

  



}





void custom_reduce_sum(float *A, int N, float *res, int block_num){
    dim3 block(BLOCK_SIZE);
    dim3 grid(block_num); //m*BLOCK_SIZE<=N, m<=BLOCK_SIZE

    __kernel_reduce_sum<<<grid, block>>>(A, N, res);
    // cudaDeviceSynchronize();
    // __kernel_reduce_sum<<<1, block>>>(res, block_num, res);

}