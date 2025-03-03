#include<cuda_runtime.h>



#define BLOCK_SIZE 256

// avoid warp divergence
// n/2->0, n/4->0, n/8->0
__global__ void __kernel_reduce_sum(float *A, int N, float *res){

    int start_id = blockIdx.x*BLOCK_SIZE+ threadIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid/32;
    int line_id = tid%32;
    int warp_num = BLOCK_SIZE/32;
    
    //add the grid 
    float val = 0;
    for(int i=start_id; i<N; i+=gridDim.x*BLOCK_SIZE){
        val += A[i];
    }

    //use the shfl instruction
    //reduce all val into fisrt thread
    for(int offset=16; offset>0; offset>>=1){
        val+=__shfl_down_sync(0xffffffff, val, offset);
    }

    __shared__ float data[32];
    if(line_id==0)
        data[warp_id] = val;
    
    __syncthreads();
    if(warp_id==0){
        if(line_id<warp_num)
            val = data[line_id];
        else
            val = 0;

        for(int offset=16; offset>0; offset>>=1){
            val+=__shfl_down_sync(0xffffffff, val, offset);
        }

        if(line_id==0)
            res[blockIdx.x] = val;
    }

    
    
    // data[tid] = val;
    

    // __syncthreads();

    // for(int offset=BLOCK_SIZE/2; offset>0; offset>>=1){
    //     if(tid<offset){
    //         data[tid]+=data[tid+offset];
    //     }
    //     __syncthreads();
    // }
    // if(tid==0)
    //     res[blockIdx.x] = data[0];

  



}





void custom_reduce_sum(float *A, int N, float *res, int block_num){
    dim3 block(BLOCK_SIZE);
    dim3 grid(block_num); //m*BLOCK_SIZE<=N, m<=BLOCK_SIZE

    __kernel_reduce_sum<<<grid, block>>>(A, N, res);
    // cudaDeviceSynchronize();
    // __kernel_reduce_sum<<<1, block>>>(res, block_num, res);

}