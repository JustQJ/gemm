#include<cuda_runtime.h>



#define BLOCK_SIZE 256

// avoid warp divergence
// n/2->0, n/4->0, n/8->0
//using register to reduce data in a warp
//using float4 to vector load
__global__ void __kernel_reduce_sum(float *A, int N, float *res){

    int warp_id = threadIdx.x / 32;
    int line_id = threadIdx.x % 32;
    int tid = threadIdx.x;
    int warp_num = blockDim.x/32;

    __shared__ float data[32]; //used to reduce data in one block

    int idx = BLOCK_SIZE*4 + 4*tid;
    float val = 0;
    if(idx+3<N){
        float4 temp = ((float4 *)(&A[idx]))[0];
        val += temp.x;
        val += temp.y;
        val += temp.z;
        val += temp.w;
    }else if(idx<N){
        for(int ii=idx; ii<N; ii++){
            val += A[ii];
        }
    }

    //reduce in one warp with warp instruction

    for(int offset=16; offset>0; offset>>=1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    //write to share data
    if(line_id==0)
        data[warp_id] = val;

    __syncthreads();

    
    if(warp_id==0){//reduce in the first warp

        val = 0;
        if(line_id<warp_num){
            val = data[line_id];
        }

        for(int offset=16; offset>0; offset>>=1){
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        //result for the block in the first thread of the block

        if(line_id==0){ //reduce for different block
            atomicAdd(res, val);
        }

    }

   

  



}





void custom_reduce_sum(float *A, int N, float *res){
    dim3 block(BLOCK_SIZE);
    dim3 grid((N+BLOCK_SIZE*4-1)/BLOCK_SIZE); //one thread deal 4 element

    __kernel_reduce_sum<<<grid, block>>>(A, N, res);
    // cudaDeviceSynchronize();
    // __kernel_reduce_sum<<<1, block>>>(res, block_num, res);

}