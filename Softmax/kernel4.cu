#include<cuda_runtime.h>




#define WARP_SIZE 32
#define BLOCK_SIZE 512
#define SHARED_SIZE 4096

//one warp process one row 
__global__ void __kernel_fused_softmax_warp_level(float *A, int m, int n){

    int warp_id = threadIdx.y;
    int line_id = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x*blockDim.y + warp_id;

    if(row<m){
        //max
        
        float *Current_Row = A+row*n;
        float max_val =  Current_Row[line_id];
        float norm_sum = 1; //exp(Current_Row[0]-max_val)
        for(int i=line_id+WARP_SIZE; i<n; i+=WARP_SIZE){
            float cur = Current_Row[i];
            if(max_val<cur){
                norm_sum = norm_sum*expf(max_val-cur);
                max_val = cur;
            }
            norm_sum += expf(cur-max_val);
        }

        //reduce the max to the first thread
        float temp_max_val = max_val;
        for(int offset=16; offset>0; offset>>=1){
            temp_max_val = max(temp_max_val, __shfl_down_sync(0xffffffff, temp_max_val, offset));
        }

        //broadcast max_val to all threads
        float global_max_val = __shfl_sync(0xffffffff, temp_max_val, 0);

        //update the norm_sum
        norm_sum = norm_sum*expf(max_val-global_max_val);

        //reduce the norm_sum to the first thread
        for(int offset=16; offset>0; offset>>=1){
            norm_sum += __shfl_down_sync(0xffffffff, norm_sum, offset);
        }

        //broadcast the normsum to all threads

        norm_sum = __shfl_sync(0xffffffff, norm_sum, 0);



        for(int i=line_id; i<n; i+=WARP_SIZE){
            float cur = Current_Row[i];
            Current_Row[i] = expf(cur-global_max_val)/norm_sum;
        }


    }
}

//one block process one row when n>1024

__global__ void __kernel_fused_softmax_block_level(float *A, int m, int n){

    int tid = threadIdx.x;
    // int warp_id = tid / WARP_SIZE;
    // int line_id = tid % WARP_SIZE;
    // int warp_num = blockDim.x/WARP_SIZE;
    int row = blockIdx.x;

    if(row<m){
        float *Current_Row = A+row*n;
        __shared__ float data[BLOCK_SIZE];
        
        float max_val = -INFINITY;
        float norm_base = 0;
        for(int i=tid; i<n; i+=BLOCK_SIZE){
            float cur = Current_Row[i];
            if(cur>max_val){
                norm_base = norm_base * expf(max_val-cur);
                max_val = cur;
            }

            norm_base += expf(cur-max_val);
        }

        data[tid] = max_val;
        __syncthreads();
        for(int offset=BLOCK_SIZE/2; offset>0; offset>>=1){
            if(tid<offset){
                data[tid] = max(data[tid], data[tid+offset]);
            }
            __syncthreads();
        }

        float global_max_val = data[0];

        norm_base = norm_base*expf(max_val-global_max_val);
        
        //reduce the norm_base
        data[tid] = norm_base;

        __syncthreads();
        for(int offset=BLOCK_SIZE/2; offset>0; offset>>=1){
            if(tid<offset){
                data[tid] += data[tid+offset];
            }
            __syncthreads();
        }

        norm_base = data[0];

        for(int i=tid; i<n; i+=BLOCK_SIZE){
            float cur = Current_Row[i];
            Current_Row[i] = expf(cur-global_max_val)/norm_base;
        }


    }



}

// use shared memory to load data first when n<=4096
__global__ void __kernel_fused_softmax_block_level_shared(float *A, int m, int n){

    int tid = threadIdx.x;
    // int warp_id = tid / WARP_SIZE;
    // int line_id = tid % WARP_SIZE;
    // int warp_num = blockDim.x/WARP_SIZE;
    int row = blockIdx.x;

    if(row<m){
        float *Current_Row = A+row*n;
        __shared__ float data[BLOCK_SIZE];

        __shared__ float AR[SHARED_SIZE];
        for(int i=tid; i<n; i+=BLOCK_SIZE){
            AR[i] = Current_Row[i]; 
        }
        
        
        float max_val = -INFINITY;
        float norm_base = 0;
        for(int i=tid; i<n; i+=BLOCK_SIZE){
            float cur = AR[i];
            if(cur>max_val){
                norm_base = norm_base * expf(max_val-cur);
                max_val = cur;
            }

            norm_base += expf(cur-max_val);
        }

        data[tid] = max_val;
        __syncthreads();
        for(int offset=BLOCK_SIZE/2; offset>0; offset>>=1){
            if(tid<offset){
                data[tid] = max(data[tid], data[tid+offset]);
            }
            __syncthreads();
        }

        float global_max_val = data[0];

        norm_base = norm_base*expf(max_val-global_max_val);
        
        //reduce the norm_base
        data[tid] = norm_base;

        __syncthreads();
        for(int offset=BLOCK_SIZE/2; offset>0; offset>>=1){
            if(tid<offset){
                data[tid] += data[tid+offset];
            }
            __syncthreads();
        }

        norm_base = data[0];

        for(int i=tid; i<n; i+=BLOCK_SIZE){
            float cur = AR[i];
            Current_Row[i] = expf(cur-global_max_val)/norm_base;
        }


    }



}




void fused_softmax(float *A, int m, int n){
    int warp_level_n = 1024;
    if(n<=warp_level_n){
         int row_per_block = 4;
        dim3 block(WARP_SIZE, row_per_block);
        dim3 grid((m+row_per_block-1)/row_per_block);

        __kernel_fused_softmax_warp_level<<<grid, block>>>(A, m, n);
    }else if(n<=SHARED_SIZE){
        dim3 block(BLOCK_SIZE);
        dim3 grid(m);
        __kernel_fused_softmax_block_level_shared<<<grid, block>>>(A, m, n);
    }else{
        dim3 block(BLOCK_SIZE);
        dim3 grid(m);
        __kernel_fused_softmax_block_level<<<grid, block>>>(A, m, n);
    }
   
    
}