
#include<cuda_runtime.h>


#define WARP_SIZE 32

//one warp process one row 
__global__ void __kernel_fused_softmax(float *A, int m, int n){

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




void fused_softmax(float *A, int m, int n){
    int row_per_block = 8;
    dim3 block(WARP_SIZE, row_per_block);
    dim3 grid((m+row_per_block-1)/row_per_block);

    __kernel_fused_softmax<<<grid, block>>>(A, m, n);
    
}