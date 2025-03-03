
#include<cuda_runtime.h>


#define BLOCK_SIZE 32

// combine max reduce and norm sum reduce
__global__ void __kernel_fused_softmax(float *A, int m, int n){
    int row = blockIdx.x*blockDim.x + threadIdx.x;

    if(row<m){
        //max
        
        float *Current_Row = A+row*n;
        float max_val =  Current_Row[0];
        float norm_sum = 1; //exp(Current_Row[0]-max_val)
        for(int i=1; i<n; i++){
            float cur = Current_Row[i];
            if(max_val<cur){
                norm_sum = norm_sum*expf(max_val-cur);
                max_val = cur;
            }
            norm_sum += expf(cur-max_val);
        }
        for(int i=0; i<n; i++){
            Current_Row[i] = expf(Current_Row[i]-max_val)/norm_sum;
        }

    }
}




void fused_softmax(float *A, int m, int n){
    dim3 block(BLOCK_SIZE);
    dim3 grid((m+BLOCK_SIZE-1)/BLOCK_SIZE);

    __kernel_fused_softmax<<<grid, block>>>(A, m, n);
    
}