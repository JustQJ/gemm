
#include<cuda_runtime.h>


#define BLOCK_SIZE 32


__global__ void __kernel_fused_softmax(float *A, int m, int n){
    int row = blockIdx.x*blockDim.x + threadIdx.x;

    if(row<m){
        //max
        float max_val = -INFINITY;
        float *Current_Row = A+row*n;
        for(int i=0; i<n; i++){
            max_val = max(max_val, Current_Row[i]);
        }

        //compute sum norm
        float norm_sum = 0.0;
        for(int i=0; i<n; i++){
            norm_sum += expf(Current_Row[i]-max_val);
        }

        //compute the ans

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