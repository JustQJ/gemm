#include<cuda_runtime.h>

#define TILE_M 16 //small when m is samll, large when m is large, such has enough blocks
#define TILE_N 32

#define FULL_MASK 0xffffffff


// each block has 32 warp, each warp process one element, each each load one element each time with float 
__global__ void __kernel_sgemv(float *A, float *x, float *y, int m, int n, float alpha, float beta){

    int wrap_id = threadIdx.y;
    int line_id = threadIdx.x%32;
    int row = blockIdx.x*TILE_M+wrap_id;

    if(row<m){
        float val = 0;
        for(int i=0; i<n; i+=TILE_N){
            int nn = i+line_id;
            if(nn<n){
                float a = A[row*n + nn];
                float b = x[nn];
                val+=a*b;
            }
           
        }

        //synchronize to the first thread in a wrap, line_id=0
        //use shfl_down_sync instruction to reduce all val to the first
        for(int offset=16; offset>0; offset>>=1){
                val+=__shfl_down_sync(FULL_MASK, val, offset);
        }
        
        if(line_id==0){
            y[row] = alpha * val + beta * y[row];
        }


        
    }

}


void custom_sgemv(float *A, float *x, float *y, int m, int n, float alpha, float beta){

    dim3 block(32, TILE_M); //each swap in same row, process one output element
    dim3 grid((m+TILE_M-1)/TILE_M);
    __kernel_sgemv<<<grid, block>>>(A, x, y, m,n, alpha, beta);

}