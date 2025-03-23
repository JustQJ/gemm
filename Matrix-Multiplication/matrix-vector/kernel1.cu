#include<cuda_runtime.h>


#define TILE_M 16 //wrap number each block, change with the m to ensure same enough number of blocks
#define TILE_N 32 

//use vector load float4 to increase memory usage
//each warp deals with deal result
__global__ void __kernel_sgemv(float *A, float *x, float *y, int m, int n, float alpha, float beta){
    int line_id = threadIdx.x % 32;
    int warp_id = threadIdx.y;
    int row = warp_id + blockIdx.x*blockDim.y;

    if(row<m){
        float val = 0;
        //int iter = n/32/4; //each time load 32*4 elemets for using float4
        for(int i=0; i<n; i+=32*4){
            int nn = i+line_id*4;
            if(nn+4<n){
                float4 a = reinterpret_cast<float4 *>(&A[row*n+nn])[0];
                float4 b = reinterpret_cast<float4 *>(&x[nn])[0];
                val+=a.x*b.x;
                val+=a.y*b.y;
                val+=a.z*b.z;
                val+=a.w*b.w;
            }else if(nn<n){ //not enough 4, so load with float
                for(int nnn=nn; nnn<n; nnn++){
                    float a = A[row*n+nnn];
                    float b = x[nnn];
                    val+=a*b;
                }
            }
            
        }

        //shf down to line_id=0;
        for(int offset=16; offset>0; offset>>=1){
            val+= __shfl_down_sync(0xffffffff, val, offset);
        }
        // for(int offset=16; offset>0; offset>>=2){
        //     val+=__shfl_down_sync(0xffffffff, val, offset)
        // }
        if(line_id==0){
            y[row] = alpha * val + beta* y[row];
        }
    }

}
//y = A*x
__global__ void __kernel_sgemv(float *A, float *x, float *y, int m, int n){

    int wrap_id =  threadIdx.y;
    int line_id =   threadIdx.x;
    int row = wrap_id+blockIdx.x*blockDim.y;

    if(row<m){
        //using float 4 to load

        float val = 0;
        for(int i=4*line_id; i<n; i+=4*32){
            float4 a = ((float4 *)(&A[row*n+i]))[0];
            float4 b = ((float4 *)(&x[i]))[0];

            val += a.x*b.x;
            val += a.y*b.y;
            val += a.z*b.z;
            val += a.w*b.w;

        } 

        //reduce all results of one warp into the first thread

        for(int offset=16; offset>0; offset>>=1){
            val += __shfl_down_sync(0xffffffff, val, offset);
        }  

        if(line_id==0)
            y[row] = val;

    }

}



void custom_sgemv(float *A, float *x, float *y, int m, int n, float alpha, float beta){
    dim3 block(32, TILE_M);
    dim3 grid((m+TILE_M-1)/TILE_M);
    __kernel_sgemv<<<grid, block>>>(A, x, y, m, n, alpha, beta);
}






