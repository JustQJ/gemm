#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/types.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define FLOAT4(val) (reinterpret_cast<float4 *>(&val)[0])
#define HALF2(val) (reinterpret_cast<half2 *>(&val)[0])
#define BFLOAT2(val) (reinterpret_cast<nv_bfloat162 *>(&val)[0])

// y[j][i] = x[i][j]
__global__ void kernel_mat_trans_fp32(float *y, float *x, const int m, const int n){
    int col = blockDim.x*blockIdx.x+threadIdx.x;
    int row = blockDim.y*blockIdx.y+threadIdx.y;

    if(col<n && row<m){
        y[col*m+row] = x[row*n+col];
    }

}

void mat_trans_fp32(torch::Tensor y, torch::Tensor x){
    int m = x.size(0);
    int n = x.size(1);
    dim3 block(32, 32);
    dim3 grid((n+32-1)/32, (m+32-1)/32);
    kernel_mat_trans_fp32<<<grid, block>>>(reinterpret_cast<float *>(y.data_ptr()), reinterpret_cast<float *>(x.data_ptr()), m, n);
}



// y[j][i] = x[i][j] float4 vector load
__global__ void kernel_mat_trans_fp32_4(float *y, float *x, const int m, const int n){
    int col = (blockDim.x*blockIdx.x + threadIdx.x)*4;
    int row = blockDim.y*blockIdx.y + threadIdx.y;

    if(col+3<n && row<m){
        float4 reg_x = FLOAT4(x[row*n+col]); //vector load, but can't store with vector
        y[(col)*m + row] = reg_x.x;
        y[(col+1)*m + row] = reg_x.y;
        y[(col+2)*m + row] = reg_x.z;
        y[(col+3)*m + row] = reg_x.w;
    }else if(col<n && row<m){
        #pragma unroll
        for(int ii=col; ii<n; ii++){
            y[ii*m+row] = x[row*n+ii];
        }
    }    
}


void mat_trans_fp32_4(torch::Tensor y, torch::Tensor x){
    int m = x.size(0);
    int n = x.size(1);

    dim3 block(32, 32);
    dim3 grid((n+32*4-1)/(32*4), (m+32-1)/32);
    
    kernel_mat_trans_fp32_4<<<grid, block>>>(reinterpret_cast<float *>(y.data_ptr()), reinterpret_cast<float *>(x.data_ptr()), m, n);
    
}


//use shared memory to make the write operation also with float4
__global__ void kernel_mat_trans_fp32_4_shared(float *y, float *x, const int m, const int n){
    int global_x = threadIdx.x+blockDim.x*blockIdx.x;
    int global_y = threadIdx.y+blockDim.y*blockIdx.y;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tid = threadIdx.x+blockDim.x*threadIdx.y;
    __shared__ float tile[WARP_SIZE][WARP_SIZE*4]; //each thread load 4 elements into the shared memory
    if(global_x*4+3<n && global_y<m){
        FLOAT4(tile[tidy][tidx*4]) = FLOAT4(x[global_y*n + global_x*4]);
    }

    __syncthreads();
    
    if(global_x*4+3<n && global_y<m){
        //load data from the shared memory with y direction 
        int num_tid_per_col = WARP_SIZE/4;
        float4 reg_x;
        reg_x.x = tile[(tid%num_tid_per_col)*4][tid/num_tid_per_col];
        reg_x.y = tile[(tid%num_tid_per_col)*4+1][tid/num_tid_per_col];
        reg_x.z = tile[(tid%num_tid_per_col)*4+2][tid/num_tid_per_col];
        reg_x.w = tile[(tid%num_tid_per_col)*4+3][tid/num_tid_per_col];

        FLOAT4(y[(blockDim.x*blockIdx.x*4+ tid/8)*m + blockDim.y*blockIdx.y+ (tid%8)]) = reg_x;

    }

}



void mat_trans_fp32_4_shared(torch::Tensor y, torch::Tensor x){
    int m = x.size(0);
    int n = x.size(1);

    dim3 block(32, 32);
    dim3 grid((n+32*4-1)/(32*4), (m+32-1)/32);
    
    kernel_mat_trans_fp32_4<<<grid, block>>>(reinterpret_cast<float *>(y.data_ptr()), reinterpret_cast<float *>(x.data_ptr()), m, n);
    
}



//use shared memory to make the write operation also with float4, solve the shared memory confict
__global__ void kernel_mat_trans_fp32_4_shared_smc(float *y, float *x, const int m, const int n){
    int global_x = threadIdx.x+blockDim.x*blockIdx.x;
    int global_y = threadIdx.y+blockDim.y*blockIdx.y;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tid = threadIdx.x+blockDim.x*threadIdx.y;
    __shared__ float tile[WARP_SIZE][WARP_SIZE*4+1]; //padding one to avoid conflict
  
    float4 reg_x;
    if(global_x*4+3<n && global_y<m){
        reg_x = FLOAT4(x[global_y*n + global_x*4]);

        FLOAT4(tile[tidy][tidx*4]) = FLOAT4(x[global_y*n + global_x*4]);


    }

    __syncthreads();
    
    if(global_x*4+3<n && global_y<m){
        //load data from the shared memory with y direction 
        int num_tid_per_col = WARP_SIZE/4;
        float4 reg_x;
        reg_x.x = tile[tidx][4*tidy];
        reg_x.y = tile[tidx][4*tidy+1];
        reg_x.z = tile[tidx][4*tidy+2];
        reg_x.w = tile[tidx][4*tidy+3];

        // FLOAT4(y[(blockDim.x*blockIdx.x*4+ tid/8)*m + blockDim.y*blockIdx.y+ (tid%8)]) = reg_x;

        y[(blockDim.x*blockIdx.x*4+ 4*tidy)*m + blockDim.y*blockIdx.y+ tidx] = reg_x.x;
        y[(blockDim.x*blockIdx.x*4+ 4*tidy+1)*m + blockDim.y*blockIdx.y+ tidx] = reg_x.y;
        y[(blockDim.x*blockIdx.x*4+ 4*tidy+2)*m + blockDim.y*blockIdx.y+ tidx] = reg_x.z;
        y[(blockDim.x*blockIdx.x*4+ 4*tidy+3)*m + blockDim.y*blockIdx.y+ tidx] = reg_x.w;

    }

}

void mat_trans_fp32_4_shared_smc(torch::Tensor y, torch::Tensor x){
    int m = x.size(0);
    int n = x.size(1);

    dim3 block(32, 32);
    dim3 grid((n+32*4-1)/(32*4), (m+32-1)/32);
    
    kernel_mat_trans_fp32_4<<<grid, block>>>(reinterpret_cast<float *>(y.data_ptr()), reinterpret_cast<float *>(x.data_ptr()), m, n);
    
}


#define STRINGFY(str) #str

#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(mat_trans_fp32)
  TORCH_BINDING_COMMON_EXTENSION(mat_trans_fp32_4)
  TORCH_BINDING_COMMON_EXTENSION(mat_trans_fp32_4_shared)
  TORCH_BINDING_COMMON_EXTENSION(mat_trans_fp32_4_shared_smc)
}