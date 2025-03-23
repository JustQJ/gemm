#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <torch/types.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define FLOAT4(val) (reinterpret_cast<float4 *>(&val)[0])
#define HALF2(val) (reinterpret_cast<half2 *>(&val)[0])
#define BFLOAT2(val) (reinterpret_cast<__nv_bfloat162*>(&val)[0])



#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
if(((T).options().dtype() != (th_type))) {                   \
  std::cout << "Tensor Info:" << (T).options() << std::endl; \
  throw std::runtime_error("values must be "#th_type);       \
}


// c = a+b, float 32
__global__ void kernel_elementwise_add_fp32(float *c, float *a, float *b, int n){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx<n){
        c[idx] = a[idx]+b[idx];
    }

}


void elementwise_add_fp32(torch::Tensor c, torch::Tensor a, torch::Tensor b){
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32);
    const int n_dim = a.dim();
    int n = 1;
    for(int i=0; i<n_dim; i++)
        n = n*a.size(i);
    dim3 block(256);
    dim3 grid((n+255)/256);
    kernel_elementwise_add_fp32<<<grid, block>>>(reinterpret_cast<float *>(c.data_ptr()), reinterpret_cast<float *>(a.data_ptr()), reinterpret_cast<float *>(b.data_ptr()), n);
}



// c = a+b float32  float4 vector load
__global__ void kernel_elementwise_add_fp32_4(float *c, float *a, float *b, int n){
    int idx = 4*(threadIdx.x + blockDim.x*blockIdx.x);
    if(idx+3<n){
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_b = FLOAT4(b[idx]);
        float4 reg_c;

        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;    

        FLOAT4(c[idx]) = reg_c;
    }else{
        
        for(int ii=idx; ii<n; ii++){
            c[ii] = a[ii]+b[ii];
        }   

    }

}


// void elementwise_add_fp32_4(float *c, float *a, float *b, int n){
//     dim3 block(256);
//     dim3 grid((n+256*4-1)/(256*4));
//     kernel_elementwise_add_fp32_4<<<grid, block>>>(c, a, b, n);
// }

void elementwise_add_fp32_4(torch::Tensor c, torch::Tensor a, torch::Tensor b){

    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32);

    const int n_dim = a.dim();
    int n = 1;
    for(int i=0; i<n_dim; i++)
        n = n*a.size(i);
    dim3 block(256);
    dim3 grid((n+256*4-1)/(256*4));
    kernel_elementwise_add_fp32_4<<<grid, block>>>(reinterpret_cast<float *>(c.data_ptr()), reinterpret_cast<float *>(a.data_ptr()), reinterpret_cast<float *>(b.data_ptr()), n);
}





//c = a+b half add 
__global__ void kernel_elementwise_add_half(half *c, half *a, half *b, int n){

    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx<n){
        c[idx] = __hadd(a[idx], b[idx]);
    }

}


// void elementwise_add_half(half *c, half *a, half *b, int n){
//     dim3 block(256);
//     dim3 grid((n+256-1)/(256));
//     kernel_elementwise_add_half<<<grid, block>>>(c, a, b, n);
// }


void elementwise_add_half(torch::Tensor c, torch::Tensor a, torch::Tensor b){

    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf);

    const int n_dim = a.dim();
    int n = 1;
    for(int i=0; i<n_dim; i++)
        n = n*a.size(i);
    dim3 block(256);
    dim3 grid((n+256-1)/(256));
    kernel_elementwise_add_half<<<grid, block>>>(reinterpret_cast<half *>(c.data_ptr()), reinterpret_cast<half *>(a.data_ptr()), reinterpret_cast<half *>(b.data_ptr()), n);
}




//c=a+b half add2
__global__ void kernel_elementwise_add_half2(half *c, half *a, half *b, int n){

    int idx = (threadIdx.x + blockDim.x*blockIdx.x)*2;
    if(idx<n){
        half2 reg_a = HALF2(a[idx]);
        half2 reg_b = HALF2(b[idx]);
        HALF2(c[idx]) = __hadd2(reg_a, reg_b);
    }
}

// void elementwise_add_half2(half *c, half *a, half *b, int n){
//     dim3 block(256);
//     dim3 grid((n+256*2-1)/(256*2));
//     kernel_elementwise_add_half2<<<grid, block>>>(c, a, b, n);
// }


void elementwise_add_half2(torch::Tensor c, torch::Tensor a, torch::Tensor b){

    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf);

    const int n_dim = a.dim();
    int n = 1;
    for(int i=0; i<n_dim; i++)
        n = n*a.size(i);
    dim3 block(256);
    dim3 grid((n+256*2-1)/(256*2));
    kernel_elementwise_add_half2<<<grid, block>>>(reinterpret_cast<half *>(c.data_ptr()), reinterpret_cast<half *>(a.data_ptr()), reinterpret_cast<half *>(b.data_ptr()), n);
}





//c=a+b half, use float4 load, each time load 8 elements
__global__ void kernel_elementwise_add_half2_4(half *c, half *a, half *b, int n){
    int idx = (threadIdx.x +blockDim.x*blockIdx.x)*8;
    if(idx+7<n){
        half2 reg_a[4];
        half2 reg_b[4];
        half2 reg_c[4];
        FLOAT4(reg_a) = FLOAT4(a[idx]);
        FLOAT4(reg_b) = FLOAT4(b[idx]);

        reg_c[0] = __hadd2(reg_a[0], reg_b[0]);
        reg_c[1] = __hadd2(reg_a[1], reg_b[1]);
        reg_c[2] = __hadd2(reg_a[2], reg_b[2]);
        reg_c[3] = __hadd2(reg_a[3], reg_b[3]);

        FLOAT4(c[idx]) = FLOAT4(reg_c);
    }else{
        half2 reg_a;
        half2 reg_b;
        for(int ii=idx; ii<n; ii+=2){
            reg_a = HALF2(a[ii]);
            reg_b = HALF2(b[ii]);
            HALF2(c[ii]) = __hadd2(reg_a, reg_b);
        }
    }
}


// void elementwise_add_half2_4(half *c, half *a, half *b, int n){
//     dim3 block(256*8);
//     dim3 grid((n+256*8-1)/(256*8));
//     kernel_elementwise_add_half2_4<<<grid, block>>>(c, a, b, n);
// }



void elementwise_add_half2_4(torch::Tensor c, torch::Tensor a, torch::Tensor b){

    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf);

    const int n_dim = a.dim();
    int n = 1;
    for(int i=0; i<n_dim; i++)
        n = n*a.size(i);
    dim3 block(256);
    dim3 grid((n+256*8-1)/(256*8));
    kernel_elementwise_add_half2_4<<<grid, block>>>(reinterpret_cast<half *>(c.data_ptr()), reinterpret_cast<half *>(a.data_ptr()), reinterpret_cast<half *>(b.data_ptr()), n);
}


#define STRINGFY(str) #str

#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_fp32)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_fp32_4)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_half)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_half2)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_half2_4)
}



