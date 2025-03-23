import time
import torch
from torch.utils.cpp_extension import load


##compile in the first time
sgemm_lib = load(name='sgemm', 
           sources=['sgemm.cu'], 
           extra_cuda_cflags=[
               "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
            ], 
           extra_cflags=['-std=c++17'])


"""

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(sgemm_fp32)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_fp32_tile)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_fp32_tile_8x8)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_fp32_tile_8x8_bcf)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_fp32_tile_8x8_bcf_offset)
}
"""

def verify(MNK):
    A = torch.randn((MNK, MNK)).cuda()
    B = torch.randn((MNK, MNK)).cuda()
    C = torch.randn((MNK, MNK)).cuda()

    
    New_C = torch.randn((MNK, MNK)).cuda()

    C = A@B


    sgemm_lib.sgemm_fp32(New_C, A, B)
    print(torch.max(torch.abs(C-New_C)))

    sgemm_lib.sgemm_fp32_tile(New_C, A, B)
    print(torch.max(torch.abs(C-New_C)))

    sgemm_lib.sgemm_fp32_tile_8x8(New_C, A, B)
    print(torch.max(torch.abs(C-New_C)))
   

    sgemm_lib.sgemm_fp32_tile_8x8_bcf(New_C, A, B)
    print(torch.max(torch.abs(C-New_C)))

    sgemm_lib.sgemm_fp32_tile_8x8_bcf_offset(New_C, A, B)
    print(torch.max(torch.abs(C-New_C)))


def bench(MNK):
    A = torch.randn((MNK, MNK)).cuda()
    B = torch.randn((MNK, MNK)).cuda()
   
    print("size: M=N=K=", MNK)
    
    New_C = torch.randn((MNK, MNK)).cuda()
    for _ in range(10):
        C = A@B

    torch.cuda.synchronize()

    t1 = time.time()
    for i in range(10):
        C = A@B
    torch.cuda.synchronize()
    t2 = time.time()
    torch_time = t2-t1
    print("torch time: ", t2-t1)


   

    t1 = time.time()
    for i in range(10):
        sgemm_lib.sgemm_fp32(New_C, A, B)
    torch.cuda.synchronize()
    t2 = time.time()

    print("sgemm_fp32 time: ", t2-t1, "speedup: ", torch_time/(t2-t1))

    t1 = time.time()
    for i in range(10):
        sgemm_lib.sgemm_fp32_tile(New_C, A, B)
    torch.cuda.synchronize()
    t2 = time.time()

    print("sgemm_fp32_tile time: ", t2-t1, "speedup: ", torch_time/(t2-t1))

    t1 = time.time()
    for i in range(10):
        sgemm_lib.sgemm_fp32_tile_8x8(New_C, A, B)
    torch.cuda.synchronize()
    t2 = time.time()

    print("sgemm_fp32_tile_8x8 time: ", t2-t1, "speedup: ", torch_time/(t2-t1))

    t1 = time.time()
    for i in range(10):
        sgemm_lib.sgemm_fp32_tile_8x8_bcf(New_C, A, B)
    torch.cuda.synchronize()
    t2 = time.time()

    print("sgemm_fp32_tile_8x8_bcf time: ", t2-t1, "speedup: ", torch_time/(t2-t1))

    t1 = time.time()
    for i in range(10):
        sgemm_lib.sgemm_fp32_tile_8x8_bcf_offset(New_C, A, B)
    torch.cuda.synchronize()
    t2 = time.time()

    print("sgemm_fp32_tile_8x8_bcf_offset time: ", t2-t1, "speedup: ", torch_time/(t2-t1))

    
    

    
   

    
   
   

    
    

    
    


for MNK in [1024, 1024*2, 1024*3, 1024*4, 1024*5, 1024*6]:
    bench(MNK)



