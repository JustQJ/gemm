import torch
from torch.utils.cpp_extension import load


##compile in the first time
matrix_trans_lib = load(name='matrix_trans_lib', 
           sources=['matrix_trans.cu'], 
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

@torch.no_grad()
def benchmark_float32():

    a = torch.randn((1024, 1024)).cuda()
    b = a.transpose(0,1)

    c = torch.randn((1024, 1024)).cuda()

    


    matrix_trans_lib.mat_trans_fp32(c, a)
    print(torch.max(torch.abs(b-c)))
    matrix_trans_lib.mat_trans_fp32_4(c, a)
    print(torch.max(torch.abs(b-c)))
    matrix_trans_lib.mat_trans_fp32_4_shared(c, a)
    print(torch.max(torch.abs(b-c)))
    matrix_trans_lib.mat_trans_fp32_4_shared_smc(c, a)
    print(torch.max(torch.abs(b-c)))
    print(a[1][2], b[1][2], c[1][2], a[2][1])



benchmark_float32()

