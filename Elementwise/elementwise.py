import torch
from torch.utils.cpp_extension import load


##compile in the first time
elementwise_lib = load(name='elementwise_lib', 
           sources=['elementwise.cu'], 
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
    b = torch.randn((1024, 1024)).cuda()

    c = torch.randn((1024, 1024)).cuda()

    torch_c = a+b
    

    elementwise_lib.elementwise_add_fp32(c, a, b)

    print(torch.max(torch.abs(torch_c-c)))

    elementwise_lib.elementwise_add_fp32_4(c, a, b)

    print(torch.max(torch.abs(torch_c-c)))


@torch.no_grad()
def benchmark_float16():

    a = torch.randn((1024, 1024)).to(torch.float16).cuda()
    b = torch.randn((1024, 1024)).to(torch.float16).cuda()

    c = torch.randn((1024, 1024)).to(torch.float16).cuda()

    torch_c = a+b

    elementwise_lib.elementwise_add_half(c, a, b)
    print(torch.max(torch.abs(torch_c-c)))

    elementwise_lib.elementwise_add_half2(c, a, b)
    print(torch.max(torch.abs(torch_c-c)))


    elementwise_lib.elementwise_add_half2_4(c, a, b)
    print(torch.max(torch.abs(torch_c-c)))

benchmark_float32()
benchmark_float16()



