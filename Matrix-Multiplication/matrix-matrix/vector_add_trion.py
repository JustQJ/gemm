import time
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, 
               y_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr          
               ):
    # Define the block size

    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE  ## 当前block的起始位置
    offsets = block_start + tl.arange(0, BLOCK_SIZE) ## 当前block所有元素的address
    mask = offsets < n_elements ## 当前block所有元素的address是否在n_elements范围内

    x = tl.load(x_ptr+offsets, mask=mask) ## 当前block的x
    y = tl.load(y_ptr+offsets, mask=mask) ## 当前block的y

    output = x + y ## 计算，这个block的内部元素的计算如何在SM上运行不需要用户关心

    tl.store(output_ptr+offsets, output, mask=mask) ## 将结果写回内存


def add(x: torch.Tensor, y: torch.Tensor):
    assert x.shape == y.shape
    assert x.is_cuda and y.is_cuda
    n_elements = x.numel()
    output = torch.empty_like(x)
    

    # Launch the kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), ) ## 这是一个grid函数，BLOCK_SIZE 来自下面kernel的参数表
    
    add_kernel[grid](x, y, output, n_elements, 1024)
    return output


def benchmark():
    x = torch.randn(98432).cuda()
    y = torch.randn(98432).cuda()


    ## warm up
    for _ in range(10):
        output = x+y

    iter_num = 100
    # torch cuda time
    
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(iter_num):
        output = x + y
    torch.cuda.synchronize()
    t2 = time.time()

    print(f'Cuda time: {(t2-t1)/iter_num}')

    # triton cuda time
    for _ in range(10):
        output = add(x, y)

    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(iter_num):
        output = add(x, y)
    torch.cuda.synchronize()
    t2 = time.time()
    print(f'Triton time: {(t2-t1)/iter_num}')



if __name__ == "__main__":
    benchmark()

    # Cuda time: 6.093978881835937e-06
    # Triton time: 9.054422378540039e-05