import torch
import triton
import triton.language as tl



@triton.jit
def fused_softmax_kernel(
    x_ptr,
    out_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):  
    pid = tl.program_id(0) #row id

    start_idx = pid * N

    offsets = tl.arange(0, BLOCK_SIZE)

    mask = offsets<N

    x = tl.load(x_ptr+start_idx+offsets, mask=mask, other=-float('inf'))

    max_x = tl.max(x, axis=0)

    mius_x = x-max_x
    
    exp_x = tl.exp(mius_x)
    sum_x = tl.sum(exp_x, axis=0)
    
    out_put = exp_x/sum_x

    output_ptrs = out_ptr + start_idx+offsets
    tl.store(output_ptrs, out_put, mask=mask)




    
@triton.jit
def fused_softmax_kernel(
    x_ptr,
    y_ptr,
    M, N,
    BLOCK_SIZE:tl.constexpr
):
    
    pid = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    
    start_idx = pid*N

    x = tl.load(x_ptr+start_idx+offsets, mask=offsets<N, other=-float('inf'))

    x_max = tl.max(x, axis=0)

    x = x-x_max
    x = tl.exp(x)

    x_sum = tl.sum(x, axis=0)

    x = x/x_sum

    tl.store(y_ptr+start_idx+offsets,x ,mask=offsets<N)



    


def fused_softmax(x):
    M = x.shape[0]
    N = x.shape[1]
    # grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']),) 
    BLOCK_SIZE = triton.next_power_of_2(N)
    out = torch.empty_like(x)

    fused_softmax_kernel[(M, 1, 1)](x,out, M, N, BLOCK_SIZE)

    return out
    


a = torch.randn((1024, 1024)).cuda()

sa = fused_softmax(a)

ta = torch.nn.functional.softmax(a)

print(torch.max(torch.abs((sa-ta))))