

import torch
import triton
import triton.language as tl


@triton.jit
def gemv_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    cur_a_ptr = a_ptr+pid*N

    offsets = tl.arange(0, BLOCK_SIZE)

    masks = offsets<N

    a_ptrs = cur_a_ptr+offsets

    v_a = tl.load(a_ptrs, mask=masks, other=0)

    v_b = tl.load(b_ptr+offsets, mask=masks, other=0)

    v_c = v_a*v_b
    c = tl.sum(v_c)


    tl.store(c_ptr+pid, c)


# def kernel_gemv(
#     a_ptr,
#     x_ptr,
#     y_ptr,
#     m, n,
#     BLOCK_SIZE: tl.constexpr,
# ):
#     pid = tl.program_id(0)
#     row_ptr = a_ptr+pid*n
#     offset = tl.arange(0, BLOCK_SIZE)

#     a_ptrs = row_ptr+offset
#     b_ptrs= x_ptr+offset
#     a = tl.load(a_ptrs, mask=offset<n, other=0)
#     b = tl.load(b_ptrs, mask=offset<n, other=0)

#     res = a*b
#     c= tl.sum(res)

#     tl.store(y_ptr+pid, c)



def gemv(a,b):
    M, N = a.shape
    assert N == b.shape[0]
    block_size = triton.next_power_of_2(N)
    
    c = torch.empty(M, dtype=a.dtype, device=a.device)

    gemv_kernel[(M, 1,1)](a,b,c,M, N, block_size)

    return c
    
    

a = torch.randn(1024, 1024).cuda()
b = torch.randn(1024).cuda()

c = a@b

print(c.shape)

g_c = gemv(a,b)

print(torch.max(torch.abs(c-g_c)))

print(c[:10])
print(g_c[:10])