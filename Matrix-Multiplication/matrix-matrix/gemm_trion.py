import torch
import triton

import triton.language as tl


# def get_autotune_config():
#     return [
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':64}, num_stage=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#     ]


# @triton.autotune(
#     configs=get_autotune_config(),
#     key=['M', 'N', 'K'],
# )

# @triton.autotune(
#         configs=[]
# )
@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K, alpha, beta,

    stride_am, stride_ak, ## 表示每个维度的地址偏移
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    
    pid = tl.program_id(0)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offset_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    offset_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offset_am[:, None] * stride_am + offset_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offset_k[:, None] * stride_bk + offset_bn[None, :] * stride_bn)

    accumulators= tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offset_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offset_k[:, None] < K - k * BLOCK_SIZE_K, other=0)

        accumulators = tl.dot(a, b, accumulators)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offset_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    c_ptrs = c_ptr + offset_cm[:, None]*stride_cm + offset_cn[None, :]*stride_cn
    c_mask = (offset_cm[:, None] < M) & (offset_cn[None, :] < N)

    c = alpha * accumulators + beta * tl.load(c_ptrs, mask=c_mask, other=0)
    c = c.to(tl.float16)
    tl.store(c_ptrs, c, mask=c_mask)





@triton.jit
def kernel_gemm(
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # compute pid
    
    pid = tl.program_id(0)

    n_per_row = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid//n_per_row
    pid_n = pid%n_per_row

    offset_a = pid_m*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_b = pid_n*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offset_a[:, None]*stride_am + offset_k[None, :])
    b_ptrs = b_ptr + (offset_k[:, None]*stride_bk + offset_b[None,:])

    res = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offset_k[None, :]<K-k*BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offset_k[:, None]<K-BLOCK_SIZE_K*k, other=0)

        res = tl.dot(a,b, res)

        a_ptrs += BLOCK_SIZE_K*stride_ak
        b_ptrs += BLOCK_SIZE_K*stride_bk

    
    c_ptrs = c_ptr + (offset_a[:, None]*stride_cm + offset_b[None, :]*stride_cn)

    mask = offset_a[:, None]<M & offset_b[None, :]<N

    tl.store(c_ptrs, res, mask=mask)
    

    

def gemm(a,b):
    M,K = a.shape

    K,N = b.shape

    c = torch.zeros((M,N),dtype=torch.float16,device=a.device)
    grid = lambda meta:(triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    # grid = lambda meta:(triton.cdiv(M, meta['BLOCK_SIZE_M'] ) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )

    gemm_kernel[grid](a, b, c, M, N, K, 1.0, 0.0, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64)

    return c

x = torch.randn((1024, 1024), device='cuda', dtype=torch.float16)
y = torch.randn((1024, 1024), device='cuda', dtype=torch.float16)
c = gemm(x, y)