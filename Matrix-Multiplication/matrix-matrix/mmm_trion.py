import torch
import triton
import triton.language as tl
import time


"""
A = [m, k]
B = [k, n]
C = [m, n]

C = A * B

"""


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]




def get_autotune_config():
    return get_cuda_autotune_config()



# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune( ## 首次调用时会搜索上面的配置找到最优的，然后后面调用的时候就不会再搜索了，依据是key是否改变
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)

@triton.jit
def mm_kerenl(
    a_ptr, b_ptr, c_ptr,

    M, N, K,

    stride_am, stride_ak, ## 表示每个维度的地址偏移
    stride_bk, stride_bn,
    stride_cm, stride_cn,

    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    pid = tl.program_id(0) ## 还是一维的grid

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M) ## 行的block数量
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n ## 当前block的行
    pid_n = pid % num_pid_n

    ## 计算A B C的对应的block的起始位置
    offset_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M ## 那些行
    offset_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N ## 那些列
    offset_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offset_am[:, None] * stride_am + offset_k[None, :]*stride_ak) ## 这里的None是为了广播，得到一个二维的地址
    b_ptrs = b_ptr + (offset_k[:, None] * stride_bk + offset_bn[None, :]*stride_bn) ## b_ptr+一个偏移矩阵，该语法在pytorch是不支持的，但是在triton是支持的

    accumulators = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0,  tl.cdiv(K, BLOCK_SIZE_K)): ## 每次处理一个block的k

        a = tl.load(a_ptrs, mask=offset_k[None, :] < K-k*BLOCK_SIZE_K, other=0) ## 这里的mask是为了处理边界情况，即最后一个block的时候，可能不够BLOCK_SIZE_K
        b = tl.load(b_ptrs, mask=offset_k[:, None] < K-k*BLOCK_SIZE_K, other=0)

        accumulators = tl.dot(a, b, accumulators) ## 

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # if ACTIVATION:
    #     accumulators =    
    
    offset_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    c_ptrs = c_ptr + offset_cm[:, None] * stride_cm + offset_cn[None, :] * stride_cn
    c_mask = (offset_cm[:, None] < M) & (offset_cn[None, :] < N)
    c = accumulators.to(tl.float16)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a,b):
    assert a.is_cuda and b.is_cuda
    assert a.shape[1] == b.shape[0]
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),) ## 一维的grid
    mm_kerenl[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), ACTIVATION="") ## block size is auto tuned
    return c


def check_correct():
    a = torch.randn((1024, 1024), device='cuda', dtype=torch.float16)
    b = torch.randn((1024, 1024), device='cuda', dtype=torch.float16)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)

    print(f"error: {torch.abs(torch_output-triton_output).max()}")


def benchmark():
    a = torch.randn((1024, 1024), device='cuda', dtype=torch.float16)
    b = torch.randn((1024, 1024), device='cuda', dtype=torch.float16)

    ## warm up
    for _ in range(10):
        output = matmul(a, b)

    iter_num = 100
    # torch cuda time
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(iter_num):
        output = torch.matmul(a, b)
    torch.cuda.synchronize()
    t2 = time.time()

    print(f'Cuda time: {(t2-t1)/iter_num}')

    # triton cuda time
    for _ in range(10):
        output = matmul(a, b)

    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(iter_num):
        output = matmul(a, b)
    torch.cuda.synchronize()
    t2 = time.time()

    print(f'Triton time: {(t2-t1)/iter_num}')

if __name__ == "__main__":
    check_correct()
    # benchmark()
    """
    error: 0.0
    Cuda time: 1.818418502807617e-05
    Triton time: 0.00014574766159057616
    """