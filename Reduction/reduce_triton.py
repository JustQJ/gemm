import torch
import triton
import triton.language as tl



@triton.jit
def reduce_kerenl(
    input_ptr,
    output_ptr,
    n,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    offsets = pid*BLOCK_SIZE+tl.arange(0, BLOCK_SIZE)

    ptrs = input_ptr+offsets

    input_data = tl.load(ptrs, mask=offsets<n, other=0)

    val = tl.sum(input_data)

    tl.store(output_ptr+pid, val)



def reduce_sum(x):
    n = x.shape[0]

    BLOCK_SIZE = 1024

    assert BLOCK_SIZE*BLOCK_SIZE>=n

    block_num = triton.cdiv(n, BLOCK_SIZE)
    y = torch.zeros((block_num, ), dtype=x.dtype, device=x.device)
    reduce_kerenl[(block_num,1,1)](x, y, n, BLOCK_SIZE=BLOCK_SIZE)

    ## reduce in one block

    final_y = torch.tensor(0, dtype=x.dtype).to(x.device)

    reduce_kerenl[(1,1,1)](y, final_y, block_num, BLOCK_SIZE=BLOCK_SIZE)

    return final_y


inputs = torch.randn((512*1024,)).cuda()

y1 = torch.sum(inputs)

y2 = reduce_sum(inputs)

print(y1, y2)
    







    
    