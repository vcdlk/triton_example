
import torch
import triton
import triton.language as tl


@triton.jit
def copy(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets,mask = mask) 
    tl.store(b_ptr + offsets, a,mask = mask)

# a, b are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    n_elements = a.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    copy[grid](a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)