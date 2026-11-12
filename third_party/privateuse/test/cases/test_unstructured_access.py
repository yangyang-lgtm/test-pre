import torch
import triton
import triton.language as tl
from ..conftest import F32_ATOL, F32_RTOL, make_inputs, device
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)

@triton.jit
def gather_kernel(psrc, pidx, pout, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    i = tl.load(pidx + offs)
    v = tl.load(psrc + i)
    tl.store(pout + offs, v)

@triton.jit
def scatter_kernel(psrc, pidx, pout, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    i = tl.load(pidx + offs)
    v = tl.load(psrc + offs)
    tl.store(pout + i, v)

@triton.jit
def gather_rows_kernel(psrc, pidx, pout, BM: tl.constexpr, BN: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    i = tl.load(pidx + rn)
    v = tl.load(tl.broadcast_to((psrc + i)[None, :], (BM, BN)))
    tl.store(pout + rm[:, None] * BN + rn[None, :], v)

def make_permutation(n: int):
    cpu_idx = torch.randperm(n, dtype=torch.int32)
    return cpu_idx, cpu_idx.to(device)

def gather_fn(x: torch.Tensor, idx: torch.Tensor):
    out = torch.empty_like(x)
    grid = lambda meta: (x.numel() // meta['BLOCK'], )
    gather_kernel[grid](x, idx, out, BLOCK=256)
    return out

def scatter_fn(x: torch.Tensor, idx: torch.Tensor):
    out = torch.empty_like(x)
    grid = lambda meta: (x.numel() // meta['BLOCK'], )
    scatter_kernel[grid](x, idx, out, BLOCK=256)
    return out

def gather_rows_fn(x: torch.Tensor, idx: torch.Tensor, bm: int, bn: int):
    out = torch.empty((bm, bn), dtype=x.dtype, device=x.device)
    gather_rows_kernel[(1, )](x, idx, out, BM=bm, BN=bn)
    return out

class TestGatherCase(TestCase):
    @parametrize("size", [256, 1024])
    def test_gather(self, size):
        cpu_x, private_x = make_inputs((size, ), torch.float32)
        cpu_idx, private_idx = make_permutation(size)
        cpu_z = cpu_x[cpu_idx.long()]
        private_z = gather_fn(private_x, private_idx)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestScatterCase(TestCase):
    @parametrize("size", [256, 1024])
    def test_scatter(self, size):
        cpu_x, private_x = make_inputs((size, ), torch.float32)
        cpu_idx, private_idx = make_permutation(size)
        cpu_z = torch.empty_like(cpu_x)
        cpu_z[cpu_idx.long()] = cpu_x
        private_z = scatter_fn(private_x, private_idx)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestGatherRowsCase(TestCase):
    @parametrize("shape", [(32, 32), (16, 64)])
    def test_gather_rows(self, shape):
        bm, bn = shape
        cpu_x, private_x = make_inputs((bn, ), torch.float32)
        cpu_idx, private_idx = make_permutation(bn)
        cpu_z = cpu_x[cpu_idx.long()].expand(bm, bn)
        private_z = gather_rows_fn(private_x, private_idx, bm, bn)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

instantiate_parametrized_tests(TestGatherCase)
instantiate_parametrized_tests(TestScatterCase)
instantiate_parametrized_tests(TestGatherRowsCase)
