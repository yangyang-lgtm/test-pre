import torch
import triton
import triton.language as tl
from ..conftest import make_inputs, F32_ATOL, F32_RTOL
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)

ROWS = 64
BLOCK = 64


@triton.jit
def sort_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    tl.store(pc + offset, tl.sort(a, 0), mask=mask)


@triton.jit
def sort_desc_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    tl.store(pc + offset, tl.sort(a, 0, descending=True), mask=mask)


@triton.jit
def median_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    index = tl.arange(0, BLOCK)
    v = tl.sort(a, 0)
    mid = tl.sum(tl.where(index == (BLOCK - 1) // 2, v, 0.0))
    tl.store(pc + pid, mid)


def _grid(x):
    return (triton.cdiv(x.numel(), BLOCK), )


def sort(x):
    out = torch.empty_like(x)
    sort_kernel[_grid(x)](x, out, x.numel(), BLOCK=BLOCK)
    return out


def topk(x, k):
    out = torch.empty_like(x)
    sort_desc_kernel[_grid(x)](x, out, x.numel(), BLOCK=BLOCK)
    return out[:, :k]


def median(x):
    out = torch.empty(x.shape[0], dtype=x.dtype, device=x.device)
    median_kernel[_grid(x)](x, out, x.numel(), BLOCK=BLOCK)
    return out


class TestSortCase(TestCase):
    @parametrize("dtype", [torch.float32])
    def test_sort(self, dtype):
        cpu_x, private_x = make_inputs((ROWS, BLOCK), dtype)
        self.assertEqual(torch.sort(cpu_x, dim=1).values, sort(private_x).cpu())


class TestTopkCase(TestCase):
    @parametrize("k", [8, 16])
    def test_topk(self, k):
        cpu_x, private_x = make_inputs((ROWS, BLOCK), torch.float32)
        self.assertEqual(torch.topk(cpu_x, k, dim=1).values, topk(private_x, k).cpu(),
                         atol=F32_ATOL, rtol=F32_RTOL)


class TestMedianCase(TestCase):
    def test_median(self):
        cpu_x, private_x = make_inputs((ROWS, BLOCK), torch.float32)
        self.assertEqual(torch.median(cpu_x, dim=1).values, median(private_x).cpu(),
                         atol=F32_ATOL, rtol=F32_RTOL)


instantiate_parametrized_tests(TestSortCase)
instantiate_parametrized_tests(TestTopkCase)
instantiate_parametrized_tests(TestMedianCase)
