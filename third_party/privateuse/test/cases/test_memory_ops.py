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
def copy_kernel(pa, pb, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    tl.store(pb + offset, a, mask=mask)

@triton.jit
def fill_kernel(pc, val, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    tl.store(pc + offset, val, mask=mask)

@triton.jit
def add_scalar_kernel(pa, scalar, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    c = a + scalar
    tl.store(pc + offset, c, mask=mask)

@triton.jit
def mul_scalar_kernel(pa, scalar, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    c = a * scalar
    tl.store(pc + offset, c, mask=mask)

def copy_fn(x: torch.Tensor):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    copy_kernel[grid](x, out, num, BLOCK=256)
    return out

def fill_fn(shape, dtype, val, dev):
    out = torch.empty(shape, dtype=dtype, device=dev)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    fill_kernel[grid](out, val, num, BLOCK=256)
    return out

def add_scalar(x: torch.Tensor, scalar):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    add_scalar_kernel[grid](x, scalar, out, num, BLOCK=256)
    return out

def mul_scalar(x: torch.Tensor, scalar):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    mul_scalar_kernel[grid](x, scalar, out, num, BLOCK=256)
    return out

class TestCopyCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_copy(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_z = cpu_x.clone()
        private_z = copy_fn(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestFillCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_fill(self, shape, dtype):
        val = 42
        cpu_z = torch.full(shape, val, dtype=dtype)
        private_z = fill_fn(shape, dtype, val, device)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestAddScalarCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_add_scalar(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        scalar = 10
        cpu_z = torch.add(cpu_x, scalar)
        private_z = add_scalar(private_x, scalar)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestMulScalarCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_mul_scalar(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        scalar = 3
        cpu_z = torch.mul(cpu_x, scalar)
        private_z = mul_scalar(private_x, scalar)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

instantiate_parametrized_tests(TestCopyCase)
instantiate_parametrized_tests(TestFillCase)
instantiate_parametrized_tests(TestAddScalarCase)
instantiate_parametrized_tests(TestMulScalarCase)
