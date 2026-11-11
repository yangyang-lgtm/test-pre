import torch
import triton
import triton.language as tl
from ..conftest import F32_ATOL, F32_RTOL, make_inputs, device, MATH_ATOL, MATH_RTOL
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)

@triton.jit
def clamp_kernel(pa, pc, min_val, max_val, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    c = tl.minimum(tl.maximum(a, min_val), max_val)
    tl.store(pc + offset, c, mask=mask)

@triton.jit
def where_kernel(pcond, pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    cond = tl.load(pcond + offset, mask=mask)
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    c = tl.where(cond, a, b)
    tl.store(pc + offset, c, mask=mask)

@triton.jit
def pow_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    c = tl.exp(b * tl.log(a))
    tl.store(pc + offset, c, mask=mask)

@triton.jit
def maximum_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    c = tl.maximum(a, b)
    tl.store(pc + offset, c, mask=mask)

@triton.jit
def minimum_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    c = tl.minimum(a, b)
    tl.store(pc + offset, c, mask=mask)

def clamp_fn(x: torch.Tensor, min_val: float, max_val: float):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    clamp_kernel[grid](x, out, min_val, max_val, num, BLOCK=256)
    return out

def where_fn(cond: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    where_kernel[grid](cond, x, y, out, num, BLOCK=256)
    return out

def pow_fn(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    pow_kernel[grid](x, y, out, num, BLOCK=256)
    return out

def maximum_fn(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    maximum_kernel[grid](x, y, out, num, BLOCK=256)
    return out

def minimum_fn(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    minimum_kernel[grid](x, y, out, num, BLOCK=256)
    return out

class TestClampCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_clamp(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        if dtype == torch.int32:
            min_val, max_val = -50, 50
        else:
            min_val, max_val = -50.0, 50.0
        cpu_z = torch.clamp(cpu_x, min_val, max_val)
        private_z = clamp_fn(private_x, min_val, max_val)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestWhereCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32])
    def test_where(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_y, private_y = make_inputs(shape, dtype)
        cpu_cond = torch.randint(0, 2, shape, dtype=torch.bool)
        private_cond = cpu_cond.to(device)
        cpu_z = torch.where(cpu_cond, cpu_x, cpu_y)
        private_z = where_fn(private_cond, private_x, private_y)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestPowCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32])
    def test_pow(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_x = torch.abs(cpu_x) + 0.1
        private_x = torch.abs(private_x) + 0.1
        cpu_y = torch.randint(1, 4, shape, dtype=torch.int32).to(dtype)
        private_y = cpu_y.to(device)
        cpu_z = torch.pow(cpu_x, cpu_y)
        private_z = pow_fn(private_x, private_y)
        self.assertEqual(cpu_z, private_z.cpu(), atol=MATH_ATOL, rtol=MATH_RTOL)

class TestMaximumCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_maximum(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_y, private_y = make_inputs(shape, dtype)
        cpu_z = torch.maximum(cpu_x, cpu_y)
        private_z = maximum_fn(private_x, private_y)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestMinimumCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_minimum(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_y, private_y = make_inputs(shape, dtype)
        cpu_z = torch.minimum(cpu_x, cpu_y)
        private_z = minimum_fn(private_x, private_y)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

instantiate_parametrized_tests(TestClampCase)
instantiate_parametrized_tests(TestWhereCase)
instantiate_parametrized_tests(TestPowCase)
instantiate_parametrized_tests(TestMaximumCase)
instantiate_parametrized_tests(TestMinimumCase)
