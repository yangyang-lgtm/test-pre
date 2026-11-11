import torch
import triton
import triton.language as tl
from ..conftest import make_inputs, device, F32_ATOL, F32_RTOL
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)

@triton.jit
def neg_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    c = -a
    tl.store(pc + offset, c, mask=mask)

@triton.jit
def abs_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    c = tl.abs(a)
    tl.store(pc + offset, c, mask=mask)

@triton.jit
def exp_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    c = tl.exp(a)
    tl.store(pc + offset, c, mask=mask)

@triton.jit
def rsqrt_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    c = tl.rsqrt(a)
    tl.store(pc + offset, c, mask=mask)

@triton.jit
def log_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    c = tl.log(a)
    tl.store(pc + offset, c, mask=mask)

@triton.jit
def sqrt_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    c = tl.sqrt(a)
    tl.store(pc + offset, c, mask=mask)

@triton.jit
def sin_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    c = tl.sin(a)
    tl.store(pc + offset, c, mask=mask)

@triton.jit
def cos_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    c = tl.cos(a)
    tl.store(pc + offset, c, mask=mask)

def neg(x: torch.Tensor):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    neg_kernel[grid](x, out, num, BLOCK=256)
    return out

def abs_fn(x: torch.Tensor):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    abs_kernel[grid](x, out, num, BLOCK=256)
    return out

def exp_fn(x: torch.Tensor):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    exp_kernel[grid](x, out, num, BLOCK=256)
    return out

def rsqrt_fn(x: torch.Tensor):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    rsqrt_kernel[grid](x, out, num, BLOCK=256)
    return out

def log_fn(x: torch.Tensor):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    log_kernel[grid](x, out, num, BLOCK=256)
    return out

def sqrt_fn(x: torch.Tensor):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    sqrt_kernel[grid](x, out, num, BLOCK=256)
    return out

def sin_fn(x: torch.Tensor):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    sin_kernel[grid](x, out, num, BLOCK=256)
    return out

def cos_fn(x: torch.Tensor):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    cos_kernel[grid](x, out, num, BLOCK=256)
    return out

class TestNegCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_neg(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_z = torch.neg(cpu_x)
        private_z = neg(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestAbsCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_abs(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_z = torch.abs(cpu_x)
        private_z = abs_fn(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestExpCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32])
    def test_exp(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_z = torch.exp(cpu_x)
        private_z = exp_fn(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestExpAccuracyCase(TestCase):
    def test_exp_dense(self):
        n = 1 << 19
        cpu_x = torch.linspace(-103.97, 88.7228, n, dtype=torch.float64).to(torch.float32)
        specials = torch.tensor([0.0, 1.0, -1.0, -86.9850006, -87.0, -90.0, -95.0, 1e-30, -1e-30,
                                 5e-45, -5e-45, 1e-40, -1e-40, 1.5e-38, -1.5e-38, 5.96e-8, -5.96e-8,
                                 -103.97207641601562, -103.97208404541016],
                                dtype=torch.float32)
        cpu_x = torch.cat([cpu_x, specials])
        private_z = exp_fn(cpu_x.to(device)).cpu()
        cpu_z = torch.exp(cpu_x)
        self.assertEqual(cpu_z, private_z, atol=F32_ATOL, rtol=F32_RTOL)

    def test_exp_range_edges(self):
        over = torch.tensor([88.7228394, 88.8, 100.0], dtype=torch.float32)
        under = torch.tensor([-103.98, -150.0, -200.0, -1000.0], dtype=torch.float32)
        special = torch.tensor([float("nan"), float("inf"), float("-inf")], dtype=torch.float32)
        cpu_x = torch.cat([over, under, special])
        private_z = exp_fn(cpu_x.to(device)).cpu()
        cpu_z = torch.exp(cpu_x)
        self.assertEqual(cpu_z, private_z, atol=0, rtol=0, equal_nan=True)

class TestRsqrtCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32])
    def test_rsqrt(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_x = torch.abs(cpu_x) + 0.1
        private_x = torch.abs(private_x) + 0.1
        cpu_z = torch.rsqrt(cpu_x)
        private_z = rsqrt_fn(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestRsqrtAccuracyCase(TestCase):
    def test_rsqrt_dense(self):
        n = 1 << 18
        cpu_x = torch.exp2(torch.linspace(-126.0, 127.0, n, dtype=torch.float64)).to(torch.float32)
        specials = torch.tensor([0.0, 1.0, 4.0, 0.25, 2.0, 1e-38, 5.9e-39, 3.4028235e38, float("inf")],
                                dtype=torch.float32)
        cpu_x = torch.cat([cpu_x, specials])
        private_z = rsqrt_fn(cpu_x.to(device)).cpu()
        cpu_z = torch.rsqrt(cpu_x)
        self.assertEqual(cpu_z, private_z, atol=F32_ATOL, rtol=F32_RTOL)

    def test_rsqrt_edges(self):
        cpu_x = torch.tensor([0.0, float("inf"), -1.0, -0.0, float("nan")], dtype=torch.float32)
        private_z = rsqrt_fn(cpu_x.to(device)).cpu()
        cpu_z = torch.rsqrt(cpu_x)
        self.assertEqual(cpu_z, private_z, atol=0, rtol=0, equal_nan=True)

class TestLogCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32])
    def test_log(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_x = torch.abs(cpu_x) + 0.1
        private_x = torch.abs(private_x) + 0.1
        cpu_z = torch.log(cpu_x)
        private_z = log_fn(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestSqrtCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32])
    def test_sqrt(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_x = torch.abs(cpu_x) + 0.1
        private_x = torch.abs(private_x) + 0.1
        cpu_z = torch.sqrt(cpu_x)
        private_z = sqrt_fn(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestSinCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32])
    def test_sin(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_z = torch.sin(cpu_x)
        private_z = sin_fn(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestCosCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32])
    def test_cos(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_z = torch.cos(cpu_x)
        private_z = cos_fn(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

instantiate_parametrized_tests(TestNegCase)
instantiate_parametrized_tests(TestAbsCase)
instantiate_parametrized_tests(TestExpCase)
instantiate_parametrized_tests(TestRsqrtCase)
instantiate_parametrized_tests(TestLogCase)
instantiate_parametrized_tests(TestSqrtCase)
instantiate_parametrized_tests(TestSinCase)
instantiate_parametrized_tests(TestCosCase)
