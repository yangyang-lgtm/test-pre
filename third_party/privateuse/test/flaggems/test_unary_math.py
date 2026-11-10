import torch
import triton
import triton.language as tl
from ..conftest import make_inputs, F32_ATOL, F32_RTOL, ACC_ATOL, ACC_RTOL
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)


@triton.jit
def exp2_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    tl.store(pc + offset, tl.exp2(a), mask=mask)


@triton.jit
def log2_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    tl.store(pc + offset, tl.log2(a), mask=mask)


@triton.jit
def log10_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    tl.store(pc + offset, tl.log(a) * 0.4342944819032518, mask=mask)


@triton.jit
def erf_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    tl.store(pc + offset, tl.erf(a), mask=mask)


@triton.jit
def floor_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    tl.store(pc + offset, tl.floor(a), mask=mask)


@triton.jit
def ceil_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    tl.store(pc + offset, tl.ceil(a), mask=mask)


@triton.jit
def trunc_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    tl.store(pc + offset, a.to(tl.int32).to(tl.float32), mask=mask)


@triton.jit
def sign_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    c = tl.where(a > 0.0, 1.0, tl.where(a < 0.0, -1.0, 0.0))
    tl.store(pc + offset, c, mask=mask)


@triton.jit
def frac_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    tl.store(pc + offset, a - a.to(tl.int32).to(tl.float32), mask=mask)


@triton.jit
def reciprocal_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    tl.store(pc + offset, 1.0 / a, mask=mask)


def _unary(x, kernel):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta["BLOCK"]), )
    kernel[grid](x, out, num, BLOCK=256)
    return out


def exp2(x):
    return _unary(x, exp2_kernel)


def log2(x):
    return _unary(x, log2_kernel)


def log10(x):
    return _unary(x, log10_kernel)


def erf(x):
    return _unary(x, erf_kernel)


def floor(x):
    return _unary(x, floor_kernel)


def ceil(x):
    return _unary(x, ceil_kernel)


def trunc(x):
    return _unary(x, trunc_kernel)


def sign(x):
    return _unary(x, sign_kernel)


def frac(x):
    return _unary(x, frac_kernel)


def reciprocal(x):
    return _unary(x, reciprocal_kernel)




class TestExp2Case(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_exp2(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        self.assertEqual(torch.exp2(cpu_x), exp2(private_x).cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestLog2Case(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_log2(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_x = torch.abs(cpu_x) + 0.1
        private_x = torch.abs(private_x) + 0.1
        self.assertEqual(torch.log2(cpu_x), log2(private_x).cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestLog10Case(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_log10(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_x = torch.abs(cpu_x) + 0.1
        private_x = torch.abs(private_x) + 0.1
        self.assertEqual(torch.log10(cpu_x), log10(private_x).cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestErfCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_erf(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        self.assertEqual(torch.erf(cpu_x), erf(private_x).cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestFloorCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_floor(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        self.assertEqual(torch.floor(cpu_x), floor(private_x).cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestCeilCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_ceil(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        self.assertEqual(torch.ceil(cpu_x), ceil(private_x).cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestTruncCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_trunc(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        self.assertEqual(torch.trunc(cpu_x), trunc(private_x).cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestSignCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_sign(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        self.assertEqual(torch.sign(cpu_x), sign(private_x).cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestFracCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_frac(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        self.assertEqual(torch.frac(cpu_x), frac(private_x).cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestReciprocalCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_reciprocal(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_x = torch.where(cpu_x.abs() < 0.1, cpu_x + 0.5, cpu_x)
        private_x = torch.where(private_x.abs() < 0.1, private_x + 0.5, private_x)
        self.assertEqual(torch.reciprocal(cpu_x), reciprocal(private_x).cpu(), atol=F32_ATOL, rtol=F32_RTOL)


instantiate_parametrized_tests(TestExp2Case)
instantiate_parametrized_tests(TestLog2Case)
instantiate_parametrized_tests(TestLog10Case)
instantiate_parametrized_tests(TestErfCase)
instantiate_parametrized_tests(TestFloorCase)
instantiate_parametrized_tests(TestCeilCase)
instantiate_parametrized_tests(TestTruncCase)
instantiate_parametrized_tests(TestSignCase)
instantiate_parametrized_tests(TestFracCase)
instantiate_parametrized_tests(TestReciprocalCase)
