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
def maximum_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    tl.store(pc + offset, tl.maximum(a, b), mask=mask)


@triton.jit
def minimum_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    tl.store(pc + offset, tl.minimum(a, b), mask=mask)


@triton.jit
def fmod_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    q = (a / b).to(tl.int32).to(tl.float32)
    tl.store(pc + offset, a - q * b, mask=mask)


@triton.jit
def remainder_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    tl.store(pc + offset, a - tl.floor(a / b) * b, mask=mask)


@triton.jit
def pow_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    tl.store(pc + offset, tl.exp(b * tl.log(a)), mask=mask)


@triton.jit
def clamp_kernel(pa, pc, lo, hi, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    tl.store(pc + offset, tl.minimum(tl.maximum(a, lo), hi), mask=mask)


@triton.jit
def where_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    tl.store(pc + offset, tl.where(a > 0.0, a, b), mask=mask)


@triton.jit
def lerp_kernel(pa, pb, pc, w, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    tl.store(pc + offset, a + w * (b - a), mask=mask)


@triton.jit
def addcmul_kernel(pa, pb, pc_in, pout, value, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    c = tl.load(pc_in + offset, mask=mask)
    tl.store(pout + offset, c + value * a * b, mask=mask)


def _binary(x, y, kernel):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta["BLOCK"]), )
    kernel[grid](x, y, out, num, BLOCK=256)
    return out


def maximum(x, y):
    return _binary(x, y, maximum_kernel)


def minimum(x, y):
    return _binary(x, y, minimum_kernel)


def fmod(x, y):
    return _binary(x, y, fmod_kernel)


def remainder(x, y):
    return _binary(x, y, remainder_kernel)


def pow_(x, y):
    return _binary(x, y, pow_kernel)


def where(x, y):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta["BLOCK"]), )
    where_kernel[grid](x, y, out, num, BLOCK=256)
    return out


def lerp(x, y, w=0.25):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta["BLOCK"]), )
    lerp_kernel[grid](x, y, out, w, num, BLOCK=256)
    return out


def addcmul(x, y, z, value=0.5):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta["BLOCK"]), )
    addcmul_kernel[grid](x, y, z, out, value, num, BLOCK=256)
    return out


def clamp(x, lo=-0.5, hi=0.5):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta["BLOCK"]), )
    clamp_kernel[grid](x, out, lo, hi, num, BLOCK=256)
    return out




class TestMaximumCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_maximum(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs(shape, torch.float32)
        self.assertEqual(torch.maximum(cpu_x, cpu_y), maximum(private_x, private_y).cpu(),
                         atol=F32_ATOL, rtol=F32_RTOL)


class TestMinimumCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_minimum(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs(shape, torch.float32)
        self.assertEqual(torch.minimum(cpu_x, cpu_y), minimum(private_x, private_y).cpu(),
                         atol=F32_ATOL, rtol=F32_RTOL)


class TestFmodCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_fmod(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs(shape, torch.float32)
        cpu_y = torch.where(cpu_y.abs() < 0.5, cpu_y + 1.5, cpu_y)
        private_y = torch.where(private_y.abs() < 0.5, private_y + 1.5, private_y)
        self.assertEqual(torch.fmod(cpu_x, cpu_y), fmod(private_x, private_y).cpu(),
                         atol=ACC_ATOL, rtol=ACC_RTOL)


class TestRemainderCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_remainder(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs(shape, torch.float32)
        cpu_y = torch.where(cpu_y.abs() < 0.5, cpu_y + 1.5, cpu_y)
        private_y = torch.where(private_y.abs() < 0.5, private_y + 1.5, private_y)
        self.assertEqual(torch.remainder(cpu_x, cpu_y), remainder(private_x, private_y).cpu(),
                         atol=ACC_ATOL, rtol=ACC_RTOL)


class TestPowCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_pow(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs(shape, torch.float32)
        cpu_x = torch.abs(cpu_x) + 0.5
        private_x = torch.abs(private_x) + 0.5
        cpu_y = torch.remainder(cpu_y, 3.0)
        private_y = torch.remainder(private_y, 3.0)
        self.assertEqual(torch.pow(cpu_x, cpu_y), pow_(private_x, private_y).cpu(),
                         atol=ACC_ATOL, rtol=ACC_RTOL)


class TestClampCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_clamp(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        self.assertEqual(torch.clamp(cpu_x, -0.5, 0.5), clamp(private_x, -0.5, 0.5).cpu(),
                         atol=F32_ATOL, rtol=F32_RTOL)


class TestWhereCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_where(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs(shape, torch.float32)
        cpu_z = torch.where(cpu_x > 0.0, cpu_x, cpu_y)
        self.assertEqual(cpu_z, where(private_x, private_y).cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestLerpCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_lerp(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs(shape, torch.float32)
        self.assertEqual(torch.lerp(cpu_x, cpu_y, 0.25), lerp(private_x, private_y, 0.25).cpu(),
                         atol=F32_ATOL, rtol=F32_RTOL)


class TestAddcmulCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_addcmul(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs(shape, torch.float32)
        cpu_z, private_z = make_inputs(shape, torch.float32)
        cpu_out = torch.addcmul(cpu_z, cpu_x, cpu_y, value=0.5)
        self.assertEqual(cpu_out, addcmul(private_x, private_y, private_z, 0.5).cpu(),
                         atol=ACC_ATOL, rtol=ACC_RTOL)


instantiate_parametrized_tests(TestMaximumCase)
instantiate_parametrized_tests(TestMinimumCase)
instantiate_parametrized_tests(TestFmodCase)
instantiate_parametrized_tests(TestRemainderCase)
instantiate_parametrized_tests(TestPowCase)
instantiate_parametrized_tests(TestClampCase)
instantiate_parametrized_tests(TestWhereCase)
instantiate_parametrized_tests(TestLerpCase)
instantiate_parametrized_tests(TestAddcmulCase)
