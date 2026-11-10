import torch
import triton
import triton.language as tl
from ..conftest import make_inputs
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)


@triton.jit
def eq_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    tl.store(pc + offset, a == b, mask=mask)


@triton.jit
def ne_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    tl.store(pc + offset, a != b, mask=mask)


@triton.jit
def lt_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    tl.store(pc + offset, a < b, mask=mask)


@triton.jit
def ge_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    tl.store(pc + offset, a >= b, mask=mask)


@triton.jit
def logical_and_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    c = (a != 0.0) & (b != 0.0)
    tl.store(pc + offset, c, mask=mask)


@triton.jit
def logical_or_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    c = (a != 0.0) | (b != 0.0)
    tl.store(pc + offset, c, mask=mask)


@triton.jit
def logical_not_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    tl.store(pc + offset, a == 0.0, mask=mask)


@triton.jit
def isnan_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    tl.store(pc + offset, a != a, mask=mask)


def _cmp2(x, y, kernel):
    out = torch.empty(x.shape, dtype=torch.bool, device=x.device)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta["BLOCK"]), )
    kernel[grid](x, y, out, num, BLOCK=256)
    return out


def _cmp1(x, kernel):
    out = torch.empty(x.shape, dtype=torch.bool, device=x.device)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta["BLOCK"]), )
    kernel[grid](x, out, num, BLOCK=256)
    return out


def eq(x, y):
    return _cmp2(x, y, eq_kernel)


def ne(x, y):
    return _cmp2(x, y, ne_kernel)


def lt(x, y):
    return _cmp2(x, y, lt_kernel)


def ge(x, y):
    return _cmp2(x, y, ge_kernel)


def logical_and(x, y):
    return _cmp2(x, y, logical_and_kernel)


def logical_or(x, y):
    return _cmp2(x, y, logical_or_kernel)


def logical_not(x):
    return _cmp1(x, logical_not_kernel)


def isnan(x):
    return _cmp1(x, isnan_kernel)


class TestEqCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_eq(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs(shape, torch.float32)
        cpu_y = torch.where(cpu_x > 0.5, cpu_x, cpu_y)
        private_y = torch.where(private_x > 0.5, private_x, private_y)
        self.assertEqual(torch.eq(cpu_x, cpu_y), eq(private_x, private_y).cpu())


class TestNeCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_ne(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs(shape, torch.float32)
        cpu_y = torch.where(cpu_x > 0.5, cpu_x, cpu_y)
        private_y = torch.where(private_x > 0.5, private_x, private_y)
        self.assertEqual(torch.ne(cpu_x, cpu_y), ne(private_x, private_y).cpu())


class TestLtCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_lt(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs(shape, torch.float32)
        self.assertEqual(torch.lt(cpu_x, cpu_y), lt(private_x, private_y).cpu())


class TestGeCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_ge(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs(shape, torch.float32)
        self.assertEqual(torch.ge(cpu_x, cpu_y), ge(private_x, private_y).cpu())


class TestLogicalAndCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_logical_and(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs(shape, torch.float32)
        self.assertEqual(torch.logical_and(cpu_x, cpu_y), logical_and(private_x, private_y).cpu())


class TestLogicalOrCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_logical_or(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs(shape, torch.float32)
        self.assertEqual(torch.logical_or(cpu_x, cpu_y), logical_or(private_x, private_y).cpu())


class TestLogicalNotCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_logical_not(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        self.assertEqual(torch.logical_not(cpu_x), logical_not(private_x).cpu())


class TestIsnanCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_isnan(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_x = cpu_x.clone()
        private_x = private_x.clone()
        cpu_x.view(-1)[0] = float("nan")
        private_x.view(-1)[0] = float("nan")
        self.assertEqual(torch.isnan(cpu_x), isnan(private_x).cpu())


instantiate_parametrized_tests(TestEqCase)
instantiate_parametrized_tests(TestNeCase)
instantiate_parametrized_tests(TestLtCase)
instantiate_parametrized_tests(TestGeCase)
instantiate_parametrized_tests(TestLogicalAndCase)
instantiate_parametrized_tests(TestLogicalOrCase)
instantiate_parametrized_tests(TestLogicalNotCase)
instantiate_parametrized_tests(TestIsnanCase)
