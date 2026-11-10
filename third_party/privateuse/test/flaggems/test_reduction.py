import torch
import triton
import triton.language as tl
from ..conftest import make_inputs, F32_ATOL, F32_RTOL, ACC_ATOL, ACC_RTOL, SUM_ATOL, SUM_RTOL
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)


@triton.jit
def sum_kernel(pa, pc, R, C: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, C)
    m = rows < R
    a = tl.load(pa + rows[:, None] * C + cols[None, :], mask=m[:, None], other=0.0)
    tl.store(pc + rows, tl.sum(a, 1), mask=m)


@triton.jit
def max_kernel(pa, pc, R, C: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, C)
    m = rows < R
    a = tl.load(pa + rows[:, None] * C + cols[None, :], mask=m[:, None], other=float("-inf"))
    tl.store(pc + rows, tl.max(a, 1), mask=m)


@triton.jit
def min_kernel(pa, pc, R, C: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, C)
    m = rows < R
    a = tl.load(pa + rows[:, None] * C + cols[None, :], mask=m[:, None], other=float("inf"))
    tl.store(pc + rows, tl.min(a, 1), mask=m)


@triton.jit
def mean_kernel(pa, pc, R, C: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, C)
    m = rows < R
    a = tl.load(pa + rows[:, None] * C + cols[None, :], mask=m[:, None], other=0.0)
    tl.store(pc + rows, tl.sum(a, 1) / C, mask=m)


@triton.jit
def mul_combine(x, y):
    return x * y


@triton.jit
def prod_kernel(pa, pc, R, C: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, C)
    m = rows < R
    a = tl.load(pa + rows[:, None] * C + cols[None, :], mask=m[:, None], other=1.0)
    tl.store(pc + rows, tl.reduce(a, 1, mul_combine), mask=m)


@triton.jit
def argmax_kernel(pa, pc, R, C: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, C)
    m = rows < R
    a = tl.load(pa + rows[:, None] * C + cols[None, :], mask=m[:, None], other=float("-inf"))
    tl.store(pc + rows, tl.argmax(a, 1), mask=m)


@triton.jit
def argmin_kernel(pa, pc, R, C: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, C)
    m = rows < R
    a = tl.load(pa + rows[:, None] * C + cols[None, :], mask=m[:, None], other=float("inf"))
    tl.store(pc + rows, tl.argmin(a, 1), mask=m)


@triton.jit
def var_kernel(pa, pc, R, C: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, C)
    m = rows < R
    a = tl.load(pa + rows[:, None] * C + cols[None, :], mask=m[:, None], other=0.0)
    mean = tl.sum(a, 1) / C
    d = a - mean[:, None]
    tl.store(pc + rows, tl.sum(d * d, 1) / C, mask=m)


def _row_reduce(x, kernel, out_dtype=torch.float32, rb=16):
    R, C = x.shape
    out = torch.empty(R, dtype=out_dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(R, meta["RB"]), )
    kernel[grid](x, out, R, C=C, RB=rb)
    return out


def sum_(x):
    return _row_reduce(x, sum_kernel)


def max_(x):
    return _row_reduce(x, max_kernel)


def min_(x):
    return _row_reduce(x, min_kernel)


def mean_(x):
    return _row_reduce(x, mean_kernel)


def prod(x):
    return _row_reduce(x, prod_kernel)


def argmax(x):
    return _row_reduce(x, argmax_kernel, torch.int32)


def argmin(x):
    return _row_reduce(x, argmin_kernel, torch.int32)


def var(x):
    return _row_reduce(x, var_kernel)


class TestSumCase(TestCase):
    @parametrize("shape", [(32, 64), (16, 128), (37, 64)])
    def test_sum(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        self.assertEqual(torch.sum(cpu_x, dim=1), sum_(private_x).cpu(), atol=SUM_ATOL, rtol=SUM_RTOL)


class TestMaxCase(TestCase):
    @parametrize("shape", [(32, 64), (16, 128), (37, 64)])
    def test_max(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        self.assertEqual(torch.max(cpu_x, dim=1).values, max_(private_x).cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestMinCase(TestCase):
    @parametrize("shape", [(32, 64), (16, 128), (37, 64)])
    def test_min(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        self.assertEqual(torch.min(cpu_x, dim=1).values, min_(private_x).cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestMeanCase(TestCase):
    @parametrize("shape", [(32, 64), (16, 128), (37, 64)])
    def test_mean(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        self.assertEqual(torch.mean(cpu_x, dim=1), mean_(private_x).cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestProdCase(TestCase):
    @parametrize("shape", [(32, 64), (16, 128), (37, 64)])
    def test_prod(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_x = torch.tanh(cpu_x) * 0.5 + 1.0
        private_x = torch.tanh(private_x) * 0.5 + 1.0
        self.assertEqual(torch.prod(cpu_x, dim=1), prod(private_x).cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestArgmaxCase(TestCase):
    @parametrize("shape", [(32, 64), (16, 128), (37, 64)])
    def test_argmax(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_z = torch.argmax(cpu_x, dim=1).to(torch.int32)
        self.assertEqual(cpu_z, argmax(private_x).cpu())


class TestArgminCase(TestCase):
    @parametrize("shape", [(32, 64), (16, 128), (37, 64)])
    def test_argmin(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_z = torch.argmin(cpu_x, dim=1).to(torch.int32)
        self.assertEqual(cpu_z, argmin(private_x).cpu())


class TestVarCase(TestCase):
    @parametrize("shape", [(32, 64), (16, 128), (37, 64)])
    def test_var(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_z = torch.var(cpu_x, dim=1, unbiased=False)
        self.assertEqual(cpu_z, var(private_x).cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


instantiate_parametrized_tests(TestSumCase)
instantiate_parametrized_tests(TestMaxCase)
instantiate_parametrized_tests(TestMinCase)
instantiate_parametrized_tests(TestMeanCase)
instantiate_parametrized_tests(TestProdCase)
instantiate_parametrized_tests(TestArgmaxCase)
instantiate_parametrized_tests(TestArgminCase)
instantiate_parametrized_tests(TestVarCase)
