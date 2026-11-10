import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from ..conftest import make_inputs, F32_ATOL, F32_RTOL, ACC_ATOL, ACC_RTOL
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)


@triton.jit
def relu_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    c = tl.maximum(a, 0.0)
    tl.store(pc + offset, c, mask=mask)


@triton.jit
def leaky_relu_kernel(pa, pc, slope, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    c = tl.where(a > 0.0, a, a * slope)
    tl.store(pc + offset, c, mask=mask)


@triton.jit
def elu_kernel(pa, pc, alpha, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask).to(tl.float32)
    c = tl.where(a > 0.0, a, alpha * (tl.exp(a) - 1.0))
    tl.store(pc + offset, c, mask=mask)


@triton.jit
def sigmoid_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask).to(tl.float32)
    c = 1.0 / (1.0 + tl.exp(-a))
    tl.store(pc + offset, c, mask=mask)


@triton.jit
def silu_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask).to(tl.float32)
    c = a / (1.0 + tl.exp(-a))
    tl.store(pc + offset, c, mask=mask)


@triton.jit
def gelu_tanh_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask).to(tl.float32)
    x3 = a * a * a
    u = 0.7978845608028654 * (a + 0.044715 * x3)
    t = 2.0 / (1.0 + tl.exp(-2.0 * u)) - 1.0
    c = 0.5 * a * (1.0 + t)
    tl.store(pc + offset, c, mask=mask)


@triton.jit
def gelu_exact_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask).to(tl.float32)
    c = 0.5 * a * (1.0 + tl.erf(a * 0.7071067811865476))
    tl.store(pc + offset, c, mask=mask)


@triton.jit
def softplus_kernel(pa, pc, beta, threshold, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask).to(tl.float32)
    bx = beta * a
    c = tl.where(bx > threshold, a, tl.log(1.0 + tl.exp(bx)) / beta)
    tl.store(pc + offset, c, mask=mask)


def _pointwise(x, kernel, *args):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta["BLOCK"]), )
    kernel[grid](x, out, *args, num, BLOCK=256)
    return out


def relu(x):
    return _pointwise(x, relu_kernel)


def leaky_relu(x, slope=0.01):
    return _pointwise(x, leaky_relu_kernel, slope)


def elu(x, alpha=1.0):
    return _pointwise(x, elu_kernel, alpha)


def sigmoid(x):
    return _pointwise(x, sigmoid_kernel)


def silu(x):
    return _pointwise(x, silu_kernel)


def gelu_tanh(x):
    return _pointwise(x, gelu_tanh_kernel)


def gelu_exact(x):
    return _pointwise(x, gelu_exact_kernel)


def softplus(x, beta=1.0, threshold=20.0):
    return _pointwise(x, softplus_kernel, beta, threshold)




class TestReluCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_relu(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        self.assertEqual(F.relu(cpu_x), relu(private_x).cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestLeakyReluCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_leaky_relu(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_z = F.leaky_relu(cpu_x, 0.01)
        private_z = leaky_relu(private_x, 0.01)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestEluCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_elu(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        self.assertEqual(F.elu(cpu_x, 1.0), elu(private_x, 1.0).cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestSigmoidCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_sigmoid(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        self.assertEqual(torch.sigmoid(cpu_x), sigmoid(private_x).cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestSiluCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_silu(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        self.assertEqual(F.silu(cpu_x), silu(private_x).cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestGeluCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_gelu_tanh(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_z = F.gelu(cpu_x, approximate="tanh")
        self.assertEqual(cpu_z, gelu_tanh(private_x).cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)

    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_gelu_exact(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        self.assertEqual(F.gelu(cpu_x), gelu_exact(private_x).cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestSoftplusCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_softplus(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        self.assertEqual(F.softplus(cpu_x), softplus(private_x).cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


instantiate_parametrized_tests(TestReluCase)
instantiate_parametrized_tests(TestLeakyReluCase)
instantiate_parametrized_tests(TestEluCase)
instantiate_parametrized_tests(TestSigmoidCase)
instantiate_parametrized_tests(TestSiluCase)
instantiate_parametrized_tests(TestGeluCase)
instantiate_parametrized_tests(TestSoftplusCase)
