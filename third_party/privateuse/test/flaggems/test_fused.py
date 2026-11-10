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
def silu_and_mul_kernel(pa, pc, R, N: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, N)
    m = rows < R
    base = rows[:, None] * (2 * N)
    x1 = tl.load(pa + base + cols[None, :], mask=m[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(pa + base + N + cols[None, :], mask=m[:, None], other=0.0).to(tl.float32)
    c = x1 / (1.0 + tl.exp(-x1)) * x2
    tl.store(pc + rows[:, None] * N + cols[None, :], c, mask=m[:, None])


@triton.jit
def gelu_and_mul_kernel(pa, pc, R, N: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, N)
    m = rows < R
    base = rows[:, None] * (2 * N)
    x1 = tl.load(pa + base + cols[None, :], mask=m[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(pa + base + N + cols[None, :], mask=m[:, None], other=0.0).to(tl.float32)
    u = 0.7978845608028654 * (x1 + 0.044715 * x1 * x1 * x1)
    t = 2.0 / (1.0 + tl.exp(-2.0 * u)) - 1.0
    g = 0.5 * x1 * (1.0 + t)
    tl.store(pc + rows[:, None] * N + cols[None, :], g * x2, mask=m[:, None])


@triton.jit
def add_rms_norm_kernel(px, pr, pw, pc, R, eps, C: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, C)
    m = rows < R
    x = tl.load(px + rows[:, None] * C + cols[None, :], mask=m[:, None], other=0.0)
    r = tl.load(pr + rows[:, None] * C + cols[None, :], mask=m[:, None], other=0.0)
    w = tl.load(pw + cols)
    t = x + r
    rstd = tl.rsqrt(tl.sum(t * t, 1) / C + eps)
    tl.store(pc + rows[:, None] * C + cols[None, :], t * rstd[:, None] * w[None, :], mask=m[:, None])


@triton.jit
def residual_layer_norm_kernel(px, pr, pw, pb, pc, pres, R, eps, C: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, C)
    m = rows < R
    x = tl.load(px + rows[:, None] * C + cols[None, :], mask=m[:, None], other=0.0)
    r = tl.load(pr + rows[:, None] * C + cols[None, :], mask=m[:, None], other=0.0)
    w = tl.load(pw + cols)
    b = tl.load(pb + cols)
    t = x + r
    mean = tl.sum(t, 1) / C
    d = t - mean[:, None]
    rstd = tl.rsqrt(tl.sum(d * d, 1) / C + eps)
    tl.store(pc + rows[:, None] * C + cols[None, :], d * rstd[:, None] * w[None, :] + b[None, :],
             mask=m[:, None])
    tl.store(pres + rows[:, None] * C + cols[None, :], t, mask=m[:, None])


def silu_and_mul(x, rb=16):
    R, total = x.shape
    N = total // 2
    out = torch.empty((R, N), dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(R, meta["RB"]), )
    silu_and_mul_kernel[grid](x, out, R, N=N, RB=rb)
    return out


def gelu_and_mul(x, rb=16):
    R, total = x.shape
    N = total // 2
    out = torch.empty((R, N), dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(R, meta["RB"]), )
    gelu_and_mul_kernel[grid](x, out, R, N=N, RB=rb)
    return out


def add_rms_norm(x, residual, w, eps=1e-5, rb=16):
    R, C = x.shape
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(R, meta["RB"]), )
    add_rms_norm_kernel[grid](x, residual, w, out, R, eps, C=C, RB=rb)
    return out


def residual_layer_norm(x, residual, w, b, eps=1e-5, rb=16):
    R, C = x.shape
    out = torch.empty_like(x)
    res = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(R, meta["RB"]), )
    residual_layer_norm_kernel[grid](x, residual, w, b, out, res, R, eps, C=C, RB=rb)
    return out, res


class TestSiluAndMulCase(TestCase):
    @parametrize("shape", [(32, 128), (16, 256), (45, 128)])
    def test_silu_and_mul(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        N = shape[1] // 2
        cpu_z = F.silu(cpu_x[..., :N]) * cpu_x[..., N:]
        self.assertEqual(cpu_z, silu_and_mul(private_x).cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestGeluAndMulCase(TestCase):
    @parametrize("shape", [(32, 128), (16, 256), (45, 128)])
    def test_gelu_and_mul(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        N = shape[1] // 2
        cpu_z = F.gelu(cpu_x[..., :N], approximate="tanh") * cpu_x[..., N:]
        self.assertEqual(cpu_z, gelu_and_mul(private_x).cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestAddRmsNormCase(TestCase):
    @parametrize("shape", [(32, 64), (16, 128), (37, 64)])
    def test_add_rms_norm(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_r, private_r = make_inputs(shape, torch.float32)
        cpu_w, private_w = make_inputs((shape[1], ), torch.float32)
        eps = 1e-5
        t = cpu_x + cpu_r
        cpu_z = t * torch.rsqrt(t.pow(2).mean(dim=-1, keepdim=True) + eps) * cpu_w
        self.assertEqual(cpu_z, add_rms_norm(private_x, private_r, private_w, eps).cpu(),
                         atol=ACC_ATOL, rtol=ACC_RTOL)


class TestResidualLayerNormCase(TestCase):
    @parametrize("shape", [(32, 64), (16, 128), (37, 64)])
    def test_residual_layer_norm(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_r, private_r = make_inputs(shape, torch.float32)
        cpu_w, private_w = make_inputs((shape[1], ), torch.float32)
        cpu_b, private_b = make_inputs((shape[1], ), torch.float32)
        eps = 1e-5
        cpu_out, _, _ = torch.native_layer_norm(cpu_x + cpu_r, [shape[1]], cpu_w, cpu_b, eps)
        out, res = residual_layer_norm(private_x, private_r, private_w, private_b, eps)
        self.assertEqual(cpu_out, out.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)
        self.assertEqual(cpu_x + cpu_r, res.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


instantiate_parametrized_tests(TestSiluAndMulCase)
instantiate_parametrized_tests(TestGeluAndMulCase)
instantiate_parametrized_tests(TestAddRmsNormCase)
instantiate_parametrized_tests(TestResidualLayerNormCase)
