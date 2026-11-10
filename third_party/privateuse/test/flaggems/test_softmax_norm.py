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
def softmax_kernel(pa, pc, R, C: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, C)
    m = rows < R
    a = tl.load(pa + rows[:, None] * C + cols[None, :], mask=m[:, None], other=float("-inf"))
    mx = tl.max(a, 1)
    e = tl.exp(a - mx[:, None])
    z = tl.sum(e, 1)
    tl.store(pc + rows[:, None] * C + cols[None, :], e / z[:, None], mask=m[:, None])


@triton.jit
def log_softmax_kernel(pa, pc, R, C: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, C)
    m = rows < R
    a = tl.load(pa + rows[:, None] * C + cols[None, :], mask=m[:, None], other=float("-inf"))
    mx = tl.max(a, 1)
    e = tl.exp(a - mx[:, None])
    lse = mx + tl.log(tl.sum(e, 1))
    tl.store(pc + rows[:, None] * C + cols[None, :], a - lse[:, None], mask=m[:, None])


@triton.jit
def logsumexp_kernel(pa, pc, R, C: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, C)
    m = rows < R
    a = tl.load(pa + rows[:, None] * C + cols[None, :], mask=m[:, None], other=float("-inf"))
    mx = tl.max(a, 1)
    e = tl.exp(a - mx[:, None])
    tl.store(pc + rows, mx + tl.log(tl.sum(e, 1)), mask=m)


@triton.jit
def layer_norm_kernel(pa, pw, pb, pc, pm, pr, R, eps, C: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, C)
    m = rows < R
    a = tl.load(pa + rows[:, None] * C + cols[None, :], mask=m[:, None], other=0.0)
    w = tl.load(pw + cols)
    b = tl.load(pb + cols)
    mean = tl.sum(a, 1) / C
    d = a - mean[:, None]
    rstd = tl.rsqrt(tl.sum(d * d, 1) / C + eps)
    out = d * rstd[:, None] * w[None, :] + b[None, :]
    tl.store(pc + rows[:, None] * C + cols[None, :], out, mask=m[:, None])
    tl.store(pm + rows, mean, mask=m)
    tl.store(pr + rows, rstd, mask=m)


@triton.jit
def rms_norm_kernel(pa, pw, pc, R, eps, C: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, C)
    m = rows < R
    a = tl.load(pa + rows[:, None] * C + cols[None, :], mask=m[:, None], other=0.0)
    w = tl.load(pw + cols)
    rstd = tl.rsqrt(tl.sum(a * a, 1) / C + eps)
    tl.store(pc + rows[:, None] * C + cols[None, :], a * rstd[:, None] * w[None, :], mask=m[:, None])


def softmax(x, rb=16):
    R, C = x.shape
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(R, meta["RB"]), )
    softmax_kernel[grid](x, out, R, C=C, RB=rb)
    return out


def log_softmax(x, rb=16):
    R, C = x.shape
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(R, meta["RB"]), )
    log_softmax_kernel[grid](x, out, R, C=C, RB=rb)
    return out


def logsumexp(x, rb=16):
    R, C = x.shape
    out = torch.empty(R, dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(R, meta["RB"]), )
    logsumexp_kernel[grid](x, out, R, C=C, RB=rb)
    return out


def layer_norm(x, w, b, eps=1e-5, rb=16):
    R, C = x.shape
    out = torch.empty_like(x)
    mean = torch.empty(R, dtype=torch.float32, device=x.device)
    rstd = torch.empty(R, dtype=torch.float32, device=x.device)
    grid = lambda meta: (triton.cdiv(R, meta["RB"]), )
    layer_norm_kernel[grid](x, w, b, out, mean, rstd, R, eps, C=C, RB=rb)
    return out, mean, rstd


def rms_norm(x, w, eps=1e-5, rb=16):
    R, C = x.shape
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(R, meta["RB"]), )
    rms_norm_kernel[grid](x, w, out, R, eps, C=C, RB=rb)
    return out


class TestSoftmaxCase(TestCase):
    @parametrize("shape", [(32, 64), (16, 128), (37, 64)])
    def test_softmax(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_z = torch.softmax(cpu_x, dim=1)
        self.assertEqual(cpu_z, softmax(private_x).cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestLogSoftmaxCase(TestCase):
    @parametrize("shape", [(32, 64), (16, 128), (37, 64)])
    def test_log_softmax(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_z = torch.log_softmax(cpu_x, dim=1)
        self.assertEqual(cpu_z, log_softmax(private_x).cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestLogsumexpCase(TestCase):
    @parametrize("shape", [(32, 64), (16, 128), (37, 64)])
    def test_logsumexp(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_z = torch.logsumexp(cpu_x, dim=1)
        self.assertEqual(cpu_z, logsumexp(private_x).cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestLayerNormCase(TestCase):
    @parametrize("shape", [(32, 64), (16, 128), (37, 64)])
    def test_layer_norm(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_w, private_w = make_inputs((shape[1], ), torch.float32)
        cpu_b, private_b = make_inputs((shape[1], ), torch.float32)
        cpu_out, cpu_mean, cpu_rstd = torch.native_layer_norm(cpu_x, [shape[1]], cpu_w, cpu_b, 1e-5)
        out, mean, rstd = layer_norm(private_x, private_w, private_b)
        self.assertEqual(cpu_out, out.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)
        self.assertEqual(cpu_mean.view(-1), mean.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)
        self.assertEqual(cpu_rstd.view(-1), rstd.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestRmsNormCase(TestCase):
    @parametrize("shape", [(32, 64), (16, 128), (37, 64)])
    def test_rms_norm(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_w, private_w = make_inputs((shape[1], ), torch.float32)
        eps = 1e-5
        cpu_z = cpu_x * torch.rsqrt(cpu_x.pow(2).mean(dim=-1, keepdim=True) + eps) * cpu_w
        self.assertEqual(cpu_z, rms_norm(private_x, private_w, eps).cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


instantiate_parametrized_tests(TestSoftmaxCase)
instantiate_parametrized_tests(TestLogSoftmaxCase)
instantiate_parametrized_tests(TestLogsumexpCase)
instantiate_parametrized_tests(TestLayerNormCase)
instantiate_parametrized_tests(TestRmsNormCase)
