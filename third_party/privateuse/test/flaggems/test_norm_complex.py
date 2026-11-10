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
def instance_norm_kernel(px, pw, pb, py, HW: tl.constexpr, C: tl.constexpr, eps):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    off = tl.arange(0, HW)
    x = tl.load(px + (n * C + c) * HW + off)
    mean = tl.sum(x, 0) / HW
    d = x - mean
    rstd = tl.rsqrt(tl.sum(d * d, 0) / HW + eps)
    w = tl.load(pw + c)
    b = tl.load(pb + c)
    tl.store(py + (n * C + c) * HW + off, d * rstd * w + b)


@triton.jit
def group_norm_kernel(px, pw, pb, py, CG: tl.constexpr, HW: tl.constexpr,
                      C: tl.constexpr, G: tl.constexpr, eps):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G
    rows = tl.arange(0, CG)
    cols = tl.arange(0, HW)
    base = n * C * HW + g * CG * HW
    x = tl.load(px + base + rows[:, None] * HW + cols[None, :])
    cnt: tl.constexpr = CG * HW
    mean = tl.sum(tl.sum(x, 1), 0) / cnt
    d = x - mean
    rstd = tl.rsqrt(tl.sum(tl.sum(d * d, 1), 0) / cnt + eps)
    ch = g * CG + rows
    w = tl.load(pw + ch)
    b = tl.load(pb + ch)
    out = d * rstd * w[:, None] + b[:, None]
    tl.store(py + base + rows[:, None] * HW + cols[None, :], out)


def instance_norm(x, w, b, eps=1e-5):
    N, C, H, W = x.shape
    y = torch.empty_like(x)
    instance_norm_kernel[(N * C,)](x, w, b, y, HW=H * W, C=C, eps=eps)
    return y


def group_norm(x, g, w, b, eps=1e-5):
    N, C, H, W = x.shape
    y = torch.empty_like(x)
    group_norm_kernel[(N * g,)](x, w, b, y, CG=C // g, HW=H * W, C=C, G=g, eps=eps)
    return y


class TestInstanceNormCase(TestCase):
    @parametrize("shape", [(2, 2, 16, 8), (1, 4, 8, 8), (3, 2, 16, 8)])
    def test_instance_norm(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_w, private_w = make_inputs((shape[1], ), torch.float32)
        cpu_b, private_b = make_inputs((shape[1], ), torch.float32)
        cpu_z = F.instance_norm(cpu_x, weight=cpu_w, bias=cpu_b, eps=1e-5)
        private_z = instance_norm(private_x, private_w, private_b)
        self.assertEqual(cpu_z, private_z.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestGroupNormCase(TestCase):
    @parametrize("groups", [1, 2])
    def test_group_norm(self, groups):
        shape = (2, 4, 8, 8)
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_w, private_w = make_inputs((shape[1], ), torch.float32)
        cpu_b, private_b = make_inputs((shape[1], ), torch.float32)
        cpu_z = F.group_norm(cpu_x, groups, weight=cpu_w, bias=cpu_b, eps=1e-5)
        private_z = group_norm(private_x, groups, private_w, private_b)
        self.assertEqual(cpu_z, private_z.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


instantiate_parametrized_tests(TestInstanceNormCase)
instantiate_parametrized_tests(TestGroupNormCase)
