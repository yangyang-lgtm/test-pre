import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from ..conftest import make_inputs, F32_ATOL, F32_RTOL, ACC_ATOL, ACC_RTOL, SUM_ATOL, SUM_RTOL
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)


@triton.jit
def layer_norm_bwd_kernel(pdy, px, pw, pdx, pdg, pdb, R: tl.constexpr, C: tl.constexpr,
                          RB: tl.constexpr, CB: tl.constexpr, eps):
    rows = tl.arange(0, RB)
    cols = tl.arange(0, CB)
    mask = (rows[:, None] < R) & (cols[None, :] < C)
    off = rows[:, None] * C + cols[None, :]
    dy = tl.load(pdy + off, mask=mask, other=0.0)
    x = tl.load(px + off, mask=mask, other=0.0)
    w = tl.load(pw + cols, mask=cols < C, other=0.0)
    mean = tl.sum(x, 1) / C
    xh = x - mean[:, None]
    rstd = tl.rsqrt(tl.sum(xh * xh, 1) / C + eps)
    xhat = xh * rstd[:, None]
    dyw = dy * w[None, :]
    m1 = tl.sum(dyw, 1) / C
    m2 = tl.sum(dyw * xhat, 1) / C
    dx = rstd[:, None] * (dyw - m1[:, None] - xhat * m2[:, None])
    tl.store(pdx + off, dx, mask=mask)
    tl.store(pdg + cols, tl.sum(dy * xhat, 0), mask=cols < C)
    tl.store(pdb + cols, tl.sum(dy, 0), mask=cols < C)


@triton.jit
def rms_norm_bwd_kernel(pdy, px, pw, pdx, pdg, R: tl.constexpr, C: tl.constexpr,
                        RB: tl.constexpr, CB: tl.constexpr, eps):
    rows = tl.arange(0, RB)
    cols = tl.arange(0, CB)
    mask = (rows[:, None] < R) & (cols[None, :] < C)
    off = rows[:, None] * C + cols[None, :]
    dy = tl.load(pdy + off, mask=mask, other=0.0)
    x = tl.load(px + off, mask=mask, other=0.0)
    w = tl.load(pw + cols, mask=cols < C, other=0.0)
    rstd = tl.rsqrt(tl.sum(x * x, 1) / C + eps)
    xhat = x * rstd[:, None]
    dyw = dy * w[None, :]
    m = tl.sum(dyw * xhat, 1) / C
    dx = rstd[:, None] * (dyw - xhat * m[:, None])
    tl.store(pdx + off, dx, mask=mask)
    tl.store(pdg + cols, tl.sum(dy * xhat, 0), mask=cols < C)


@triton.jit
def group_norm_bwd_kernel(pdy, px, pw, pdx, CG: tl.constexpr, HW: tl.constexpr,
                          C: tl.constexpr, G: tl.constexpr, eps):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G
    rows = tl.arange(0, CG)
    cols = tl.arange(0, HW)
    base = n * C * HW + g * CG * HW
    off = base + rows[:, None] * HW + cols[None, :]
    dy = tl.load(pdy + off)
    x = tl.load(px + off)
    w = tl.load(pw + g * CG + rows)
    cnt: tl.constexpr = CG * HW
    mean = tl.sum(tl.sum(x, 1), 0) / cnt
    xh = x - mean
    rstd = tl.rsqrt(tl.sum(tl.sum(xh * xh, 1), 0) / cnt + eps)
    xhat = xh * rstd
    dyw = dy * w[:, None]
    m1 = tl.sum(tl.sum(dyw, 1), 0) / cnt
    m2 = tl.sum(tl.sum(dyw * xhat, 1), 0) / cnt
    dx = rstd * (dyw - m1 - xhat * m2)
    tl.store(pdx + off, dx)


@triton.jit
def batch_norm_bwd_kernel(pdy, px, pg, pdx, pdg, pdb, N: tl.constexpr, C: tl.constexpr,
                          HW: tl.constexpr, eps):
    c = tl.program_id(0)
    nrows = tl.arange(0, N)
    cols = tl.arange(0, HW)
    off = (nrows[:, None] * C + c) * HW + cols[None, :]
    dy = tl.load(pdy + off)
    x = tl.load(px + off)
    g = tl.load(pg + c)
    cnt: tl.constexpr = N * HW
    mean = tl.sum(tl.sum(x, 1), 0) / cnt
    xh = x - mean
    rstd = tl.rsqrt(tl.sum(tl.sum(xh * xh, 1), 0) / cnt + eps)
    xhat = xh * rstd
    dyw = dy * g
    m1 = tl.sum(tl.sum(dyw, 1), 0) / cnt
    m2 = tl.sum(tl.sum(dyw * xhat, 1), 0) / cnt
    dx = rstd * (dyw - m1 - xhat * m2)
    tl.store(pdx + off, dx)
    tl.store(pdg + c, tl.sum(tl.sum(dy * xhat, 1), 0))
    tl.store(pdb + c, tl.sum(tl.sum(dy, 1), 0))


def layer_norm_bwd(dy, x, w, eps=1e-5):
    R, C = x.shape
    dx = torch.empty_like(x)
    dg = torch.empty(C, dtype=torch.float32, device=x.device)
    db = torch.empty(C, dtype=torch.float32, device=x.device)
    layer_norm_bwd_kernel[(1,)](dy, x, w, dx, dg, db, R=R, C=C,
                                RB=triton.next_power_of_2(R), CB=triton.next_power_of_2(C), eps=eps)
    return dx, dg, db


def rms_norm_bwd(dy, x, w, eps=1e-5):
    R, C = x.shape
    dx = torch.empty_like(x)
    dg = torch.empty(C, dtype=torch.float32, device=x.device)
    rms_norm_bwd_kernel[(1,)](dy, x, w, dx, dg, R=R, C=C,
                              RB=triton.next_power_of_2(R), CB=triton.next_power_of_2(C), eps=eps)
    return dx, dg


def group_norm_bwd(dy, x, w, groups, eps=1e-5):
    N, C, H, W = x.shape
    dx = torch.empty_like(x)
    group_norm_bwd_kernel[(N * groups,)](dy, x, w, dx, CG=C // groups, HW=H * W,
                                         C=C, G=groups, eps=eps)
    return dx


def batch_norm_bwd(dy, x, g, eps=1e-5):
    N, C, H, W = x.shape
    dx = torch.empty_like(x)
    dg = torch.empty(C, dtype=torch.float32, device=x.device)
    db = torch.empty(C, dtype=torch.float32, device=x.device)
    batch_norm_bwd_kernel[(C,)](dy, x, g, dx, dg, db, N=N, C=C, HW=H * W, eps=eps)
    return dx, dg, db


def _grads(fn, dy, *tensors):
    ts = [t.clone().requires_grad_(True) for t in tensors]
    out = fn(*ts)
    if isinstance(out, tuple):
        out = out[0]
    out.backward(dy)
    return out, [t.grad for t in ts]


class TestLayerNormBackwardCase(TestCase):
    @parametrize("shape", [(16, 64), (8, 128), (19, 64)])
    def test_layer_norm_bwd(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_dy, private_dy = make_inputs(shape, torch.float32)
        cpu_w, private_w = make_inputs((shape[1], ), torch.float32)
        _, cpu_g = _grads(lambda x, w: F.layer_norm(x, (shape[1], ), w), cpu_dy, cpu_x, cpu_w)
        dx, dg, db = layer_norm_bwd(private_dy, private_x, private_w)
        self.assertEqual(cpu_g[0], dx.cpu(), atol=SUM_ATOL, rtol=SUM_RTOL)
        self.assertEqual(cpu_g[1], dg.cpu(), atol=SUM_ATOL, rtol=SUM_RTOL)


class TestRmsNormBackwardCase(TestCase):
    @parametrize("shape", [(16, 64), (8, 128), (19, 64)])
    def test_rms_norm_bwd(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_dy, private_dy = make_inputs(shape, torch.float32)
        cpu_w, private_w = make_inputs((shape[1], ), torch.float32)
        eps = 1e-5
        ref = lambda x, w: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
        _, cpu_g = _grads(ref, cpu_dy, cpu_x, cpu_w)
        dx, dg = rms_norm_bwd(private_dy, private_x, private_w)
        self.assertEqual(cpu_g[0], dx.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)
        self.assertEqual(cpu_g[1], dg.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestGroupNormBackwardCase(TestCase):
    @parametrize("groups", [1, 2])
    def test_group_norm_bwd(self, groups):
        shape = (2, 4, 8, 8)
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_dy, private_dy = make_inputs(shape, torch.float32)
        cpu_w, private_w = make_inputs((shape[1], ), torch.float32)
        _, cpu_g = _grads(lambda x, w: F.group_norm(x, groups, weight=w), cpu_dy, cpu_x, cpu_w)
        dx = group_norm_bwd(private_dy, private_x, private_w, groups)
        self.assertEqual(cpu_g[0], dx.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


def ref_batch_norm_bwd(dy, x, g, eps=1e-5):
    N, C, H, W = x.shape
    xc = x.permute(1, 0, 2, 3).reshape(C, -1)
    dyc = dy.permute(1, 0, 2, 3).reshape(C, -1)
    mean = xc.mean(1, keepdim=True)
    var = xc.var(1, unbiased=False, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    xhat = (xc - mean) * rstd
    dyw = dyc * g[:, None]
    m1 = dyw.mean(1, keepdim=True)
    m2 = (dyw * xhat).mean(1, keepdim=True)
    dxc = rstd * (dyw - m1 - xhat * m2)
    dx = dxc.reshape(C, N, H, W).permute(1, 0, 2, 3)
    dg = (dyc * xhat).sum(1)
    db = dyc.sum(1)
    return dx, dg, db


class TestBatchNormBackwardCase(TestCase):
    def test_batch_norm_bwd(self):
        shape = (2, 4, 8, 8)
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_dy, private_dy = make_inputs(shape, torch.float32)
        cpu_g, private_g = make_inputs((shape[1], ), torch.float32)
        ref_dx, ref_dg, ref_db = ref_batch_norm_bwd(cpu_dy, cpu_x, cpu_g)
        dx, dg, db = batch_norm_bwd(private_dy, private_x, private_g)
        self.assertEqual(ref_dx, dx.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)
        self.assertEqual(ref_dg, dg.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)
        self.assertEqual(ref_db, db.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


instantiate_parametrized_tests(TestLayerNormBackwardCase)
instantiate_parametrized_tests(TestRmsNormBackwardCase)
instantiate_parametrized_tests(TestGroupNormBackwardCase)
instantiate_parametrized_tests(TestBatchNormBackwardCase)
