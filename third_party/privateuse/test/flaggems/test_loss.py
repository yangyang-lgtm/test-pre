import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from ..conftest import make_inputs, device, F32_ATOL, F32_RTOL
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)


@triton.jit
def mse_kernel(pa, pb, pout, R, C: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, C)
    m = rows < R
    a = tl.load(pa + rows[:, None] * C + cols[None, :], mask=m[:, None], other=0.0)
    b = tl.load(pb + rows[:, None] * C + cols[None, :], mask=m[:, None], other=0.0)
    d = a - b
    tl.store(pout + pid, tl.sum(tl.sum(d * d, 1), 0))


@triton.jit
def l1_kernel(pa, pb, pout, R, C: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, C)
    m = rows < R
    a = tl.load(pa + rows[:, None] * C + cols[None, :], mask=m[:, None], other=0.0)
    b = tl.load(pb + rows[:, None] * C + cols[None, :], mask=m[:, None], other=0.0)
    d = tl.abs(a - b)
    tl.store(pout + pid, tl.sum(tl.sum(d, 1), 0))


@triton.jit
def nll_kernel(plp, ptgt, pout, R, C: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, C)
    m = rows < R
    lp = tl.load(plp + rows[:, None] * C + cols[None, :], mask=m[:, None], other=0.0)
    tgt = tl.load(ptgt + rows, mask=m, other=0)
    onehot = (cols[None, :] == tgt[:, None]).to(tl.float32)
    picked = tl.sum(lp * onehot, 1)
    loss = tl.where(m, -picked, 0.0)
    tl.store(pout + pid, tl.sum(loss, 0))


@triton.jit
def cross_entropy_kernel(pa, ptgt, pout, R, C: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, C)
    m = rows < R
    a = tl.load(pa + rows[:, None] * C + cols[None, :], mask=m[:, None], other=float("-inf"))
    tgt = tl.load(ptgt + rows, mask=m, other=0)
    mx = tl.max(a, 1)
    lse = mx + tl.log(tl.sum(tl.exp(a - mx[:, None]), 1))
    onehot = (cols[None, :] == tgt[:, None]).to(tl.float32)
    picked = tl.sum(a * onehot, 1)
    loss = tl.where(m, lse - picked, 0.0)
    tl.store(pout + pid, tl.sum(loss, 0))


def _reduce_loss(x, y, kernel, rb=16):
    R, C = x.shape
    nblocks = triton.cdiv(R, rb)
    partial = torch.zeros(nblocks, dtype=torch.float32, device=x.device)
    grid = lambda meta: (triton.cdiv(R, meta["RB"]), )
    kernel[grid](x, y, partial, R, C=C, RB=rb)
    return partial.sum()


def mse_loss(x, y):
    return _reduce_loss(x, y, mse_kernel) / x.numel()


def l1_loss(x, y):
    return _reduce_loss(x, y, l1_kernel) / x.numel()


def nll_loss(log_probs, target):
    R, C = log_probs.shape
    nblocks = triton.cdiv(R, 16)
    partial = torch.zeros(nblocks, dtype=torch.float32, device=log_probs.device)
    grid = lambda meta: (triton.cdiv(R, meta["RB"]), )
    nll_kernel[grid](log_probs, target, partial, R, C=C, RB=16)
    return partial.sum() / R


def cross_entropy(logits, target):
    R, C = logits.shape
    nblocks = triton.cdiv(R, 16)
    partial = torch.zeros(nblocks, dtype=torch.float32, device=logits.device)
    grid = lambda meta: (triton.cdiv(R, meta["RB"]), )
    cross_entropy_kernel[grid](logits, target, partial, R, C=C, RB=16)
    return partial.sum() / R


class TestMseLossCase(TestCase):
    @parametrize("shape", [(32, 32), (16, 128), (45, 32)])
    def test_mse_loss(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs(shape, torch.float32)
        self.assertEqual(F.mse_loss(cpu_x, cpu_y), mse_loss(private_x, private_y).cpu(),
                         atol=F32_ATOL, rtol=F32_RTOL)


class TestL1LossCase(TestCase):
    @parametrize("shape", [(32, 32), (16, 128), (45, 32)])
    def test_l1_loss(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs(shape, torch.float32)
        self.assertEqual(F.l1_loss(cpu_x, cpu_y), l1_loss(private_x, private_y).cpu(),
                         atol=F32_ATOL, rtol=F32_RTOL)


class TestNllLossCase(TestCase):
    @parametrize("shape", [(32, 16), (16, 64), (37, 16)])
    def test_nll_loss(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        C = shape[1]
        cpu_lp = torch.log_softmax(cpu_x, dim=1)
        private_lp = cpu_lp.to(device)
        cpu_tgt = torch.randint(0, C, (shape[0], ), dtype=torch.int64)
        private_tgt = cpu_tgt.to(torch.int32).to(device)
        cpu_z = F.nll_loss(cpu_lp, cpu_tgt)
        self.assertEqual(cpu_z, nll_loss(private_lp, private_tgt).cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestCrossEntropyCase(TestCase):
    @parametrize("shape", [(32, 16), (16, 64), (37, 16)])
    def test_cross_entropy(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        C = shape[1]
        cpu_tgt = torch.randint(0, C, (shape[0], ), dtype=torch.int64)
        private_tgt = cpu_tgt.to(torch.int32).to(device)
        cpu_z = F.cross_entropy(cpu_x, cpu_tgt)
        self.assertEqual(cpu_z, cross_entropy(private_x, private_tgt).cpu(), atol=F32_ATOL, rtol=F32_RTOL)


instantiate_parametrized_tests(TestMseLossCase)
instantiate_parametrized_tests(TestL1LossCase)
instantiate_parametrized_tests(TestNllLossCase)
instantiate_parametrized_tests(TestCrossEntropyCase)
