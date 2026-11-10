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
def moe_sum_kernel(px, pw, py, T, E: tl.constexpr, H: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, H)
    m = rows < T
    acc = tl.zeros((RB, H), dtype=tl.float32)
    for e in tl.static_range(0, E):
        w = tl.load(pw + rows * E + e, mask=m, other=0.0)
        x = tl.load(px + (e * T + rows[:, None]) * H + cols[None, :], mask=m[:, None], other=0.0)
        acc += w[:, None] * x
    tl.store(py + rows[:, None] * H + cols[None, :], acc, mask=m[:, None])


@triton.jit
def topk_softmax_kernel(pl, pw, T, E: tl.constexpr, K: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, E)
    m = rows < T
    logits = tl.load(pl + rows[:, None] * E + cols[None, :], mask=m[:, None], other=float("-inf"))
    s = tl.sort(logits, 1, descending=True)
    topk = tl.where(cols[None, :] < K, s, float("-inf"))
    mx = tl.max(topk, 1)
    e = tl.exp(topk - mx[:, None])
    z = tl.sum(e, 1)
    tl.store(pw + rows[:, None] * E + cols[None, :], e / z[:, None], mask=m[:, None])


def moe_sum(x, w, rb=16):
    E, T, H = x.shape
    y = torch.empty((T, H), dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(T, meta["RB"]), )
    moe_sum_kernel[grid](x, w, y, T, E=E, H=H, RB=rb)
    return y


def topk_softmax(logits, k, rb=16):
    T, E = logits.shape
    w = torch.empty_like(logits)
    grid = lambda meta: (triton.cdiv(T, meta["RB"]), )
    topk_softmax_kernel[grid](logits, w, T, E=E, K=k, RB=rb)
    return w


class TestMoeSumCase(TestCase):
    @parametrize("shape", [(4, 16, 32), (2, 32, 64), (3, 13, 32)])
    def test_moe_sum(self, shape):
        E, T, H = shape
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_w, private_w = make_inputs((T, E), torch.float32)
        cpu_z = torch.einsum("te,eth->th", cpu_w, cpu_x)
        self.assertEqual(cpu_z, moe_sum(private_x, private_w).cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestTopkSoftmaxCase(TestCase):
    @parametrize("k", [2, 4])
    def test_topk_softmax(self, k):
        T, E = 32, 32
        cpu_x, private_x = make_inputs((T, E), torch.float32)
        topk_vals = torch.topk(cpu_x, k, dim=-1).values
        cpu_w = torch.softmax(topk_vals, dim=-1)
        private_w = topk_softmax(private_x, k)[:, :k].cpu()
        self.assertEqual(cpu_w, private_w, atol=ACC_ATOL, rtol=ACC_RTOL)


instantiate_parametrized_tests(TestMoeSumCase)
instantiate_parametrized_tests(TestTopkSoftmaxCase)
