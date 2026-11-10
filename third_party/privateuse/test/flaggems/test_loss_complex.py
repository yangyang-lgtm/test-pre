import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from ..conftest import make_inputs, F32_ATOL, F32_RTOL
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)


@triton.jit
def ce_label_smoothing_kernel(pa, ptgt, pout, R, smooth, C: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, C)
    m = rows < R
    a = tl.load(pa + rows[:, None] * C + cols[None, :], mask=m[:, None], other=float("-inf"))
    tgt = tl.load(ptgt + rows, mask=m, other=0)
    mx = tl.max(a, 1)
    lse = mx + tl.log(tl.sum(tl.exp(a - mx[:, None]), 1))
    onehot = (cols[None, :] == tgt[:, None]).to(tl.float32)
    nll = tl.sum((lse[:, None] - a) * onehot, 1)
    smooth_term = tl.sum(lse[:, None] - a, 1) / C
    loss = (1.0 - smooth) * nll + smooth * smooth_term
    loss = tl.where(m, loss, 0.0)
    tl.store(pout + pid, tl.sum(loss, 0))


def cross_entropy_label_smoothing(logits, target, smoothing=0.1):
    R, C = logits.shape
    nblocks = triton.cdiv(R, 16)
    partial = torch.zeros(nblocks, dtype=torch.float32, device=logits.device)
    grid = lambda meta: (triton.cdiv(R, meta["RB"]), )
    ce_label_smoothing_kernel[grid](logits, target, partial, R, smoothing, C=C, RB=16)
    return partial.sum() / R


class TestCrossEntropyLabelSmoothingCase(TestCase):
    @parametrize("smoothing", [0.0, 0.1])
    def test_ce_label_smoothing(self, smoothing):
        shape = (32, 16)
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_tgt = torch.randint(0, shape[1], (shape[0], ), dtype=torch.int64)
        private_tgt = cpu_tgt.to(torch.int32).to(private_x.device)
        cpu_z = F.cross_entropy(cpu_x, cpu_tgt, label_smoothing=smoothing)
        private_z = cross_entropy_label_smoothing(private_x, private_tgt, smoothing)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)


instantiate_parametrized_tests(TestCrossEntropyLabelSmoothingCase)
