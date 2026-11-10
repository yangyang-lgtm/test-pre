import torch
import triton
import triton.language as tl
from ..conftest import make_inputs, device
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)


@triton.jit
def flip_kernel(pa, pc, N: tl.constexpr, NB: tl.constexpr):
    offset = tl.arange(0, NB)
    mask = offset < N
    a = tl.load(pa + N - 1 - offset, mask=mask, other=0.0)
    tl.store(pc + offset, a, mask=mask)


@triton.jit
def pad_kernel(pa, pc, N, P: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < P
    a = tl.load(pa + offset, mask=mask & (offset < N), other=0.0)
    tl.store(pc + offset, a, mask=mask)


@triton.jit
def masked_fill_kernel(px, pmask, pc, value, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    m = offset < num
    a = tl.load(px + offset, mask=m)
    mask = tl.load(pmask + offset, mask=m)
    tl.store(pc + offset, tl.where(mask != 0, value, a), mask=m)


def flip(x):
    out = torch.empty_like(x)
    flip_kernel[(1,)](x, out, N=x.numel(), NB=triton.next_power_of_2(x.numel()))
    return out


def constant_pad(x, p):
    out = torch.empty(p, dtype=x.dtype, device=x.device)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta["BLOCK"]), )
    pad_kernel[grid](x, out, x.numel(), P=p, BLOCK=256)
    return out


def masked_fill(x, mask, value):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta["BLOCK"]), )
    masked_fill_kernel[grid](x, mask, out, value, num, BLOCK=256)
    return out


class TestFlipCase(TestCase):
    @parametrize("shape", [(256, ), (243, )])
    def test_flip(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        self.assertEqual(torch.flip(cpu_x, [0]), flip(private_x).cpu())


class TestPadCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_constant_pad(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        num = cpu_x.numel()
        p = num + 128
        cpu_z = torch.cat([cpu_x.reshape(-1), torch.zeros(128, dtype=cpu_x.dtype)])
        self.assertEqual(cpu_z, constant_pad(private_x, p).cpu())


class TestMaskedFillCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_masked_fill(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_mask = (cpu_x > 0).to(torch.int8)
        private_mask = cpu_mask.to(device)
        cpu_z = cpu_x.masked_fill(cpu_mask.bool(), 0.0)
        self.assertEqual(cpu_z, masked_fill(private_x, private_mask, 0.0).cpu())


instantiate_parametrized_tests(TestFlipCase)
instantiate_parametrized_tests(TestPadCase)
instantiate_parametrized_tests(TestMaskedFillCase)
