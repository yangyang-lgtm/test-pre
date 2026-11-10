import torch
import triton
import triton.language as tl
from ..conftest import make_inputs, F32_ATOL, F32_RTOL
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)


@triton.jit
def rope_interleaved_kernel(px, pcos, psin, py, HD: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    t = pid // HD
    base = pid * D
    off = tl.arange(0, D // 2)
    x1 = tl.load(px + base + 2 * off)
    x2 = tl.load(px + base + 2 * off + 1)
    c = tl.load(pcos + t * (D // 2) + off)
    s = tl.load(psin + t * (D // 2) + off)
    tl.store(py + base + 2 * off, x1 * c - x2 * s)
    tl.store(py + base + 2 * off + 1, x1 * s + x2 * c)


@triton.jit
def rope_neox_kernel(px, pcos, psin, py, HD: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    t = pid // HD
    base = pid * D
    off = tl.arange(0, D // 2)
    x1 = tl.load(px + base + off)
    x2 = tl.load(px + base + D // 2 + off)
    c = tl.load(pcos + t * (D // 2) + off)
    s = tl.load(psin + t * (D // 2) + off)
    tl.store(py + base + off, x1 * c - x2 * s)
    tl.store(py + base + D // 2 + off, x1 * s + x2 * c)


def rope(x, cos, sin, interleaved=True):
    T, H, D = x.shape
    y = torch.empty_like(x)
    grid = (T * H, )
    kernel = rope_interleaved_kernel if interleaved else rope_neox_kernel
    kernel[grid](x, cos, sin, y, HD=H, D=D)
    return y


def make_freqs(T, D, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, D, 2).float() / D))
    t = torch.arange(T).float()
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def ref_rope(x, cos, sin, interleaved=True):
    T, H, D = x.shape
    if interleaved:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos[:, None, :] - x2 * sin[:, None, :]
        out[..., 1::2] = x1 * sin[:, None, :] + x2 * cos[:, None, :]
        return out
    half = D // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    out = torch.empty_like(x)
    out[..., :half] = x1 * cos[:, None, :] - x2 * sin[:, None, :]
    out[..., half:] = x1 * sin[:, None, :] + x2 * cos[:, None, :]
    return out


class TestRotaryInterleavedCase(TestCase):
    @parametrize("shape", [(16, 2, 32), (8, 4, 64), (13, 3, 32)])
    def test_interleaved(self, shape):
        T, H, D = shape
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_c, cpu_s = make_freqs(T, D)
        cpu_z = ref_rope(cpu_x, cpu_c, cpu_s, True)
        private_z = rope(private_x, cpu_c.to(private_x.device), cpu_s.to(private_x.device), True)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestRotaryNeoxCase(TestCase):
    @parametrize("shape", [(16, 2, 32), (8, 4, 64), (13, 3, 32)])
    def test_neox(self, shape):
        T, H, D = shape
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_c, cpu_s = make_freqs(T, D)
        cpu_z = ref_rope(cpu_x, cpu_c, cpu_s, False)
        private_z = rope(private_x, cpu_c.to(private_x.device), cpu_s.to(private_x.device), False)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)


instantiate_parametrized_tests(TestRotaryInterleavedCase)
instantiate_parametrized_tests(TestRotaryNeoxCase)
