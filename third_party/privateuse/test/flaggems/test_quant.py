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
def per_token_quant_kernel(px, py, R, C: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, C)
    m = rows < R
    x = tl.load(px + rows[:, None] * C + cols[None, :], mask=m[:, None], other=0.0)
    amax = tl.max(tl.abs(x), 1)
    scale = amax / 127.0
    scale = tl.where(scale == 0.0, 1.0, scale)
    v = x / scale[:, None]
    q = tl.where(v >= 0.0, tl.floor(v + 0.5), -tl.floor(-v + 0.5))
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(py + rows[:, None] * C + cols[None, :], q * scale[:, None], mask=m[:, None])


@triton.jit
def per_tensor_fake_quant_kernel(px, py, num, scale, zero_point, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    m = offset < num
    x = tl.load(px + offset, mask=m, other=0.0)
    v = x / scale + zero_point
    q = tl.where(v >= 0.0, tl.floor(v + 0.5), -tl.floor(-v + 0.5))
    q = tl.minimum(tl.maximum(q, 0.0), 255.0)
    tl.store(py + offset, (q - zero_point) * scale, mask=m)


def per_token_quant(x, rb=16):
    R, C = x.shape
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(R, meta["RB"]), )
    per_token_quant_kernel[grid](x, y, R, C=C, RB=rb)
    return y


def per_tensor_fake_quant(x, scale, zero_point):
    y = torch.empty_like(x)
    num = y.numel()
    grid = lambda meta: (triton.cdiv(num, meta["BLOCK"]), )
    per_tensor_fake_quant_kernel[grid](x, y, num, scale, zero_point, BLOCK=256)
    return y


def _round_away(v):
    return torch.where(v >= 0, torch.floor(v + 0.5), -torch.floor(-v + 0.5))


def ref_per_token_quant(x):
    scale = x.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = _round_away(x / scale).clamp(-127, 127)
    return q * scale


def ref_per_tensor_fake_quant(x, scale, zero_point):
    q = _round_away(x / scale + zero_point).clamp(0, 255)
    return (q - zero_point) * scale


class TestPerTokenQuantCase(TestCase):
    @parametrize("shape", [(32, 64), (16, 128), (37, 64)])
    def test_per_token_quant(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        self.assertEqual(ref_per_token_quant(cpu_x), per_token_quant(private_x).cpu(),
                         atol=F32_ATOL, rtol=F32_RTOL)


class TestPerTensorFakeQuantCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 32), (13, 29, 128)])
    def test_per_tensor_fake_quant(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        xmin = cpu_x.min().item()
        xmax = cpu_x.max().item()
        scale = (xmax - xmin) / 255.0
        zero_point = float(round(0 - xmin / scale))
        self.assertEqual(ref_per_tensor_fake_quant(cpu_x, scale, zero_point),
                         per_tensor_fake_quant(private_x, scale, zero_point).cpu(),
                         atol=F32_ATOL, rtol=F32_RTOL)


instantiate_parametrized_tests(TestPerTokenQuantCase)
instantiate_parametrized_tests(TestPerTensorFakeQuantCase)
