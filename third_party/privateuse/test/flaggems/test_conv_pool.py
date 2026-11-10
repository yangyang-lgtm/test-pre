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
def conv2d_kernel(px, pw, py,
                  C_IN: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                  C_OUT: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
                  OT: tl.constexpr):
    pid = tl.program_id(0)
    n = pid // C_OUT
    co = pid % C_OUT
    OH: tl.constexpr = H - KH + 1
    OW: tl.constexpr = W - KW + 1
    oh = tl.arange(0, OT)
    ow = tl.arange(0, OT)
    valid = (oh[:, None] < OH) & (ow[None, :] < OW)
    acc = tl.zeros((OT, OT), dtype=tl.float32)
    for ci in tl.static_range(0, C_IN):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh + kh
                iw = ow + kw
                lm = valid & (ih[:, None] < H) & (iw[None, :] < W)
                x = tl.load(px + ((n * C_IN + ci) * H + ih[:, None]) * W + iw[None, :],
                            mask=lm, other=0.0)
                w = tl.load(pw + ((co * C_IN + ci) * KH + kh) * KW + kw)
                acc += x * w
    out_off = (n * C_OUT + co) * OH * OW + oh[:, None] * OW + ow[None, :]
    tl.store(py + out_off, acc, mask=valid)


@triton.jit
def max_pool2d_kernel(px, py, pi,
                      C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                      OH: tl.constexpr, OW: tl.constexpr):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    oh = tl.arange(0, OH) * 2
    ow = tl.arange(0, OW) * 2
    base = (n * C + c) * H * W
    v00 = tl.load(px + base + oh[:, None] * W + ow[None, :])
    v01 = tl.load(px + base + oh[:, None] * W + ow[None, :] + 1)
    v10 = tl.load(px + base + (oh[:, None] + 1) * W + ow[None, :])
    v11 = tl.load(px + base + (oh[:, None] + 1) * W + ow[None, :] + 1)
    m0 = tl.maximum(v00, v01)
    i0 = tl.where(v01 > v00, oh[:, None] * W + ow[None, :] + 1, oh[:, None] * W + ow[None, :])
    m1 = tl.maximum(v10, v11)
    i1 = tl.where(v11 > v10, (oh[:, None] + 1) * W + ow[None, :] + 1, (oh[:, None] + 1) * W + ow[None, :])
    m = tl.maximum(m0, m1)
    idx = tl.where(m1 > m0, i1, i0)
    out_off = (n * C + c) * OH * OW + tl.arange(0, OH)[:, None] * OW + tl.arange(0, OW)[None, :]
    tl.store(py + out_off, m)
    tl.store(pi + out_off, idx)


@triton.jit
def avg_pool2d_kernel(px, py, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                      OH: tl.constexpr, OW: tl.constexpr):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    oh = tl.arange(0, OH) * 2
    ow = tl.arange(0, OW) * 2
    base = (n * C + c) * H * W
    s = tl.load(px + base + oh[:, None] * W + ow[None, :])
    s += tl.load(px + base + oh[:, None] * W + ow[None, :] + 1)
    s += tl.load(px + base + (oh[:, None] + 1) * W + ow[None, :])
    s += tl.load(px + base + (oh[:, None] + 1) * W + ow[None, :] + 1)
    out_off = (n * C + c) * OH * OW + tl.arange(0, OH)[:, None] * OW + tl.arange(0, OW)[None, :]
    tl.store(py + out_off, s * 0.25)


def conv2d(x, w, pad=0):
    KH, KW = w.shape[2], w.shape[3]
    if pad:
        x = F.pad(x, (pad, pad, pad, pad))
    N, C, H, W = x.shape
    OH, OW = H - KH + 1, W - KW + 1
    ot = 1 << (max(OH, OW) - 1).bit_length()
    y = torch.empty((N, w.shape[0], OH, OW), dtype=x.dtype, device=x.device)
    conv2d_kernel[(N * w.shape[0],)](x, w, y, C_IN=C, H=H, W=W, C_OUT=w.shape[0],
                                     KH=KH, KW=KW, OT=ot)
    return y


def max_pool2d(x):
    N, C, H, W = x.shape
    OH, OW = H // 2, W // 2
    y = torch.empty((N, C, OH, OW), dtype=x.dtype, device=x.device)
    idx = torch.empty((N, C, OH, OW), dtype=torch.int32, device=x.device)
    max_pool2d_kernel[(N * C,)](x, y, idx, C=C, H=H, W=W, OH=OH, OW=OW)
    return y, idx


def avg_pool2d(x):
    N, C, H, W = x.shape
    OH, OW = H // 2, W // 2
    y = torch.empty((N, C, OH, OW), dtype=x.dtype, device=x.device)
    avg_pool2d_kernel[(N * C,)](x, y, C=C, H=H, W=W, OH=OH, OW=OW)
    return y


class TestConv2dCase(TestCase):
    @parametrize("shape", [(2, 2, 16, 16), (3, 2, 14, 18)])
    @parametrize("pad", [0, 1])
    def test_conv2d(self, shape, pad):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_w, private_w = make_inputs((2, 2, 3, 3), torch.float32)
        cpu_z = F.conv2d(cpu_x, cpu_w, padding=pad)
        private_z = conv2d(private_x, private_w, pad)
        self.assertEqual(cpu_z, private_z.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestMaxPool2dCase(TestCase):
    @parametrize("shape", [(2, 2, 16, 16), (3, 2, 8, 16)])
    def test_max_pool2d(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_z, cpu_idx = F.max_pool2d(cpu_x, 2, return_indices=True)
        private_z, private_idx = max_pool2d(private_x)
        self.assertEqual(cpu_z, private_z.cpu())
        self.assertEqual(cpu_idx.to(torch.int32).view(-1), private_idx.cpu().view(-1))


class TestAvgPool2dCase(TestCase):
    @parametrize("shape", [(2, 2, 16, 16), (3, 2, 8, 16)])
    def test_avg_pool2d(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_z = F.avg_pool2d(cpu_x, 2)
        self.assertEqual(cpu_z, avg_pool2d(private_x).cpu(), atol=F32_ATOL, rtol=F32_RTOL)


instantiate_parametrized_tests(TestConv2dCase)
instantiate_parametrized_tests(TestMaxPool2dCase)
instantiate_parametrized_tests(TestAvgPool2dCase)
