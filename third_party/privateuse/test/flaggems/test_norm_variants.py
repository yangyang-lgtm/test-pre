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
def gemma_rms_norm_kernel(px, pw, py, R, eps, C: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, C)
    m = rows < R
    x = tl.load(px + rows[:, None] * C + cols[None, :], mask=m[:, None], other=0.0)
    w = tl.load(pw + cols)
    rstd = tl.rsqrt(tl.sum(x * x, 1) / C + eps)
    tl.store(py + rows[:, None] * C + cols[None, :], x * rstd[:, None] * (1.0 + w[None, :]),
             mask=m[:, None])


@triton.jit
def weight_norm_kernel(pv, pg, py, M: tl.constexpr, K: tl.constexpr):
    r = tl.program_id(0)
    cols = tl.arange(0, K)
    v = tl.load(pv + r * K + cols)
    g = tl.load(pg + r)
    norm = tl.sqrt(tl.sum(v * v, 0))
    tl.store(py + r * K + cols, g * v / norm)


@triton.jit
def vector_norm_kernel(px, py, R, ORD: tl.constexpr, C: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, C)
    m = rows < R
    x = tl.load(px + rows[:, None] * C + cols[None, :], mask=m[:, None], other=0.0)
    if ORD == 1:
        v = tl.sum(tl.abs(x), 1)
    elif ORD == 2:
        v = tl.sqrt(tl.sum(x * x, 1))
    else:
        v = tl.max(tl.abs(x), 1)
    tl.store(py + rows, v, mask=m)


@triton.jit
def matrix_norm_kernel(px, py, R: tl.constexpr, C: tl.constexpr, RB: tl.constexpr, CB: tl.constexpr):
    rows = tl.arange(0, RB)
    cols = tl.arange(0, CB)
    mask = (rows[:, None] < R) & (cols[None, :] < C)
    x = tl.load(px + rows[:, None] * C + cols[None, :], mask=mask, other=0.0)
    tl.store(py, tl.sqrt(tl.sum(tl.sum(x * x, 1), 0)))


def gemma_rms_norm(x, w, eps=1e-5, rb=16):
    R, C = x.shape
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(R, meta["RB"]), )
    gemma_rms_norm_kernel[grid](x, w, y, R, eps, C=C, RB=rb)
    return y


def weight_norm(v, g):
    M, K = v.shape
    y = torch.empty_like(v)
    weight_norm_kernel[(M,)](v, g, y, M=M, K=K)
    return y


def vector_norm(x, ord, rb=16):
    R, C = x.shape
    y = torch.empty(R, dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(R, meta["RB"]), )
    vector_norm_kernel[grid](x, y, R, ORD=ord, C=C, RB=rb)
    return y


def matrix_norm(x):
    R, C = x.shape
    y = torch.empty(1, dtype=x.dtype, device=x.device)
    matrix_norm_kernel[(1,)](x, y, R=R, C=C, RB=triton.next_power_of_2(R), CB=triton.next_power_of_2(C))
    return y[0]


class TestGemmaRmsNormCase(TestCase):
    @parametrize("shape", [(32, 64), (16, 128), (37, 64)])
    def test_gemma_rms_norm(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_w, private_w = make_inputs((shape[1], ), torch.float32)
        eps = 1e-5
        cpu_z = cpu_x * torch.rsqrt(cpu_x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + cpu_w)
        self.assertEqual(cpu_z, gemma_rms_norm(private_x, private_w, eps).cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestWeightNormCase(TestCase):
    @parametrize("shape", [(16, 32), (32, 64), (13, 32)])
    def test_weight_norm(self, shape):
        M, K = shape
        cpu_v, private_v = make_inputs(shape, torch.float32)
        cpu_g, private_g = make_inputs((M, ), torch.float32)
        cpu_z = cpu_g[:, None] * cpu_v / cpu_v.norm(dim=1, keepdim=True)
        self.assertEqual(cpu_z, weight_norm(private_v, private_g).cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestVectorNormCase(TestCase):
    @parametrize("ord", [1, 2, float("inf")])
    def test_vector_norm(self, ord):
        shape = (32, 64)
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_z = torch.linalg.vector_norm(cpu_x, ord=ord, dim=1)
        self.assertEqual(cpu_z, vector_norm(private_x, ord).cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestMatrixNormCase(TestCase):
    @parametrize("shape", [(32, 64), (16, 128), (37, 64)])
    def test_matrix_norm(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_z = torch.linalg.matrix_norm(cpu_x, ord="fro")
        self.assertEqual(cpu_z, matrix_norm(private_x).cpu(), atol=F32_ATOL, rtol=F32_RTOL)


instantiate_parametrized_tests(TestGemmaRmsNormCase)
instantiate_parametrized_tests(TestWeightNormCase)
instantiate_parametrized_tests(TestVectorNormCase)
instantiate_parametrized_tests(TestMatrixNormCase)
