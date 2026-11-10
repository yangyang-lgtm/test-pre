import torch
import triton
import triton.language as tl
from ..conftest import make_inputs, F32_ATOL, F32_RTOL, ACC_ATOL, ACC_RTOL, SUM_ATOL, SUM_RTOL
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)

BLOCK = 32


@triton.jit
def mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
              stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_bn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


@triton.jit
def addmm_kernel(a_ptr, b_ptr, bias_ptr, c_ptr, M, N, K, beta, alpha,
                 stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_bn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    bias = tl.load(bias_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn,
                   mask=mask, other=0.0)
    acc = beta * bias + alpha * acc
    tl.store(c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn, acc, mask=mask)


@triton.jit
def matmul_relu_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                       stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_bn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    c = tl.maximum(acc, 0.0)
    tl.store(c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn, c, mask=mask)


@triton.jit
def bmm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
               stride_ab, stride_am, stride_ak,
               stride_bb, stride_bk, stride_bn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + pid * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + pid * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
    acc = tl.dot(a, b)
    c_ptrs = c_ptr + pid * M * N + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def mv_kernel(a_ptr, x_ptr, y_ptr, M, K, stride_am, stride_ak, BK: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB)
    cols = tl.arange(0, BK)
    m = rows < M
    a = tl.load(a_ptr + rows[:, None] * stride_am + cols[None, :] * stride_ak, mask=m[:, None], other=0.0)
    x = tl.load(x_ptr + cols, mask=cols < K, other=0.0)
    tl.store(y_ptr + rows, tl.sum(a * x[None, :], 1), mask=m)


def matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    mm_kernel[grid](a, b, c, M, N, K,
                    a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                    BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_K=BLOCK)
    return c


def addmm(bias, a, b, beta=1.0, alpha=1.0):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    addmm_kernel[grid](a, b, bias, c, M, N, K, beta, alpha,
                       a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                       BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_K=BLOCK)
    return c


def matmul_relu(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    matmul_relu_kernel[grid](a, b, c, M, N, K,
                             a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                             BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_K=BLOCK)
    return c


def bmm(a, b):
    B, M, K = a.shape
    K, N = b.shape[1], b.shape[2]
    c = torch.empty((B, M, N), dtype=a.dtype, device=a.device)
    bmm_kernel[(B,)](a, b, c, M, N, K,
                     a.stride(0), a.stride(1), a.stride(2),
                     b.stride(0), b.stride(1), b.stride(2),
                     BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_K=BLOCK)
    return c


def mv(a, x):
    M, K = a.shape
    y = torch.empty(M, dtype=a.dtype, device=a.device)
    grid = lambda meta: (triton.cdiv(M, meta["RB"]), )
    mv_kernel[grid](a, x, y, M, K, a.stride(0), a.stride(1), BK=BLOCK if K <= BLOCK else K, RB=16)
    return y


class TestMmCase(TestCase):
    @parametrize("shape", [(32, 32), (64, 64), (64, 100), (37, 32)])
    def test_mm(self, shape):
        M, K = shape
        N = 64
        cpu_a, private_a = make_inputs((M, K), torch.float32)
        cpu_b, private_b = make_inputs((K, N), torch.float32)
        self.assertEqual(torch.matmul(cpu_a, cpu_b), matmul(private_a, private_b).cpu(),
                         atol=SUM_ATOL, rtol=SUM_RTOL)


class TestAddmmCase(TestCase):
    @parametrize("shape", [(32, 32), (64, 64), (37, 32)])
    def test_addmm(self, shape):
        M, K = shape
        N = 64
        cpu_a, private_a = make_inputs((M, K), torch.float32)
        cpu_b, private_b = make_inputs((K, N), torch.float32)
        cpu_bias, private_bias = make_inputs((M, N), torch.float32)
        cpu_z = torch.addmm(cpu_bias, cpu_a, cpu_b, beta=0.5, alpha=2.0)
        self.assertEqual(cpu_z, addmm(private_bias, private_a, private_b, 0.5, 2.0).cpu(),
                         atol=SUM_ATOL, rtol=SUM_RTOL)


class TestMatmulReluCase(TestCase):
    @parametrize("shape", [(32, 32), (64, 64), (37, 32)])
    def test_matmul_relu(self, shape):
        M, K = shape
        N = 64
        cpu_a, private_a = make_inputs((M, K), torch.float32)
        cpu_b, private_b = make_inputs((K, N), torch.float32)
        cpu_z = torch.relu(torch.matmul(cpu_a, cpu_b))
        self.assertEqual(cpu_z, matmul_relu(private_a, private_b).cpu(), atol=SUM_ATOL, rtol=SUM_RTOL)


class TestBmmCase(TestCase):
    @parametrize("batch", [2, 4, 3])
    def test_bmm(self, batch):
        M = K = N = 32
        cpu_a, private_a = make_inputs((batch, M, K), torch.float32)
        cpu_b, private_b = make_inputs((batch, K, N), torch.float32)
        self.assertEqual(torch.bmm(cpu_a, cpu_b), bmm(private_a, private_b).cpu(),
                         atol=SUM_ATOL, rtol=SUM_RTOL)


class TestMvCase(TestCase):
    @parametrize("shape", [(32, 32), (64, 64), (37, 32)])
    def test_mv(self, shape):
        M, K = shape
        cpu_a, private_a = make_inputs((M, K), torch.float32)
        cpu_x, private_x = make_inputs((K, ), torch.float32)
        self.assertEqual(torch.mv(cpu_a, cpu_x), mv(private_a, private_x).cpu(),
                         atol=SUM_ATOL, rtol=SUM_RTOL)


instantiate_parametrized_tests(TestMmCase)
instantiate_parametrized_tests(TestAddmmCase)
instantiate_parametrized_tests(TestMatmulReluCase)
instantiate_parametrized_tests(TestBmmCase)
instantiate_parametrized_tests(TestMvCase)
