import torch
import triton
import triton.language as tl
from ..conftest import F32_ATOL, F32_RTOL, make_inputs, device
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)


@triton.jit
def copy2d_kernel(x_ptr, y_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    a = tl.load(x_ptr + rm[:, None] * N + rn[None, :], mask=mask)
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], a, mask=mask)


@triton.jit
def fill2d_kernel(y_ptr, val, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], val, mask=mask)


@triton.jit
def add2d_kernel(x_ptr, y_ptr, z_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    a = tl.load(x_ptr + rm[:, None] * N + rn[None, :], mask=mask)
    b = tl.load(y_ptr + rm[:, None] * N + rn[None, :], mask=mask)
    tl.store(z_ptr + rm[:, None] * N + rn[None, :], a + b, mask=mask)


@triton.jit
def scale2d_kernel(x_ptr, y_ptr, scale, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    a = tl.load(x_ptr + rm[:, None] * N + rn[None, :], mask=mask)
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], a * scale, mask=mask)


@triton.jit
def transpose2d_kernel(x_ptr, y_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    a = tl.load(x_ptr + rm[:, None] * N + rn[None, :], mask=mask)
    block_t = tl.trans(a)
    t_mask = (rn[:, None] < N) & (rm[None, :] < M)
    tl.store(y_ptr + rn[:, None] * M + rm[None, :], block_t, mask=t_mask)


def copy2d_fn(x: torch.Tensor):
    M, N = x.shape
    BLOCK_M = 16
    BLOCK_N = 16
    out = torch.empty_like(x)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    copy2d_kernel[grid](x, out, M, N, BLOCK_M, BLOCK_N)
    return out


def fill2d_fn(shape, dtype, val):
    M, N = shape
    BLOCK_M = 16
    BLOCK_N = 16
    out = torch.empty(shape, dtype=dtype, device=device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    fill2d_kernel[grid](out, val, M, N, BLOCK_M, BLOCK_N)
    return out


def add2d_fn(x: torch.Tensor, y: torch.Tensor):
    M, N = x.shape
    BLOCK_M = 16
    BLOCK_N = 16
    out = torch.empty_like(x)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    add2d_kernel[grid](x, y, out, M, N, BLOCK_M, BLOCK_N)
    return out


def scale2d_fn(x: torch.Tensor, scale: float):
    M, N = x.shape
    BLOCK_M = 16
    BLOCK_N = 16
    out = torch.empty_like(x)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    scale2d_kernel[grid](x, out, scale, M, N, BLOCK_M, BLOCK_N)
    return out


def transpose2d_fn(x: torch.Tensor):
    M, N = x.shape
    BLOCK_M = 16
    BLOCK_N = 16
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    transpose2d_kernel[grid](x, out, M, N, BLOCK_M, BLOCK_N)
    return out


class TestCopy2DCase(TestCase):
    @parametrize("shape", [(32, 32), (64, 128), (33, 65)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_copy2d(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_z = cpu_x.clone()
        private_z = copy2d_fn(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestFill2DCase(TestCase):
    @parametrize("shape", [(32, 32), (64, 128), (33, 65)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_fill2d(self, shape, dtype):
        val = 7
        cpu_z = torch.full(shape, val, dtype=dtype)
        private_z = fill2d_fn(shape, dtype, val)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestAdd2DCase(TestCase):
    @parametrize("shape", [(32, 32), (64, 128), (33, 65)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_add2d(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_y, private_y = make_inputs(shape, dtype)
        cpu_z = torch.add(cpu_x, cpu_y)
        private_z = add2d_fn(private_x, private_y)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestScale2DCase(TestCase):
    @parametrize("shape,dtype,scale", [
        ((32, 32), torch.float32, 2.5),
        ((64, 128), torch.float32, 2.5),
        ((33, 65), torch.float32, 2.5),
        ((32, 32), torch.int32, 3),
        ((64, 128), torch.int32, 3),
        ((33, 65), torch.int32, 3),
    ])
    def test_scale2d(self, shape, dtype, scale):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_z = cpu_x * scale
        private_z = scale2d_fn(private_x, scale)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestTranspose2DCase(TestCase):
    @parametrize("shape", [(32, 32), (64, 128)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_transpose2d(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_z = cpu_x.t().contiguous()
        private_z = transpose2d_fn(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)


instantiate_parametrized_tests(TestCopy2DCase)
instantiate_parametrized_tests(TestFill2DCase)
instantiate_parametrized_tests(TestAdd2DCase)
instantiate_parametrized_tests(TestScale2DCase)
instantiate_parametrized_tests(TestTranspose2DCase)

