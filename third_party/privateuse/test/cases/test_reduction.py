import torch
import triton
import triton.language as tl
from ..conftest import make_inputs, device, F32_ATOL, F32_RTOL, ACC_ATOL, ACC_RTOL, LOWP_ATOL, LOWP_RTOL, SUM_ATOL, SUM_RTOL
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)

@triton.jit
def sum_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask, other=0.0)
    c = tl.sum(a, axis=0)
    tl.store(pc + pid, c)

@triton.jit
def max_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask, other=float('-inf'))
    c = tl.max(a, axis=0)
    tl.store(pc + pid, c)

@triton.jit
def min_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask, other=float('inf'))
    c = tl.min(a, axis=0)
    tl.store(pc + pid, c)

def sum_fn(x: torch.Tensor):
    num = x.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    out = torch.empty(grid({'BLOCK': 256}), dtype=x.dtype, device=x.device)
    sum_kernel[grid](x, out, num, BLOCK=256)
    return out.sum()

def max_fn(x: torch.Tensor):
    num = x.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    out = torch.empty(grid({'BLOCK': 256}), dtype=x.dtype, device=x.device)
    max_kernel[grid](x, out, num, BLOCK=256)
    return out.max()

def min_fn(x: torch.Tensor):
    num = x.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    out = torch.empty(grid({'BLOCK': 256}), dtype=x.dtype, device=x.device)
    min_kernel[grid](x, out, num, BLOCK=256)
    return out.min()

class TestSumCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_sum(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_z = torch.sum(cpu_x)
        private_z = sum_fn(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=SUM_ATOL, rtol=SUM_RTOL)

class TestMaxCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_max(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_z = torch.max(cpu_x)
        private_z = max_fn(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=0, rtol=0)

class TestMinCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_min(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_z = torch.min(cpu_x)
        private_z = min_fn(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=0, rtol=0)
@triton.jit
def max_abs_combine_fn(x, y):
    return tl.maximum(tl.abs(x), tl.abs(y))

@triton.jit
def max_abs_1d_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask, other=0.0)
    c = tl.reduce(a, 0, max_abs_combine_fn)
    tl.store(pc + pid, c)

@triton.jit
def max_abs_2d_kernel(pa, pc, axis: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    a = tl.load(pa + offs_m[:, None] * N + offs_n[None, :])
    c = tl.reduce(a, axis, max_abs_combine_fn)
    if axis == 0:
        tl.store(pc + offs_n, c)
    else:
        tl.store(pc + offs_m, c)

@triton.jit
def max_abs_3d_kernel(pa, pc, axis: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    a = tl.load(pa + offs_m[:, None, None] * N * K + offs_n[None, :, None] * K + offs_k[None, None, :])
    c = tl.reduce(a, axis, max_abs_combine_fn)
    if axis == 0:
        tl.store(pc + offs_n[:, None] * K + offs_k[None, :], c)
    elif axis == 1:
        tl.store(pc + offs_m[:, None] * K + offs_k[None, :], c)
    else:
        tl.store(pc + offs_m[:, None] * N + offs_n[None, :], c)

def max_abs_fn_1d(x: torch.Tensor):
    num = x.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    out = torch.empty(grid({'BLOCK': 256}), dtype=x.dtype, device=x.device)
    max_abs_1d_kernel[grid](x, out, num, BLOCK=256)
    return out.max()

def max_abs_fn_2d(x: torch.Tensor, axis: int):
    m, n = x.shape
    out_shape = (n, ) if axis == 0 else (m, )
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    max_abs_2d_kernel[(1, )](x, out, axis, M=m, N=n)
    return out

def max_abs_fn_3d(x: torch.Tensor, axis: int):
    m, n, k = x.shape
    out_shape = [m, n, k]
    del out_shape[axis]
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    max_abs_3d_kernel[(1, )](x, out, axis, M=m, N=n, K=k)
    return out

@triton.jit
def max_abs_4d_kernel(pa, pc, axis: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                      L: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    offs_l = tl.arange(0, L)
    a = tl.load(pa + offs_m[:, None, None, None] * N * K * L + offs_n[None, :, None, None] * K * L
                + offs_k[None, None, :, None] * L + offs_l[None, None, None, :])
    c = tl.reduce(a, axis, max_abs_combine_fn)
    if axis == 0:
        tl.store(pc + (offs_n[:, None, None] * K + offs_k[None, :, None]) * L + offs_l[None, None, :], c)
    elif axis == 1:
        tl.store(pc + (offs_m[:, None, None] * K + offs_k[None, :, None]) * L + offs_l[None, None, :], c)
    elif axis == 2:
        tl.store(pc + (offs_m[:, None, None] * N + offs_n[None, :, None]) * L + offs_l[None, None, :], c)
    else:
        tl.store(pc + (offs_m[:, None, None] * N + offs_n[None, :, None]) * K + offs_k[None, None, :], c)

def max_abs_fn_4d(x: torch.Tensor, axis: int):
    out_shape = list(x.shape)
    del out_shape[axis]
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    max_abs_4d_kernel[(1, )](x, out, axis, M=x.shape[0], N=x.shape[1], K=x.shape[2], L=x.shape[3])
    return out

@triton.jit
def max_2d_kernel(pa, pc, axis: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    a = tl.load(pa + offs_m[:, None] * N + offs_n[None, :])
    c = tl.max(a, axis)
    if axis == 0:
        tl.store(pc + offs_n, c)
    else:
        tl.store(pc + offs_m, c)

@triton.jit
def max_3d_kernel(pa, pc, axis: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    a = tl.load(pa + offs_m[:, None, None] * N * K + offs_n[None, :, None] * K + offs_k[None, None, :])
    c = tl.max(a, axis)
    if axis == 0:
        tl.store(pc + offs_n[:, None] * K + offs_k[None, :], c)
    elif axis == 1:
        tl.store(pc + offs_m[:, None] * K + offs_k[None, :], c)
    else:
        tl.store(pc + offs_m[:, None] * N + offs_n[None, :], c)

def max_nd_fn_2d(x: torch.Tensor, axis: int):
    m, n = x.shape
    out_shape = (n, ) if axis == 0 else (m, )
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    max_2d_kernel[(1, )](x, out, axis, M=m, N=n)
    return out

def max_nd_fn_3d(x: torch.Tensor, axis: int):
    m, n, k = x.shape
    out_shape = [m, n, k]
    del out_shape[axis]
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    max_3d_kernel[(1, )](x, out, axis, M=m, N=n, K=k)
    return out

@triton.jit
def max_full_2d_kernel(pa, pc, M: tl.constexpr, N: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    a = tl.load(pa + offs_m[:, None] * N + offs_n[None, :])
    tl.store(pc, tl.max(a))

@triton.jit
def max_full_3d_kernel(pa, pc, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    a = tl.load(pa + offs_m[:, None, None] * N * K + offs_n[None, :, None] * K + offs_k[None, None, :])
    tl.store(pc, tl.max(a))

def max_full_fn_2d(x: torch.Tensor):
    m, n = x.shape
    out = torch.empty(1, dtype=x.dtype, device=x.device)
    max_full_2d_kernel[(1, )](x, out, M=m, N=n)
    return out[0]

def max_full_fn_3d(x: torch.Tensor):
    m, n, k = x.shape
    out = torch.empty(1, dtype=x.dtype, device=x.device)
    max_full_3d_kernel[(1, )](x, out, M=m, N=n, K=k)
    return out[0]

class TestCustomCombineReduceCase(TestCase):
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_max_abs_1d(self, dtype):
        cpu_x, private_x = make_inputs((256, ), dtype)
        cpu_z = torch.max(torch.abs(cpu_x))
        private_z = max_abs_fn_1d(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=0, rtol=0)

    @parametrize("axis", [0, 1])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_max_abs_2d(self, dtype, axis):
        cpu_x, private_x = make_inputs((32, 64), dtype)
        cpu_z = torch.max(torch.abs(cpu_x), dim=axis).values
        private_z = max_abs_fn_2d(private_x, axis)
        self.assertTrue(torch.equal(cpu_z, private_z.cpu()))

    @parametrize("axis", [0, 1, 2])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_max_abs_3d(self, dtype, axis):
        cpu_x, private_x = make_inputs((8, 16, 32), dtype)
        cpu_z = torch.max(torch.abs(cpu_x), dim=axis).values
        private_z = max_abs_fn_3d(private_x, axis)
        self.assertTrue(torch.equal(cpu_z, private_z.cpu()))

    @parametrize("axis", [0, 1, 2, 3])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_max_abs_4d(self, dtype, axis):
        cpu_x, private_x = make_inputs((4, 8, 16, 32), dtype)
        cpu_z = torch.max(torch.abs(cpu_x), dim=axis).values
        private_z = max_abs_fn_4d(private_x, axis)
        self.assertTrue(torch.equal(cpu_z, private_z.cpu()))
@triton.jit
def sum_f32_combine_fn(x, y):
    return x + y

@triton.jit
def sum_lowp_f32out_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask, other=0.0)
    c = tl.reduce(a.to(tl.float32), 0, sum_f32_combine_fn)
    tl.store(pc + pid, c)

@triton.jit
def sum_fp16_f16out_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask, other=0.0)
    c = tl.reduce(a.to(tl.float32), 0, sum_f32_combine_fn)
    tl.store(pc + pid, c.to(tl.float16))

@triton.jit
def sum_bf16_bf16out_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask, other=0.0)
    c = tl.reduce(a.to(tl.float32), 0, sum_f32_combine_fn)
    tl.store(pc + pid, c.to(tl.bfloat16))

def sum_lowp_out_fn(x: torch.Tensor, out_dtype: torch.dtype):
    num = x.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    out = torch.empty(grid({'BLOCK': 256}), dtype=out_dtype, device=x.device)
    if out_dtype == torch.float16:
        sum_fp16_f16out_kernel[grid](x, out, num, BLOCK=256)
    elif out_dtype == torch.bfloat16:
        sum_bf16_bf16out_kernel[grid](x, out, num, BLOCK=256)
    else:
        sum_lowp_f32out_kernel[grid](x, out, num, BLOCK=256)
    return out.sum()

class TestLowpCastReduceCase(TestCase):
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("shape", [(256, ), (32, 64, 128)])
    def test_sum_lowp_f32out(self, shape, dtype):
        cpu_x, _ = make_inputs(shape, torch.float32)
        cpu_x = cpu_x.to(dtype)
        private_x = cpu_x.to(device)
        cpu_z = torch.sum(cpu_x, dtype=torch.float32)
        private_z = sum_lowp_out_fn(private_x, torch.float32)
        self.assertTrue(torch.allclose(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL))

    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("shape", [(256, ), (32, 64, 128)])
    def test_sum_lowp_lowpout(self, shape, dtype):
        cpu_x, _ = make_inputs(shape, torch.float32)
        cpu_x = cpu_x.to(dtype)
        private_x = cpu_x.to(device)
        cpu_z = torch.sum(cpu_x)
        private_z = sum_lowp_out_fn(private_x, dtype)
        self.assertTrue(torch.allclose(cpu_z, private_z.cpu(), atol=LOWP_ATOL, rtol=LOWP_RTOL))

class TestMaxNDCase(TestCase):
    @parametrize("axis", [0, 1])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_max_2d(self, dtype, axis):
        cpu_x, private_x = make_inputs((32, 64), dtype)
        cpu_z = torch.max(cpu_x, dim=axis).values
        private_z = max_nd_fn_2d(private_x, axis)
        self.assertTrue(torch.equal(cpu_z, private_z.cpu()))

    @parametrize("axis", [0, 1, 2])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_max_3d(self, dtype, axis):
        cpu_x, private_x = make_inputs((8, 16, 32), dtype)
        cpu_z = torch.max(cpu_x, dim=axis).values
        private_z = max_nd_fn_3d(private_x, axis)
        self.assertTrue(torch.equal(cpu_z, private_z.cpu()))

class TestFullReduceCase(TestCase):
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_max_full_2d(self, dtype):
        cpu_x, private_x = make_inputs((32, 64), dtype)
        cpu_z = torch.max(cpu_x)
        private_z = max_full_fn_2d(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=0, rtol=0)

    @parametrize("dtype", [torch.float32, torch.int32])
    def test_max_full_3d(self, dtype):
        cpu_x, private_x = make_inputs((8, 16, 32), dtype)
        cpu_z = torch.max(cpu_x)
        private_z = max_full_fn_3d(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=0, rtol=0)

instantiate_parametrized_tests(TestSumCase)
instantiate_parametrized_tests(TestMaxCase)
instantiate_parametrized_tests(TestMinCase)
instantiate_parametrized_tests(TestCustomCombineReduceCase)
@triton.jit
def argmax_combine_fn(vx, ix, vy, iy):
    take_x = (vx > vy) | ((vx == vy) & (ix < iy))
    return tl.where(take_x, vx, vy), tl.where(take_x, ix, iy)

@triton.jit
def argmin_combine_fn(vx, ix, vy, iy):
    take_x = (vx < vy) | ((vx == vy) & (ix < iy))
    return tl.where(take_x, vx, vy), tl.where(take_x, ix, iy)

@triton.jit
def argmax_1d_kernel(pa, pc, N: tl.constexpr):
    offs = tl.arange(0, N)
    a = tl.load(pa + offs)
    _, c = tl.reduce((a, offs), 0, argmax_combine_fn)
    tl.store(pc, c)

@triton.jit
def argmin_1d_kernel(pa, pc, N: tl.constexpr):
    offs = tl.arange(0, N)
    a = tl.load(pa + offs)
    _, c = tl.reduce((a, offs), 0, argmin_combine_fn)
    tl.store(pc, c)

@triton.jit
def argmax_2d_kernel(pa, pc, axis: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    a = tl.load(pa + offs_m[:, None] * N + offs_n[None, :])
    if axis == 0:
        idx = tl.broadcast_to(offs_m[:, None], (M, N))
        _, c = tl.reduce((a, idx), axis, argmax_combine_fn)
        tl.store(pc + offs_n, c)
    else:
        idx = tl.broadcast_to(offs_n[None, :], (M, N))
        _, c = tl.reduce((a, idx), axis, argmax_combine_fn)
        tl.store(pc + offs_m, c)

@triton.jit
def argmin_2d_kernel(pa, pc, axis: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    a = tl.load(pa + offs_m[:, None] * N + offs_n[None, :])
    if axis == 0:
        idx = tl.broadcast_to(offs_m[:, None], (M, N))
        _, c = tl.reduce((a, idx), axis, argmin_combine_fn)
        tl.store(pc + offs_n, c)
    else:
        idx = tl.broadcast_to(offs_n[None, :], (M, N))
        _, c = tl.reduce((a, idx), axis, argmin_combine_fn)
        tl.store(pc + offs_m, c)

def argmax_fn_1d(x: torch.Tensor, N: int):
    out = torch.empty(1, dtype=torch.int32, device=x.device)
    argmax_1d_kernel[(1, )](x, out, N=N)
    return out[0]

def argmin_fn_1d(x: torch.Tensor, N: int):
    out = torch.empty(1, dtype=torch.int32, device=x.device)
    argmin_1d_kernel[(1, )](x, out, N=N)
    return out[0]

def argmax_fn_2d(x: torch.Tensor, axis: int):
    m, n = x.shape
    out_shape = (n, ) if axis == 0 else (m, )
    out = torch.empty(out_shape, dtype=torch.int32, device=x.device)
    argmax_2d_kernel[(1, )](x, out, axis, M=m, N=n)
    return out

def argmin_fn_2d(x: torch.Tensor, axis: int):
    m, n = x.shape
    out_shape = (n, ) if axis == 0 else (m, )
    out = torch.empty(out_shape, dtype=torch.int32, device=x.device)
    argmin_2d_kernel[(1, )](x, out, axis, M=m, N=n)
    return out

class TestArgMaxCase(TestCase):
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_argmax_1d(self, dtype):
        cpu_x, private_x = make_inputs((256, ), dtype)
        cpu_z = torch.argmax(cpu_x)
        private_z = argmax_fn_1d(private_x, 256)
        self.assertEqual(cpu_z, private_z.cpu().to(torch.int64))

    @parametrize("axis", [0, 1])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_argmax_2d(self, dtype, axis):
        cpu_x, private_x = make_inputs((32, 64), dtype)
        cpu_z = torch.argmax(cpu_x, dim=axis)
        private_z = argmax_fn_2d(private_x, axis)
        self.assertTrue(torch.equal(cpu_z, private_z.cpu().to(torch.int64)))

class TestArgMinCase(TestCase):
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_argmin_1d(self, dtype):
        cpu_x, private_x = make_inputs((256, ), dtype)
        cpu_z = torch.argmin(cpu_x)
        private_z = argmin_fn_1d(private_x, 256)
        self.assertEqual(cpu_z, private_z.cpu().to(torch.int64))

    @parametrize("axis", [0, 1])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_argmin_2d(self, dtype, axis):
        cpu_x, private_x = make_inputs((32, 64), dtype)
        cpu_z = torch.argmin(cpu_x, dim=axis)
        private_z = argmin_fn_2d(private_x, axis)
        self.assertTrue(torch.equal(cpu_z, private_z.cpu().to(torch.int64)))

instantiate_parametrized_tests(TestArgMaxCase)
instantiate_parametrized_tests(TestArgMinCase)

@triton.jit
def sum_2d_dim0_kernel(pa, pc, M: tl.constexpr, N: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    a = tl.load(pa + offs_m[:, None] * N + offs_n[None, :])
    c = tl.sum(a, axis=0)
    tl.store(pc + offs_n, c)

class TestSum2DDim0Case(TestCase):
    @parametrize("shape", [(16, 64), (32, 128)])
    def test_sum_dim0(self, shape):
        m, n = shape
        cpu_x, private_x = make_inputs(shape, torch.float32)
        out = torch.empty(n, dtype=torch.float32, device=private_x.device)
        sum_2d_dim0_kernel[(1, )](private_x, out, M=m, N=n)
        cpu_z = torch.sum(cpu_x, dim=0)
        self.assertTrue(torch.allclose(cpu_z, out.cpu(), atol=SUM_ATOL, rtol=SUM_RTOL))

instantiate_parametrized_tests(TestSum2DDim0Case)
instantiate_parametrized_tests(TestLowpCastReduceCase)
instantiate_parametrized_tests(TestMaxNDCase)
instantiate_parametrized_tests(TestFullReduceCase)
