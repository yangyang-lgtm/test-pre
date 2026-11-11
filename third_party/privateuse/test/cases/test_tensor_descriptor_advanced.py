import torch
import triton
import triton.language as tl
from ..conftest import F32_ATOL, F32_RTOL, make_inputs, device, ACC_ATOL, ACC_RTOL, SUM_ATOL, SUM_RTOL
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        a = a_desc.load([pid_m * BLOCK_M, k_start])
        b = b_desc.load([k_start, pid_n * BLOCK_N])
        acc += tl.dot(a, b)

    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc)

def matmul_with_descriptor(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    grid = (M // BLOCK_M, N // BLOCK_N)
    matmul_kernel[grid](a, b, c, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
    return c

@triton.jit
def row_sum_kernel(x_ptr, y_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    in_desc = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])
    out_desc = tl.make_tensor_descriptor(y_ptr, shape=[M], strides=[1], block_shape=[BLOCK_M])

    pid_m = tl.program_id(0)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        acc += tl.sum(in_desc.load([pid_m * BLOCK_M, n_start]), axis=1)

    out_desc.store([pid_m * BLOCK_M], acc)

def row_sum_with_descriptor(x: torch.Tensor):
    M, N = x.shape
    out = torch.empty((M,), dtype=x.dtype, device=x.device)

    BLOCK_M = 32
    BLOCK_N = 32

    grid = (M // BLOCK_M,)
    row_sum_kernel[grid](x, out, M, N, BLOCK_M, BLOCK_N)
    return out

@triton.jit
def col_sum_kernel(x_ptr, y_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    in_desc = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])
    out_desc = tl.make_tensor_descriptor(y_ptr, shape=[N], strides=[1], block_shape=[BLOCK_N])

    pid_n = tl.program_id(0)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        acc += tl.sum(in_desc.load([m_start, pid_n * BLOCK_N]), axis=0)

    out_desc.store([pid_n * BLOCK_N], acc)

def col_sum_with_descriptor(x: torch.Tensor):
    M, N = x.shape
    out = torch.empty((N,), dtype=x.dtype, device=x.device)

    BLOCK_M = 32
    BLOCK_N = 32

    grid = (N // BLOCK_N,)
    col_sum_kernel[grid](x, out, M, N, BLOCK_M, BLOCK_N)
    return out

@triton.jit
def accumulate_kernel(x_ptr, y_ptr, M, N, scale, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    in_desc = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])
    out_desc = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])

    m_offset = tl.program_id(0) * BLOCK_M
    n_offset = tl.program_id(1) * BLOCK_N
    acc = out_desc.load([m_offset, n_offset]) + in_desc.load([m_offset, n_offset]) * scale
    out_desc.store([m_offset, n_offset], acc)

def accumulate_with_descriptor(x: torch.Tensor, scale: float):
    M, N = x.shape
    out = torch.zeros_like(x)

    BLOCK_M = 32
    BLOCK_N = 32

    grid = (M // BLOCK_M, N // BLOCK_N)
    accumulate_kernel[grid](x, out, M, N, scale, BLOCK_M, BLOCK_N)
    return out

@triton.jit
def multi_pass_kernel(x_ptr, y_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    in_desc = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])
    out_desc = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])

    m_offset = tl.program_id(0) * BLOCK_M
    n_offset = tl.program_id(1) * BLOCK_N
    block = in_desc.load([m_offset, n_offset]) * 2.0 + 1.0
    out_desc.store([m_offset, n_offset], block)

def multi_pass_with_descriptor(x: torch.Tensor):
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK_M = 32
    BLOCK_N = 32

    grid = (M // BLOCK_M, N // BLOCK_N)
    multi_pass_kernel[grid](x, out, M, N, BLOCK_M, BLOCK_N)
    return out

class TestTensorDescriptorMatmul(TestCase):
    @parametrize("shape", [(64, 64), (128, 128)])
    @parametrize("dtype", [torch.float32])
    def test_matmul(self, shape, dtype):
        M, K = shape
        K, N = K, 64
        cpu_a, private_a = make_inputs((M, K), dtype)
        cpu_b, private_b = make_inputs((K, N), dtype)
        cpu_c = torch.matmul(cpu_a, cpu_b)
        private_c = matmul_with_descriptor(private_a, private_b)
        self.assertEqual(cpu_c, private_c.cpu(), atol=SUM_ATOL, rtol=SUM_RTOL)

class TestTensorDescriptorRowSum(TestCase):
    @parametrize("shape", [(64, 128), (128, 256)])
    @parametrize("dtype", [torch.float32])
    def test_row_sum(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_z = torch.sum(cpu_x, dim=1)
        private_z = row_sum_with_descriptor(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=SUM_ATOL, rtol=SUM_RTOL)

class TestTensorDescriptorColSum(TestCase):
    @parametrize("shape", [(128, 64), (256, 128)])
    @parametrize("dtype", [torch.float32])
    def test_col_sum(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_z = torch.sum(cpu_x, dim=0)
        private_z = col_sum_with_descriptor(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=SUM_ATOL, rtol=SUM_RTOL)

class TestTensorDescriptorAccumulate(TestCase):
    @parametrize("shape", [(64, 64), (128, 128)])
    @parametrize("dtype", [torch.float32])
    def test_accumulate(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        scale = 0.5
        cpu_z = cpu_x * scale
        private_z = accumulate_with_descriptor(private_x, scale)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestTensorDescriptorMultiPass(TestCase):
    @parametrize("shape", [(64, 64), (128, 128)])
    @parametrize("dtype", [torch.float32])
    def test_multi_pass(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_z = (cpu_x * 2.0) + 1.0
        private_z = multi_pass_with_descriptor(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

instantiate_parametrized_tests(TestTensorDescriptorMatmul)
instantiate_parametrized_tests(TestTensorDescriptorRowSum)
instantiate_parametrized_tests(TestTensorDescriptorColSum)
instantiate_parametrized_tests(TestTensorDescriptorAccumulate)
instantiate_parametrized_tests(TestTensorDescriptorMultiPass)
