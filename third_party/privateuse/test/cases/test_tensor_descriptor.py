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
def copy_kernel(x_ptr, y_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    in_desc = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])
    out_desc = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])

    m_offset = tl.program_id(0) * BLOCK_M
    n_offset = tl.program_id(1) * BLOCK_N
    out_desc.store([m_offset, n_offset], in_desc.load([m_offset, n_offset]))

def copy_with_descriptor(x: torch.Tensor):
    M, N = x.shape
    BLOCK_M = 32
    BLOCK_N = 32
    out = torch.empty_like(x)

    grid = (M // BLOCK_M, N // BLOCK_N)
    copy_kernel[grid](x, out, M, N, BLOCK_M, BLOCK_N)
    return out

@triton.jit
def transpose_kernel(x_ptr, y_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    in_desc = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])
    out_desc = tl.make_tensor_descriptor(y_ptr, shape=[N, M], strides=[M, 1], block_shape=[BLOCK_N, BLOCK_M])

    m_offset = tl.program_id(0) * BLOCK_M
    n_offset = tl.program_id(1) * BLOCK_N
    block_t = tl.trans(in_desc.load([m_offset, n_offset]))
    out_desc.store([n_offset, m_offset], block_t)

def transpose_with_descriptor(x: torch.Tensor):
    M, N = x.shape
    BLOCK_M = 32
    BLOCK_N = 32
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)

    grid = (M // BLOCK_M, N // BLOCK_N)
    transpose_kernel[grid](x, out, M, N, BLOCK_M, BLOCK_N)
    return out

@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])
    b_desc = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])
    c_desc = tl.make_tensor_descriptor(z_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])

    m_offset = tl.program_id(0) * BLOCK_M
    n_offset = tl.program_id(1) * BLOCK_N
    c_desc.store([m_offset, n_offset], a_desc.load([m_offset, n_offset]) + b_desc.load([m_offset, n_offset]))

def add_with_descriptor(x: torch.Tensor, y: torch.Tensor):
    M, N = x.shape
    BLOCK_M = 32
    BLOCK_N = 32
    out = torch.empty_like(x)

    grid = (M // BLOCK_M, N // BLOCK_N)
    add_kernel[grid](x, y, out, M, N, BLOCK_M, BLOCK_N)
    return out

@triton.jit
def scale_kernel(x_ptr, y_ptr, scale, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    in_desc = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])
    out_desc = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])

    m_offset = tl.program_id(0) * BLOCK_M
    n_offset = tl.program_id(1) * BLOCK_N
    out_desc.store([m_offset, n_offset], in_desc.load([m_offset, n_offset]) * scale)

def scale_with_descriptor(x: torch.Tensor, scale: float):
    M, N = x.shape
    BLOCK_M = 32
    BLOCK_N = 32
    out = torch.empty_like(x)

    grid = (M // BLOCK_M, N // BLOCK_N)
    scale_kernel[grid](x, out, scale, M, N, BLOCK_M, BLOCK_N)
    return out

class TestTensorDescriptorCopy(TestCase):
    @parametrize("shape", [(64, 64), (128, 128), (256, 256)])
    @parametrize("dtype", [torch.float32])
    def test_copy(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_z = cpu_x.clone()
        private_z = copy_with_descriptor(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestTensorDescriptorTranspose(TestCase):
    @parametrize("shape", [(64, 64), (128, 128)])
    @parametrize("dtype", [torch.float32])
    def test_transpose(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_z = cpu_x.t()
        private_z = transpose_with_descriptor(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestTensorDescriptorAdd(TestCase):
    @parametrize("shape", [(64, 64), (128, 128)])
    @parametrize("dtype", [torch.float32])
    def test_add(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_y, private_y = make_inputs(shape, dtype)
        cpu_z = torch.add(cpu_x, cpu_y)
        private_z = add_with_descriptor(private_x, private_y)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestTensorDescriptorScale(TestCase):
    @parametrize("shape", [(64, 64), (128, 128)])
    @parametrize("dtype", [torch.float32])
    def test_scale(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        scale = 2.5
        cpu_z = cpu_x * scale
        private_z = scale_with_descriptor(private_x, scale)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

instantiate_parametrized_tests(TestTensorDescriptorCopy)
instantiate_parametrized_tests(TestTensorDescriptorTranspose)
instantiate_parametrized_tests(TestTensorDescriptorAdd)
instantiate_parametrized_tests(TestTensorDescriptorScale)
