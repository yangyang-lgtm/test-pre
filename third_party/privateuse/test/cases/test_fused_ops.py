import torch
import triton
import triton.language as tl
from ..conftest import F32_ATOL, F32_RTOL, make_inputs, device, ACC_ATOL, ACC_RTOL, MATH_ATOL, MATH_RTOL, COMP_ATOL, COMP_RTOL
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)

@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_start = input_ptr + row_idx * n_cols
    row = tl.load(row_start + col_offsets, mask=mask, other=float('-inf'))

    row_max = tl.max(row, axis=0)
    row = row - row_max
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_start = output_ptr + row_idx * n_cols
    tl.store(output_start + col_offsets, softmax_output, mask=mask)

@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x_start = x_ptr + row_idx * n_cols
    x = tl.load(x_start + col_offsets, mask=mask, other=0.0)

    mean = tl.sum(x, axis=0) / n_cols
    x_centered = x - mean
    variance = tl.sum(x_centered * x_centered, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(variance + eps)
    x_norm = x_centered * rstd

    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    output = x_norm * weight + bias

    output_start = output_ptr + row_idx * n_cols
    tl.store(output_start + col_offsets, output, mask=mask)

@triton.jit
def rms_norm_kernel(
    x_ptr, weight_ptr, output_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x_start = x_ptr + row_idx * n_cols
    x = tl.load(x_start + col_offsets, mask=mask, other=0.0)

    rms = tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    x_norm = x / rms

    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    output = x_norm * weight

    output_start = output_ptr + row_idx * n_cols
    tl.store(output_start + col_offsets, output, mask=mask)

def softmax(x: torch.Tensor):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    out = torch.empty_like(x)
    softmax_kernel[(n_rows,)](x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out

def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    out = torch.empty_like(x)
    layer_norm_kernel[(n_rows,)](x, weight, bias, out, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    out = torch.empty_like(x)
    rms_norm_kernel[(n_rows,)](x, weight, out, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out

class TestSoftmaxCase(TestCase):
    @parametrize("shape", [(32, 64), (64, 128), (128, 256)])
    @parametrize("dtype", [torch.float32])
    def test_softmax(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_z = torch.softmax(cpu_x, dim=-1)
        private_z = softmax(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=MATH_ATOL, rtol=MATH_RTOL)

class TestLayerNormCase(TestCase):
    @parametrize("shape", [(32, 64), (64, 128), (128, 256)])
    @parametrize("dtype", [torch.float32])
    def test_layer_norm(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        n_cols = shape[-1]
        cpu_weight = torch.ones(n_cols, dtype=dtype)
        cpu_bias = torch.zeros(n_cols, dtype=dtype)
        private_weight = cpu_weight.to(device)
        private_bias = cpu_bias.to(device)
        cpu_z = torch.nn.functional.layer_norm(cpu_x, [n_cols], cpu_weight, cpu_bias)
        private_z = layer_norm(private_x, private_weight, private_bias)
        self.assertEqual(cpu_z, private_z.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)

class TestRMSNormCase(TestCase):
    @parametrize("shape", [(32, 64), (64, 128), (128, 256)])
    @parametrize("dtype", [torch.float32])
    def test_rms_norm(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        n_cols = shape[-1]
        cpu_weight = torch.ones(n_cols, dtype=dtype)
        private_weight = cpu_weight.to(device)
        cpu_z = cpu_x * torch.rsqrt(torch.mean(cpu_x ** 2, dim=-1, keepdim=True) + 1e-5) * cpu_weight
        private_z = rms_norm(private_x, private_weight)
        self.assertEqual(cpu_z, private_z.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)

instantiate_parametrized_tests(TestSoftmaxCase)
instantiate_parametrized_tests(TestLayerNormCase)
instantiate_parametrized_tests(TestRMSNormCase)
