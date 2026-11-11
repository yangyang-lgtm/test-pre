import torch
import triton
import triton.language as tl
from ..conftest import make_inputs, device, F32_ATOL, F32_RTOL, F16_ATOL, F16_RTOL
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)

@triton.jit
def sub_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    c = a - b
    tl.store(pc + offset, c, mask=mask)

@triton.jit
def mul_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    c = a * b
    tl.store(pc + offset, c, mask=mask)

@triton.jit
def div_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    c = a / b
    tl.store(pc + offset, c, mask=mask)

def sub(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    sub_kernel[grid](x, y, out, num, BLOCK=256)
    return out

def mul(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    mul_kernel[grid](x, y, out, num, BLOCK=256)
    return out

def div(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    div_kernel[grid](x, y, out, num, BLOCK=256)
    return out

class TestSubCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_sub(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_y, private_y = make_inputs(shape, dtype)
        cpu_z = torch.sub(cpu_x, cpu_y)
        private_z = sub(private_x, private_y)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestMulCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_mul(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_y, private_y = make_inputs(shape, dtype)
        cpu_z = torch.mul(cpu_x, cpu_y)
        private_z = mul(private_x, private_y)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestDivCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32, torch.float16])
    def test_div(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_y, private_y = make_inputs(shape, dtype)
        cpu_y = cpu_y + 1.0
        private_y = private_y + 1.0
        cpu_z = torch.div(cpu_x, cpu_y)
        private_z = div(private_x, private_y)
        if dtype == torch.float16:
            self.assertEqual(cpu_z.float(), private_z.float().cpu(), atol=F16_ATOL, rtol=F16_RTOL)
        else:
            self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

instantiate_parametrized_tests(TestSubCase)
instantiate_parametrized_tests(TestMulCase)
instantiate_parametrized_tests(TestDivCase)
