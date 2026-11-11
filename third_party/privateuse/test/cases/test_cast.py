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
def cast_kernel(pa, pb, num, BLOCK: tl.constexpr, OUT_DTYPE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = a.to(OUT_DTYPE)
    tl.store(pb + offset, b, mask=mask)

def cast_fn(x: torch.Tensor, out_dtype: torch.dtype):
    tl_dtype = {
        torch.float32: tl.float32,
        torch.int32: tl.int32,
        torch.float64: tl.float64,
        torch.int64: tl.int64,
        torch.float16: tl.float16,
        torch.int16: tl.int16,
        torch.int8: tl.int8,
        torch.bool: tl.int1,
    }[out_dtype]

    out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    cast_kernel[grid](x, out, num, BLOCK=256, OUT_DTYPE=tl_dtype)
    return out

class TestCastFloatToInt(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    def test_float_to_int(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_z = cpu_x.to(torch.int32)
        private_z = cast_fn(private_x, torch.int32)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestCastIntToFloat(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    def test_int_to_float(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.int32)
        cpu_z = cpu_x.to(torch.float32)
        private_z = cast_fn(private_x, torch.float32)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestCastFloatToHalf(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    def test_float_to_half(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_z = cpu_x.to(torch.float16)
        private_z = cast_fn(private_x, torch.float16)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestCastFloatToInt16(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    def test_float_to_int16(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_z = cpu_x.to(torch.int16)
        private_z = cast_fn(private_x, torch.int16)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestCastFloatToInt8(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    def test_float_to_int8(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_z = cpu_x.to(torch.int8)
        private_z = cast_fn(private_x, torch.int8)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestCastFloatToDouble(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    def test_float_to_double(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_z = cpu_x.to(torch.float64)
        private_z = cast_fn(private_x, torch.float64)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestCastIntToInt16(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    def test_int_to_int16(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.int32)
        cpu_z = cpu_x.to(torch.int16)
        private_z = cast_fn(private_x, torch.int16)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestCastIntToInt8(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    def test_int_to_int8(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.int32)
        cpu_z = cpu_x.to(torch.int8)
        private_z = cast_fn(private_x, torch.int8)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestCastDoubleToFloat(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    def test_double_to_float(self, shape):
        cpu_x = torch.randn(shape, dtype=torch.float64)
        private_x = cpu_x.to(device)
        cpu_z = cpu_x.to(torch.float32)
        private_z = cast_fn(private_x, torch.float32)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestCastCrossWidth(TestCase):
    @parametrize("in_dtype,out_dtype", [
        (torch.float32, torch.float64),
        (torch.float64, torch.float32),
        (torch.float32, torch.float16),
        (torch.float16, torch.float32),
        (torch.int32, torch.int64),
        (torch.int64, torch.int32),
        (torch.int32, torch.int8),
        (torch.int8, torch.int32),
    ])
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    def test_cross_width(self, in_dtype, out_dtype, shape):
        if in_dtype.is_floating_point:
            cpu_x = torch.randn(shape, dtype=in_dtype)
        else:
            cpu_x = torch.randint(-100, 100, shape, dtype=in_dtype)
        cpu_z = cpu_x.to(out_dtype)
        private_z = cast_fn(cpu_x.to(device), out_dtype)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

instantiate_parametrized_tests(TestCastFloatToInt)
instantiate_parametrized_tests(TestCastIntToFloat)
instantiate_parametrized_tests(TestCastCrossWidth)
instantiate_parametrized_tests(TestCastFloatToHalf)
instantiate_parametrized_tests(TestCastFloatToInt16)
instantiate_parametrized_tests(TestCastFloatToInt8)
instantiate_parametrized_tests(TestCastFloatToDouble)
instantiate_parametrized_tests(TestCastIntToInt16)
instantiate_parametrized_tests(TestCastIntToInt8)
instantiate_parametrized_tests(TestCastDoubleToFloat)
