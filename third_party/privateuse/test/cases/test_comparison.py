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
def eq_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    c = a == b
    tl.store(pc + offset, c, mask=mask)

@triton.jit
def ne_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    c = a != b
    tl.store(pc + offset, c, mask=mask)

@triton.jit
def lt_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    c = a < b
    tl.store(pc + offset, c, mask=mask)

@triton.jit
def gt_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    c = a > b
    tl.store(pc + offset, c, mask=mask)

@triton.jit
def le_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    c = a <= b
    tl.store(pc + offset, c, mask=mask)

@triton.jit
def ge_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    c = a >= b
    tl.store(pc + offset, c, mask=mask)

def eq(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty(x.shape, dtype=torch.bool, device=x.device)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    eq_kernel[grid](x, y, out, num, BLOCK=256)
    return out

def ne(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty(x.shape, dtype=torch.bool, device=x.device)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    ne_kernel[grid](x, y, out, num, BLOCK=256)
    return out

def lt(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty(x.shape, dtype=torch.bool, device=x.device)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    lt_kernel[grid](x, y, out, num, BLOCK=256)
    return out

def gt(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty(x.shape, dtype=torch.bool, device=x.device)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    gt_kernel[grid](x, y, out, num, BLOCK=256)
    return out

def le(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty(x.shape, dtype=torch.bool, device=x.device)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    le_kernel[grid](x, y, out, num, BLOCK=256)
    return out

def ge(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty(x.shape, dtype=torch.bool, device=x.device)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    ge_kernel[grid](x, y, out, num, BLOCK=256)
    return out

class TestEqCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_eq(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_y, private_y = make_inputs(shape, dtype)
        cpu_z = torch.eq(cpu_x, cpu_y)
        private_z = eq(private_x, private_y)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestNeCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_ne(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_y, private_y = make_inputs(shape, dtype)
        cpu_z = torch.ne(cpu_x, cpu_y)
        private_z = ne(private_x, private_y)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestLtCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_lt(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_y, private_y = make_inputs(shape, dtype)
        cpu_z = torch.lt(cpu_x, cpu_y)
        private_z = lt(private_x, private_y)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestGtCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_gt(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_y, private_y = make_inputs(shape, dtype)
        cpu_z = torch.gt(cpu_x, cpu_y)
        private_z = gt(private_x, private_y)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestLeCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_le(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_y, private_y = make_inputs(shape, dtype)
        cpu_z = torch.le(cpu_x, cpu_y)
        private_z = le(private_x, private_y)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

class TestGeCase(TestCase):
    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_ge(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_y, private_y = make_inputs(shape, dtype)
        cpu_z = torch.ge(cpu_x, cpu_y)
        private_z = ge(private_x, private_y)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

@triton.jit
def bool_copy_kernel(px, py, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    x = tl.load(px + offset, mask=mask, other=0)
    tl.store(py + offset, x, mask=mask)

def bool_copy(x: torch.Tensor):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    bool_copy_kernel[grid](x, out, num, BLOCK=256)
    return out

class TestSameWidthBitcastStoreRewrite(TestCase):
    def test_bool_copy_through_bitcast(self):
        cpu_x, private_x = make_inputs((256,), torch.float32)
        cpu_mask = cpu_x > 0
        private_mask = private_x > 0
        self.assertEqual(cpu_mask, bool_copy(private_mask).cpu())


instantiate_parametrized_tests(TestEqCase)
instantiate_parametrized_tests(TestNeCase)
instantiate_parametrized_tests(TestLtCase)
instantiate_parametrized_tests(TestGtCase)
instantiate_parametrized_tests(TestLeCase)
instantiate_parametrized_tests(TestGeCase)
