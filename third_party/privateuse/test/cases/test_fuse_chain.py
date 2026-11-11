import re

import torch
import triton
import triton.language as tl
from ..conftest import make_inputs, device, F32_ATOL, F32_RTOL, ACC_ATOL, ACC_RTOL, COMP_ATOL, COMP_RTOL
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)

@triton.jit
def fuse_chain_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    a = x * 2.0 - y
    b = a / (y * y + 1.0)
    c = tl.exp(-b)
    d = c * tl.rsqrt(x * x + 1.0)
    o = d * a + c
    tl.store(o_ptr + offs, o, mask=mask)


@triton.jit
def exp_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, tl.exp(x), mask=mask)


@triton.jit
def div_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, x / y, mask=mask)


@triton.jit
def rsqrt_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, tl.rsqrt(x), mask=mask)


@triton.jit
def mixed_width_kernel(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    c = tl.exp(a.to(tl.float32))
    tl.store(o_ptr + offs, b * tl.rsqrt(c), mask=mask)


def fuse_chain(x, y):
    out = torch.empty_like(x)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']), )
    fuse_chain_kernel[grid](x, y, out, n, BLOCK=64)
    return out


def run_unary(kernel, x):
    out = torch.empty_like(x)
    n = out.numel()
    kernel[(triton.cdiv(n, 64),)](x, out, n, 64)
    return out


def run_binary(kernel, x, y):
    out = torch.empty_like(x)
    n = out.numel()
    kernel[(triton.cdiv(n, 64),)](x, y, out, n, 64)
    return out


def run_mixed_width(a, b):
    out = torch.empty(b.shape, dtype=torch.float32, device=device)
    n = out.numel()
    mixed_width_kernel[(triton.cdiv(n, 64),)](a, b, out, n, 64)
    return out


def torch_fuse_chain(x, y):
    a = x * 2.0 - y
    b = a / (y * y + 1.0)
    c = torch.exp(-b)
    d = c * torch.rsqrt(x * x + 1.0)
    return d * a + c


class TestFuseChainCase(TestCase):

    @parametrize("shape", [(32, 32), (32, 64, 128), (37, 53), (13, 29, 71)])
    def test_fuse_chain(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs(shape, torch.float32)

        cpu_z = torch_fuse_chain(cpu_x, cpu_y)
        private_z = fuse_chain(private_x, private_y).cpu()

        self.assertTrue(torch.allclose(private_z, cpu_z, atol=ACC_ATOL, rtol=ACC_RTOL))

    @parametrize("shape", [(37, 53), (5, 7, 11)])
    def test_fuse_chain_groups(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs(shape, torch.float32)

        n = private_x.numel()
        cpp = fuse_chain_kernel.warmup(private_x, private_y, torch.empty_like(private_x), n, 64,
                                       grid=(triton.cdiv(n, 64), )).asm['puc']
        groups = set(re.findall(r"outlined_linalg_generic_([a-z_0-9]+)", cpp))

        self.assertTrue(any("subf" in name and "mulf" in name for name in groups))
        self.assertTrue(any("divf" in name and "exp" in name for name in groups))
        self.assertTrue(any("rsqrt" in name and "mulf" in name for name in groups))

        cpu_z = torch_fuse_chain(cpu_x, cpu_y)
        private_z = fuse_chain(private_x, private_y).cpu()
        self.assertTrue(torch.allclose(private_z, cpu_z, atol=ACC_ATOL, rtol=ACC_RTOL))

    def test_exp_precision(self):
        cpu_x = torch.linspace(-87.0, 88.0, 37 * 53)
        out = run_unary(exp_kernel, cpu_x.to(device)).cpu()
        self.assertTrue(torch.allclose(out, torch.exp(cpu_x), atol=F32_ATOL, rtol=F32_RTOL))

    def test_div_precision(self):
        cpu_x, private_x = make_inputs((37, 53), torch.float32)
        cpu_y, private_y = make_inputs((37, 53), torch.float32)
        cpu_y = torch.where(cpu_y.abs() < 0.1, cpu_y + 0.5, cpu_y)
        private_y = cpu_y.to(device)

        out = run_binary(div_kernel, private_x, private_y).cpu()
        self.assertTrue(torch.allclose(out, cpu_x / cpu_y, atol=F32_ATOL, rtol=F32_RTOL))

    def test_rsqrt_precision(self):
        torch.manual_seed(0)
        exponents = torch.empty(1 << 16, dtype=torch.float64).uniform_(-85.0, 85.0)
        x = torch.exp(exponents).float().to(device)

        out = run_unary(rsqrt_kernel, x).cpu()

        self.assertTrue(torch.allclose(out, torch.rsqrt(x.cpu()), atol=F32_ATOL, rtol=F32_RTOL))

    @parametrize("shape", [(37, 53), (16, 32), (5, 7, 11)])
    def test_mixed_width_exp_rsqrt(self, shape):
        torch.manual_seed(0)
        a16 = (torch.rand(shape, dtype=torch.float32) * 8.0 - 4.0).to(torch.float16)
        b = torch.randn(shape, dtype=torch.float32)

        got = run_mixed_width(a16.to(device), b.to(device)).cpu()
        ref = b * torch.rsqrt(torch.exp(a16.to(torch.float32)))

        self.assertTrue(torch.allclose(got, ref, atol=COMP_ATOL, rtol=COMP_RTOL))

    def test_special_values(self):
        values = torch.tensor([0.0, -0.0, 1e-40, 1.0, 3.4e38, float('inf'), float('-inf'), float('nan'), -1.0],
                              device=device)
        for kernel, ref_fn in ((exp_kernel, torch.exp), (rsqrt_kernel, torch.rsqrt)):
            out = run_unary(kernel, values).cpu()
            ref = ref_fn(values.cpu())
            self.assertTrue(torch.allclose(out, ref, atol=F32_ATOL, rtol=F32_RTOL, equal_nan=True))


instantiate_parametrized_tests(TestFuseChainCase)
