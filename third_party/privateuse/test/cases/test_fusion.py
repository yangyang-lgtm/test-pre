import re

import torch
import triton
import triton.language as tl
from ..conftest import F32_ATOL, F32_RTOL, make_inputs, device, ACC_ATOL, ACC_RTOL, SUM_ATOL, SUM_RTOL
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)

# Note: every kernel stores its chain inputs again at the end. `t = a + b` is
# then read by a second consumer, which is exactly what makes the fusion refuse
# (see test_chain2); the three-op kernels below keep their intermediates private
# so the fusion actually fires.

def fusion_ops(compiled):
    """How many fused generics the fusion pass left in the kernel.

    Every linalg op carries a single-entry `pumem.fusion_recipe` after
    fuse-and-bufferize, so only recipes with two or more origin entries count
    as real chain fusions.
    """
    recipe = re.compile(r"pumem\.fusion_recipe = \[\{[^\n]*\}, \{")
    return len(recipe.findall(compiled.asm["bufferize"]))


def fused_ops(compiled):
    """How many generics are real chain fusions."""
    return fusion_ops(compiled)


def has_reduce_op(compiled):
    """Reductions are lowered to `linalg.generic`, so `linalg.reduce` must be gone."""
    return "linalg.reduce" in compiled.asm["privateuseir"]

# --- pointwise + pointwise (chain of 2) -----------------------------------

@triton.jit
def chain2_kernel(pa, pb, pc, pkeep, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask, other=0.0)
    b = tl.load(pb + offset, mask=mask, other=0.0)
    t = a + b
    c = t * t
    tl.store(pc + offset, c, mask=mask)
    tl.store(pkeep + offset, a + b, mask=mask)

def chain2(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty_like(x)
    keep = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    k = chain2_kernel[grid](x, y, out, keep, num, BLOCK=256)
    return out, keep, k

# --- pointwise x3 (chain of 3, no reduction) ------------------------------

@triton.jit
def chain3_ew_kernel(pa, pb, pc, pkeep, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask, other=0.0)
    b = tl.load(pb + offset, mask=mask, other=0.0)
    t1 = a + b
    t2 = t1 * t1
    c = t2 - a
    tl.store(pc + offset, c, mask=mask)
    tl.store(pkeep + offset, a + b, mask=mask)

def chain3_ew(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty_like(x)
    keep = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    k = chain3_ew_kernel[grid](x, y, out, keep, num, BLOCK=256)
    return out, keep, k

# --- pointwise + all-dimension reduce (chain of 2) ------------------------

@triton.jit
def pwise_reduce_kernel(pa, pb, pc, pkeepa, pkeepb, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask, other=0.0)
    b = tl.load(pb + offset, mask=mask, other=0.0)
    t = a * b
    c = tl.sum(t, axis=0)
    tl.store(pc + pid, c)
    tl.store(pkeepa + offset, a, mask=mask)
    tl.store(pkeepb + offset, b, mask=mask)

def pwise_reduce(x: torch.Tensor, y: torch.Tensor, block: int = 256):
    n = x.numel()
    out = torch.empty((triton.cdiv(n, block), ), dtype=x.dtype, device=x.device)
    keep = torch.empty_like(x)
    grid = (out.numel(), )
    k = pwise_reduce_kernel[grid](x, y, out, keep, keep, n, BLOCK=block)
    return out, keep, k

# --- pointwise x2 + all-dimension reduce (chain of 3) ---------------------

@triton.jit
def chain3_reduce_kernel(pa, pc, pkeep, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask, other=0.0)
    t1 = a + a
    t2 = t1 * t1
    c = tl.sum(t2, axis=0)
    tl.store(pc + pid, c)
    tl.store(pkeep + offset, a, mask=mask)

def chain3_reduce(x: torch.Tensor, block: int = 256):
    n = x.numel()
    out = torch.empty((triton.cdiv(n, block), ), dtype=x.dtype, device=x.device)
    keep = torch.empty_like(x)
    grid = (out.numel(), )
    k = chain3_reduce_kernel[grid](x, out, keep, n, BLOCK=block)
    return out, keep, k

# --- intermediate with a second consumer ---------------------------------
# `t` feeds the reduce and an elementwise op, so it is live beyond the reduce:
# the fusion must be refused and the result must still be correct.

@triton.jit
def two_consumers_kernel(pa, pb, psum, pplus, pkeep, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask, other=0.0)
    b = tl.load(pb + offset, mask=mask, other=0.0)
    t = a * b
    s = tl.sum(t, axis=0)
    p = t + 1.0
    tl.store(psum + pid, s)
    tl.store(pplus + offset, p, mask=mask)
    tl.store(pkeep + offset, a, mask=mask)

def two_consumers(x: torch.Tensor, y: torch.Tensor, block: int = 256):
    n = x.numel()
    part = torch.empty((triton.cdiv(n, block), ), dtype=x.dtype, device=x.device)
    plus = torch.empty_like(x)
    keep = torch.empty_like(x)
    grid = (part.numel(), )
    k = two_consumers_kernel[grid](x, y, part, plus, keep, n, BLOCK=block)
    return part, plus, keep, k

def block_sum(cpu_x: torch.Tensor, block: int):
    n = cpu_x.numel()
    pad = (-n) % block
    if pad:
        cpu_x = torch.cat([cpu_x, torch.zeros(pad, dtype=cpu_x.dtype)])
    return cpu_x.reshape(-1, block).sum(dim=1)

# --- pointwise + 2D reduce (reduce over dim 0 / dim 1) --------------------
# M and N stay constexpr so the reduce keeps a static shape.

@triton.jit
def pwise_reduce_2d_dim0_kernel(pa, pb, pc, pka, pkb, M: tl.constexpr, N: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    idx = offs_m[:, None] * N + offs_n[None, :]
    a = tl.load(pa + idx)
    b = tl.load(pb + idx)
    t = a * b
    s = tl.sum(t, axis=0)
    tl.store(pc + offs_n, s)
    tl.store(pka + idx, a)
    tl.store(pkb + idx, b)

def pwise_reduce_2d_dim0(x: torch.Tensor, y: torch.Tensor, m: int, n: int):
    out = torch.empty((n, ), dtype=x.dtype, device=x.device)
    keep_a = torch.empty_like(x)
    keep_b = torch.empty_like(y)
    k = pwise_reduce_2d_dim0_kernel[(1, )](x, y, out, keep_a, keep_b, M=m, N=n)
    return out, k

@triton.jit
def pwise_reduce_2d_dim1_kernel(pa, pb, pc, pka, pkb, M: tl.constexpr, N: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    idx = offs_m[:, None] * N + offs_n[None, :]
    a = tl.load(pa + idx)
    b = tl.load(pb + idx)
    t = a * b
    s = tl.sum(t, axis=1)
    tl.store(pc + offs_m, s)
    tl.store(pka + idx, a)
    tl.store(pkb + idx, b)

def pwise_reduce_2d_dim1(x: torch.Tensor, y: torch.Tensor, m: int, n: int):
    out = torch.empty((m, ), dtype=x.dtype, device=x.device)
    keep_a = torch.empty_like(x)
    keep_b = torch.empty_like(y)
    k = pwise_reduce_2d_dim1_kernel[(1, )](x, y, out, keep_a, keep_b, M=m, N=n)
    return out, k


class TestPointwiseFusion(TestCase):
    @parametrize("shape", [(256, ), (32, 64, 128)])
    def test_chain2(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs(shape, torch.float32)
        cpu_z = (cpu_x + cpu_y) ** 2
        private_z, private_keep, k = chain2(private_x, private_y)
        # `t = a + b` is stored as well, so it escapes the chain and the chain
        # fusion must be refused; both ops still get their own container.
        self.assertEqual(fused_ops(k), 0)
        self.assertFalse(has_reduce_op(k))
        self.assertEqual(cpu_z, private_z.cpu(), atol=SUM_ATOL, rtol=SUM_RTOL)
        self.assertEqual(cpu_x + cpu_y, private_keep.cpu(), atol=SUM_ATOL, rtol=SUM_RTOL)

    @parametrize("shape", [(256, ), (32, 64, 128)])
    def test_chain3(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs(shape, torch.float32)
        cpu_z = (cpu_x + cpu_y) ** 2 - cpu_x
        private_z, private_keep, k = chain3_ew(private_x, private_y)
        # pointwise x3, every intermediate private -> one fused op.
        self.assertGreaterEqual(fused_ops(k), 1)
        self.assertFalse(has_reduce_op(k))
        self.assertEqual(cpu_z, private_z.cpu(), atol=SUM_ATOL, rtol=SUM_RTOL)
        self.assertEqual(cpu_x + cpu_y, private_keep.cpu(), atol=SUM_ATOL, rtol=SUM_RTOL)


class TestPointwiseReduceFusion(TestCase):
    @parametrize("size", [256, 1024, 1000])
    def test_pointwise_reduce(self, size):
        cpu_x, private_x = make_inputs((size, ), torch.float32)
        cpu_y, private_y = make_inputs((size, ), torch.float32)
        cpu_z = block_sum(cpu_x * cpu_y, 256)
        private_z, _, k = pwise_reduce(private_x, private_y)
        # pointwise + all-dimension reduce.
        self.assertGreaterEqual(fused_ops(k), 1)
        self.assertFalse(has_reduce_op(k))
        self.assertEqual(cpu_z, private_z.cpu(), atol=SUM_ATOL, rtol=SUM_RTOL)

    @parametrize("size", [256, 1024, 1000])
    def test_chain3_reduce(self, size):
        cpu_x, private_x = make_inputs((size, ), torch.float32)
        cpu_z = block_sum((cpu_x + cpu_x) ** 2, 256)
        private_z, _, k = chain3_reduce(private_x)
        # pointwise x2 + reduce: the whole chain folds into one fused op.
        self.assertGreaterEqual(fused_ops(k), 1)
        self.assertFalse(has_reduce_op(k))
        self.assertEqual(cpu_z, private_z.cpu(), atol=SUM_ATOL, rtol=SUM_RTOL)

    @parametrize("size", [256, 1000])
    def test_two_consumers(self, size):
        cpu_x, private_x = make_inputs((size, ), torch.float32)
        cpu_y, private_y = make_inputs((size, ), torch.float32)
        cpu_sum = block_sum(cpu_x * cpu_y, 256)
        private_sum, private_plus, _, k = two_consumers(private_x, private_y)
        # `t` has a second consumer, so the chain fusion must be refused; every
        # op still gets its own container.
        self.assertEqual(fused_ops(k), 0)
        self.assertFalse(has_reduce_op(k))
        self.assertEqual(cpu_sum, private_sum.cpu(), atol=SUM_ATOL, rtol=SUM_RTOL)
        self.assertEqual(cpu_x * cpu_y + 1.0, private_plus.cpu(), atol=SUM_ATOL, rtol=SUM_RTOL)


class TestPointwiseReduce2DFusion(TestCase):
    @parametrize("shape", [(32, 64), (64, 32)])
    def test_dim0(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs(shape, torch.float32)
        cpu_z = (cpu_x * cpu_y).sum(dim=0)
        private_z, k = pwise_reduce_2d_dim0(private_x, private_y, *shape)
        # pointwise + 2D reduce over the outer dim; the reduce stays the last op.
        self.assertGreaterEqual(fused_ops(k), 1)
        self.assertFalse(has_reduce_op(k))
        self.assertEqual(cpu_z, private_z.cpu(), atol=SUM_ATOL, rtol=SUM_RTOL)

    @parametrize("shape", [(32, 64), (64, 32)])
    def test_dim1(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs(shape, torch.float32)
        cpu_z = (cpu_x * cpu_y).sum(dim=1)
        private_z, k = pwise_reduce_2d_dim1(private_x, private_y, *shape)
        # pointwise + 2D reduce over the inner dim.
        self.assertGreaterEqual(fused_ops(k), 1)
        self.assertFalse(has_reduce_op(k))
        self.assertEqual(cpu_z, private_z.cpu(), atol=SUM_ATOL, rtol=SUM_RTOL)


instantiate_parametrized_tests(TestPointwiseFusion)
instantiate_parametrized_tests(TestPointwiseReduceFusion)
instantiate_parametrized_tests(TestPointwiseReduce2DFusion)
