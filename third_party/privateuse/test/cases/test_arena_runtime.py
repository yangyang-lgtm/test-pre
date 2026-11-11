import re

import torch
import triton
import triton.language as tl
from ..conftest import device, F32_ATOL, F32_RTOL

BLOCK = 64
SIZES = (997, 2051)


@triton.jit
def arena_static_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(x_ptr + offs, mask=mask)
    c = a * 2.0 + 1.0
    tl.store(o_ptr + offs, c, mask=mask)


def run_kernel(kernel, n):
  grid = (triton.cdiv(n, BLOCK),)
  x = torch.randn(n, device=device)
  out = torch.empty(n, device=device)
  kernel[grid](x, out, n, BLOCK)
  puc = kernel.warmup(x, out, n, BLOCK, grid=grid).asm["puc"]
  return x.cpu(), out.cpu(), puc


def test_kernel_plans_static_slots(monkeypatch):
  monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")

  for n in SIZES:
    x, out, puc = run_kernel(arena_static_kernel, n)
    assert "arena_take(" not in puc
    assert "arena_give(" not in puc
    assert "MEM0 uint8_t" in puc
    assert len(re.findall(r"MEM\d uint8_t", puc)) <= 5
    torch.testing.assert_close(out, x * 2.0 + 1.0, atol=F32_ATOL, rtol=F32_RTOL)
