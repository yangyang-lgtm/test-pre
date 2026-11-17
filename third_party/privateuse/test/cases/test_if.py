import torch
import triton
import triton.language as tl
from ..conftest import F32_ATOL, F32_RTOL, make_inputs
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)

BLOCK_SIZE = 256


@triton.jit
def if_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask, other=0.0)
    b = tl.load(pb + offset, mask=mask, other=0.0)
    if pid % 2 == 0:
        c = a + b
    else:
        c = a - b
    tl.store(pc + offset, c, mask=mask)


def if_fn(x: torch.Tensor, y: torch.Tensor):
    num = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    if_kernel[grid](x, y, out, num, BLOCK=BLOCK_SIZE)
    return out


class TestIfCase(TestCase):

    @parametrize("shape", [(1000, ), (32, 32)])
    def test_if(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs(shape, torch.float32)

        n = cpu_x.numel()
        nblocks = triton.cdiv(n, BLOCK_SIZE)
        cpu_z = torch.empty_like(cpu_x)
        flat_z = cpu_z.view(-1)
        flat_x = cpu_x.view(-1)
        flat_y = cpu_y.view(-1)
        for block in range(nblocks):
            lo = block * BLOCK_SIZE
            hi = min(lo + BLOCK_SIZE, n)
            if block % 2 == 0:
                flat_z[lo:hi] = flat_x[lo:hi] + flat_y[lo:hi]
            else:
                flat_z[lo:hi] = flat_x[lo:hi] - flat_y[lo:hi]

        private_z = if_fn(private_x, private_y)

        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

instantiate_parametrized_tests(TestIfCase)
