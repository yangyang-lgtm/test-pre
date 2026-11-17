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
def masked_fill_kernel(pa, pc, num, FILL: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    value = tl.load(pa + offset, mask=mask, other=FILL)
    tl.store(pc + offset, value)


def masked_fill(x: torch.Tensor, fill):
    num = x.numel()
    n_out = triton.cdiv(num, BLOCK_SIZE) * BLOCK_SIZE
    out = torch.empty(n_out, dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    masked_fill_kernel[grid](x, out, num, FILL=fill, BLOCK=BLOCK_SIZE)
    return out


class TestMaskedFillCase(TestCase):

    @parametrize("num", [1, BLOCK_SIZE, 1000])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_masked_fill(self, num, dtype):
        fill = 3.5 if dtype == torch.float32 else -7
        cpu_x, private_x = make_inputs((num,), dtype)

        n_out = triton.cdiv(num, BLOCK_SIZE) * BLOCK_SIZE
        cpu_z = torch.cat([cpu_x, torch.full((n_out - num, ), fill, dtype=dtype)])
        private_z = masked_fill(private_x, fill)

        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

instantiate_parametrized_tests(TestMaskedFillCase)
