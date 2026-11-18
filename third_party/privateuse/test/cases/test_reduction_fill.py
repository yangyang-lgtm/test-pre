import torch
import triton
import triton.language as tl
from ..conftest import make_inputs, device, F32_ATOL, F32_RTOL, ACC_ATOL, ACC_RTOL, LOWP_ATOL, LOWP_RTOL, SUM_ATOL, SUM_RTOL
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)


@triton.jit
def sum_kernel(pa, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask, other=0.0)
    c = tl.sum(a, axis=0)
    tl.store(pc + pid, c)


def sum_fn(x: torch.Tensor):
    num = x.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    out = torch.empty(grid({'BLOCK': 256}), dtype=x.dtype, device=x.device)
    sum_kernel[grid](x, out, num, BLOCK=256)
    return out.sum()


class TestSumCase(TestCase):
    @parametrize("shape", [(1000, 13)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_sum(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_z = torch.sum(cpu_x)
        private_z = sum_fn(private_x)
        self.assertEqual(cpu_z, private_z.cpu(), atol=SUM_ATOL, rtol=SUM_RTOL)

instantiate_parametrized_tests(TestSumCase)
