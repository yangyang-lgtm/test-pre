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
def kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num

    a = tl.load(pa + offset, mask=mask)
    b = tl.load(pb + offset, mask=mask)
    c = a + b
    tl.store(pc + offset, c, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty_like(x)
    num = out.numel()
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    kernel[grid](x, y, out, num, BLOCK=256)
    return out


class TestAddCase(TestCase):

    @parametrize("shape", [(32, 32), (32, 64, 128)])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_add(self, shape, dtype):
        cpu_x, private_x = make_inputs(shape, dtype)
        cpu_y, private_y = make_inputs(shape, dtype)

        cpu_z = torch.add(cpu_x, cpu_y)
        private_z = add(private_x, private_y)

        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

instantiate_parametrized_tests(TestAddCase)
