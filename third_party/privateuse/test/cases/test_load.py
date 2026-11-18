import torch
import triton
import triton.language as tl
from ..conftest import F32_ATOL, F32_RTOL, make_inputs
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)


BLOCK_SIZE = 64


@triton.jit
def load_kernel(pa, pb, pc, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num
    a = tl.load(pa + offset, mask=mask, other=0.0)
    b = tl.load(pb + tl.arange(0, BLOCK))
    c = a * b
    tl.store(pc + offset, c, mask=mask)


def load_fn(x: torch.Tensor, y: torch.Tensor):
    num = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK']), )
    load_kernel[grid](x, y, out, num, BLOCK=BLOCK_SIZE)
    return out


class TestLoadCase(TestCase):

    @parametrize("shape", [(64, 64)])
    def test_load(self, shape):
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_y, private_y = make_inputs((1, shape[-1]), torch.float32)

        cpu_z = cpu_x * cpu_y
        private_z = load_fn(private_x, private_y)

        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)

instantiate_parametrized_tests(TestLoadCase)
