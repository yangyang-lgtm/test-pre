import torch
import triton
import triton.language as tl
from ..conftest import (
    make_inputs, F32_ATOL, F32_RTOL, F16_ATOL, F16_RTOL,
    ACC_ATOL, ACC_RTOL, SUM_ATOL, SUM_RTOL
)
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)
from .test_2d_load_store import copy2d_fn, transpose2d_fn
from .test_add import add
from .test_cast import cast_fn
from .test_elementwise_unary import exp_fn, neg
from .test_fill import masked_fill
from .test_flash_attention import flash_attention, reference_attention
from .test_fused_ops import layer_norm, softmax
from .test_if import if_fn
from .test_matmul import matmul
from .test_reduction import max_fn, sum_fn
from .test_tensor_descriptor import copy_with_descriptor
from .test_unstructured_access import gather_fn, scatter_fn

BLOCK = 256
FLAT_SHAPE = (64, BLOCK)
SQUARE_SHAPE = (128, 128)


@triton.jit
def dynamic_loop_kernel(pa, pc, K, R: tl.constexpr, N: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, N)
    acc = tl.zeros((N, ), dtype=tl.float32)
    for k in range(0, K):
        acc += tl.load(pa + (k * R + pid) * N + cols)
    tl.store(pc + pid * N + cols, acc)


class TestPointwiseLargeGrid(TestCase):
    def test_add(self):
        cpu_x, private_x = make_inputs(FLAT_SHAPE, torch.float32)
        cpu_y, private_y = make_inputs(FLAT_SHAPE, torch.float32)
        self.assertEqual(cpu_x + cpu_y, add(private_x, private_y).cpu(),
                         atol=F32_ATOL, rtol=F32_RTOL)

    @parametrize("name", ["exp", "neg"])
    def test_unary(self, name):
        fn, ref = {"exp": (exp_fn, torch.exp), "neg": (neg, torch.neg)}[name]
        cpu_x, private_x = make_inputs(FLAT_SHAPE, torch.float32)
        self.assertEqual(ref(cpu_x), fn(private_x).cpu(), atol=F32_ATOL, rtol=F32_RTOL)

    def test_cast(self):
        cpu_x, private_x = make_inputs(FLAT_SHAPE, torch.float32)
        self.assertEqual(cpu_x.to(torch.float16), cast_fn(private_x, torch.float16).cpu(),
                         atol=F16_ATOL, rtol=F16_RTOL)

    def test_masked_fill(self):
        num = 16000
        fill = 3.5
        cpu_x, private_x = make_inputs((num, ), torch.float32)
        n_out = triton.cdiv(num, BLOCK) * BLOCK
        cpu_z = torch.cat([cpu_x, torch.full((n_out - num, ), fill)])
        self.assertEqual(cpu_z, masked_fill(private_x, fill).cpu(),
                         atol=F32_ATOL, rtol=F32_RTOL)


class TestTwoDimLargeGrid(TestCase):
    def test_copy2d(self):
        cpu_x, private_x = make_inputs(SQUARE_SHAPE, torch.float32)
        self.assertEqual(cpu_x, copy2d_fn(private_x).cpu(), atol=F32_ATOL, rtol=F32_RTOL)

    def test_transpose2d(self):
        cpu_x, private_x = make_inputs(SQUARE_SHAPE, torch.float32)
        self.assertEqual(cpu_x.T, transpose2d_fn(private_x).cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestReductionLargeGrid(TestCase):
    @parametrize("name", ["sum", "max"])
    def test_reduce(self, name):
        fn, ref, atol, rtol = {
            "sum": (sum_fn, torch.sum, SUM_ATOL, SUM_RTOL),
            "max": (max_fn, torch.max, F32_ATOL, F32_RTOL),
        }[name]
        cpu_x, private_x = make_inputs(FLAT_SHAPE, torch.float32)
        self.assertEqual(ref(cpu_x), fn(private_x).cpu(), atol=atol, rtol=rtol)


class TestMatmulLargeGrid(TestCase):
    def test_matmul_dynamic_k(self):
        cpu_a, private_a = make_inputs((1024, 96), torch.float32)
        cpu_b, private_b = make_inputs((96, 64), torch.float32)
        self.assertEqual(torch.matmul(cpu_a, cpu_b), matmul(private_a, private_b).cpu(),
                         atol=SUM_ATOL, rtol=SUM_RTOL)


class TestFusedLargeGrid(TestCase):
    def test_softmax(self):
        cpu_x, private_x = make_inputs((64, 128), torch.float32)
        self.assertEqual(torch.softmax(cpu_x, dim=1), softmax(private_x).cpu(),
                         atol=ACC_ATOL, rtol=ACC_RTOL)

    def test_layer_norm(self):
        cpu_x, private_x = make_inputs((64, 128), torch.float32)
        cpu_w, private_w = make_inputs((128, ), torch.float32)
        cpu_b, private_b = make_inputs((128, ), torch.float32)
        ref = torch.nn.functional.layer_norm(cpu_x, [128], cpu_w, cpu_b, 1e-5)
        self.assertEqual(ref, layer_norm(private_x, private_w, private_b).cpu(),
                         atol=ACC_ATOL, rtol=ACC_RTOL)


class TestFlashAttentionLargeGrid(TestCase):
    def test_flash_attention_dynamic_loop(self):
        shape = (8, 8, 64, 64)
        cpu_q, private_q = make_inputs(shape, torch.float32)
        cpu_k, private_k = make_inputs(shape, torch.float32)
        cpu_v, private_v = make_inputs(shape, torch.float32)
        cpu_z = reference_attention(cpu_q, cpu_k, cpu_v, False)
        private_z = flash_attention(private_q, private_k, private_v, False)
        self.assertEqual(cpu_z, private_z.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestDescriptorLargeGrid(TestCase):
    def test_copy_with_descriptor(self):
        cpu_x, private_x = make_inputs((256, 256), torch.float32)
        self.assertEqual(cpu_x, copy_with_descriptor(private_x).cpu(),
                         atol=F32_ATOL, rtol=F32_RTOL)


class TestUnstructuredLargeGrid(TestCase):
    @parametrize("name", ["gather", "scatter"])
    def test_unstructured_access(self, name):
        size = 16384
        cpu_x, private_x = make_inputs((size, ), torch.float32)
        cpu_idx = torch.randperm(size, dtype=torch.int32)
        private_idx = cpu_idx.to(private_x.device)
        if name == "gather":
            cpu_z = cpu_x[cpu_idx.long()]
            private_z = gather_fn(private_x, private_idx)
        else:
            cpu_z = torch.empty_like(cpu_x)
            cpu_z[cpu_idx.long()] = cpu_x
            private_z = scatter_fn(private_x, private_idx)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestIfLargeGrid(TestCase):
    def test_if(self):
        num = 64 * BLOCK
        cpu_x, private_x = make_inputs((num, ), torch.float32)
        cpu_y, private_y = make_inputs((num, ), torch.float32)
        cpu_z = torch.empty_like(cpu_x)
        for block in range(num // BLOCK):
            lo = block * BLOCK
            hi = lo + BLOCK
            if block % 2 == 0:
                cpu_z[lo:hi] = cpu_x[lo:hi] + cpu_y[lo:hi]
            else:
                cpu_z[lo:hi] = cpu_x[lo:hi] - cpu_y[lo:hi]
        self.assertEqual(cpu_z, if_fn(private_x, private_y).cpu(),
                         atol=F32_ATOL, rtol=F32_RTOL)


class TestDynamicLoopLargeGrid(TestCase):
    def test_dynamic_inner_loop(self):
        R = 64
        num_cols = 64
        K = 7
        cpu_x, private_x = make_inputs((K, R, num_cols), torch.float32)
        out = torch.empty((R, num_cols), dtype=torch.float32, device=private_x.device)
        dynamic_loop_kernel[(R, )](private_x, out, K, R=R, N=num_cols)
        self.assertEqual(cpu_x.sum(dim=0).reshape(-1), out.cpu().reshape(-1),
                         atol=SUM_ATOL, rtol=SUM_RTOL)


instantiate_parametrized_tests(TestPointwiseLargeGrid)
instantiate_parametrized_tests(TestTwoDimLargeGrid)
instantiate_parametrized_tests(TestReductionLargeGrid)
instantiate_parametrized_tests(TestMatmulLargeGrid)
instantiate_parametrized_tests(TestFusedLargeGrid)
instantiate_parametrized_tests(TestFlashAttentionLargeGrid)
instantiate_parametrized_tests(TestDescriptorLargeGrid)
instantiate_parametrized_tests(TestUnstructuredLargeGrid)
instantiate_parametrized_tests(TestIfLargeGrid)
instantiate_parametrized_tests(TestDynamicLoopLargeGrid)
