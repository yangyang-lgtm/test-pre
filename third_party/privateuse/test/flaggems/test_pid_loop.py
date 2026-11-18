import torch
import torch.nn.functional as F
import triton
from ..conftest import make_inputs, F32_ATOL, F32_RTOL, ACC_ATOL, ACC_RTOL, SUM_ATOL, SUM_RTOL
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)
from .test_activation import relu, sigmoid, silu
from .test_attention import flash_attention, _ref_attention
from .test_binary_math import maximum, minimum, where, clamp
from .test_compare_logic import eq, lt, logical_and
from .test_conv_pool import conv2d, max_pool2d, avg_pool2d
from .test_fused import silu_and_mul, gelu_and_mul, add_rms_norm, residual_layer_norm
from .test_indexing import masked_fill, constant_pad
from .test_loss import mse_loss, l1_loss, nll_loss, cross_entropy
from .test_matmul import matmul, bmm
from .test_moe import moe_sum, topk_softmax
from .test_norm_backward import batch_norm_bwd, group_norm_bwd, ref_batch_norm_bwd, _grads
from .test_norm_batch import batch_norm_train
from .test_norm_complex import instance_norm
from .test_norm_variants import gemma_rms_norm
from .test_quant import per_token_quant, ref_per_token_quant
from .test_reduction import sum_, max_, mean_
from .test_rope import rope, make_freqs, ref_rope
from .test_softmax_norm import softmax, log_softmax, layer_norm, rms_norm
from .test_sort_topk import sort, topk, median
from .test_unary_math import exp2, erf, floor

FLAT_SHAPE = (64, 256)
ROW_SHAPE = (1024, 64)


class TestActivationLargeGrid(TestCase):
    @parametrize("name", ["relu", "sigmoid", "silu"])
    def test_activation(self, name):
        fn, ref = {"relu": (relu, torch.relu),
                   "sigmoid": (sigmoid, torch.sigmoid),
                   "silu": (silu, F.silu)}[name]
        cpu_x, private_x = make_inputs(FLAT_SHAPE, torch.float32)
        self.assertEqual(ref(cpu_x), fn(private_x).cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestBinaryLargeGrid(TestCase):
    @parametrize("name", ["maximum", "minimum", "where", "clamp"])
    def test_binary(self, name):
        cpu_x, private_x = make_inputs(FLAT_SHAPE, torch.float32)
        if name == "clamp":
            self.assertEqual(torch.clamp(cpu_x, -0.5, 0.5), clamp(private_x, -0.5, 0.5).cpu(),
                             atol=F32_ATOL, rtol=F32_RTOL)
            return
        cpu_y, private_y = make_inputs(FLAT_SHAPE, torch.float32)
        fn, ref = {"maximum": (maximum, torch.maximum),
                   "minimum": (minimum, torch.minimum),
                   "where": (where, lambda x, y: torch.where(x > 0.0, x, y))}[name]
        self.assertEqual(ref(cpu_x, cpu_y), fn(private_x, private_y).cpu(),
                         atol=F32_ATOL, rtol=F32_RTOL)


class TestCompareLargeGrid(TestCase):
    @parametrize("name", ["eq", "lt", "logical_and"])
    def test_compare(self, name):
        cpu_x, private_x = make_inputs(FLAT_SHAPE, torch.float32)
        cpu_y, private_y = make_inputs(FLAT_SHAPE, torch.float32)
        fn, ref = {"eq": (eq, torch.eq),
                   "lt": (lt, torch.lt),
                   "logical_and": (logical_and, torch.logical_and)}[name]
        self.assertEqual(ref(cpu_x, cpu_y), fn(private_x, private_y).cpu())


class TestConvPoolLargeGrid(TestCase):
    def test_conv2d(self):
        shape = (32, 2, 16, 16)
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_w, private_w = make_inputs((2, 2, 3, 3), torch.float32)
        self.assertEqual(F.conv2d(cpu_x, cpu_w), conv2d(private_x, private_w).cpu(),
                         atol=SUM_ATOL, rtol=SUM_RTOL)

    def test_max_pool2d(self):
        cpu_x, private_x = make_inputs((8, 8, 16, 16), torch.float32)
        cpu_z, cpu_idx = F.max_pool2d(cpu_x, 2, return_indices=True)
        private_z, private_idx = max_pool2d(private_x)
        self.assertEqual(cpu_z, private_z.cpu())
        self.assertEqual(cpu_idx.to(torch.int32).view(-1), private_idx.cpu().view(-1))

    def test_avg_pool2d(self):
        cpu_x, private_x = make_inputs((8, 8, 16, 16), torch.float32)
        self.assertEqual(F.avg_pool2d(cpu_x, 2), avg_pool2d(private_x).cpu(),
                         atol=F32_ATOL, rtol=F32_RTOL)


class TestFusedLargeGrid(TestCase):
    def test_silu_and_mul(self):
        cpu_x, private_x = make_inputs((1024, 128), torch.float32)
        cpu_z = F.silu(cpu_x[..., :64]) * cpu_x[..., 64:]
        self.assertEqual(cpu_z, silu_and_mul(private_x).cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)

    def test_gelu_and_mul(self):
        cpu_x, private_x = make_inputs((1024, 128), torch.float32)
        cpu_z = F.gelu(cpu_x[..., :64], approximate="tanh") * cpu_x[..., 64:]
        self.assertEqual(cpu_z, gelu_and_mul(private_x).cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)

    def test_add_rms_norm(self):
        cpu_x, private_x = make_inputs(ROW_SHAPE, torch.float32)
        cpu_r, private_r = make_inputs(ROW_SHAPE, torch.float32)
        cpu_w, private_w = make_inputs((ROW_SHAPE[1], ), torch.float32)
        t = cpu_x + cpu_r
        cpu_z = t * torch.rsqrt(t.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * cpu_w
        self.assertEqual(cpu_z, add_rms_norm(private_x, private_r, private_w).cpu(),
                         atol=ACC_ATOL, rtol=ACC_RTOL)

    def test_residual_layer_norm(self):
        cpu_x, private_x = make_inputs(ROW_SHAPE, torch.float32)
        cpu_r, private_r = make_inputs(ROW_SHAPE, torch.float32)
        cpu_w, private_w = make_inputs((ROW_SHAPE[1], ), torch.float32)
        cpu_b, private_b = make_inputs((ROW_SHAPE[1], ), torch.float32)
        cpu_out, _, _ = torch.native_layer_norm(cpu_x + cpu_r, [ROW_SHAPE[1]], cpu_w, cpu_b, 1e-5)
        out, res = residual_layer_norm(private_x, private_r, private_w, private_b)
        self.assertEqual(cpu_out, out.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)
        self.assertEqual(cpu_x + cpu_r, res.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestIndexingLargeGrid(TestCase):
    def test_masked_fill(self):
        cpu_x, private_x = make_inputs(FLAT_SHAPE, torch.float32)
        cpu_mask = (cpu_x > 0).to(torch.int8)
        private_mask = cpu_mask.to(private_x.device)
        cpu_z = cpu_x.masked_fill(cpu_mask.bool(), 0.0)
        self.assertEqual(cpu_z, masked_fill(private_x, private_mask, 0.0).cpu())

    def test_constant_pad(self):
        cpu_x, private_x = make_inputs(FLAT_SHAPE, torch.float32)
        num = cpu_x.numel()
        p = num + 128
        cpu_z = torch.cat([cpu_x.reshape(-1), torch.zeros(128, dtype=cpu_x.dtype)])
        self.assertEqual(cpu_z, constant_pad(private_x, p).cpu())


class TestLossLargeGrid(TestCase):
    def test_mse_loss(self):
        cpu_x, private_x = make_inputs(ROW_SHAPE, torch.float32)
        cpu_y, private_y = make_inputs(ROW_SHAPE, torch.float32)
        self.assertEqual(F.mse_loss(cpu_x, cpu_y), mse_loss(private_x, private_y).cpu(),
                         atol=F32_ATOL, rtol=F32_RTOL)

    def test_l1_loss(self):
        cpu_x, private_x = make_inputs(ROW_SHAPE, torch.float32)
        cpu_y, private_y = make_inputs(ROW_SHAPE, torch.float32)
        self.assertEqual(F.l1_loss(cpu_x, cpu_y), l1_loss(private_x, private_y).cpu(),
                         atol=F32_ATOL, rtol=F32_RTOL)

    def test_nll_loss(self):
        shape = (1024, 16)
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_lp = torch.log_softmax(cpu_x, dim=1)
        private_lp = cpu_lp.to(private_x.device)
        cpu_tgt = torch.randint(0, shape[1], (shape[0], ), dtype=torch.int64)
        private_tgt = cpu_tgt.to(torch.int32).to(private_x.device)
        self.assertEqual(F.nll_loss(cpu_lp, cpu_tgt), nll_loss(private_lp, private_tgt).cpu(),
                         atol=F32_ATOL, rtol=F32_RTOL)

    def test_cross_entropy(self):
        shape = (1024, 16)
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_tgt = torch.randint(0, shape[1], (shape[0], ), dtype=torch.int64)
        private_tgt = cpu_tgt.to(torch.int32).to(private_x.device)
        self.assertEqual(F.cross_entropy(cpu_x, cpu_tgt), cross_entropy(private_x, private_tgt).cpu(),
                         atol=F32_ATOL, rtol=F32_RTOL)


class TestMatmulLargeGrid(TestCase):
    def test_matmul_dynamic_k(self):
        cpu_a, private_a = make_inputs((1024, 96), torch.float32)
        cpu_b, private_b = make_inputs((96, 64), torch.float32)
        self.assertEqual(torch.matmul(cpu_a, cpu_b), matmul(private_a, private_b).cpu(),
                         atol=SUM_ATOL, rtol=SUM_RTOL)

    def test_bmm(self):
        cpu_a, private_a = make_inputs((64, 32, 32), torch.float32)
        cpu_b, private_b = make_inputs((64, 32, 32), torch.float32)
        self.assertEqual(torch.bmm(cpu_a, cpu_b), bmm(private_a, private_b).cpu(),
                         atol=SUM_ATOL, rtol=SUM_RTOL)


class TestMoeLargeGrid(TestCase):
    def test_moe_sum(self):
        shape = (8, 1024, 64)
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_w, private_w = make_inputs((shape[1], shape[0]), torch.float32)
        cpu_z = torch.einsum("te,eth->th", cpu_w, cpu_x)
        self.assertEqual(cpu_z, moe_sum(private_x, private_w).cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)

    def test_topk_softmax(self):
        k = 4
        cpu_x, private_x = make_inputs((1024, 16), torch.float32)
        topk_vals = torch.topk(cpu_x, k, dim=-1).values
        cpu_z = torch.softmax(topk_vals, dim=-1)
        private_z = topk_softmax(private_x, k)[:, :k].cpu()
        self.assertEqual(cpu_z, private_z, atol=ACC_ATOL, rtol=ACC_RTOL)


class TestNormLargeGrid(TestCase):
    def test_batch_norm_train(self):
        shape = (4, 64, 4, 4)
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_w, private_w = make_inputs((shape[1], ), torch.float32)
        cpu_b, private_b = make_inputs((shape[1], ), torch.float32)
        cpu_y, cpu_mean, cpu_invstd = torch.native_batch_norm(
            cpu_x, cpu_w, cpu_b, None, None, True, 0.1, 1e-5)
        y, mean, invstd = batch_norm_train(private_x, private_w, private_b)
        self.assertEqual(cpu_y, y.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)
        self.assertEqual(cpu_mean, mean.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)
        self.assertEqual(cpu_invstd, invstd.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)

    def test_batch_norm_bwd(self):
        shape = (2, 64, 4, 4)
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_dy, private_dy = make_inputs(shape, torch.float32)
        cpu_g, private_g = make_inputs((shape[1], ), torch.float32)
        ref_dx, ref_dg, ref_db = ref_batch_norm_bwd(cpu_dy, cpu_x, cpu_g)
        dx, dg, db = batch_norm_bwd(private_dy, private_x, private_g)
        self.assertEqual(ref_dx, dx.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)
        self.assertEqual(ref_dg, dg.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)
        self.assertEqual(ref_db, db.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)

    def test_group_norm_bwd(self):
        groups = 8
        shape = (8, 8, 4, 4)
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_dy, private_dy = make_inputs(shape, torch.float32)
        cpu_w, private_w = make_inputs((shape[1], ), torch.float32)
        _, cpu_g = _grads(lambda x, w: F.group_norm(x, groups, weight=w), cpu_dy, cpu_x, cpu_w)
        dx = group_norm_bwd(private_dy, private_x, private_w, groups)
        self.assertEqual(cpu_g[0], dx.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)

    def test_instance_norm(self):
        shape = (8, 8, 16, 16)
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_w, private_w = make_inputs((shape[1], ), torch.float32)
        cpu_b, private_b = make_inputs((shape[1], ), torch.float32)
        cpu_z = F.instance_norm(cpu_x, weight=cpu_w, bias=cpu_b, eps=1e-5)
        self.assertEqual(cpu_z, instance_norm(private_x, private_w, private_b).cpu(),
                         atol=ACC_ATOL, rtol=ACC_RTOL)

    def test_gemma_rms_norm(self):
        cpu_x, private_x = make_inputs(ROW_SHAPE, torch.float32)
        cpu_w, private_w = make_inputs((ROW_SHAPE[1], ), torch.float32)
        cpu_z = cpu_x * torch.rsqrt(cpu_x.pow(2).mean(-1, keepdim=True) + 1e-5) * (1.0 + cpu_w)
        self.assertEqual(cpu_z, gemma_rms_norm(private_x, private_w).cpu(),
                         atol=ACC_ATOL, rtol=ACC_RTOL)


class TestQuantLargeGrid(TestCase):
    def test_per_token_quant(self):
        cpu_x, private_x = make_inputs(ROW_SHAPE, torch.float32)
        self.assertEqual(ref_per_token_quant(cpu_x), per_token_quant(private_x).cpu(),
                         atol=F32_ATOL, rtol=F32_RTOL)


class TestReductionLargeGrid(TestCase):
    @parametrize("name", ["sum", "max", "mean"])
    def test_reduction(self, name):
        cpu_x, private_x = make_inputs(ROW_SHAPE, torch.float32)
        fn, ref, atol, rtol = {
            "sum": (sum_, lambda x: torch.sum(x, dim=1), SUM_ATOL, SUM_RTOL),
            "max": (max_, lambda x: torch.max(x, dim=1).values, F32_ATOL, F32_RTOL),
            "mean": (mean_, lambda x: torch.mean(x, dim=1), ACC_ATOL, ACC_RTOL),
        }[name]
        self.assertEqual(ref(cpu_x), fn(private_x).cpu(), atol=atol, rtol=rtol)


class TestRopeLargeGrid(TestCase):
    def test_rope_interleaved(self):
        shape = (8, 8, 64)
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_c, cpu_s = make_freqs(shape[0], shape[2])
        cpu_z = ref_rope(cpu_x, cpu_c, cpu_s, True)
        private_z = rope(private_x, cpu_c.to(private_x.device), cpu_s.to(private_x.device), True)
        self.assertEqual(cpu_z, private_z.cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestSoftmaxNormLargeGrid(TestCase):
    @parametrize("name", ["softmax", "log_softmax"])
    def test_softmax(self, name):
        cpu_x, private_x = make_inputs(ROW_SHAPE, torch.float32)
        fn, ref = {"softmax": (softmax, torch.softmax),
                   "log_softmax": (log_softmax, torch.log_softmax)}[name]
        self.assertEqual(ref(cpu_x, dim=1), fn(private_x).cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)

    def test_layer_norm(self):
        cpu_x, private_x = make_inputs(ROW_SHAPE, torch.float32)
        cpu_w, private_w = make_inputs((ROW_SHAPE[1], ), torch.float32)
        cpu_b, private_b = make_inputs((ROW_SHAPE[1], ), torch.float32)
        cpu_out, cpu_mean, cpu_rstd = torch.native_layer_norm(
            cpu_x, [ROW_SHAPE[1]], cpu_w, cpu_b, 1e-5)
        out, mean, rstd = layer_norm(private_x, private_w, private_b)
        self.assertEqual(cpu_out, out.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)
        self.assertEqual(cpu_mean.view(-1), mean.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)
        self.assertEqual(cpu_rstd.view(-1), rstd.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)

    def test_rms_norm(self):
        cpu_x, private_x = make_inputs(ROW_SHAPE, torch.float32)
        cpu_w, private_w = make_inputs((ROW_SHAPE[1], ), torch.float32)
        eps = 1e-5
        cpu_z = cpu_x * torch.rsqrt(cpu_x.pow(2).mean(dim=-1, keepdim=True) + eps) * cpu_w
        self.assertEqual(cpu_z, rms_norm(private_x, private_w, eps).cpu(),
                         atol=ACC_ATOL, rtol=ACC_RTOL)


class TestSortTopkLargeGrid(TestCase):
    def test_sort(self):
        cpu_x, private_x = make_inputs((64, 64), torch.float32)
        self.assertEqual(torch.sort(cpu_x, dim=1).values, sort(private_x).cpu())

    @parametrize("k", [8, 16])
    def test_topk(self, k):
        cpu_x, private_x = make_inputs((64, 64), torch.float32)
        self.assertEqual(torch.topk(cpu_x, k, dim=1).values, topk(private_x, k).cpu(),
                         atol=F32_ATOL, rtol=F32_RTOL)

    def test_median(self):
        cpu_x, private_x = make_inputs((64, 64), torch.float32)
        self.assertEqual(torch.median(cpu_x, dim=1).values, median(private_x).cpu(),
                         atol=F32_ATOL, rtol=F32_RTOL)


class TestUnaryLargeGrid(TestCase):
    @parametrize("name", ["exp2", "erf", "floor"])
    def test_unary(self, name):
        fn, ref = {"exp2": (exp2, torch.exp2),
                   "erf": (erf, torch.erf),
                   "floor": (floor, torch.floor)}[name]
        cpu_x, private_x = make_inputs(FLAT_SHAPE, torch.float32)
        self.assertEqual(ref(cpu_x), fn(private_x).cpu(), atol=F32_ATOL, rtol=F32_RTOL)


class TestAttentionLargeGrid(TestCase):
    def test_flash_attention_dynamic_loop(self):
        shape = (2, 16, 64, 32)
        cpu_q, private_q = make_inputs(shape, torch.float32)
        cpu_k, private_k = make_inputs(shape, torch.float32)
        cpu_v, private_v = make_inputs(shape, torch.float32)
        cpu_z = _ref_attention(cpu_q, cpu_k, cpu_v, False, 0, 1)
        private_z = flash_attention(private_q, private_k, private_v)
        self.assertEqual(cpu_z, private_z.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


instantiate_parametrized_tests(TestActivationLargeGrid)
instantiate_parametrized_tests(TestBinaryLargeGrid)
instantiate_parametrized_tests(TestCompareLargeGrid)
instantiate_parametrized_tests(TestConvPoolLargeGrid)
instantiate_parametrized_tests(TestFusedLargeGrid)
instantiate_parametrized_tests(TestIndexingLargeGrid)
instantiate_parametrized_tests(TestLossLargeGrid)
instantiate_parametrized_tests(TestMatmulLargeGrid)
instantiate_parametrized_tests(TestMoeLargeGrid)
instantiate_parametrized_tests(TestNormLargeGrid)
instantiate_parametrized_tests(TestQuantLargeGrid)
instantiate_parametrized_tests(TestReductionLargeGrid)
instantiate_parametrized_tests(TestRopeLargeGrid)
instantiate_parametrized_tests(TestSoftmaxNormLargeGrid)
instantiate_parametrized_tests(TestSortTopkLargeGrid)
instantiate_parametrized_tests(TestUnaryLargeGrid)
instantiate_parametrized_tests(TestAttentionLargeGrid)
