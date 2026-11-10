import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from ..conftest import make_inputs, F32_ATOL, F32_RTOL, ACC_ATOL, ACC_RTOL
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)


@triton.jit
def batch_norm_train_kernel(px, pw, pb, py, pmean, pinvstd,
                            N: tl.constexpr, C: tl.constexpr, HW: tl.constexpr, eps):
    c = tl.program_id(0)
    nrows = tl.arange(0, N)
    cols = tl.arange(0, HW)
    off = (nrows[:, None] * C + c) * HW + cols[None, :]
    x = tl.load(px + off)
    w = tl.load(pw + c)
    b = tl.load(pb + c)
    cnt: tl.constexpr = N * HW
    mean = tl.sum(tl.sum(x, 1), 0) / cnt
    d = x - mean
    rstd = tl.rsqrt(tl.sum(tl.sum(d * d, 1), 0) / cnt + eps)
    tl.store(py + off, d * rstd * w + b)
    tl.store(pmean + c, mean)
    tl.store(pinvstd + c, rstd)


@triton.jit
def batch_norm_eval_kernel(px, pw, pb, prm, prv, py,
                           N: tl.constexpr, C: tl.constexpr, HW: tl.constexpr, eps):
    c = tl.program_id(0)
    nrows = tl.arange(0, N)
    cols = tl.arange(0, HW)
    off = (nrows[:, None] * C + c) * HW + cols[None, :]
    x = tl.load(px + off)
    w = tl.load(pw + c)
    b = tl.load(pb + c)
    rm = tl.load(prm + c)
    rv = tl.load(prv + c)
    y = (x - rm) * tl.rsqrt(rv + eps) * w + b
    tl.store(py + off, y)


@triton.jit
def batch_norm_update_stats_kernel(px, prm, prv, pmean, pvar,
                                   N: tl.constexpr, C: tl.constexpr, HW: tl.constexpr,
                                   momentum):
    c = tl.program_id(0)
    nrows = tl.arange(0, N)
    cols = tl.arange(0, HW)
    off = (nrows[:, None] * C + c) * HW + cols[None, :]
    x = tl.load(px + off)
    cnt: tl.constexpr = N * HW
    mean = tl.sum(tl.sum(x, 1), 0) / cnt
    d = x - mean
    var = tl.sum(tl.sum(d * d, 1), 0) / cnt
    unbiased = var * cnt / (cnt - 1)
    rm = tl.load(prm + c)
    rv = tl.load(prv + c)
    tl.store(pmean + c, (1.0 - momentum) * rm + momentum * mean)
    tl.store(pvar + c, (1.0 - momentum) * rv + momentum * unbiased)


def batch_norm_train(x, w, b, eps=1e-5):
    N, C, H, W = x.shape
    y = torch.empty_like(x)
    mean = torch.empty(C, dtype=torch.float32, device=x.device)
    invstd = torch.empty(C, dtype=torch.float32, device=x.device)
    batch_norm_train_kernel[(C,)](x, w, b, y, mean, invstd, N=N, C=C, HW=H * W, eps=eps)
    return y, mean, invstd


def batch_norm_eval(x, w, b, rm, rv, eps=1e-5):
    N, C, H, W = x.shape
    y = torch.empty_like(x)
    batch_norm_eval_kernel[(C,)](x, w, b, rm, rv, y, N=N, C=C, HW=H * W, eps=eps)
    return y


def batch_norm_update_stats(x, rm, rv, momentum=0.1):
    N, C, H, W = x.shape
    new_rm = torch.empty_like(rm)
    new_rv = torch.empty_like(rv)
    batch_norm_update_stats_kernel[(C,)](x, rm, rv, new_rm, new_rv,
                                         N=N, C=C, HW=H * W, momentum=momentum)
    return new_rm, new_rv


class TestBatchNormTrainCase(TestCase):
    def test_batch_norm_train(self):
        shape = (2, 4, 8, 8)
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_w, private_w = make_inputs((shape[1], ), torch.float32)
        cpu_b, private_b = make_inputs((shape[1], ), torch.float32)
        cpu_y, cpu_mean, cpu_invstd = torch.native_batch_norm(cpu_x, cpu_w, cpu_b,
                                                              None, None, True, 0.1, 1e-5)
        y, mean, invstd = batch_norm_train(private_x, private_w, private_b)
        self.assertEqual(cpu_y, y.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)
        self.assertEqual(cpu_mean, mean.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)
        self.assertEqual(cpu_invstd, invstd.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestBatchNormEvalCase(TestCase):
    def test_batch_norm_eval(self):
        shape = (2, 4, 8, 8)
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_w, private_w = make_inputs((shape[1], ), torch.float32)
        cpu_b, private_b = make_inputs((shape[1], ), torch.float32)
        cpu_rm, private_rm = make_inputs((shape[1], ), torch.float32)
        cpu_rv, private_rv = make_inputs((shape[1], ), torch.float32)
        cpu_rv = torch.abs(cpu_rv) + 0.5
        private_rv = cpu_rv.clone().to(private_x.device)
        cpu_y = F.batch_norm(cpu_x, cpu_rm, cpu_rv, cpu_w, cpu_b,
                             False, 0.1, 1e-5)
        y = batch_norm_eval(private_x, private_w, private_b, private_rm, private_rv)
        self.assertEqual(cpu_y, y.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestBatchNormUpdateStatsCase(TestCase):
    def test_batch_norm_update_stats(self):
        shape = (4, 4, 8, 8)
        cpu_x, private_x = make_inputs(shape, torch.float32)
        cpu_rm, private_rm = make_inputs((shape[1], ), torch.float32)
        cpu_rv, private_rv = make_inputs((shape[1], ), torch.float32)
        cpu_rv = torch.abs(cpu_rv) + 0.5
        private_rv = cpu_rv.clone().to(private_x.device)
        momentum = 0.1
        cnt = shape[0] * shape[2] * shape[3]
        batch_mean = cpu_x.mean(dim=(0, 2, 3))
        batch_var = cpu_x.var(dim=(0, 2, 3), unbiased=False) * cnt / (cnt - 1)
        ref_rm = (1 - momentum) * cpu_rm + momentum * batch_mean
        ref_rv = (1 - momentum) * cpu_rv + momentum * batch_var
        rm, rv = batch_norm_update_stats(private_x, private_rm, private_rv, momentum)
        self.assertEqual(ref_rm, rm.cpu(), atol=F32_ATOL, rtol=F32_RTOL)
        self.assertEqual(ref_rv, rv.cpu(), atol=F32_ATOL, rtol=F32_RTOL)


instantiate_parametrized_tests(TestBatchNormTrainCase)
instantiate_parametrized_tests(TestBatchNormEvalCase)
instantiate_parametrized_tests(TestBatchNormUpdateStatsCase)
