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
def flash_attention_kernel(
    Q, K, V, O, sm_scale,
    N_HEADS,
    sqz, sqh, sqm, sqk,
    skz, skh, skn, skk,
    svz, svh, svn, svk,
    soz, soh, som, sok,
    GROUP: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // N_HEADS
    off_h = off_hz % N_HEADS
    kv_h = off_h // GROUP

    q_off = off_z * sqz + off_h * sqh
    k_off = off_z * skz + kv_h * skh
    v_off = off_z * svz + kv_h * svh
    o_off = off_z * soz + off_h * soh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + q_off + offs_m[:, None] * sqm + offs_d[None, :] * sqk,
                mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n_cur = start_n + offs_n
        k = tl.load(K + k_off + offs_n_cur[:, None] * skn + offs_d[None, :] * skk)
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        if CAUSAL:
            mask = offs_m[:, None] >= offs_n_cur[None, :]
            if WINDOW > 0:
                mask = mask & (offs_m[:, None] - offs_n_cur[None, :] < WINDOW)
            qk = tl.where(mask, qk, -1.0e6)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        v = tl.load(V + v_off + offs_n_cur[:, None] * svn + offs_d[None, :] * svk)
        acc = tl.dot(p, v, acc)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(O + o_off + offs_m[:, None] * som + offs_d[None, :] * sok, acc,
             mask=offs_m[:, None] < N_CTX)


def flash_attention(q, k, v, causal=False, window=0, group=1):
    B, H, N, D = q.shape
    o = torch.empty_like(q)
    sm_scale = 1.0 / (D ** 0.5)
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    flash_attention_kernel[grid](
        q, k, v, o, sm_scale,
        H,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        GROUP=group, N_CTX=N, HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        CAUSAL=causal, WINDOW=window,
    )
    return o


def _ref_attention(q, k, v, causal, window, group):
    B, H, N, D = q.shape
    kvh = H // group
    k2 = k.repeat_interleave(group, dim=1)
    v2 = v.repeat_interleave(group, dim=1)
    scores = torch.matmul(q, k2.transpose(-1, -2)) / (D ** 0.5)
    qm = torch.arange(N)[:, None]
    kn = torch.arange(N)[None, :]
    mask = qm >= kn
    if window > 0:
        mask = mask & (qm - kn < window)
    if causal:
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v2)


class TestFlashAttentionCase(TestCase):
    @parametrize("shape", [(1, 4, 64, 32), (2, 2, 128, 32), (1, 3, 96, 32)])
    @parametrize("causal", [False, True])
    def test_flash_attention(self, shape, causal):
        B, H, N, D = shape
        cpu_q, private_q = make_inputs(shape, torch.float32)
        cpu_k, private_k = make_inputs(shape, torch.float32)
        cpu_v, private_v = make_inputs(shape, torch.float32)
        cpu_z = _ref_attention(cpu_q, cpu_k, cpu_v, causal, 0, 1)
        private_z = flash_attention(private_q, private_k, private_v, causal=causal)
        self.assertEqual(cpu_z, private_z.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestFlashAttentionGqaCase(TestCase):
    def test_gqa(self):
        B, H, N, D = 1, 6, 96, 32
        group = 2
        cpu_q, private_q = make_inputs((B, H, N, D), torch.float32)
        cpu_k, private_k = make_inputs((B, H // group, N, D), torch.float32)
        cpu_v, private_v = make_inputs((B, H // group, N, D), torch.float32)
        cpu_z = _ref_attention(cpu_q, cpu_k, cpu_v, True, 0, group)
        private_z = flash_attention(private_q, private_k, private_v, causal=True, group=group)
        self.assertEqual(cpu_z, private_z.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


class TestFlashAttentionWindowCase(TestCase):
    def test_sliding_window(self):
        B, H, N, D = 1, 3, 96, 32
        window = 32
        cpu_q, private_q = make_inputs((B, H, N, D), torch.float32)
        cpu_k, private_k = make_inputs((B, H, N, D), torch.float32)
        cpu_v, private_v = make_inputs((B, H, N, D), torch.float32)
        cpu_z = _ref_attention(cpu_q, cpu_k, cpu_v, True, window, 1)
        private_z = flash_attention(private_q, private_k, private_v, causal=True, window=window)
        self.assertEqual(cpu_z, private_z.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


instantiate_parametrized_tests(TestFlashAttentionCase)
instantiate_parametrized_tests(TestFlashAttentionGqaCase)
instantiate_parametrized_tests(TestFlashAttentionWindowCase)
