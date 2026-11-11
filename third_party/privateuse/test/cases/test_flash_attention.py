import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from ..conftest import F32_ATOL, F32_RTOL, make_inputs, device, ACC_ATOL, ACC_RTOL
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize
)


@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    sm_scale,
    BATCH, N_HEADS, N_CTX,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // N_HEADS
    off_h = off_hz % N_HEADS

    q_offset = off_z * stride_qz + off_h * stride_qh + start_m * BLOCK_M * stride_qm
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh + start_m * BLOCK_M * stride_om

    Q_desc = tl.make_tensor_descriptor(
        Q_ptr + q_offset,
        shape=[BLOCK_M, HEAD_DIM],
        strides=[stride_qm, stride_qk],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    K_desc = tl.make_tensor_descriptor(
        K_ptr + k_offset,
        shape=[N_CTX, HEAD_DIM],
        strides=[stride_kn, stride_kk],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    V_desc = tl.make_tensor_descriptor(
        V_ptr + v_offset,
        shape=[N_CTX, HEAD_DIM],
        strides=[stride_vn, stride_vk],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    O_desc = tl.make_tensor_descriptor(
        O_ptr + o_offset,
        shape=[BLOCK_M, HEAD_DIM],
        strides=[stride_om, stride_ok],
        block_shape=[BLOCK_M, HEAD_DIM],
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = Q_desc.load([0, 0])
    qk_scale = sm_scale * 1.44269504

    if CAUSAL:
        lo, hi = 0, (start_m + 1) * BLOCK_M
    else:
        lo, hi = 0, N_CTX

    for start_n in range(lo, hi, BLOCK_N):
        k = K_desc.load([start_n, 0])
        qk = tl.dot(q, tl.trans(k))

        if CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(causal_mask, 0, -1.0e6)
        else:
            qk = qk * qk_scale

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)

        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)

        acc = acc * alpha[:, None]

        v = V_desc.load([start_n, 0])
        acc = tl.dot(p, v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    acc = acc / l_i[:, None]
    O_desc.store([0, 0], acc)


def flash_attention(q, k, v, causal=False):
    BATCH, N_HEADS, N_CTX, HEAD_DIM = q.shape

    o = torch.empty_like(q)

    BLOCK_M = 64
    BLOCK_N = 64

    grid = (triton.cdiv(N_CTX, BLOCK_M), BATCH * N_HEADS)
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)

    flash_attention_kernel[grid](
        q, k, v, o,
        sm_scale,
        BATCH, N_HEADS, N_CTX,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CAUSAL=causal,
    )
    return o


def reference_attention(q, k, v, causal=False):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


class TestFlashAttention(TestCase):
    @parametrize("shape", [(2, 4, 64, 64), (1, 8, 128, 64)])
    @parametrize("causal", [False, True])
    def test_flash_attention(self, shape, causal):
        BATCH, N_HEADS, N_CTX, HEAD_DIM = shape
        cpu_q, private_q = make_inputs(shape, torch.float32)
        cpu_k, private_k = make_inputs(shape, torch.float32)
        cpu_v, private_v = make_inputs(shape, torch.float32)

        cpu_out = reference_attention(cpu_q, cpu_k, cpu_v, causal)
        private_out = flash_attention(private_q, private_k, private_v, causal)

        self.assertEqual(cpu_out, private_out.cpu(), atol=ACC_ATOL, rtol=ACC_RTOL)


instantiate_parametrized_tests(TestFlashAttention)
