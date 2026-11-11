"""用 torch.compile 编译 Qwen 风格 GQA eager attention,生成并收集 Triton kernel。

与 Qwen2.5-0.5B 同构: hidden=896, 14 Q head / 2 KV head (GQA), head_dim=64, RoPE, causal mask。
eager 数学实现 → inductor 会把 softmax/RoPE/mask/scale/repeat_kv 编译成 Triton kernel,
matmul 走 cuBLAS (extern kernel)。

用法:
  TORCH_LOGS=output_code TORCHINDUCTOR_CACHE_DIR=<fresh> python -u compile_attention.py > attn_log.txt 2>&1
  python collect_triton_kernels.py attn_log.txt attn
"""

import math

import torch
import torch.nn as nn
import torch._inductor.utils as _iu

_iu.is_big_gpu = lambda *a, **k: True

DEV = "cuda"
BF16 = torch.bfloat16
H, KVH, D, HID, S, B = 14, 2, 64, 896, 512, 2
REP = H // KVH


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(x):
    """(B, KVH, S, D) -> (B, H, S, D) — GQA KV head 复制"""
    b, h, s, d = x.shape
    x = x[:, :, None, :, :].expand(b, h, REP, s, d)
    return x.reshape(b, h * REP, s, d)


class GQAAttention(nn.Module):
    """Qwen2 eager attention (与 modeling_qwen2 的 Qwen2Attention 等价的数学实现)"""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HID, H * D, bias=True, dtype=BF16, device=DEV)
        self.k_proj = nn.Linear(HID, KVH * D, bias=True, dtype=BF16, device=DEV)
        self.v_proj = nn.Linear(HID, KVH * D, bias=True, dtype=BF16, device=DEV)
        self.o_proj = nn.Linear(H * D, HID, bias=False, dtype=BF16, device=DEV)

    def forward(self, x, cos, sin, mask):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, H, D).transpose(1, 2)
        k = self.k_proj(x).view(b, s, KVH, D).transpose(1, 2)
        v = self.v_proj(x).view(b, s, KVH, D).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        k, v = repeat_kv(k), repeat_kv(v)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D)
        scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(b, s, HID)
        return self.o_proj(out)


model = GQAAttention()


@torch.compile(mode="max-autotune-no-cudagraphs")
def attn_forward(x, cos, sin, mask):
    return model(x, cos, sin, mask)


def main():
    cos = torch.randn(1, 1, S, D, dtype=BF16, device=DEV)
    sin = torch.randn(1, 1, S, D, dtype=BF16, device=DEV)
    mask = torch.triu(
        torch.full((1, 1, S, S), -torch.inf, device=DEV), diagonal=1
    ).to(BF16)
    for step in range(2):
        x = torch.randn(B, S, HID, dtype=BF16, device=DEV, requires_grad=True)
        out = attn_forward(x, cos, sin, mask).clone()
        torch.cuda.synchronize()
        if step == 0:
            print(f"OK   ex_attn_forward -> Tensor {tuple(out.shape)}")
        g = torch.randn_like(out)
        out.backward(g)
        torch.cuda.synchronize()
        if step == 0:
            print("OK   ex_attn_backward -> Tensor")
        else:
            print("OK   ex_attn_step2 -> Tensor")


if __name__ == "__main__":
    main()
