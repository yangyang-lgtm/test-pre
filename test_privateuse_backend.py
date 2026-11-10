#!/usr/bin/env python3
import os
os.environ['LLVM_BINARY_DIR'] = '/home/yaoshi16ultra/workspace/llvm-project/build/bin'

import triton
import triton.language as tl
import torch
from triton.backends.privateuse.driver import PrivateUseDriver

triton.runtime.driver.set_active(PrivateUseDriver())


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    input_ptr += row_idx * n_cols
    output_ptr += row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(input_ptr + col_offsets, mask=mask, other=float('-inf'))
    row_max = tl.max(row)
    row = row - row_max
    numerator = tl.exp(row)
    denominator = tl.sum(numerator)
    result = numerator / denominator
    tl.store(output_ptr + col_offsets, result, mask=mask)


def test_vector_add():
    print("\n=== Test: Vector Addition (privateuse backend) ===")
    n = 1024
    x = torch.randn(n, dtype=torch.float32)
    y = torch.randn(n, dtype=torch.float32)
    output = torch.zeros(n, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(n, 128),)
    vector_add_kernel[grid](x, y, output, n, BLOCK_SIZE=128)

    expected = x + y
    if torch.allclose(output, expected, rtol=1e-5):
        print("PASS: Vector add")
        return True
    else:
        print("FAIL: Vector add")
        print(f"Max diff: {(output - expected).abs().max().item()}")
        return False


def test_softmax():
    print("\n=== Test: Softmax (privateuse backend) ===")
    n_rows = 32
    n_cols = 256
    input_tensor = torch.randn((n_rows, n_cols), dtype=torch.float32)
    output = torch.zeros((n_rows, n_cols), dtype=torch.float32)

    grid = (n_rows,)
    softmax_kernel[grid](input_tensor, output, n_cols, BLOCK_SIZE=256)

    expected = torch.softmax(input_tensor, dim=1)
    if torch.allclose(output, expected, rtol=1e-4, atol=1e-4):
        print("PASS: Softmax")
        return True
    else:
        print("FAIL: Softmax")
        print(f"Max diff: {(output - expected).abs().max().item()}")
        return False


if __name__ == "__main__":
    print("PrivateUse Backend Kernel Tests")
    print(f"Triton version: {triton.__version__}")
    print(f"Active driver: {triton.runtime.driver.active.__class__.__name__}")

    results = []
    results.append(test_vector_add())
    results.append(test_softmax())

    print("\n=== Summary ===")
    if all(results):
        print("All tests PASSED")
    else:
        print("Some tests FAILED")

# ● 全量测试：198/198 全部通过（默认缓存和全新缓存各跑一遍均通过；之前那次单失败是 .triton 缓存竞态，test_ne_shape0_float32 重跑即过）。
#
# 未向量化 case 分析：全部 11 个未向量化 op 都来自 test_flash_attention（其余 12 个测试文件的 outlined linalg op 全部成功 merge + 向量化），分三类：
#
# 类别: broadcast 操作数（7 个）
# op: mulf_subf_exp2、mulf_addf×2、divf×2、addi_cmpi_select_mulf_addf×2
# kernel 来源: qk - m_ij[:, None]、acc * alpha[:, None]、acc / l_i[:, None]、causal mask 的 where
# 原因: 操作数 indexing map 不同：一个 (d0,d1)->(d0,d1)，广播的 rank-1/[:, None] 操作数是 (d0,d1)->(d0)。当前 pass 要求 payload map 完全一致
# ────────────────────────────────────────
# 类别: in-place 更新（2 个）
# op: maxnumf_subf_exp2_mulf_addf
# kernel 来源: m_i = maximum(m_i, ...)、l_i = l_i*alpha + l_ij
# 原因: input aliases output（in==out 的 running max/sum）。纯 elementwise 下其实可安全向量化（每个元素只依赖自身），可以放宽 alias 检查：in==out 且 map/layout 相同时，把 in 的
# block arg 直接映射到 out 的 init transfer_read
# ────────────────────────────────────────
# 类别: linalg.index（2 个）
# op: index_index_cast、index_index_cast_addi
# kernel 来源: offs_m = start_m*BLOCK_M + arange(...)、causal mask 偏移
# 原因: body 用了循环 IV（index→index_cast），展平/向量化会改变 IV 语义，走标量管线是正确的——这些是一次性算 mask 的 op，向量化收益也低
#
# 后续可做的（按收益排序）：
# 1. 放宽 alias 检查（改动小，直接消灭 2 个 op）：in==out 且 identical map/layout 时安全；
# 2. broadcast 支持（消灭 7 个 op）：允许 rank<numLoops 的广播操作数——非广播维按行 splat（内层循环里 transfer_read rank-1 切片 + vector.broadcast），需要新的匹配分支；
# 3. linalg.index 的 2 个 op 建议保持标量路径。
