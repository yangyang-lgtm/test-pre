# DRAM↔SRAM 数据流分析（DataflowAnalysis）设计方案

> **状态：设计稿（v5.3）。** 语义模型 v5.1 经两轮参考会话评审，本文档已合入
> 两轮全部修订（见第 6、7 节）；工程实现尚未落地。
> 分析点为 `backend/compiler.py` 的 `_software_pipeline`（bufferize 之后、
> `linalg→runtime_call` 之前）。分析**纯只读、纯描述**：不修改 IR，不做
> stage/slot 决策，不建模拷贝大小、范围、offset 与 mask 表达式。

本文把数学模型与工程实现合并为一份规范：第 2 节给出与实现无关的数学定义
（v5.3），第 3 节给出该模型在真实语料上的支撑事实，第 4 节给出 C++ 落地方案
（数据结构、算法、不变量、输出、集成、测试），第 5 节是数学对象与代码结构的
一一映射，第 6 节记录设计决策，第 7 节记录评审结论与修订，第 9 节给出
flash-attention 的完整走查附录。

---

## 1. 背景、目标与边界

### 1.1 管线位置

`third_party/privateuse/backend/compiler.py` 的编译流程：

```python
_specialize      # SPECIALIZE + AnalyzeAccess + 结构化/非结构化访存降级
                 # + ptr→memref + arith→linalg + PidLoop
_fuse_and_bufferize  # empty-tensor→alloc-tensor + bufferize
_software_pipeline   # 当前：print(module); return module   <-- 本分析挂载点
_transform       # linalg→runtime_call + outline + reduce→scf
                 # + convert_linalg_to_loops + scf→pumem + 方言降级
_lower_to_pumem
_emit_cpp
```

挂载点的 IR 形态：所有访存已变成 `memref.copy` / `pumem.tile_gather` /
`pumem.tile_scatter`；SRAM 缓冲已由 bufferize 分配为 `memref.alloc`（space 128，
见 `include/privateuse/Utils/Constants.h` 的 `kSramSpace`）；DRAM 侧是函数参数与
`reinterpret_cast` 视图；计算仍是 linalg。这正是模型需要的全部信息。

### 1.2 目标

1. 精确描述每个 SRAM 站点的数据来源（谁写入、从 DRAM 还是其他 SRAM）与数据去向
   （谁读取、写回 DRAM 还是被原地修改）。
2. 给出每个站点的**生命周期区间**——生产者最早写时刻到消费者最晚读时刻——作为
   双缓冲 / 多 stage 排布求解器的约束输入。
3. 把跨迭代携带（`scf.for iter_args`）表达为显式的 φ 中继，保证生命周期不被低估。

### 1.3 非目标（明确不做）

| 不做 | 理由 |
|---|---|
| 修改 IR / 插入 buffer | 分析只描述；排布是后续 pass 的事 |
| 拷贝大小、范围、offset、mask 表达式 | 用户明确不关心；双缓冲只需要方向与时刻 |
| 重建 ScalarExpr / strided box | 直接分析 `memref.copy` 即可，指针值本就未知 |
| 决定 stage 数、槽位数、调度顺序 | 属于求解器；分析端不预置任何 slot 公式 |
| 分析 ptr 数值 | kernel 中除 ptr 值外全部静态已知；ptr 只影响真正的内存别名，不影响流向 |

---

## 2. 数学模型（规范层，v5.3）

一句话概括：**把程序抽象成一个带坐标和移位标签的有向数据流图，每个 SRAM 站点
在其上得到一条生命周期区间；双缓冲排布 = 在这个图上求解。**

### 2.1 基本域

存储空间 `Ω = {D, S}`（DRAM = 0，SRAM = 128）。

Place 集合划分为三类：

```
P = T ⊎ S ⊎ C
  T: Terminal，函数参数及其视图根，σ(t) = D
  S: Station，memref.alloc 站点，σ(s) = S，唯一有生命周期/槽位的实体
  C: Carrier，scf.for 的 iter_args，σ(c) = ⊥，纯 φ 中继，无存储
```

根解析 `ρ : Value ⇀ P`：沿 `reinterpret_cast / subview / cast / view /
collapse_shape / expand_shape / transpose` 折叠，命中 alloc / 函数参数 /
iter_arg 即返回值，否则未定义（`unresolved`）。这是模型中唯一一处视图处理，
不做偏移、大小、步长。Station 按 alloc 站点建立，**不做 DSU 合并**：多个视图
共享同一 alloc 时仍是同一个 Place，但两个 alloc 永不合一。

### 2.2 程序坐标与序

循环树路径集合 `Γ`。事件坐标为

```
π = (γ; k₁,…,k_d; i) ∈ Π = Γ × ∏ K_j × B
  γ: interned 循环路径
  k_j: 第 j 层归一化迭代变量；静态循环 ∈ {0,…,trip_j−1}，其余记符号。
       trip_j 由 lb/ub/step 的 ScalarExpr 区间计算（常量、program id、
       physical id、外层 IV）；physical id 的上界来自 privateuse.num_pids，
       lb/ub/step 共享符号时区间法给出保守上界
  i: 块内位置
```

`Π` 上定义**字典序全序** `≺`：

1. 取两条路径的最长公共前缀；
2. 分歧层的比较对象是"该层循环 op 在公共父块内的位置"；
3. 若一方在分歧层已无循环（叶 op 就在该块内），比较其叶位置与另一方的循环 op 位置；
4. 位置相同（一方是循环语句本身、另一方在其体内）时，约定体内事件晚于循环语句；
5. 叶层 `blockPos.back()` 就是事件在叶块内的位置，与 `Transfer.blockOrder` /
   `Access.blockOrder` 严格相等（同一数据，不设两套坐标）。

**LoopExit 的序。** `LoopExit(ℓ)` 表示第 ℓ 层循环的退出点，它必须参与全序，
否则原地 RMW 的退休点无从比较、会被低估。规定第 ℓ 层循环体子树是开区间
`(pos(ℓ), LoopExit(ℓ))`：

- 对子树内任意点 π：`π ≺ LoopExit(ℓ)`；
- 对父块中位置大于 `pos(ℓ)` 的任意点 π（含循环之后的 epilogue 读者）：
  `LoopExit(ℓ) ≺ π`；
- 对父块中位置小于 `pos(ℓ)` 的任意点：`π ≺ LoopExit(ℓ)`。

这条规则保证"退出后读者 > LoopExit > 循环体内事件"，retire 不会被截断。

语料 0 个 `scf.if`、除 pid 外所有 bounds 为常量，因此 `≺` 是执行顺序的精确刻画，
不是偏序近似。

移位算子：对第 ℓ 层循环

```
S_ℓ(π):  k_ℓ ↦ k_ℓ − 1，其余分量不变
```

`S_ℓ` **只用于归属判定**：确定第 k 轮读者消费的是第 k−1 轮生产者实例。
`δ` 是流边上的标签（语料仅 0/1），表达读者相对其消费的生产者实例晚 δ 个迭代
执行；**任何寿命坐标都不做回退**（见 2.4）。

### 2.3 数据流图

边集 `E = E_c ∪ E_r ∪ E_w ∪ E_φ`。

**传输边 `E_c`**：每条 `memref.copy` / `pumem.tile_gather` / `pumem.tile_scatter`
对应一条带标注的边

```
e = (ρ(src), ρ(dst), κ_e, π_e, δ_e),   κ_e ∈ {L, S, X}

κ_e = L  当 (σ(src), σ(dst)) = (D, S)      # DRAM → SRAM
κ_e = S  当 (σ(src), σ(dst)) = (S, D)      # SRAM → DRAM
κ_e = X  当 (σ(src), σ(dst)) = (S, S)      # SRAM → SRAM
```

gather / scatter 是 `L` / `S` 的**间接变体**（索引操作数额外记入 `E_r`）。
DRAM→DRAM 不是模型边，记 anomaly；未知空间 hard fail。

**读/写边 `E_r` / `E_w`**：linalg 的 `ins` / `outs`、`memref.load` / `memref.store`
（标量通道）、pumem 操作的 memref 操作数，标注 `(ρ, role, π)`。标量通道只记录，
不产生 Station 生命周期。

**carrier 边 `E_φ`**：对每个 iter_arg `c`，设 init 值 `p₀ = ρ(init)`、
yield 值 `p₁ = ρ(yield)`：

```
p₀ --(δ=0, 仅 k_ℓ = 0)--> c
p₁ --(δ=1, 仅 k_ℓ ≥ 1)--> c
p₀ = p₁  → inPlace（RMW）；否则 → rotation
```

**carrier 解析 `μ`**：block arg 的每个使用按迭代分裂，把 φ 消回具体存储。μ 只
决定**归属**，不改变读者的执行坐标。设使用点在第 ℓ 层的迭代为 `r`，μ 的第二
项记录**归属后的生产者实例坐标**：

```
μ(c, r) = (p₀, r,      δ=0)    当 r = 0     # 读者消费 init
        = (p₁, S_ℓ(r), δ=1)    当 r ≥ 1     # 读者消费 yield 的 r−1 实例
```

即 k ≥ 1 时实例坐标 `k = S_ℓ(r) = r−1`。`δ` 是该读者相对该实例的执行滞后量
（实例在 k，读者在 k+δ = r 执行）。把读者记入寿命区间时，用实例坐标
`k ⊕_ℓ δ` 得到读者执行坐标（就是原始使用点 r）；`S_ℓ` 只回答"这个读者读的是
哪个生产者实例"，不参与寿命坐标计算。

### 2.4 生命周期

对每个 Station `s ∈ S`：

**生产** `Prod(s)`：命中 `s` 的传输边、fill 写事件，以及经 carrier 归属到 `s` 的
block arg 写使用。所有生产者事件都记**本迭代执行坐标**，不做任何回退。fill 与其
同组 copy 共享 `group_id`（分组限同站、同迭代），组 token 取组内最早写——fill
真实先占用缓冲，start 提前只会让寿命更长，是保守方向；因此只设单 token：

```
token(s) = min≺ { π : π ∈ Prod(s) }        # 生产者执行坐标，δ = 0
```

**消费** `Cons(s)`：命中 `s` 的读取、标量读、出方向传输。直接读者记执行坐标；
carrier 读者晚于其消费的生产者实例 δ 个迭代执行，因此其归一化坐标为
`π ⊕_ℓ δ`（`π` 为归属后的实例坐标，第 ℓ 层迭代分量 +δ，即读者实际执行
迭代），**绝不回退成 π ⊖ 1**：

```
⊕_ℓ:  k_ℓ ↦ k_ℓ + δ，其余分量不变      # 归属性前移，只用于 carrier 读者

retire(s) = max≺ ( { π } ∪ { π ⊕_ℓ δ : (π, δ) ∈ Cons(s), δ > 0 }
                   ∪ { LoopExit(ℓ) | s 为 inPlace 站点 } )
```

符号约定：`S_ℓ`（减一）只出现在 μ 的归属判定里；寿命计算只用 `⊕_ℓ`（前移）。
相对区间跨 δ 个迭代边界，由 solver 按 2.7 的 `S^m` 归一化。

`LoopExit` 只是一个候选，**退出后的 epilogue 读者坐标通常更大**（matmul 累加器、
flash-attention 的 m/l 都在循环后读取），max 会取到它，因此寿命不会被截断。
inPlace 站点用 `wroteBody` 区分 RMW（body 内有写）与 pass-through（只读）：
两者都按上式保守计算，solver 可据此决定是否放宽。

**区间与标志**：

```
I(s) = [ token(s), retire(s) ]_≺
dead     : 无消费者
through  : 生产/消费均为纯 copy，且站点无直接访问
indirect : 经 gather / scatter
carried  : 值进入过 carrier
```

### 2.5 模型本体

```
M = (P, σ, E, μ, I, F)
```

- `P, σ`：Place 与空间；
- `E`：全部 DRAM↔SRAM、SRAM↔SRAM、SRAM↔DRAM 边（拷贝 + gather/scatter + 标量通道）；
- `μ`：carrier 消解，把 φ 归约到具体存储；
- `I`：每站生命周期区间；
- `F`：`dead / through / indirect / carried / masked` 标志。

两个性质保证模型可用：

1. **完备性**：语料中跨空间流动只有 `memref.copy` 与 `tile_gather/scatter`
   三种形式（965 + 579 + 15 条 copy，6 + 2 条 gather/scatter），全部入边；
   标量通道单独记录。
2. **保守性**：`token` 取下界，`retire` 取上界并沿 carry 边闭包；任何求解器用
   `I` 推得的槽位数只会偏多、不会偏少。

### 2.6 示例

**matmul（test/cases，单 pid 迭代 + 1 层 k 循环）**：

```
E_c:  (arg_A, a14, L, pid@2)             # fill + 部分 copy 进 SRAM
      (arg_B, b23, L, pid@4)
      (accum, out, S, pid@9)
E_r:  matmul 读 a14 @ (pid;k)@5，b23 @ (pid;k)@5
E_w:  matmul 写 accum @ (pid;k)@5（in-place 计算站）
I(a14)    = [pid@2, (pid;k)@5]                   # start 在循环外，消费在循环内
I(accum)  = [pid@3(fill), max(pid@9, LoopExit(k))]  # 退出后读者 > LoopExit
```

若 epilogue 在 k 循环之后读 accum，该读者坐标大于 `LoopExit(k)`，retire 取 max
后落在读者上；若没有 epilogue 读者，才由 `LoopExit(k)` 兜底。

**flash-attention（旋转 carrier）**：

```
c = scf.for iter_args(%m_i_17 = %m_i)；p₀ = ρ(%m_i)，p₁ = ρ(%m_ij)，rotation
block arg 的使用经 μ 归属到 p₁；生产者写迭代 k，消费它的读者在 k+1 执行
I(p₁) = [k, k+1] → 跨一个迭代边界，下界 2 槽
绝不把读者回退成 [k, k]（v5.2 的错误写法）
```

### 2.7 双缓冲 = 求解

求解器要找槽位函数与阶段函数

```
slot : S → N,    stage : S → Z
```

使任意两个经迭代平移后区间重叠的站点不共用槽位：

```
I(s) ∩ S^m(I(s')) ≠ ∅  ⇒  slot(s) ≠ slot(s')
```

旋转 carrier 的站点（生产于 body(k)、消费于 body(k+1)）区间必跨迭代边界，
下界 2 槽；跨 k 循环存活的 accumulator 同理。这些都是 `I` 的直接读数，
分析端不预置任何槽位公式。

### 2.8 为什么是这个模型（关键取舍）

| 决策 | 理由 |
|---|---|
| Station = alloc 站点，不做 DSU 合并 | 生命周期与槽位是物理缓冲的属性；视图只是同一缓冲的窗口 |
| 必须保留 φ carrier | 若把 block arg 的使用同时记到 init 与 yield 两个站点而不带 δ，会把 body(k) 的消费放到同一迭代，生命周期缩成 [k,k]，**低估**槽位，不是保守近似 |
| δ 只做归属，寿命不回退 | 把 carrier 读者回退到生产者迭代会把区间缩成 [k,k]、低估槽位；读者按其执行迭代记 +δ |
| LoopExit 参与全序（子树开区间右端点） | 只存标志会让比较落到循环头，原地 RMW retire 被低估 |
| retire = max(消费者执行坐标, LoopExit 候选) | epilogue 读者通常大于退出点，硬编 LoopExit 会截断寿命；也不硬编码"+1 迭代" |
| 槽位公式不进模型 | 分析只给约束，求解器决定 stage/slot |
| JSON 输出每站有序 access 列表 | solver 由事件序推 WAR/WAW；显式反依赖边属于求解决策，分析不越界 |
| 不做 ScalarExpr / 视图偏移折叠 | 用户明确不需要；方向与坐标已足以排布 |

---

## 3. 语料支撑（test/cases + test/flaggems 全量）

全量 597 个测试中 578 通过；19 个失败均为既有问题（`Unknown key: 'bufferize'`，
`compiler.py:397`），与数据流无关。共收集 587 份 `_software_pipeline` 处模块 IR。

### 3.1 规模

| 指标 | 数量 |
|---|---|
| 模块数 | 587 |
| SRAM alloc（Station 候选） | 2787 |
| `memref.copy` D→S / S→D / S→S | 965 / 579 / 15 |
| `pumem.tile_gather` / `tile_scatter` | 6 / 2 |
| 条件传输（`scf.if`） | 0 |
| alloc/arg 逃逸 | 0 |
| 第三方空间（非 0/128） | 0 |
| dealloc / DRAM→DRAM 拷贝 | 0 / 0 |
| 拷贝根类型 | 仅 {函数参数, alloc} |

### 3.2 结构事实

| 事实 | 数量 |
|---|---|
| load 位于 pid 深度 / 用户循环内 | 923 / 42 |
| store 位于 pid 深度 | 579（全部） |
| 读位于 pid 深度 / 内层循环 | 5289 / 324 |
| load 深度 == alloc 深度 | 965/965 |
| 每站 load 生产者最大值 | 1 |
| 每站 store 消费者最大值 | 2 |
| 每站读取者最大值 | 54 |
| 最大嵌套 | pid + 1 层用户循环 |
| 单次 fill+copy 组（masked load） | 268 条（动态 subview）+ 2 条（整块） |
| 无 mask 整块 copy（load） | 695 条 |
| store 直接写 alloc / 经 subview | 439 / 140 |
| D→S 直接 alloc / view-of-alloc | 697 / 268 |
| S→D 直接 alloc / view-of-alloc | 439 / 140 |
| S→S view→alloc / alloc→alloc | 13 / 2 |
| SRAM 标量读站点（memref.load） | 78 个站点、102 次 |
| DRAM 标量 store（溯源到 SRAM load） | 62 次（全部可溯源） |
| 未落 DRAM 的 DRAM 标量 load | 90 次 |
| 循环携带 memref（flash-attention） | 10 个类，全部 rotation，δ=1 |
| 只写无读（dead）站点 | 18（模型语义；如 argmax 的 f32 reduce 结果。朴素 SSA 扫描的 31 中另 13 个经 carrier / gather 间接读后并非 dead：10 个 flash `%m_i`、2 个 gather_rows `%v_1`、1 个 flip `%mask_2`） |

### 3.3 站点原型（和为 2787）

| 原型 | 数量 |
|---|---|
| resident（无 D2S/S2D 边界拷贝） | 1521 |
| input tile（仅 D2S 边界） | 690 |
| output tile（仅 S2D 边界） | 295 |
| in-place 计算（D2S + S2D，有直接访问） | 262 |
| through（D2S + S2D，纯 copy 且无直接访问） | 11 |
| gather/scatter 辅助（无直接访问） | 8 |

分区按优先级互斥统计：gather/scatter → through → in-place → input → output →
resident。`resident` 含 18 个 dead 站点（全部无边界拷贝）以及 18 个仅 S2S
拷贝的站点；`in-place` 含 6 个只读中转站（D2S + S2D，只有读访问，因此不是
through）；`gather/scatter 辅助` 为 6 个 gather 端点 + 2 个 scatter 端点。

---

## 4. 工程方案（实现层）

### 4.1 文件与构建布局

对齐现有 `AccessAnalysis` 约定：

```
include/privateuse/Analysis/DataflowAnalysis.h
lib/Analysis/DataflowAnalysis.cpp                  # 并入 PrivateUseAnalysis
include/privateuse/Utils/DataflowAnalysis/Passes.td
include/privateuse/Utils/DataflowAnalysis/Passes.h
include/privateuse/Utils/DataflowAnalysis/CMakeLists.txt
lib/Utils/DataflowAnalysis/DataflowReportPass.cpp
lib/Utils/DataflowAnalysis/CMakeLists.txt          # 并入 PrivateUseDataflowAnalysis
test/DataflowAnalysis/{basic,masked_load,s2s_carried,carrier_retire,
                       carrier_write,carrier_access,gather_scatter,
                       scalar_channel,dead_station,inplace_rmw,
                       unsupported}.mlir
privateuse.cc                                      # 新增 add_analyze_dataflow
backend/compiler.py                                # _software_pipeline 接入
```

构建接线：

- `lib/Analysis/CMakeLists.txt` 追加 `DataflowAnalysis.cpp`；
  `LINK_LIBS` 追加 `MLIRLinalgDialect MLIRMemRefDialect PUMemIR`
  （现有已含 `MLIRArithDialect MLIRIR MLIRSCFDialect MLIRSupport
  PrivateUseUtils TritonIR`）。
- `lib/Utils/CMakeLists.txt` 追加 `add_subdirectory(DataflowAnalysis)`。
- `include/privateuse/Utils/DataflowAnalysis/CMakeLists.txt`：

```cmake
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name DataflowAnalysis)
add_public_tablegen_target(PrivateUseDataflowAnalysisPassIncGen)
```

- `lib/Utils/DataflowAnalysis/CMakeLists.txt` 的 `DEPENDS`
  `PrivateUseDataflowAnalysisPassIncGen`，`LINK_LIBS` 照抄
  `PrivateUseAccessAnalysis` 并追加 `PrivateUseAnalysis`。

Pass 定义（`Passes.td`）：

```tablegen
def AnalyzeDataflow : Pass<"privateuse-analyze-dataflow", "mlir::ModuleOp"> {
  let summary = "Report the static DRAM<->SRAM dataflow model";
  let description = [{
    Prints places, transfers, carriers, station start/retire coordinates and
    invariant checks. Never mutates the IR.
  }];
  let constructor = "mlir::triton::createAnalyzeDataflowPass()";
  let options = [
    Option<"jsonPath", "json", "std::string", "",
           "Write the dataflow model as JSON to this path">,
    Option<"strict", "strict", "bool", "false",
           "Promote empirical anomalies to errors">
  ];
}
```

### 4.2 数据结构与 API

```cpp
// include/privateuse/Analysis/DataflowAnalysis.h
namespace mlir { namespace triton {

enum class Space : uint8_t { Dram = 0, Sram = 128, Unknown = 255 };
enum class PlaceKind : uint8_t { Terminal, Station, Carrier };
enum class Dir : uint8_t { L, S, X };      // D2S / S2D / S2S
enum class OpKind : uint8_t { Copy, Gather, Scatter };
enum class Role : uint8_t { Read, Write, ScalarRead, ScalarWrite, Atomic };

struct LoopInfo {
  scf::ForOp op;
  LoopInfo *parent = nullptr;
  unsigned id = 0;
  unsigned pathId = 0;              // interned 路径（root→this）
  int64_t blockPosInParent = 0;
  int64_t trip = -1;                // 符号区间上的最大迭代数；-1 = 符号
  bool isPid = false;
};

// 执行坐标：path 各级 loop op 的块内位置 + 叶 op 位置
// blockPos.back() == Transfer/Access.blockOrder；LoopExit 按 2.2 的右端点规则参与全序
struct ProgramPoint {
  unsigned pathId = 0;
  SmallVector<int64_t> blockPos;    // blockPos[i] 对应 path 第 i 层
  bool isLoopExit = false;
  unsigned exitLoopId = 0;
};

struct Transfer {
  Operation *op;
  unsigned src = 0, dst = 0;        // Place id
  Dir dir;
  OpKind opKind;
  unsigned pathId = 0;
  int64_t blockOrder = 0;
  bool masked = false, indirect = false;
  int64_t delta = 0;                // +δ 前移；端点命中 carrier 时生成 init/yield 两条
  int64_t groupId = -1;
};

struct Access {
  Operation *op;
  unsigned place = 0;
  Role role;
  unsigned pathId = 0;
  int64_t blockOrder = 0;
  int64_t delta = 0;                // +δ 前移：读者在生产者实例之后 δ 轮执行
  int64_t groupId = -1;             // fill 所在组
  // carrier 的每个读写使用生成两条归属：initPlace δ=0、yieldPlace δ=1
  // （普通读写为 Access，copy 端点为 block arg 时为 Transfer）
};

struct Carrier {
  BlockArgument arg;
  unsigned loopId = 0;
  unsigned initPlace = 0;           // k = 0 来源
  unsigned yieldPlace = 0;          // k ≥ 1 来源（δ = 1）
  bool rotation = false;            // yieldPlace != initPlace
  bool inPlace = false;             // yieldPlace == initPlace
  bool wroteBody = false;           // body 内对 block arg 有写（RMW）；否则 pass-through
};

struct EventRef {
  enum Kind : uint8_t { TransferK, AccessK } kind;
  unsigned id = 0;
};

struct Place {
  unsigned id = 0;
  PlaceKind kind;
  Space space;
  Value anchor;
  std::string name;                 // IR 名，报告用
  unsigned defPathId = 0;
  unsigned allocDepth = 0;          // alloc 所在循环深度（pid 内为 1）
  SmallVector<EventRef> producers, consumers;  // 含 transfer 与 access 事件
  SmallVector<Access> accesses;     // 按坐标排序，solver 据此推 WAR/WAW
  bool dead = false, through = false, indirect = false, carried = false;
  ProgramPoint start, retire;
  SmallVector<unsigned> carries;    // Carrier id
};

struct DataflowModel {
  SmallVector<LoopInfo> loops;
  SmallVector<Place> places;
  SmallVector<Transfer> transfers;
  SmallVector<Carrier> carriers;
  SmallVector<std::string> warnings;   // empirical
  SmallVector<std::string> errors;     // correctness
  LogicalResult toJSON(llvm::raw_ostream &os) const;
};

class DataflowAnalysis {
public:
  DataflowAnalysis(ModuleOp module, bool strict);
  LogicalResult run();
  const DataflowModel &getModel() const;

private:
  LogicalResult buildLoopTree();
  LogicalResult buildPlaces();
  LogicalResult collectTransfers();
  LogicalResult groupFills();
  LogicalResult attachAccesses();
  LogicalResult computeLifetimes();
  LogicalResult checkInvariants();
  LogicalResult resolvePlace(Value v, unsigned &place,
                             unsigned pathId, int64_t &delta);
  ...
};

}}
```

### 4.3 算法阶段

**P0 循环树与 pid 标记。** 遍历每个 func 的 `scf.for`，从 lb/ub/step 的 def 链
找 `pumem.get_physical_id` 判定 pid 循环（每函数至多 1 个，否则 fail）。
`ScalarExprBuilder` 把 lb/ub/step 折成 ScalarExpr，`evaluateInterval` 用
`privateuse.grid` 与 `privateuse.num_pids` 求区间；step 区间不跨 0 时，按
lb 下界、ub 上界与最小步长取最大迭代数作为 trip。区间法会丢失共享符号的
相关性（例如 `[pid, pid + 8)` 只能给保守上界）；无法定界的循环记 empirical
warn（模型用相对 δ + 符号 LoopExit，不依赖绝对 trip）；pid 循环承载数据则
hard fail（C5）。intern 路径，记录每层 loop op 的块内位置。

**P1 Place 表与 φ。** 遍历 `memref.alloc` → Station（space 非 0/128 → fail）；
Station 的 `defPath` 必须包含 pid 层，否则该 alloc 跨 pid 实例共享，直接 bail
（语料 alloc 全在 pid 内）。函数参数及视图根 → Terminal；`scf.for` 的
`iter_args`/`scf.yield` 配对 → Carrier（数量不匹配 fail；`yield` 根等于 `init` 根
→ `inPlace`，否则 `rotation`；body 内对 block arg 有写 → `wroteBody`）。
`resolvePlace` 沿视图上溯；block arg 的每个读写使用命中 carrier 后按 μ 生成
init/yield 两条归属，再路由回具体 Station。Station 按 alloc 建，不做 DSU 合并。

**P2 Transfer 提取。** 扫描 `memref.copy`、`pumem.tile_gather`、
`pumem.tile_scatter`：解析 src/dst Place；端点命中 carrier 时先按 μ 拆成
init(δ=0)/yield(δ=1) 两条，再按空间决定 `L/S/X`；gather 的索引操作数记 SRAM
读 Access，值目标记 SRAM 写；scatter 对称。两条共享 op/opKind/groupId，填
`pathId/blockOrder/masked/indirect`；同一 block 内用 op 序号保证确定性。
`memref.dealloc` 若出现，记为寿命终止（retire 候选）并记 empirical warn，
不 bail、不归入 C3；语料 0 例。

**P3 fill 分组。** `linalg.fill outs(%alloc 或视图)` 与**同一 Station、同一迭代**
内写该站的 copy 归入同 group；fill 与 copy 都是写事件，单 token 起点取组内最早写
（fill 先占缓冲，提前 start 是保守方向）。`groupId` 暴露给 solver 以定位"数据
就绪"的那条 copy。268 条 masked load 全部走 `fill + subview + copy`，2 条整块
fill+copy；695 条无 fill 的 load 不进组。

**P4 Access 归集。** linalg 的 `getDpsInputs()` → Read、`getDpsOutputs()` →
Write；gather/scatter 操作数按语义拆分；`memref.load` → ScalarRead、
`memref.store` → ScalarWrite（标量通道只记 Access，不生成 Transfer；标量写
Terminal 是合法通道，不属于逃逸）。carrier block arg 的每个读写使用生成两条
Access：initPlace 分支 δ=0、yieldPlace 分支 δ=1，供 solver 展开 k=0/k≥1。
白名单外的 memref 值传递（call/return/写入其他缓冲）视为逃逸（C3）。

**P5 生命周期。** 对每个 Station：

1. 生产者写 = 入方向 Transfer ∪ fill 写 ∪ carrier 归属到该站的 block arg 写；
   全部记本迭代执行坐标（δ=0），不做回退；
2. `start = min≺(生产者写)`（fill 组按组算一次，单 token）；
3. 候选消费者 = 出方向 Transfer ∪ Read/ScalarRead；carrier 读者带 δ=+1，
   `π` 取归属后的实例坐标，归一化坐标为 `π ⊕_ℓ δ`（读者实际执行迭代，
   比实例晚 δ 轮）；
4. 沿 carrier 边做传递闭包（δ 累加；语料只有 1 层、δ∈{0,1}）；
5. `retire = max≺(归一化消费者坐标 ∪ {LoopExit(ℓ) | inPlace 站点})`——epilogue
   读者坐标通常大于 LoopExit，max 会正确取到；**绝不把消费者回退到生产者迭代**；
6. 无消费者 → `dead`；生产/消费均为纯 copy 且站点无直接访问 → `through`。

**P6 不变量。** 见 4.4。

**P7 序列化。** Place id 按稳定 walk 序（func → block → op 序号）分配，事件按
func → Place → 坐标排序输出，保证两次运行字节一致。

### 4.4 不变量与 bail 策略

**正确性（诊断 + `signalPassFailure`）**：

| 编号 | 约束 | 当前语料 |
|---|---|---|
| C1 | copy/gather/scatter 操作数可解析到 Place | 100% |
| C2 | space ∈ {0,128}，无第三方空间 | 0 例 |
| C3 | alloc/arg 不逃逸（无 call/return/将 memref 值写入其他缓冲）；标量通道 `memref.store` 到 Terminal 豁免；dealloc 不属逃逸 | 0 例 |
| C4 | carrier 良构（数量、单路由、旋转/原地一致） | 10/10 |
| C5 | pid 循环不得承载 memref（破坏跨实例独立性） | 0 例 |
| C6 | SRAM alloc 的 defPath 含 pid（无跨实例共享） | 全部 |
| C7 | 方向唯一属于 {L,S,X}，无 DRAM→DRAM | 0 例 |

C3 的标量通道豁免是必须的：3.2 节的 62 次 DRAM 标量 store 就是写 Terminal，
若按"store 到别处"一律判逃逸，语料回归会 errors ≠ 0。

**经验性（记录 + warn，`--strict` 升级为错误）**：

- 每 Station ≤1 个 load 生产者、≤2 个 store 消费者；
- load 深度 == alloc 深度；store 深度 == pid 深度；
- 承载数据的循环 bounds 静态（模型本身用相对 δ + 符号 LoopExit，不依赖绝对 trip）；
- 无 `scf.if` / 条件 transfer、无 dealloc、无嵌套 carrier、δ ∈ {0,1}；
- 读取者上限、dead 站点数、标量通道计数等**绝对量不进不变量**，只作为第 3 节
  的语料快照与回归基线。

诊断格式：`[privateuse-dataflow] error: ...`，正确性失败时 pass 失败并中止。

### 4.5 输出格式

**文本报告**（`llvm::errs()`，与 `AccessAnalysis` 一致，FileCheck 友好）：

```
[privateuse-dataflow] func @matmul_kernel loops={0:pid,1:static(trip=2)} places=6 transfers=6 carriers=0
[privateuse-dataflow] place p0 kind=terminal space=dram anchor=%arg0
[privateuse-dataflow] place p1 kind=station space=sram anchor=%a_14 path=pid[3] producers=2 consumers=1
[privateuse-dataflow] transfer t0 dir=D2S src=p0 dst=p1 masked=1 group=0 at=pid[2]
[privateuse-dataflow] transfer t1 dir=S2D src=p4 dst=p2 at=pid[9]
[privateuse-dataflow] carrier c0 loop=1 arg=%m_i_17 init=p3 yield=p4 rotation=1 wroteBody=1
[privateuse-dataflow] station p1 start=pid[3] retire=pid.k[7] carried=1 dead=0 through=0 indirect=0
[privateuse-dataflow] access p1 role=write at=pid[3] d=0
[privateuse-dataflow] access p1 role=read at=pid.k[7] d=+1
[privateuse-dataflow] check no_escape ok; check single_load warn count=0 ...
[privateuse-dataflow] summary funcs=1 places=6 stations=4 transfers={D2S=2,S2D=1,S2S=1} carriers=0 dead=0 unresolved=0 errors=0
```

文本报告逐条打印 `access` 行（role、路径坐标、`d=+δ`），用于 FileCheck 钉住
carrier 方向与 init/body 双分支；`station` 行的 start/retire 已是归一化执行坐标。

**JSON**（`--json=<path>`，求解器接口）：

```json
{"functions":[{
  "name":"matmul_kernel",
  "loops":[{"id":0,"pid":true,"trip":"symbolic"},{"id":1,"parent":0,"trip":2}],
  "places":[{"id":1,"kind":"station","space":128,"anchor":"%a_14",
             "defPath":[0],"allocDepth":1}],
  "transfers":[{"src":0,"dst":1,"dir":"D2S","opKind":"copy","masked":true,"group":0,
                "at":{"path":[0],"pos":[3],"delta":0}}],
  "carriers":[{"arg":"%m_i_17","loop":1,"init":3,"yield":4,
               "rotation":true,"wroteBody":true}],
  "stations":[{"place":1,
               "start":{"path":[0],"pos":[3],"delta":0},
               "retire":{"path":[0,1],"pos":[4,7],"delta":1},
               "dead":false,"carried":true,"through":false,"indirect":false,
               "accesses":[
                 {"role":"write","at":{"path":[0],"pos":[3],"delta":0}},
                 {"role":"read","at":{"path":[0,1],"pos":[4,6],"delta":0}},
                 {"role":"read","at":{"path":[0,1],"pos":[4,6],"delta":1}}]}]
}],
 "summary":{"stations":4,"transfers":{"D2S":2,"S2D":1,"S2S":1},
            "carriers":0,"dead":0,"unresolved":0,"errors":0}}
```

`trip` 输出为符号区间上的最大迭代数；pid loop 的 physical id 上界来自
`privateuse.num_pids`，缺失该属性时退化为 `"symbolic"`。

标志以 `stations[]` 为权威，`summary` 只放聚合计数（派生数据，不双写）。每站
`accesses` 必须按坐标排序输出：solver 由"同站 read/write 事件序"自行推导
WAR/WAW 反依赖，分析不输出显式反依赖边。carrier 的每个读写使用出现两条
（init δ=0 / yield δ=1；copy 端点为 block arg 时是两条 Transfer，普通读写是
两条 Access）；`delta` 是 **+δ 前移量**（读者在生产者实例之后 δ 轮执行），
不是回退映射。

`LoopExit` 用 `{"exit": <loopId>}` 表示。文本报告里 `producers` / `consumers`
是事件计数：Transfer + fill/读写 Access（masked load 的 `producers=2` = fill
Access + copy Transfer）。

### 4.6 流水线集成

`privateuse.cc`（对照 `add_analyze_access`，当前 110 行）：

```cpp
#include "privateuse/Utils/DataflowAnalysis/Passes.h"
...
m.def("add_analyze_dataflow", [](PassManager& pm) {
  pm.addPass(triton::createAnalyzeDataflowPass());
}, py::arg("pm"));
```

`backend/compiler.py` 的 `_software_pipeline`（当前 193 行）：

```python
def _software_pipeline(self, module):
    pm = ir.pass_manager(module.context)
    pm.enable_debug()
    print(module)  # 验证期保留 IR dump；正式接入时移除或改为 env 控制
    privateuse.add_analyze_dataflow(pm)
    pm.run(module, 'software_pipeline')
    return module
```

接口分层（对齐 `AccessAnalysis` + `analyze-access` 模式）：`DataflowAnalysis` 是
纯只读 C++ 类，报告 pass 只是薄包装。双缓冲 pass 直接 in-process 构造
`DataflowAnalysis` 拿 `DataflowModel`，不走 JSON；`--json` 仅供测试与回归。
`run()` 返回 `LogicalResult`，正确性错误同时进入 `model.errors`；求解器遇到
correctness 错误应优雅 bail（回退非流水版本）。不要把 model 缓存进 IR 属性
（IR 一变即失效）。

launcher cache 语义不变：测试需 `TRITON_ALWAYS_COMPILE=1` 或新
`TRITON_CACHE_DIR`。

### 4.7 测试方案

**lit**（`test/DataflowAnalysis/`，RUN 模式与 `test/AccessAnalysis/basic.mlir` 一致）：

```
// RUN: privateuse-opt %s -privateuse-analyze-dataflow -o /dev/null 2>&1 | FileCheck %s
```

| 文件 | 覆盖点 |
|---|---|
| `basic.mlir` | 直连 D2S/S2D、station start/retire |
| `masked_load.mlir` | fill+subview+copy、masked=1、group、组起点 |
| `s2s_carried.mlir` | 旋转 carrier、init/yield 双 Place、区间跨迭代 |
| `carrier_retire.mlir` | 旋转 retire 钉在读者执行迭代 k+1（非回退 k） |
| `carrier_write.mlir` | copy dst=block arg：路由回 station 并计入生产者 |
| `carrier_access.mlir` | carrier 读写的 init/body 双分支 Access |
| `gather_scatter.mlir` | indirect L/S、索引 SRAM 读 |
| `scalar_channel.mlir` | SRAM 标量 load、DRAM 标量 store 记录 |
| `dead_station.mlir` | 只写站点 dead=1 |
| `inplace_rmw.mlir` | 原地 RMW + 循环后读者：retire > LoopExit |
| `unsupported.mlir` | 第三方空间 / 逃逸 → 诊断 + 非零退出 |

只读性检查：

```
// RUN: privateuse-opt %s -privateuse-analyze-dataflow -o - | diff - %s
```

**语料回归**（`test/DataflowAnalysis/corpus_check.py`，不进 lit/CI）：

1. 新 `TRITON_CACHE_DIR` + `TRITON_ALWAYS_COMPILE=1`，全量跑 `test/cases` 与
   `test/flaggems`；
2. 断言 `unresolved = 0`、`errors = 0`；
3. 快照计数：965 D2S / 579 S2D / 15 S2S / 6 gather / 2 scatter / 10 carrier /
   18 dead / 78 标量读站点（作为基线，变化需显式重基线；18 与朴素 SSA
   扫描的 31 之差见 3.2 的 carrier/gather 读说明）；
4. 两次运行报告逐字节一致；
5. 已知 19 个与本次无关的 `Unknown key: 'bufferize'` 失败排除。

### 4.8 分期与验收

| 阶段 | 内容 | 出口条件 |
|---|---|---|
| M1 | 骨架 + 循环树 + Place + copy 分类 + 文本报告 + binding | `basic.mlir` 通过 |
| M2 | 视图折叠、fill 分组、masked、标量通道、dead | matmul / argmax lit 通过 |
| M3 | carrier / δ / retire 闭包、LoopExit | flash-attention lit 通过 |
| M4 | 不变量 C1–C7、JSON、语料基线与确定性 | 587 模块 0 unresolved / 0 error |
| M5 | （独立任务）求解器消费 JSON 做双缓冲排布 | 不在本方案内 |

整体复杂度 O(op 数)；无 IR 变更、无跨 pass 状态、输出确定性；循环 trip 复用
`ValueRange` 的 `ScalarExprBuilder` / `evaluateInterval`，pid 标记来自
`pumem.get_physical_id` 的 def 链。

---

## 5. 数学对象与实现的映射

| 数学对象 | 代码结构 |
|---|---|
| `P, σ` | `Place{kind, space, anchor, defPathId, allocDepth}` |
| `E_c` | `Transfer{src, dst, dir, opKind, pathId, blockOrder, delta}` |
| `E_r / E_w` | `Access{place, role, pathId, blockOrder, delta=+δ}`；carrier 使用双条；`Place.accesses` 按坐标排序 |
| 事件引用（计数） | `EventRef{kind, id}`；`Place.producers/consumers` 含 transfer 与 access |
| `E_φ, μ` | `Carrier{initPlace, yieldPlace, rotation, inPlace, wroteBody}` |
| `Π, ≺, S_ℓ` | `ProgramPoint{pathId, blockPos[], isLoopExit}` + 比较函数 |
| `I, F` | `Place{start, retire, dead, through, indirect, carried}` |
| `M` | `DataflowModel`（solver 走内存接口）+ `toJSON()`（仅测试/回归） |
| `token` 的 fill 组 | `Transfer/Access{groupId}` |

---

## 6. 设计决策记录（v5 → v5.3）

1. **Station 按 alloc 站点建立，不做 DSU 合并。** 物理缓冲才有生命周期；
   视图只是同一缓冲的窗口。
2. **φ carrier 必须保留。** 评审反例：把 block arg 的使用同时记到 init 与 yield
   两个站点、却不带 δ，会把 body(k) 的消费放进同一迭代，生命周期缩为 [k,k]，
   **低估**槽位——这是不安全近似，不是保守近似。
3. **δ 是 k⃗ 上的移位标签，不新增坐标轴。** 执行序仍是字典序全序；δ 只改变
   数据来源坐标。
4. **retire 取消费者执行坐标；carrier 归属只决定实例配对，寿命绝不回退。**
   生产者实例 i 的读者在 i+δ 执行；若把读者回退到 i，区间缩成 [i,i]、低估槽位。
   `LoopExit` 只是消费者候选，退出后读者坐标通常更大，取 max 后才是不截断的寿命。
5. **槽位公式移出模型。** 分析给区间与顺序，求解器决定 stage/slot。
6. **fill 是有序写事件 + group_id。** 单 token 起点取组内最早写；分组限同站同迭代。
7. **gather/scatter 是 L/S 的间接变体**（v5 曾漏掉，语料 6 + 2 条）。
8. **不变量分层。** 正确性（assert + bail）与经验事实（仅记录）分开，避免把
   语料偶然性固化成正确性约束。
9. **LoopExit 是右端点，不是标记。** 只存 `isLoopExit` 标志会让比较落到循环头，
   低估原地 RMW 的 retire；现显式规定 LoopExit ≻ 子树内所有点、≺ 父块中其后
   所有点（子图为开区间）。
10. **JSON 输出每站有序 access 列表。** solver 由事件序推 WAR/WAW；分析不输出
    显式反依赖边，避免越界替求解器做决策。
11. **不变量不含语料绝对计数。** 965/579/18/78 等是回归快照，不是每模块约束；
    只有结构性事实进入 empirical。
12. **生产者写事件记本迭代执行坐标。** 经 carrier 写入站点的写使用归属到
    init/yield 分支，但写事件坐标不回退；只有读者的 +δ 前移参与 retire。
13. **函数级 SRAM alloc 直接 bail，dealloc 视为寿命终止。** 前者跨 pid 实例共享
    破坏模型；后者不能误判为逃逸（语料 0，先 bail 可接受）。
14. **Place id 按稳定 walk 序分配。** 在排序输出之外保证 id 本身可复现。
15. **carrier 使用生成两条归属。** initPlace(δ=0) 与 yieldPlace(δ=1) 各一条
    （普通读写为 Access，copy 端点为 block arg 时为 Transfer）；单条事件无法
    表达 k=0/k≥1 两分支。
16. **producers/consumers 是 EventRef 而非 Transfer id。** fill 写与读写 Access
    也计入；masked load 的 `producers=2` = fill Access + copy Transfer。
17. **C3 豁免标量通道。** `memref.store` 标量数据到 Terminal 是合法 DRAM 写，
    否则语料 62 次标量 store 会被误判为逃逸、回归 errors ≠ 0。
18. **dealloc 统一为 empirical warn + retire 候选。** 不 bail、不归入逃逸，
    与 empirical 列表一致。

---

## 7. 评审结论（v5.1 → v5.3）

两轮评审确认语义模型已收敛，修订均为落地 correctness / 保守性 / 可决策性问题，
已全部合入正文：

| 项 | 结论 |
|---|---|
| Q1 坐标全序 | 采用显式比较谓词：`LoopExit(ℓ)` 位于循环体子树开区间的右端点（≻ 体内所有点、≺ 父块中其后所有点）；叶层 `blockPos.back()` 与 `blockOrder` 统一；新增 `inplace_rmw` lit |
| Q2 carrier 双边 | Carrier 表（init/yield/rotation/inPlace/wroteBody）已含双边，不显式存 δ 边；block arg 的每个读写使用生成 init/yield 两条归属（copy 为两条 Transfer、普通读写为两条 Access），写使用计入生产者；区分 RMW 与 pass-through |
| Q3 LoopExit | `retire = max(消费者执行坐标, LoopExit 候选)`；carrier 读者按其执行迭代 `π ⊕ δ` 记录，**不做 `S_ℓ` 回退**；退出后读者通常更大，不得硬编 LoopExit |
| Q4 fill token | 单 token、组内最早写正确且保守；不设双 token；`groupId` 暴露给 solver；分组限同站同迭代 |
| Q5 标志位置 | `stations[]` 为权威，JSON 顶层只放聚合计数；`through` 不存 producer→consumer 对 |
| Q6 不变量 | C1–C4 保持硬失败；C5 拆为"pid 承载硬失败 + bounds-static 经验性"；新增"alloc 在 pid 内"硬失败；绝对计数移出 empirical |
| Q7 JSON | 增加每站有序 `accesses`（carrier 双分支）、`allocDepth/defPath`、transfer `opKind`；不输出显式 WAR/WAW 边 |
| Q8 接入 | 分析类 + 薄报告 pass 两层；双缓冲 pass 走 C++ 内存接口，JSON 仅回归；`run()` 返回 `LogicalResult`，solver 遇 correctness 错误优雅 bail；不缓存 model 进 IR 属性 |

**终审（v5.2 → v5.3）另发现并修复一处 critical 不一致**：v5.2 的 retire 公式
对 carrier 读者又做了一次 `S_ℓ` 回退，等于把 δ 抵消，旋转 station 的区间被缩成
`[k,k]`（1 槽）——正是 6.2 警告过的低估。v5.3 改为 retire 取消费者执行坐标
（carrier 读者 `+δ` 前移），`S_ℓ` 只做归属；并新增 `carrier_retire` /
`carrier_write` / `carrier_access` 三个 lit 钉住方向。

终审同时采纳：carrier 读写生成两条归属事件、producers/consumers 改为 EventRef、
C3 豁免标量通道、M4 出口改 C1–C7、dealloc 严重度与 empirical 统一。

首轮额外风险项（生产者坐标 δ、dealloc 语义、函数级 alloc bail、稳定 Place id）
继续有效，其中"生产者坐标回退"已按终审意见改为"生产者不回退、只有读者 +δ"。

---

## 8. 参考

- `third_party/privateuse/backend/compiler.py`：`_specialize` / `_fuse_and_bufferize`
  / `_software_pipeline` / `_transform`。
- `third_party/privateuse/privateuse.cc`：`add_analyze_access` 绑定模式。
- `include/privateuse/Analysis/AccessAnalysis.h`、
  `lib/Analysis/AccessAnalysis.cpp`：分析类与 `PrivateUseAnalysis` 约定。
- `include/privateuse/Utils/AccessAnalysis/`、
  `lib/Utils/AccessAnalysis/AccessReportPass.cpp`：报告 pass 与 lit 约定。
- `include/privateuse/Utils/Constants.h`：`kSramSpace = 128`。
- 语料采集与校验脚本（仓库外）：`/tmp/pu_analyze.py`、`/tmp/pu_alias.py`、
  `/tmp/pu_ir_all/`。

---

## 9. 附录 A：flash-attention 全链路走查

本附录用 `test/cases/test_flash_attention.py`（第 3 节语料快照中的
`mod_8420_00122.mlir`）把第 2–5 节的对象逐一落到真实 IR 上。块内位置均为
0 基 op 序号（含 arith 等所有 op）。坐标写法：

```
pid@25          path [0]，blockPos [25]
pid.k@[26,13]   path [0,1]，blockPos [26,13]
                26 是 k 循环在 pid 体内的位置，13 是事件在 k 体内的位置
```

下面的 IR 是摘录，`...` 与省略的 op 不计入位置；位置以完整模块的 op 序列为准。

### A.1 分析点 IR

去掉无关细节后：

```mlir
tt.func public @flash_attention_kernel(
    %Q_ptr: memref<?xf32>, %K_ptr: memref<?xf32>,
    %V_ptr: memref<?xf32>, %O_ptr: memref<?xf32>) {
  %pid = pumem.get_physical_id : index
  scf.for %arg4 = %pid to %c16 step %c32 {              // L0: pid
    %qk   = memref.alloc() : memref<64x64xf32, 128>     // pid[4]
    %qk_3 = memref.alloc() : memref<64x64xf32, 128>     // pid[5]
    linalg.fill ... outs(%qk_3)                         // pid[6]
    %l_i  = memref.alloc() : memref<64xf32, 128>        // pid[7]
    %l_i_4= memref.alloc() : memref<64xf32, 128>        // pid[8]
    linalg.fill ... outs(%l_i_4)                        // pid[9]
    %m_i  = memref.alloc() : memref<64xf32, 128>        // pid[10]
    linalg.fill ... outs(%m_i)                          // pid[11]
    %alloc= memref.alloc() : memref<64x64xf32, 128>     // pid[12]
    linalg.fill ... outs(%alloc)                        // pid[13]
    ...
    %q_13 = memref.reinterpret_cast %Q_ptr ...          // pid[23]
    %q_14 = memref.alloc() : memref<64x64xf32, 128>     // pid[24]
    memref.copy %q_13, %q_14                            // pid[25] D→S

    %acc = scf.for %acc_16 = %c0_i32 to %c128_i32 step %c64_i32
        iter_args(%m_i_17 = %m_i) -> (memref<64xf32, 128>) {  // pid[26] L1
      ...
      %k_24 = memref.alloc() : memref<64x64xf32, 128>   // k[7]
      linalg.fill ... outs(%k_24)                       // k[8]
      %k_25 = memref.subview %k_24 ...                  // k[9]
      memref.copy %k_23, %k_25                          // k[10] D→S, masked
      linalg.transpose ins(%k_24) outs(%qk)             // k[11]
      ...
      %m_ij = memref.alloc() : memref<64xf32, 128>      // k[16]
      linalg.generic ... ins(%m_i_17, %l_i) outs(%m_ij) // k[17] 生产 m_ij
      ...
      linalg.generic ... ins(%m_i_17, %m_ij) outs(%m_i_17) // k[22] RMW φ
      linalg.generic ... ins(%m_i_17) outs(%m_i_17)        // k[23] RMW φ
      ...
      linalg.generic ... ins(%l_i_4, %m_i_17) outs(%l_i_4) // k[35] 读 φ
      scf.yield %m_ij : memref<64xf32, 128>             // k[37]
    }
    ...
    memref.copy %alloc, %reinterpret_cast : ...         // pid[31] S→D
  }
}
```

四个关键点：

- 所有跨空间流动已经是 `memref.copy`，没有隐藏的 load/store；
- 11 个 SRAM alloc（space 128），4 个 DRAM Terminal；
- 一个 pid 循环（`L0`），嵌套一个静态 `trip=2` 的 k 循环（`L1`）；
- 唯一的跨迭代对象是 `%m_i_17`，它是 `L1` 的 `iter_args`。

`scf.yield`（k[37]）只是 carrier 边，不是一次存储写。`%m_ij` 的真实生产写
是 k[17] 的 linalg.generic；这一点在 A.5 会直接影响寿命方向。

### A.2 循环树与坐标

```
L0: scf.for %arg4 = %pid to %c16 step %c32        # pid；trip 由 num_pids 上界计算
└─ L1: scf.for %acc_16 = 0 to 128 step 64          # trip = 2
      iter_args(%m_i_17 = %m_i) -> memref<64xf32, 128>
```

路径与代表事件：

| 事件 | 坐标 |
|---|---|
| Q 的 D→S copy | `pid@25` |
| K 的 fill | `pid.k@[26,8]` |
| K 的 D→S copy | `pid.k@[26,10]` |
| `%m_ij` 的生产写 | `pid.k@[26,17]` |
| 对 `%m_i_17` 的最后一次 φ 读 | `pid.k@[26,35]` |
| `%m_ij` 的 `scf.yield`（carrier 边） | `pid.k@[26,37]` |
| epilogue 对 `%alloc` / `%qk` 的读 | `pid@29` |
| O 的 S→D copy | `pid@31` |

字典序：

```
pid@25 ≺ pid@26 (L1 op)
  ≺ pid.k@[26,8] ≺ pid.k@[26,10] ≺ pid.k@[26,17]
  ≺ pid.k@[26,22] ≺ pid.k@[26,23] ≺ pid.k@[26,26]
  ≺ pid.k@[26,35] ≺ pid.k@[26,37]
  ≺ LoopExit(L1)
  ≺ pid@29 ≺ pid@31
```

`LoopExit(L1)` 是 L1 体子树开区间的右端点：

- 大于体内所有点（含 k[37]）；
- 小于父块中 L1 之后的点（`pid@27` 起的 epilogue 和 `pid@31` 的 S→D）。

这就是 `%alloc`、`%qk`、`%l_i_4` 的 retire 能落在循环之后，而不是被截断到
`LoopExit` 的原因。

### A.3 Place 与视图折叠

| Place | kind | space | anchor / def | 说明 |
|---|---|---|---|---|
| `p_Q` | Terminal | D | `%Q_ptr` | `%q_13` 的视图根 |
| `p_K` | Terminal | D | `%K_ptr` | `%k_23` 的视图根 |
| `p_V` | Terminal | D | `%V_ptr` | `%v` 的视图根 |
| `p_O` | Terminal | D | `%O_ptr` | `%reinterpret_cast` 的视图根 |
| `p_qk` | Station | S | `pid[4]` | qk 矩阵；循环内原地更新 |
| `p_qk3` | Station | S | `pid[5]` | 1.0 mask |
| `p_li` | Station | S | `pid[7]` | reduction 中间量 |
| `p_li4` | Station | S | `pid[8]` | l 累加器；循环后读 |
| `p_mi` | Station | S | `pid[10]` | carrier init |
| `p_alloc` | Station | S | `pid[12]` | acc 累加器；循环后读 + S→D |
| `p_q14` | Station | S | `pid[24]` | Q tile |
| `p_k24` | Station | S | `k[7]` | K tile |
| `p_qk26` | Station | S | `k[12]` | qk 临时结果 |
| `p_mij` | Station | S | `k[16]` | carrier yield |
| `p_v29` | Station | S | `k[29]` | V tile |
| `c0` | Carrier | ⊥ | `%m_i_17` | init `p_mi`，yield `p_mij` |

视图折叠只做根解析：

```mlir
%q_13  = reinterpret_cast %Q_ptr      → p_Q
%k_23  = reinterpret_cast %K_ptr      → p_K
%k_25  = subview %k_24                → p_k24
%qk_27 = expand_shape %m_ij           → p_mij
%acc_28= expand_shape %m_i_17         → carrier，按 μ 分成 p_mi / p_mij
```

两条容易写错的规则：

1. **视图 op 本身不是 Access。** `%qk_27` 在 k[18]，但 `p_mij` 的读 Access
   记在消费它的 linalg op k[19]；`%acc_28` 在 k[25]，读 Access 记在 k[26]。
2. **Station 按 alloc 建立，不做 DSU 合并。** `%k_25` 和 `%k_24` 是同一个
   Station；两个不同 alloc 永不合一。

### A.4 Transfer 与 Access

跨空间边：

| id | op | src → dst | 方向 | 属性 | at |
|---|---|---|---|---|---|
| `t0` | Q copy | `p_Q` → `p_q14` | `L` | masked=0 | `pid@25` |
| `t1` | K copy | `p_K` → `p_k24` | `L` | masked=1, group=0 | `pid.k@[26,10]` |
| `t2` | V copy | `p_V` → `p_v29` | `L` | masked=1, group=1 | `pid.k@[26,32]` |
| `t3` | O copy | `p_alloc` → `p_O` | `S` | masked=0 | `pid@31` |

本 kernel 没有 S→S、gather/scatter、标量通道；这些形态见第 3 节语料和第 4.7
节的 lit。

fill 与 linalg 读写是 Access：

| op | Place | role | at |
|---|---|---|---|
| k[8] fill | `p_k24` | Write | `[26,8]`，group=0 |
| k[11] transpose | `p_k24` | Read | `[26,11]` |
| k[13] matmul | `p_q14`, `p_qk` | Read | `[26,13]` |
| k[13] matmul | `p_qk26` | Write | `[26,13]` |
| k[17] generic | `p_mi`/`p_mij` (φ) | Read | `[26,17]` |
| k[17] generic | `p_mij` | Write | `[26,17]` |
| k[22] alpha | `p_mi`/`p_mij` (φ) | Read + Write | `[26,22]` |
| k[23] exp | `p_mi`/`p_mij` (φ) | Read + Write | `[26,23]` |
| k[26] generic | `p_mi`/`p_mij` (经 `%acc_28`) | Read | `[26,26]` |
| k[35] generic | `p_mi`/`p_mij` (φ) | Read | `[26,35]` |
| pid[29] generic | `p_alloc`, `p_qk` | Read | `[0,29]` |

K/V 的 copy 使用动态大小的 subview（`%k_20 = min(128−k, 64)`），因此
`masked=1`。`masked` 是本分析的标志（动态 subview / 部分拷贝），不是 IR
属性；动态 extent 表达式本身不分析。两者各自与同迭代的 fill 组成一组：

```
p_k24: fill k[8] (group 0) + copy k[10] (group 0) → start = k[8]
p_v29: fill k[30] (group 1) + copy k[32] (group 1) → start = k[30]
```

因此 masked load 的 `producers=2` = fill Access + copy Transfer；组 token 取
fill，因为 fill 真实先占用缓冲。

### A.5 carrier、μ 与 δ

```
c0: arg=%m_i_17  loop=L1
    init=p_mi   yield=p_mij
    rotation=1  inPlace=0  wroteBody=1
```

`%m_i_17` 的使用：

| 静态 op | 使用 | `p_mi` 分支 | `p_mij` 分支 |
|---|---|---|---|
| k[17] | 读 block arg | δ=0，仅 k=0 | δ=1，k≥1 |
| k[22] | 读 + 写 block arg | δ=0 | δ=1 |
| k[23] | 读 + 写 block arg | δ=0 | δ=1 |
| k[25] | `expand_shape`（视图） | 无 Access | 无 Access |
| k[26] | 经 `%acc_28` 读 | δ=0 | δ=1 |
| k[35] | 读 block arg | δ=0 | δ=1 |

`wroteBody=1` 来自 k[22] 和 k[23] 的 `outs(%m_i_17)`：这是 RMW，不是
pass-through。

#### A.5.1 两条归属是必须的

k[22] 同一个 op 里同时有 `p_mij` 的两个读和一个写：

```mlir
linalg.generic ... ins(%m_i_17, %m_ij) outs(%m_i_17)
//                 ^ p_mij read δ=1（上一轮 yield 的实例）
//                            ^ p_mij read δ=0（本轮 k[17] 刚写的实例）
//                                      ^ p_mij write δ=1（RMW 旧实例）
```

k[17] 也同时有 `p_mij` 的读和写：

```mlir
linalg.generic ... ins(%m_i_17, %l_i) outs(%m_ij)
//                 ^ p_mij read δ=1（旧实例）
//                                      ^ p_mij write δ=0（新实例）
```

单条 `Access.place` 无法表达“同一 op 对不同迭代实例的读写”；这就是第 4.2
节每条 carrier 使用生成两条归属（init δ=0 / yield δ=1）的原因。

#### A.5.2 时间线与 δ 方向

`p_mij` 的生产写是 k[17]（不是 k[37] 的 `scf.yield`）。实例 k 的生命周期：

```
W_k   @ (k,   17)          # linalg.generic 写 %m_ij
R_k   @ (k+1, 17)          # k+1 轮读 block arg（以及 22/23/26/35）
I_k = [ (k,17), (k+1,35) ] # 跨一个迭代边界
```

同一 k+1 轮里，`p_mij` 的新实例在 k[17] 就开写，而旧实例的最后读者在
k[35]：

```
instance k:    W_k @(k,17) ..................... R_k @(k+1,35)
instance k+1:                 W_{k+1} @(k+1,17) ...
                              └──── overlap ────┘
```

所以两个实例在 k+1 轮内共存，`p_mij` 下界是 2 槽。

坐标约定（避免把 δ 加两次）：

```
使用点原始坐标 r
归属实例 k = S_ℓ(r) = r−1        # 只回答“读的是哪个实例”
读者执行坐标 = k ⊕ δ = r          # 真正进入寿命的坐标
区间 = [k, k+1]
```

**绝不把读者记到 k。** 写成 `[k,k]` 会让相邻实例看起来不重叠，槽位被低估成
1——这是 v5.3 修掉的 critical bug。附录的 k[35] 就是钉住这个方向的锚点。

#### A.5.3 生产写不回退

k[22]/k[23] 对 block arg 的写发生在第 k 轮，写的是上一轮 yield 实例的存储
（k=0 时写 init `p_mi`）；它们作为生产写记录在本轮 k，不参与任何回退。
`retire` 只对读者使用 +δ。同理，`p_mi` 的生产写是 pid[11] 的 fill，不是
L1 的 `iter_args` 边。这就是第 6 节决策 12：**生产者记本迭代，只有读者
前移。**

### A.6 生命周期

| Station | start | retire | 标志 / 说明 |
|---|---|---|---|
| `p_q14` | `pid@25` | `pid.k@[26,13]` | input tile；retire 取最后一轮 k 的 `k[13]`；无 epilogue |
| `p_k24` | `pid.k@[26,8]` | `pid.k@[26,11]` | masked；group 0；fill 为组起点 |
| `p_v29` | `pid.k@[26,30]` | `pid.k@[26,33]` | masked；group 1 |
| `p_qk3` | `pid@6` | `pid.k@[26,14]` | resident；跨 k 循环 |
| `p_li` | `pid.k@[26,15]` | `pid.k@[26,36]` | 迭代内 scratch；alloc 在 pid[7] |
| `p_li4` | `pid@9` | `pid@28` | 原地累加器；epilogue 读 |
| `p_mi` | `pid@11` | `pid.k@[26,35]`（仅 k=0） | carried；朴素扫描会误判 dead |
| `p_mij` | `pid.k@[26,17]` | `pid.k@[26,35]` +1 | carried, rotation；`[k,k+1]`，2 槽 |
| `p_alloc` | `pid@13` | `pid@31` | 原地累加器 + S→D；epilogue > LoopExit |
| `p_qk` | `pid.k@[26,11]`（k=0） | `pid@29` | 原地 RMW；epilogue 读 |
| `p_qk26` | `pid.k@[26,13]` | `pid.k@[26,33]` | 迭代内 scratch |

三个关键读数：

1. **`p_k24` 的 start 是 fill 不是 copy。** fill 与 masked copy 同组，组
   token 取最早写；copy 的数据就绪位置由 `groupId` 留给 solver 定位。
2. **`p_mi` 不是 dead。** 只按 SSA 值统计的朴素扫描会看到 `%m_i` 的
   `reads=0`（因为使用都经过 `%m_i_17`）；carrier 归属后它在 k=0 被
   k[17]/[22]/[23]/[26]/[35] 消费，retire 落在 k[35]。
3. **`p_alloc` / `p_qk` / `p_li4` 的 retire 在循环之后。** epilogue 读
   （`pid@29` / `pid@28`）和 S→D（`pid@31`）都大于 `LoopExit(L1)`；max
   会取到它们，不会截断到退出点。

本 kernel 没有 dead / through / indirect 站点；这些标志的样例分别见第 3 节
的 argmax、纯 copy 中转和 gather/scatter 模块。

### A.7 solver 读法

solver 对 `p_mij` 看到的核心事实：

```
I_k = [ (k,17), (k+1,35) ]
I_{k+1} = [ (k+1,17), (k+2,35) ]

(k+1,17) < (k+1,35)  ⇒  I_k ∩ I_{k+1} ≠ ∅
                     ⇒  slot(k) ≠ slot(k+1)
```

因此至少 2 槽。这个结论来自“新实例在 k+1 的 k[17] 已开写，而旧实例的最后
读者在 k[35]”，不是来自任何硬编码的“carrier 需要 2 槽”。

其余站点的读法：

- `p_q14`：一个 pid 实例一份，跨整个 k 循环，无旋转 → 通常 1 槽；
- `p_alloc`：循环内串行 RMW，生命周期覆盖整个 k 循环 → 不能靠双缓冲并行化
  累加器；
- `p_qk`：循环内原地更新，epilogue 读 → 反依赖由有序 Access 推导，分析不
  输出显式 WAR 边；
- 最后一轮：`trip=2` 时 `p_mij` 的实例 1 没有 k+1 读者，solver 用 trip 与
  `LoopExit` 截断；分析仍给保守的跨迭代上界。

### A.8 这个例子钉住的回归点

| lit | 本例子中的锚点 |
|---|---|
| `basic.mlir` | `p_q14` 的 D→S `pid[25]`、`p_alloc` 的 S→D `pid[31]` |
| `masked_load.mlir` | `p_k24` / `p_v29` 的 fill+copy 分组、`masked=1`、`producers=2` |
| `carrier_retire.mlir` | `p_mij` 的 retire 在 k+1 的 `[26,35]`，不是 k |
| `carrier_write.mlir` | k[22]/[23] 的 block arg 写 → 生产写在 k，不回退 |
| `carrier_access.mlir` | k[22] 同时出现 `p_mij` 的 δ=0 / δ=1 两条读 |
| `inplace_rmw.mlir` | `p_alloc` / `p_qk` 的 epilogue 读 > `LoopExit(L1)` |

`gather_scatter.mlir`、`scalar_channel.mlir`、`dead_station.mlir`、
`unsupported.mlir` 的形态不在本 kernel 中，分别由第 3 节语料和第 4.4 节的
反例覆盖。

一句话总结这份走查：**`%m_ij` 的生产写在 k[17]，读者在 k+1 的 k[35]；
carrier 归属只负责把读者配到实例 k，寿命坐标必须停在读者执行迭代 k+1。**
把这条钉住，flash-attention 的 2 槽结论和 §2.4 的 retire 方向就同时成立了。
