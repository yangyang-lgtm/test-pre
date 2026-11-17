# DataflowAnalyzer 设计总览与数据结构参考

本文从整体上说明 `DataflowAnalyzer` 的设计原理、数学模型、分析流水线，以及每个公开结构和内部辅助结构的作用。

相关文档：

- [dataflow-analysis.md](dataflow-analysis.md)：完整设计方案、语料支撑与示例走查。
- [DataflowAnalysis.h](../include/privateuse/Analysis/DataflowAnalysis.h)：公开数据结构定义。
- [DataflowAnalysis.cpp](../lib/Analysis/DataflowAnalysis.cpp)：分析器实现。
- [DataflowReportPass.cpp](../lib/Utils/DataflowAnalysis/DataflowReportPass.cpp)：报告 pass。

---

## 1. 总体设计原理

`DataflowAnalyzer` 不是传统的“沿 CFG 迭代求 reaching definition”，而是把 bufferize 之后的函数抽象成一张**静态的存储事件图**：

> **节点 = 物理存储 Place；边 = Transfer / Access；事件挂在循环树坐标上；`scf.for` 的 iter_arg 用显式 φ（Carrier）消解；最后给每个 SRAM alloc 计算一条生命周期区间。**

它只负责回答：

- 谁在什么时候写入了哪块存储；
- 谁在什么时候读取了它；
- 这块存储从第一次被写到最后一次被读写跨越了哪些迭代；
- 哪些站点经过 gather/scatter、进入过 carrier、是纯 copy 中转等。

它**不负责**决定 slot 数量、stage 数或调度顺序。后者由软件流水 solver 根据这些约束求解。

整体模型可以写成：

```text
M = (P, σ, E, μ, I, F)
```

- `P`：Place 集合。
- `σ`：Place 的存储空间。
- `E`：传输边、读写边、carrier 边。
- `μ`：carrier 消解函数，把 φ 的 block arg 使用归约到具体存储实例。
- `I`：每个 SRAM 站点的生命周期区间。
- `F`：`dead / through / indirect / carried / masked / dynamic` 等标志。

核心目标有三个：

1. 精确描述每个 SRAM 站点的数据来源和去向。
2. 给出 `[首次写, 最后一次读写]` 的生命周期区间。
3. 显式处理跨迭代 `iter_args`，保证生命周期不被低估。

### 1.1 数学对象与代码结构的映射

数学层是 `(P, σ, E, μ, I, F)` 这组集合与函数，代码层是若干 record。两者不是一一对应的同构表，而是**物化关系**：同一个 record 可以同时承载多个数学对象，同一个数学对象也可能体现在多个字段或后处理结果中。

| 数学对象 | 代码中的物化位置 | 说明 |
|---|---|---|
| `P` | `Place` 行本身；`id` 是元素标识 | 每个 `Place` 是 `P` 的一个元素 |
| `σ` | `Place.space` | 以字段形式表出的函数 `σ: P → Ω` |
| `E_c` | `Transfer` 行：`src/dst/dir/opKind/pathId/blockOrder` | 每条展开后的传输边一行 |
| `E_r / E_w` | `Access` 行：`place/role/pathId/blockOrder` | 每条读写事件一行 |
| 事件引用 | `Place::EventRef` 与 `Place.producers/consumers` | 把事件按 Place 重新索引，便于求 `Prod/Cons` |
| `E_φ` | `Carrier{initPlace, yieldPlace, rotation, inPlace, wroteBody}` | φ 的结构与分支事实 |
| `μ` | `Carrier` + 事件上的 `carrierId/branch/guard` | `μ` 是消解规则，不是单独存储的映射表 |
| `Π` | `ProgramPoint{pathId, blockPos}` + `FunctionInfo.paths` | `pathId` 指向循环或 `scf.if` 分支路径节点 |
| `≺` | `comparePoints` + `ProgramPoint{delta, shiftLoopId}` + `PathInfo` | 循环按位置；同一 `scf.if` 的 then/else 保守串行化 |
| `S_ℓ` | `Carrier.loopId` + `delta/shiftLoopId` | 归属判定时使用，寿命计算只做 `⊕_ℓ` |
| `I` | `Place.start/retire` | `computeLifetimes` 写入的派生字段 |
| `F` | `Place.dead/through/indirect/carried` | 派生标志 |
| `M` | `AnalysisResult.info` | `M` 是整个 `FunctionInfo` 的语义读法；`deferred` 是分析状态，不属于 `M` |
| fill 分组 | `Transfer/Access.groupId` | 附加在事件上的分组标签 |

因此，`Place` 同时出现在 `P, σ` 和 `I, F` 两行，`Transfer/Access` 同时承载边和 `μ` 的端点标签，并不表示数学对象不唯一。它表示代码把**基对象、派生结果和报告元数据**打包在同一个 record 中；只要投影关系固定，模型本身仍然是唯一确定的。

数学上的“唯一”指：给定 IR 和上述定义，`ρ`、`P`、`σ`、`E`、`μ`、`I`、`F` 的语义是唯一确定的；代码允许不同的等价物化方式，例如：

- `σ` 可以是一张独立的 `Place → Space` 表，也可以是 `Place.space` 字段；
- `I` 可以是一张独立的 `SramAlloc → Interval` 表，也可以是 `Place.start/retire`；
- `E_φ` 可以存成独立的 φ 边集合，也可以存成 `Carrier` 加事件上的 `branch/guard`；
- `≺` 可以是一个独立的比较器，也可以体现为 `ProgramPoint.delta/shiftLoopId` 加 `comparePoints`。

这和“图可以用邻接表或边表表示”是同一类表示自由度，不改变图本身。
---

## 2. 分析流水线

每个函数独立分析，`FunctionInfo` 中的 id 都是**函数本地下标**，函数之间没有 memref 数据流。

```text
buildLoopTree
  → checkModeledRegions
  → checkEscapes
  → buildPlaces
  → buildCarriers
  → collectTransfers
  → collectAccesses
  → groupFills
  → finalizePlaceAccesses
  → computeLifetimes
  → checkEmpirical
  → checkInvariants
```

| 阶段 | 作用 |
|---|---|
| `buildLoopTree` | 建立循环树、控制路径树（含 `scf.if` then/else）、循环坐标、pid 循环标记和 trip 上界 |
| `checkModeledRegions` | 检查未建模区域中的 memref 操作，避免静默丢事件 |
| `checkEscapes` | 检查 call/return 等 memref 逃逸 |
| `buildPlaces` | 建立 KernelArg、SramAlloc、LoopCarried 三类 Place |
| `buildCarriers` | 识别 `scf.for` 的 memref iter_arg，建立 φ |
| `collectTransfers` | 收集 copy/gather/scatter，生成 Transfer 边 |
| `collectAccesses` | 收集 linalg、load/store、gather/scatter 操作数、carrier 使用 |
| `groupFills` | 把 fill 与同站同迭代的 copy 分组 |
| `finalizePlaceAccesses` | 生成每个 Place 的有序 `accesses` 列表 |
| `computeLifetimes` | 计算 `start/retire/dead/through/...` |
| `checkEmpirical` | 检查当前语料经验约束 |
| `checkInvariants` | 检查模型不变量，违背时 hard error |

失败分两级：

- **deferred**：IR 合法，但当前分析器不支持或未考虑，返回 `DataflowDeferredReason`，pass 继续分析其他函数。
- **hard error**：分析器内部假设被破坏，例如 carrier arity 不匹配、transfer 方向与 space 不一致、branch/guard 不一致等，发出诊断并使 pass 失败。

---

## 3. 数学模型

### 3.1 基本域

```text
Ω = {D, S}
```

- `D`：DRAM，空间值 0。
- `S`：SRAM，空间值 128。

Place 划分为三类：

```text
P = T ⊎ S ⊎ C
```

- `T`：`KernelArg`，函数参数及其视图根，`σ(t)=D`。
- `S`：`SramAlloc`，`memref.alloc/alloca` 站点，`σ(s)=S`，**只有这一类有生命周期和槽位**。
- `C`：`LoopCarried`，`scf.for` 的 memref iter_arg，`σ(c)=⊥`，只是 φ 中继，没有真实存储。

根解析函数：

```text
ρ : Value ⇀ P
```

它沿 `reinterpret_cast / subview / cast / view / collapse_shape / expand_shape / transpose` 等视图链上溯，命中 alloc、函数参数或 iter_arg 就返回对应 Place。

**视图只做折叠，不重建 offset/size/stride，也不做 DSU 合并**：多个视图共享同一个 alloc 时仍然是同一个 Place，两个 alloc 永远不会合并。

### 3.2 程序坐标与序

控制路径集合记为 `Γ`。路径节点可以是循环或 `scf.if` 的 then/else 分支；一个事件坐标是：

```text
π = (γ; k₁,…,k_d; i) ∈ Π = Γ × ∏ K_j × B
```

- `γ`：控制路径；`PathInfo` 显式记录每个节点的父路径、父块内位置和节点种类。
- `k_j`：第 j 层归一化迭代变量。
- `i`：事件在叶 block 内的位置，也就是 `Transfer.blockOrder` / `Access.blockOrder`。

`scf.if` 的两个分支互斥，严格意义上的执行序是偏序。分析器选择**保守全序**：
同一 `scf.if` 的 then 分支整体排在 else 分支之前，分支内部仍按真实块内顺序。
因此任何真实执行路径上的顺序都保持；跨分支的区间端点取并集，只会让生命周期更宽，
不会低估。

`Π` 上定义字典序全序 `≺`：

1. 取两条路径的最长公共节点前缀。
2. 在公共循环层，仅当该层等于 `shiftLoopId` 时比较 `delta`（carrier 读侧滞后）。
3. 在第一个不同的路径节点处：先比较节点在父 block 内的位置；位置相同时
   then 分支在 else 分支之前；双方都是循环时回落到 `blockPos` 字典序。
4. 叶层比较 block 内位置。
5. 一方是循环语句本身、另一方在循环体内时，约定体内事件更晚。
6. `blockPos.back()` 严格等于 `Transfer/Access.blockOrder`，不维护第二套坐标。

`LoopExit(ℓ)` 表示第 ℓ 层循环的退出点，它也被嵌入同一套坐标：

```text
LoopExit(ℓ) = ( path(parent(ℓ)); pos(ℓ); +∞ )
```

实现上表示为：

```text
parentPath 的 blockPos ++ [loop.blockPosInParent, INT64_MAX]
```

这样循环体子树是开区间：

```text
body(k) ≺ LoopExit(ℓ) ≺ 循环之后的 epilogue 事件
```

这条规则保证原地 RMW 或循环后读取不会被提前截断。

移位算子：

```text
S_ℓ(π): k_ℓ ↦ k_ℓ − 1，其余分量不变
```

它**只用于归属判定**：确定第 k 轮读者消费的是第 k−1 轮生产者实例。寿命坐标绝不回退，见后面的 `⊕_ℓ`。

### 3.3 数据流图

边集为：

```text
E = E_c ∪ E_r ∪ E_w ∪ E_φ
```

#### 传输边 `E_c`

每条 `memref.copy` / `pumem.tile_gather` / `pumem.tile_scatter` 对应一条带标注的边：

```text
e = (ρ(src), ρ(dst), κ_e, π_e, δ_e)
```

方向 `κ_e` 由两端空间决定：

```text
κ_e = L  当 (σ(src), σ(dst)) = (D, S)   # DRAM → SRAM
κ_e = S  当 (σ(src), σ(dst)) = (S, D)   # SRAM → DRAM
κ_e = X  当 (σ(src), σ(dst)) = (S, S)   # SRAM → SRAM
```

- gather 是 `L` 的间接变体，scatter 是 `S` 的间接变体。
- gather/scatter 的索引操作数会额外记成 SRAM 读 Access。
- DRAM→DRAM 不进入模型，返回 deferred。
- space 未知是 hard error。

对每个 Transfer：

- `dst` 是 dst Place 的**生产者**。
- `src` 是 src Place 的**消费者**。

#### 读写边 `E_r / E_w`

linalg 的 `ins` 记 Read，`outs` 记 Write；`memref.load` 记 ScalarRead，`memref.store` 记 ScalarWrite；`memref.atomic_rmw` 记 Atomic。

- 标量通道**不生成 Transfer 边**，但它的 Access 仍然参与生产/消费集合。
- 所有写事件进入 `Prod`，所有读事件进入 `Cons`。
- `Atomic` 同时算读和写。

#### carrier 边 `E_φ`

对每个 memref iter_arg：

```text
c = φ(p₀, p₁)
```

- `p₀ = ρ(init)`：第 0 轮来源。
- `p₁ = ρ(yield)`：第 k≥1 轮来源。
- `p₀ = p₁` 是 inPlace，也就是原地 RMW。
- `p₀ ≠ p₁` 是 rotation，也就是跨迭代旋转。

每条 carrier 边带：

```text
branch ∈ {init, yield}
guard  ∈ {k=0, k≥1}
```

`branch/guard` 是显式标签，报告和 solver 不需要只靠 `δ` 猜语义。不变量要求：

```text
Init  ⇔ FirstIter
Yield ⇔ BodyIter
```

### 3.4 carrier 消解 μ

block arg 的每个使用都要经过 `μ` 消解回具体存储：

```text
μ(c, r) = (p₀, r,      δ=0)   当 r = 0
        = (p₁, S_ℓ(r), δ=1)   当 r ≥ 1
```

也就是：

- 第 0 轮读 init。
- 第 r≥1 轮读 yield 的第 r−1 个实例。

`S_ℓ` 只回答“这个读者读的是哪个生产者实例”，不改变读者的执行坐标。对寿命计算，读者实际执行坐标是：

```text
π ⊕_ℓ δ : k_ℓ ↦ k_ℓ + δ
```

例如 rotation carrier 中，生产者在第 k 轮写，消费者在第 k+1 轮读：

```text
生产者实例坐标：k
读者执行坐标：  k+1
寿命区间跨一个迭代边界 → 至少需要 2 个槽
```

绝不能把读者回退成 `[k, k]`，否则会低估槽位。

对写使用：`δ` 不参与执行坐标。第 0 轮写 init 存储，第 r≥1 轮写 yield 的 r−1 实例，事件仍记在第 r 轮，由 `branch/guard` 记录所写实例。

### 3.5 生命周期

对每个 `SramAlloc s ∈ S`：

#### 生产集合

```text
Prod(s) = 命中 s 的全部写事件
```

包括：

- 入方向 Transfer，例如 D2S 的 dst、S2S 的 dst。
- linalg/fill 的 `outs` 写。
- `memref.store` 标量写。
- 经 carrier 归属到 `s` 的 block arg 写使用。

所有生产者都记本迭代执行坐标，`δ=0`，不做回退。

#### 消费集合

```text
Cons(s) = 命中 s 的全部读事件
```

包括：

- 出方向 Transfer，例如 D2S 的 src、S2S 的 src。
- linalg 的 `ins` 读。
- `memref.load` 标量读。
- carrier 读者：其坐标用 `π ⊕_ℓ δ` 归一化到实际执行迭代。

#### 区间

```text
start(s)    = min≺ ( Prod(s) ∪ Cons(s) )
Touch(s)    = Prod(s) ∪ Cons(s)

retire(s)   = max≺ ( { π ⊕_ℓ δ : (π,δ) ∈ Cons(s) }
                   ∪ Prod(s)
                   ∪ { LoopExit(ℓ) | s 是 inPlace carrier 站点 } )

I(s)        = [ start(s), retire(s) ]_≺
```

几个关键点：

- `start = min≺(Prod ∪ Cons)`；没有 `scf.if` 时通常就是第一次**写**，不是 alloc
  指令的位置。
- 引入 `scf.if` 后，then/else 被保守串行化；某条分支的读者可能排在另一条分支的
  生产者之前（例如 then 读、else 写）。这时 `start` 取 `Prod ∪ Cons` 的最早点，
  否则读者会落在区间外。真实执行序里的首次写仍不小于该保守起点。
- `retire` 取 `Prod ∪ Cons` 的最晚事件，不能只取消费者，否则尾写会被漏掉。
- 对 inPlace carrier 站点，额外加入 `LoopExit(ℓ)` 作为候选；但 epilogue 读者通常坐标更大，`max` 会正确取到读者。
- 没有生产者时，`start` 暂时设为 alloc 点，并返回 `StationWithoutProducer`。

#### 标志

```text
dead     : 无消费者，区间退化为 [首写, 末写]
through  : 生产/消费均为纯 copy，且站点没有直接 Access
indirect : 经 gather/scatter
carried  : 值进入过 carrier
```

### 3.6 双缓冲 = 对区间求解

分析器不决定槽位。Solver 要找：

```text
slot  : S → N
stage : S → Z
```

对正在流水化的第 ℓ 层循环，把区间按迭代平移：

```text
S^m_ℓ(I) : 只对第 ℓ 层迭代分量 +m
```

约束是：

```text
I(s) ∩ S^m_ℓ(I(s')) ≠ ∅  ⇒  slot(s) ≠ slot(s')
```

分析只提供 `I`、事件顺序和标志；stage 数、slot 数、调度都由 solver 决定。

---

## 4. 公开数据结构

### 4.1 主要枚举

| 枚举 | 值 | 作用 |
|---|---|---|
| `Space` | `Dram=0, Sram=128, Unknown=255` | memref 空间；只接受 0/128 |
| `PlaceKind` | `KernelArg, SramAlloc, LoopCarried` | Place 三类 |
| `PathKind` | `Loop, IfThen, IfElse` | 路径节点种类：循环体或 `scf.if` 分支 |
| `Dir` | `DramToSram, SramToDram, SramToSram` | 传输方向 |
| `OpKind` | `Copy, Gather, Scatter` | 传输操作种类 |
| `Role` | `Read, Write, ScalarRead, ScalarWrite, Atomic` | Access 角色 |
| `Branch` | `None, Init, Yield` | carrier 端点属于 init 还是 yield 分支 |
| `Guard` | `None, FirstIter, BodyIter` | `k=0` 还是 `k≥1`，是 Branch 的显式冗余 |
| `DataflowDeferredReason` | 见下 | deferred 原因码 |

`DataflowDeferredReason` 分组：

| 类别 | 原因码 |
|---|---|
| 控制流/区域 | `UnsupportedControlFlow`（其它未建模控制流）、`UnmodeledRegion`、`MultiplePidLoops` |
| Place/空间 | `UnsupportedMemorySpace`、`SramAllocOutsidePid`、`PidCarriesMemref` |
| φ/根解析 | `NestedCarrier`、`TransferBetweenCarriers`、`UnresolvedMemref`、`SymbolicTrip` |
| 生命周期/逃逸 | `MemrefDealloc`、`MemrefEscape`、`ScalarWriteToSram`、`StationWithoutProducer` |
| 经验约束 | `LoadDepthMismatch`、`StoreDepthMismatch`、`MultipleLoadProducers`、`MultipleStoreConsumers` |
| 哨兵 | `None`、`Count` |

### 4.2 `PathInfo` 与 `LoopInfo`

`PathInfo` 是所有控制路径节点的统一表；根路径为 0，节点 `id` 从 0 开始，`pathId = id + 1`。

| 字段 | 作用 |
|---|---|
| `op` | 对应的循环或 `scf.if` op |
| `kind` | `Loop / IfThen / IfElse` |
| `id` | 路径节点 id |
| `parentPathId` | 父路径 id；根路径为 0 |
| `blockPosInParent` | 该节点 op 在父 block 内的位置 |
| `loopId` | 仅 `Loop` 节点有效，指向 `FunctionInfo.loops` |
| `ifId` | 仅 if 分支节点有效；同一 `scf.if` 的 then/else 共享该 id |

`LoopInfo`：

| 字段 | 作用 |
|---|---|
| `op` | 对应的 `scf.for` |
| `parentId` | 父循环 id，`-1` 表示顶层 |
| `parentPathId` | 循环 op 所在路径 id；可能指向循环或 if 分支 |
| `id` | 函数内循环 id |
| `pathId` | 该循环的 `PathInfo.pathId`；无 if 时等于 `id + 1` |
| `blockPosInParent` | 循环 op 在父 block 内的位置 |
| `trip` | 由 lb/ub/step 区间求得的最大迭代数；`-1` 表示符号 trip |
| `isPid` | 是否是 pid 循环；lb/ub/step 任一依赖 `pumem.get_physical_id` 即为 pid 循环 |

### 4.3 `ProgramPoint`

| 字段 | 作用 |
|---|---|
| `pathId` | 事件所在控制路径 `γ`，指向循环或 if 分支节点 |
| `blockPos` | 从根到叶的完整位置向量；包含循环 op 和 if op 在父 block 内的位置；末项是叶 block 位置；LoopExit 的末项是 `INT64_MAX` |
| `isLoopExit` | 是否表示循环退出点；只用于报告/调试，比较不依赖此标志 |
| `exitLoopId` | 退出的是哪个循环 |
| `delta` | 读侧相对生产者实例的 `+δ`，用于比较 |
| `shiftLoopId` | `delta` 作用在哪一层循环 |

`pointFor` 构造普通事件点：

```text
path 上各级节点的 blockPosInParent
  ++ [事件在当前叶 block 的 blockOrder]
```

`loopExit` 构造退出点：

```text
parentPath 的各级位置
  ++ [loop.blockPosInParent, INT64_MAX]
```

`comparePoints` 的顺序：

1. 展开 `pathId` 为从根到叶的路径节点。
2. 在公共循环层，仅当该层等于 `shiftLoopId` 时比较 `delta`。
3. 在第一个不同的路径节点处比较父块内位置；位置相同时 then 在 else 前；
   双方都是循环时回落到 `blockPos` 字典序。
4. 然后按 `blockPos` 字典序比较。
5. 若一个是另一个的前缀，较短者在前。
6. role、operandIndex、opOrdinal 不参与这里，它们是 `finalizePlaceAccesses` 的 tie-breaker。

### 4.4 `Transfer`

一条 Transfer 表示一次跨 Place 的数据搬运。

| 字段 | 作用 |
|---|---|
| `op` | 源 IR 操作，用于诊断和报告 |
| `src` | 源 Place id |
| `dst` | 目标 Place id |
| `dir` | `D2S / S2D / S2S` |
| `opKind` | `Copy / Gather / Scatter` |
| `pathId` | 事件所在循环路径 |
| `blockOrder` | 事件在叶 block 内的位置 |
| `masked` | 是否有真实 mask 操作数；gather/scatter 可能为 true，copy 恒为 false |
| `dynamic` | 是否存在动态 shape 或动态视图 offset/size/stride |
| `indirect` | 是否是 gather/scatter 间接访问 |
| `delta` | 仅 src 端读侧滞后；src 命中 yield carrier 时为 1，否则 0 |
| `carrierId` | 端点命中 carrier 时的 carrier id；无则为 `-1` |
| `carrierAtDst` | carrier 是否在 dst 端；src 无 carrier、dst 有 carrier 时为 true |
| `branch` | `Init / Yield`，标识归属端点 |
| `guard` | `FirstIter / BodyIter`，与 branch 配对 |
| `groupId` | fill 分组 id；用于标识同站同迭代的 fill + copy |

注意：

- `masked` 和 `dynamic` 是两个独立维度，不能合并。
- 如果 src 是 carrier，`carrierId/branch/guard` 描述 src 端点。
- 如果 dst 是 carrier，`carrierAtDst=true`，`delta=0`，所写实例由 branch/guard 表示。
- `src` 和 `dst` 同时是 carrier 会被 deferred。

### 4.5 `Access`

一条 Access 表示一次读、写、标量读写或原子操作。

| 字段 | 作用 |
|---|---|
| `op` | 源 IR 操作 |
| `place` | 被访问的 Place id |
| `role` | `Read / Write / ScalarRead / ScalarWrite / Atomic` |
| `pathId` | 事件所在循环路径 |
| `blockOrder` | 事件在叶 block 内的位置 |
| `delta` | 仅读侧有意义：读者相对其消费实例晚 δ 轮执行；写恒为 0 |
| `carrierId` | 经 μ 归属到 carrier 时记录 |
| `branch` | `Init / Yield` |
| `guard` | `FirstIter / BodyIter` |
| `groupId` | fill 所在分组 |
| `operandIndex` | 操作数序号，用于同一 op 多个访问的稳定排序 |

`Atomic` 同时计入生产者和消费者。若 Atomic 作用于 carrier，`addAccess` 会把它拆成一条 Read 和一条 Write，避免丢掉读侧的 `δ`。

### 4.6 `Carrier`

表示一个 `scf.for` 的 memref iter_arg 的 φ。

| 字段 | 作用 |
|---|---|
| `arg` | 对应的 block argument |
| `loopId` | 所属循环 id |
| `initPlace` | init 值解析到的物理 Place |
| `yieldPlace` | yield 值解析到的物理 Place |
| `rotation` | `yieldPlace != initPlace` |
| `inPlace` | `yieldPlace == initPlace` |
| `wroteBody` | body 内是否对 block arg 有写，区分 RMW 与 pass-through |
| `id` | 函数内 carrier id |
| `placeId` | 为该 carrier 创建的 `LoopCarried` 占位 Place id |

`LoopCarried` Place 没有存储，也不参与生命周期计算；它的作用是作为 φ 端点在 Place 表中的锚点。

### 4.7 `Place` 与 `Place::EventRef`

`Place` 是模型的核心节点。

| 字段 | 作用 |
|---|---|
| `id` | 函数内 Place id |
| `kind` | `KernelArg / SramAlloc / LoopCarried` |
| `space` | `Dram / Sram / Unknown` |
| `anchor` | 用于报告时现场打印名字的 Value |
| `defPathId` | 定义点所在循环路径 |
| `allocDepth` | 定义点所在循环深度 |
| `allocOrder` | 定义点在叶 block 内的位置；LoopCarried 存循环位置 |
| `producers` | 所有写事件引用 |
| `consumers` | 所有读事件引用 |
| `accesses` | 排序后的直接 Access 列表 |
| `start` | 生命周期起点，`min≺ Prod` |
| `retire` | 生命周期终点，`max≺(Prod ∪ Cons ∪ LoopExit 候选)` |
| `carries` | 涉及该 Place 的 carrier id 列表 |
| `dead` | 无消费者 |
| `through` | 生产/消费均为纯 copy，且无直接 Access |
| `indirect` | 经 gather/scatter |
| `carried` | 值进入过 carrier |

`EventRef`：

| 字段 | 作用 |
|---|---|
| `kind` | `TransferK` 或 `AccessK` |
| `id` | 在 `function.transfers` 或 `function.accesses` 中的下标 |

关键区别：

- `producers/consumers` 包含 Transfer 和 Access，是生命周期计算的事件全集。
- `accesses` 只包含 Access，不包含 Transfer；它是 solver 推导 WAR/WAW 的事件序列。
- `producers` 是“写入该 Place 的事件”，`consumers` 是“从该 Place 读取的事件”。

`Place.accesses` 的排序规则：

1. 按 `sortPoint` 的坐标比较。
2. `branch == Yield` 的事件在排序时归一化为 `δ=1`，所以 yield 的读和写会落到执行坐标 `+1` 上。
3. role rank：read < atomic < write。
4. `operandIndex`。
5. op 的稳定序号。

这样同一个 op 内 RMW 的读会排在写前面，符合 `ins` 先读、`outs` 后写的语义。

### 4.8 `FunctionInfo`

| 字段 | 作用 |
|---|---|
| `op` | 当前函数 |
| `loops` | 循环表 |
| `paths` | 控制路径节点表（循环与 `scf.if` then/else） |
| `places` | Place 表 |
| `transfers` | 所有 Transfer |
| `carriers` | 所有 Carrier |
| `accesses` | 所有 Access |
| `unresolved` | 无法解析到根的 memref value 数量 |
| `tailWrites` | 存在尾写的 station 数量 |

`tailWrites` 的判定：最后一个生产者不早于最后一个消费者，或者没有消费者，就认为该 station 有尾写。

### 4.9 `DataflowAnalyzer` 与 `AnalysisResult`

```cpp
class DataflowAnalyzer {
public:
  struct AnalysisResult {
    FunctionInfo info;
    DataflowDeferredReason deferred = DataflowDeferredReason::None;
    bool isDeferred() const;
  };

  explicit DataflowAnalyzer(FunctionOpInterface func);
  FailureOr<AnalysisResult> analyze(DenseI64ArrayAttr grid);
};
```

- 一个 `DataflowAnalyzer` 只分析一个函数。
- 使用 pimpl 隐藏 `Impl`，因此提供移动构造/移动赋值，禁止拷贝。
- `analyze(grid)` 中的 `grid` 来自 `privateuse.grid`，用于给 program id 轴定上界，从而计算 loop trip。
- 外层 `FailureOr` 失败表示 hard error。
- 内层 `AnalysisResult::deferred` 非 `None` 表示合法但未支持的 IR。
- 一旦 deferred，只有 `info.op` 有效，其他字段是半成品；消费者必须先检查 `isDeferred()`。

---

## 5. 实现层辅助结构

这些结构不在数学模型中，但支撑实现。

### 5.1 `Site`

```cpp
struct Site {
  unsigned pathId;
  int64_t blockOrder;
};
```

临时记录一个 IR 操作的循环路径和块内位置。`walkBlock` 为每个已建模 block 中的 op 建立 Site，`checkModeledRegions` 通过“找不到 Site”识别未建模区域。

### 5.2 `RootRef`

```cpp
struct RootRef {
  unsigned place;
  int64_t carrierId;
  bool isCarrier() const;
};
```

根解析结果：

- `carrierId == -1`：直接指向具体 Place。
- `carrierId >= 0`：指向一个 φ carrier。

### 5.3 `RootResolution`

```cpp
struct RootResolution {
  std::optional<RootRef> root;
  DataflowDeferredReason deferred;
};
```

表示“解析成功得到 RootRef”或者“解析失败并给出 deferred 原因”。

### 5.4 `CarrierEndpoint`

```cpp
struct CarrierEndpoint {
  unsigned place;
  int64_t carrierId;
  Branch branch;
  Guard guard;
};
```

`expandEndpoints` 把一个 carrier 根展开成两个端点：

```text
initPlace  + Branch::Init  + Guard::FirstIter
yieldPlace + Branch::Yield + Guard::BodyIter
```

Transfer 和 Access 都会通过这个展开生成带 branch/guard 的事件。

### 5.5 `Impl` 的状态成员

| 成员 | 作用 |
|---|---|
| `func` / `function` | 当前函数和正在构建的模型 |
| `loopIndices` | `Operation* → loopId` |
| `sites` | `Operation* → Site`，只记录已建模 block 中的 op |
| `opOrdinals` | `Operation* → 稳定序号`，用于最终排序 tie-break |
| `argPlaces` | 函数参数 → Place id |
| `allocPlaces` | alloc op → Place id |
| `carrierArgs` | block arg → carrier id |
| `rootCache` | Value → RootRef，根解析缓存 |
| `unresolvedValues` | 已判定无法解析的 Value 集合，避免重复计数 |
| `ivLoopIds` | induction variable → 符号 id，用于 trip 分析 |
| `loopBounds` | `SymbolBounds`，保存 pid 轴和循环 IV 的区间 |
| `nextOrdinal` | 稳定 op 序号分配器 |
| `nextLoopSym` | 循环符号 id 分配器 |

`loopBounds` 初始化时给 physical id 一个 `[0, +∞)` 的下界，因此以 physical id 为上界的循环只能得到符号 trip。

### 5.6 `ValueRange.h` 中的辅助结构

这些结构只用于**计算 loop bounds**，不重建指针、视图 offset 或 strided box。

| 结构 | 作用 |
|---|---|
| `SymKind` | `PidX, PidY, PidZ, LoopIv` |
| `ScalarExpr` | lb/ub/step 的表达式 AST；包含常量、符号、算术/位运算节点 |
| `ExprPtr` | `shared_ptr<const ScalarExpr>` |
| `Interval` | 可选上下界 `[lo, hi]`，支持 `point/unknown/isPoint` |
| `SymbolBounds` | 每个 pid 轴和每个循环 IV 的区间，用于 `evaluateInterval` |
| `SymbolValues` | 具体符号取值，用于精确求值；DataflowAnalyzer 主要使用 `SymbolBounds` |

`ScalarExpr` 的字段：

```text
kind     节点类型：Const / Sym / Add / Sub / Mul / Div / Rem / Shl / Shr / Min / Max / And / Or / Xor
value    常量值
symKind  符号种类
symId    符号 id
lhs/rhs  子表达式
```

`evaluateInterval` 会用区间算术给出保守上界；它不保留多个符号之间的相关性，因此像 `[pid, pid+8)` 这样的范围只能给保守结果。

---

## 6. 容易混淆但很关键的点

### 6.1 `Place.accesses` 和 `producers/consumers` 不是一回事

- `producers/consumers`：生命周期用的**全部事件**，包含 Transfer 和 Access。
- `accesses`：只包含 Access，按执行坐标排序，供 solver 推导 WAR/WAW。

### 6.2 三种 delta 不要混淆

1. **`Transfer.delta` / `Access.delta`**：语义上的读侧滞后，只有读事件可能为 1，写恒为 0。
2. **`ProgramPoint.delta` + `shiftLoopId`**：比较器使用的坐标移位；只在指定循环层生效。
3. **`finalizePlaceAccesses` 的排序 delta**：`branch == Yield ? 1 : 0`，即使 yield 写事件的 `Access.delta` 是 0，排序时也按执行坐标 `+1` 放置。

### 6.3 `retire` 为什么包含写事件

如果只取消费者，最后一次读之后可能还有一次尾写，例如 buffer 复用或原地更新。按区间判据，这会把槽位提前释放，随后被该写覆盖，属于低估。因此：

```text
retire = max≺(全部读事件, 全部写事件, LoopExit 候选)
```

### 6.4 `LoopExit` 为什么必须参与全序

如果只存一个“循环退出”标志，比较两个点时容易退回到循环头位置，导致原地 RMW 的 retire 被低估。把 `LoopExit` 表示成 `(parentPath; loopPos, +∞)` 后，所有点对复用同一个字典序比较器，不需要第二套特判。

### 6.5 为什么不能把 carrier 读者回退一轮

rotation carrier 的生产者在第 k 轮写，读者在第 k+1 轮读。如果把读者错误地记成第 k 轮，区间会缩成 `[k,k]`，低估槽位。正确做法是：

```text
生产者实例坐标：S_ℓ(r) = r−1
读者执行坐标：  r = (r−1) + 1
寿命区间跨迭代边界
```

也就是只用 `⊕_ℓ` 前移读者，绝不做 `⊖ 1` 回退。

### 6.6 “Station” 不是独立结构

`Station` 是 `Place` 的一个视图：

```text
kind == PlaceKind::SramAlloc
```

它的生命周期字段就是 `Place.start / retire / dead / through / indirect / carried`。`KernelArg` 和 `LoopCarried` 不是 station，不参与槽位生命周期计算。

---

## 7. 设计取舍总结

| 决策 | 原因 |
|---|---|
| 只为物理 alloc 建 station，不做 DSU 合并 | 生命周期和槽位属于物理缓冲；视图只是同一缓冲的窗口 |
| 显式保留 φ carrier | 否则跨迭代消费会被记到同一轮，生命周期被低估 |
| `δ` 只定义在消费方向 | 写事件按 branch/guard 归属，寿命坐标不回退 |
| `LoopExit` 作为 `+∞` 坐标嵌入 | 原地 RMW 和循环后读者都能正确排序 |
| `scf.if` then/else 保守串行化 | 真实执行序是偏序；串行化保持所有真实顺序，只放宽区间 |
| `start = min≺(Prod ∪ Cons)` | 跨分支读者可能排在写者前，取并集避免区间漏事件 |
| `retire` 纳入全部写事件 | 尾写不能被漏掉 |
| 完整收集所有 memref 操作数 | 避免未建模区域或标量通道静默丢事件 |
| 分析只给区间和事件序 | slot/stage 是 solver 的决策，分析不越界 |
| deferred vs hard error 分离 | 合法但不支持的 IR 与内部 bug 需要不同处理 |
| Place/事件 id 稳定分配 | 两次运行报告字节一致，便于回归测试 |

---

## 8. 一句话总结

> `DataflowAnalyzer` 把 bufferized IR 变成一张“物理存储 + 循环坐标 + φ 消解”的静态数据流图；每个 SRAM alloc 在这张图上得到 `[首次写, 最后读写]` 的保守生命周期，连同有序访问事件和标志一起交给 solver，由 solver 决定双缓冲/多 stage 的具体排布。
