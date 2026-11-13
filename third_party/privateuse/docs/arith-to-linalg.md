# 值算子降级：privateuse-arith-to-linalg Pass 设计方案

> **状态：已实现并验证——采用「Triton 方言默认非法 + 白名单」模型。**
> `privateuse-arith-to-linalg` 不再「整方言 legal + 逐个 addIllegalOp」，而是
> `addIllegalDialect<triton>` 之后只把结构性 / program-info 的 `tt.func / tt.return /
> tt.get_program_id / tt.get_num_programs` 显式标 legal；其余 tt 算子一律非法——有 converter
> 的转成 linalg/tensor，没 converter 的 legalize 直接硬报错（lower-or-error）。
> 已落地 converter：核心 6 值算子（splat / broadcast / expand_dims / reshape / make_range /
> bitcast）+ dense splat 常量（DenseConstant）+ **tt.reduce / tt.dot / tt.trans**（本轮从参考
> tmp 裁剪移植）。精简：删指针保留守卫与末尾 warning walk；`ModuleOp /
> UnrealizedConversionCastOp` 不再显式标 legal（partial conversion + 无 TypeConverter 下冗余）；
> `tt.call` 的整模块提前 return 已删除；`tt.gather` 从访存分析移出、归为本 pass 的值算子
> （converter 待补）。目录已按参考式子目录拆分（`Conversion/AccessLowering`、`Utils/AccessAnalysis`，
> 见 `docs/access-lowering.md`）。构建通过，privateuse 全套 lit 12/12，corpus 在 arith-to-linalg
> 处零 legalize 失败。下文「文件清单 / 复用 / 验证」对应已实现代码；历史评审与实测细节见第 7 节。

本文解释这个 pass 要解决什么问题、它在管线中的位置、转换目标（ConversionTarget）
与转换模式（ConversionPattern）如何设计、有哪些必须保留的不变量与已知陷阱，以及
日后实现时的文件接线与验证方法。方案已与参考仓库会话 triton-b2 评审一轮，第 7 节
记录采纳的修订。阅读顺序建议：先看第 1 节的宏观模型与范围，再看第 3 节的核心设计，
实现前查第 4 节的文件接线与第 6 节的验证。

---

## 1. 宏观模型

### 1.1 在管线中的位置

工作目录 `test-pre` 的 privateuse 后端是精简重构版。`third_party/privateuse/backend/compiler.py`
的 `_specialize_ir` 当前按以下顺序装配（访问降级三层已就位）：

```python
privateuse.add_specialize_kernel(pm)
passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
pm.add_pass(passes.common.add_canonicalizer(pm))
privateuse.add_analyze_access(pm)            # 只诊断，不改 IR
privateuse.add_lower_structured_access(pm)   # 规整访问 → memref.copy
privateuse.add_lower_unstructured_access(pm) # 逐元素访问 → pumem.gather/scatter
privateuse.add_ptr_to_memref(pm)             # 函数签名 / 地址链 → memref
pm.add_pass(passes.common.add_canonicalizer(pm))
pm.add_pass(passes.common.add_cse(pm))
```

新 pass 插在 `add_ptr_to_memref` 之后、收尾 canonicalize / cse 之前：

```python
privateuse.add_ptr_to_memref(pm)
privateuse.add_arith_to_linalg(pm)           # 本 pass：残留张量值算子 → linalg/tensor
pm.add_pass(passes.common.add_canonicalizer(pm))
pm.add_pass(passes.common.add_cse(pm))
```

分层关系是「访问降级先把指针世界清干净，值算子降级再处理残留的张量计算」。访问降级
（见 `docs/access-lowering.md`）把所有 `tt.load / tt.store` 与指针链降成
memref + pumem + `bufferization.to_tensor / to_buffer`；此后 IR 里剩下的纯值计算交给本 pass。

### 1.2 访问降级后残留的张量值算子

访问降级完成后，IR 里仍残留这些**张量值算子**：

- elementwise `arith` / `math`（`arith.mulf`、`arith.addf`、`math.sqrt` 等，操作数是 ranked tensor）；
- `arith.constant dense<...>`（dense splat 张量常量）；
- `tt.make_range`（生成索引张量）；
- 值用途的 `tt.splat / tt.broadcast / tt.expand_dims / tt.reshape / tt.bitcast`；
- 访问降级遗留的值 `tt.splat`（结构化/非结构化 pass 都不再改写值 splat，原样保留 `tt.splat`，统一留到本 pass 转 `linalg.fill`）。

这些算子目前**只靠 lit RUN 行临时挂的上游 pass** 处理：

```
-privateuse-lower-structured-access -privateuse-ptr-to-memref -canonicalize
-convert-elementwise-to-linalg -one-shot-bufferize="allow-unknown-ops=1" -canonicalize
```

`backend/compiler.py` 的真实管线并未接入这一步（`_emit_cpp` / `_compile_object` 仍是
`NotImplementedError`），所以真实编译路径上这些残留算子无人处理。

### 1.3 为什么上游 `-convert-elementwise-to-linalg` 不够

上游 `convert-elementwise-to-linalg` 只认识带 `ElementwiseMappable` trait 的标准算子
（arith / math），**不认识 Triton 的值算子**：`tt.make_range`、`tt.splat`、`tt.broadcast`、
`tt.expand_dims`、`tt.reshape`、`tt.bitcast` 都不会被它转换。

这在 unstructured 路径上尤其致命：`LowerUnstructuredAccessPass` 把索引 / 值张量包进
`bufferization.to_buffer`，但 `make_range / arith.subi / dense 常量 / 值 tt.splat / cmpi mask`
仍残留为张量算子。上游 pass 转不动它们，`-one-shot-bufferize` 也就无法把这些索引 / 值张量
bufferize，管线不完整。

本 pass 用 privateuse 原生的 `privateuse-arith-to-linalg` 补齐：既复用上游 elementwise helper
处理 arith / math，又自带 Triton 值算子的 converter，使残留张量值算子全部收敛到 linalg / tensor。

### 1.4 范围界定：默认全非法，逐个补 converter

逻辑参考 `tmp/triton/third_party/privateuse/lib/Conversion/ArithToLinalg`，但采用「方言默认非法」
模型（见 3.2）：任何到达本 pass 的 tt 算子都必须有 converter，否则硬报错。

**已移植**：

- 核心值算子 6 个（splat / broadcast / expand_dims / reshape / make_range / bitcast）+ dense splat 常量；
- **`tt.reduce` → `linalg.generic`（reduction iterator）**、**`tt.dot` → `linalg.matmul` /
  `linalg.batch_matmul`**、**`tt.trans` → `linalg.transpose`**（本轮从参考裁剪移植）。

**仍未移植**（一旦到达即 legalize 失败；当前 corpus 未覆盖）：
`scan / cat / join / split / gather / histogram / unsplat / map_elementwise / dot_scaled /
fp_to_fp / clampf / precise_sqrt / precise_divf / mulhiui / extern_elementwise /
elementwise_inline_asm / print / assert`。

**走白名单放行**（标量、不可 linalg 化，留到 GPU→LLVM 才降）：`tt.get_program_id`、
`tt.get_num_programs`；结构性 `tt.func`、`tt.return`。

参考里的 load / store / addptr、`PtrAnalysis / MaskAnalysis / TritonStructured / TritonTilingExt`
等机制**不移植**：访问由降级三层覆盖，工作目录无那些组件。因此仍不整体搬参考
`ConversionPatterns.hpp`（它顶部 include 了缺失组件），而用 trimmed 头逐个加 converter。
移植时对参考做了两处裁剪：`tt.reduce` 的 fp16/bf16 高精度提升（参考的 `highPrecisionReduce`
option）与 `tt.dot` 的 `inputPrecision`（tf32 / bf16x3）都**未移植**，一律走朴素 lowering。

---

## 2. 关键约束

### 2.1 指针由上游处理，本 pass 只认纯值算子

`PtrToMemrefPass` 转换所有函数：把 `!tt.ptr` 签名转成 memref，指针链
（`tt.splat / tt.addptr / tt.broadcast / tt.expand_dims`）塌缩成 base token。
lower-or-error 契约保证到达它的函数访存均已降级、无残留 `tt.load`/`tt.store`，
指针链只喂降级访存（后者用 base token），因此塌缩永远正确，无需门控
（旧的 `hasLoweredAccess` / keepChains 分支均已删除）。

指针 `tt.bitcast` 不被 ptr-to-memref 塌缩（它改 pointee 类型，语义上不能塌成 base），
但访问降级的 `findBase`/`buildChain` 穿过它取 base，降级后它 dead，被 ptr-to-memref
之后的 canonicalize DCE 掉。因此**真实管线里没有指针值算子到达本 pass**。

本 pass 据此只处理纯值算子，不再带指针保留守卫：

- 整个 Triton 方言默认 illegal（见 3.2），值算子（splat/broadcast/expand_dims/reshape/make_range/
  bitcast/reduce/dot/trans）由各自 converter 处理；`tt.gather` 已从访存分析（`AccessAnalysis`）
  与 `classifyOp` 移出、改归值算子，但 **converter 待补**——真到达即硬报错；
- converter 不再对指针 bail；若竟有指针 op 漏到，partial conversion 直接硬报错（lower-or-error 取向）；
- `unrealized_conversion_cast` 桥**保留**（不像参考实现那样 erase）。

前提：管线在 ptr-to-memref 与 arith-to-linalg 之间有一次 canonicalize（DCE dead 指针 bitcast）。
lit 的 lower.mlir / unstructured.mlir RUN 行已对齐此顺序。旧的 `pointer_preserve.mlir`
（单独跑 arith-to-linalg、不经 ptr-to-memref 的人工指针保留测试）随守卫一并删除。

### 2.2 测试 RUN 行约束

- `test/AccessLowering/unstructured.mlir` 的 RUN 行加入 `-privateuse-arith-to-linalg`
  （ptr-to-memref 之后、canonicalize 之前），`--implicit-check-not` 改为
  `tt.load / tt.store / tt.splat / tt.bitcast`（不再断言 no-linalg）；
- `test/AccessLowering/lower.mlir` 的 RUN 行同样接入新 pass（见 4.2）；
- mixed / skip / call / unstructured_skip 的 RUN 行不改；unstructured 形态另加专门测试覆盖（见 6）。

### 2.3 工程规范（AGENTS.md）

- 原生 / 编译器改动先 `make` 再测；lit 无需 GPU；
- C++ `class` 成员尾下划线（`foo_`），`struct` 成员_plain；
- `assert` 内不得有副作用（断言可能被编译掉）。

---

## 3. 设计

### 3.1 机制

```cpp
applyPartialConversion(module, target, patterns);
```

- **无 TypeConverter**：类型保持 tensor。上游 elementwise helper 是 plain `RewritePattern`，
  不需要 converter；本 pass 的 converter 也都在 tensor 世界内改写。
- **Triton 方言默认非法**：`addIllegalDialect<triton>` + 极小白名单（见 3.2）。有 converter 的
  tt 值算子被转，没 converter 的 legalize 失败——无需再逐个 `addIllegalOp`，也不再需要末尾
  warning walk（见 3.4）。
- **不再对 `tt.call` 整模块提前 return**：含 call 的模块现在会走转换，`CallOp` 非法且无
  converter → 到达即报错（lower-or-error；当前测试 / corpus 无 call 到达本 pass）。

### 3.2 ConversionTarget

**非法默认 + 白名单放行**

```cpp
// 不含 triton
target.addLegalDialect<arith::ArithDialect, math::MathDialect, linalg::LinalgDialect,
                       tensor::TensorDialect, bufferization::BufferizationDialect,
                       memref::MemRefDialect, scf::SCFDialect, func::FuncDialect,
                       cf::ControlFlowDialect, pumem::PUMemDialect, affine::AffineDialect>();
target.addIllegalDialect<triton::TritonDialect>();
target.addLegalOp<triton::FuncOp, triton::ReturnOp,
                  triton::GetProgramIdOp, triton::GetNumProgramsOp>();
```

- **`ModuleOp` / `UnrealizedConversionCastOp` 不再显式 `addLegalOp`**：二者属 builtin，没被任何
  `addIllegal*` 覆盖，`ConversionTarget` 默认即 legal；且本 pass 无 TypeConverter，partial
  conversion 既不产生 cast、也不会因残留 cast 报错（正是「cast 桥保留」的由来）。故显式列出是冗余。
- 白名单是「原样穿过本 pass 的 tt 算子」：结构性的 `tt.func` / `tt.return`，以及标量 program-info
  `tt.get_program_id` / `tt.get_num_programs`（不可 linalg 化，留到 GPU→LLVM 降）。

**arith / math 动态 legal**

legality lambda 只有一行 `return !isElementwiseMappableOnRankedTensors(op);`，判定全部集中在
helper `isElementwiseMappableOnRankedTensors`（返回 true 表示需转换 / illegal）。它合并两类：

1. **elementwise**：逐字对齐上游 `isElementwiseMappableOpOnRankedTensors`
   （`mlir/lib/Dialect/Linalg/Transforms/ElementwiseToLinalg.cpp:24-36`）：
   `OpTrait::hasElementwiseMappableTraits(op)`（不是单一 trait，是 TraitList 判定函数）
   && 任一 operand 是 RankedTensorType && 所有 operand ∈ {RankedTensor, 标量 int/float/index/complex}。
2. **dense splat 张量常量**：`arith.constant` 结果为 RankedTensor 且值是 dense splat → 需转换
   （交 DenseConstantConverter → scalar + `linalg.fill`）。**非 splat** 的 dense 张量常量
   （如 ReshapeConverter 生成的 shape 常量 `dense<[4,2]> : tensor<2xi64>`）保持 legal。

**只看 operand，不查 result**（elementwise 部分）。legality 必须是 pattern 覆盖范围的子集（⊆）；
若造出「被判 illegal 却没有 pattern 可用」的 op，partial conversion 失败。常量分支的 **splat 守卫
不能省**：converter 只处理 splat，若 legality 把所有张量常量标 illegal，非 splat 的 shape 常量
被标 illegal 后 converter bail 就会 legalize 失败（实测 `@reshape_nonassoc` 因此报错）。

**tt 值算子非法**：不再显式列举——`addIllegalDialect<triton>` 已把 `splat / broadcast /
expand_dims / reshape / make_range / bitcast / reduce / dot / trans` 等全部标非法，各自由
converter 处理。注意 **ReshapeOp 恒非法**：converter 仍对动态 shape bail，若动态 reshape 到达会
hard error（真实管线 / corpus 无动态 shape reshape 到达）。

**配对不变量（模型反转后）**：以前是「每加 converter 要补一条 illegal 标记」；现在方言默认非法，
风险反过来——

> **一个必须原样穿过本 pass 的 tt 算子（如 program-info、func/return）必须进白名单，否则被判
> 非法又无 converter → legalize 失败。** 反之，任何**有** converter 的 tt 算子天然被默认非法覆盖，
> 无需再标 illegal。

### 3.3 Conversion Patterns

trimmed 移植，统一约定：用 `adaptor.getSrc()` 取操作数、`report_fatal_error` 改
`notifyMatchFailure`。

| Converter | 转换 | 要点 |
|---|---|---|
| 上游 helper | arith / math → `linalg.generic` | 收尾调 `linalg::populateElementwiseToLinalgConversionPatterns(patterns)`（`mlir/Dialect/Linalg/Passes.h`），核心 elementwise 全交给它 |
| SplatConverter | `tt.splat` → `linalg.fill` | 参考 tmp 实现；统一用 `linalg.fill`，canonicalize 稳定（见 3.6） |
| DenseConstantConverter | `arith.constant dense<splat>` → scalar + `linalg.fill` | 只处理 dense splat（非 splat `notifyMatchFailure`）；与 legality 的 splat 守卫配对（见 3.2）。bufferize 产 alloc + fill，不产 global |
| BitcastConverter | `tt.bitcast`（值）→ `arith.bitcast` | `arith.bitcast` 本身 ElementwiseMappable，上游 helper 接力转 `linalg.generic`。**评审新增**，覆盖 `.to(bitcast=True)` 常见残留 |
| MakeRangeConverter | `tt.make_range` → `linalg.generic` + `linalg.index` + `index_cast` | start 常量用外层 rewriter hoist（参考形态） |
| BroadcastConverter | `tt.broadcast` → `linalg.generic` + broadcast affine map | **删掉 getBroadcastDims 及 broadcastDims attr**（下游无消费者，同 shape broadcast 的 fatal 退化点随之消失）；`tt.broadcast` verifier 要求同 rank，`getBroadcastAffineMap` 的前导维展开对合法 TTIR 是死代码 |
| ExpandDimsConverter | `tt.expand_dims` → `tensor.expand_shape` | 参考手写 reassoc 对 srcRank==0 已正确（空 reassociation 合法），无需额外 guard |
| ReshapeConverter | `tt.reshape` → `tensor.collapse/expand_shape` 否则 `tensor.reshape` | 用 `getReassociationIndicesForReshape`；动态 shape bail；allow_reorder / efficient_layout 转 row-major 语义成立，无需特判 |
| ReduceConverter | `tt.reduce` → `linalg.generic`（reduction iterator） | 参考 `convertToLinalgReduce` 裁剪移植（去掉 fp16/bf16 `highPrecisionReduce`）。rank-1 全归约 → rank-0 init + `tensor.extract` 取标量；更高 rank 保留并行维；用 `IRMapping` 把 `combineOp` region 克隆进 body，`tt.reduce.return` → `linalg.yield` |
| DotConverter | `tt.dot` → `linalg.matmul`（rank≤2）/ `linalg.batch_matmul`（rank 3） | 参考 `MatmulConverter`：检测全零累加器（原始 `op.getC()` 的 splat/dense-zero）则丢掉 add-C，否则 `arith.addf/addi` 加回（随即被 elementwise helper 转成 `linalg.generic`）；rank>3 `notifyMatchFailure`。**忽略 `inputPrecision`**（tf32/bf16x3 未移植） |
| TransposeConverter | `tt.trans` → `linalg.transpose` | `permutation` 取自 `getOrder()`（test-pre 的 order 是必填 `DenseI32ArrayAttr`，比参考的空 order 交换末两维分支更简单）；size 与 rank 不符则 `notifyMatchFailure` |

**ConversionTools.h 只移植两个函数**：`getNParallelLoopsAttrs`（ReduceConverter 也复用
`utils::IteratorType`）+ `getBroadcastAffineMap`（删 `getBroadcastDims`）。

### 3.4 漏转处理（warning walk 已删）

Triton 方言默认非法：任何**有** converter 的值算子被转，**无** converter 且不在白名单的 tt 算子
（`gather / scan / cat / join / split / fp_to_fp / print / assert …`）一旦到达即 `applyPartialConversion`
hard error，不会静默——故无需额外诊断防护。

旧的末尾 `warnUnconvertedValueOps` walk（对范围外算子发非致命 warning）已删除；
随之删除专门测试 `warning.mlir`。

### 3.5 getDependentDialects

```cpp
registry.insert<arith::ArithDialect, math::MathDialect, linalg::LinalgDialect,
                tensor::TensorDialect, affine::AffineDialect, triton::TritonDialect>();
```

### 3.6 已知交互与陷阱（写测试时注意）

- **SplatConverter 产物用 `linalg.fill`（不是 `tensor.splat`）**：
  `tensor.splat(常量)` 会被 canonicalize 折回 dense 常量（`SplatOp::fold`），而真实管线
  arith-to-linalg 紧跟 canonicalize，会把 converter 产物撤销。改用 `linalg.fill`（参考 tmp 实现）
  后形态 canonicalize 稳定，bufferize 产物是 alloc + fill。这是评审后由 `tensor.splat` 改过来的取舍。
- **dense splat 张量常量转 scalar + `linalg.fill`**：DenseConstantConverter 保留。常量源
  `tt.splat(常量)` 在 ptr-to-memref 后的 canonicalize 里 fold 成 dense 常量，本 pass 转成
  scalar + `linalg.fill`，bufferize 产 alloc + fill（如 lower.mlir @loop_acc 的累加器初始化），
  不产 `memref.global`。
- 访问降级不再产 `tensor.splat`：结构化/非结构化 pass 都把值 splat 留作 `tt.splat`，
  由本 pass 的 SplatConverter 统一转成 `linalg.fill`。管线里不再有任何 `tensor.splat`。
- 上游 helper `getOrCreateOperandsMatchingResultTypes` 对「mixed scalar+tensor 且 scalar 在前」
  的写法有潜在崩溃路径；Triton 一般 tensor 在前，pass 前先 canonicalize 即可，不依赖 scalar-first。
- **unstructured 路径不是 no-op**：`lowerUnstructuredAccess` 把索引 / 值张量包进 `to_buffer`，
  但 `make_range / arith.subi / 值 tt.splat / cmpi mask` 仍残留为张量算子，新 pass 会转它们
  （dense 常量转 fill）。`unstructured.mlir` 的 RUN 行已接入 `-privateuse-arith-to-linalg`
  （ptr-to-memref + canonicalize 之后），i32→i64 加宽的 CHECK 改为 `linalg.generic`
  （ins `tensor<4xi32>` outs `tensor<4xi64>`）；另加专门测试覆盖此形态。
- **reduce / dot 累加器种子**：ReduceConverter 的 `linalg.generic`（reduction iterator）与
  DotConverter 的 `linalg.matmul` 都用 `tensor.empty` 作 outs/init。参考实现同样用未初始化
  buffer（`alloc_tensor`）；因 `_emit_cpp` 未实现，corpus 无法验证数值，等有 codegen 后需确认
  归约种子语义。dot 里 `tensor.empty` 作 matmul init 是安全的（命名 `linalg.matmul` 覆盖写、
  不 fold init），C 的累加靠后面显式 `addf/addi`。
- **dot 零累加器检测**依赖匹配时原始 `op.getC()` 的 def-use（splat/dense-zero），与转换先后
  无关；命中后留下的死 `linalg.fill` 由 arith-to-linalg 之后的 canonicalize DCE。
- **reduce / dot / trans 都忽略了部分语义**：dot 不处理 `inputPrecision`（tf32/bf16x3）、
  reduce 不做 fp16/bf16 高精度提升（参考的 `highPrecisionReduce`，test-pre 无该 option）。

---

## 4. 文件清单与构建接线（日后实现）

路径相对 `third_party/privateuse/`。采用参考式子目录：独立的
`lib/Conversion/ArithToLinalg/` + `include/privateuse/Conversion/ArithToLinalg/`，
自带 Passes.td / .h / tablegen / CMake，独立 `add_triton_library`。

### 4.1 新建文件

1. `include/privateuse/Conversion/ArithToLinalg/Passes.td` —
   `def ArithToLinalg : Pass<"privateuse-arith-to-linalg","mlir::ModuleOp">`，
   constructor `createArithToLinalgPass()`，无 options。
2. `include/privateuse/Conversion/ArithToLinalg/Passes.h` —
   factory 声明 + `GEN_PASS_REGISTRATION`（置于 `mlir::triton` 内）。
3. `include/privateuse/Conversion/ArithToLinalg/CMakeLists.txt` —
   `mlir_tablegen(Passes.h.inc -gen-pass-decls --name ArithToLinalg)` → `PrivateUseArithToLinalgPassIncGen`。
4. `include/privateuse/Conversion/ArithToLinalg/ConversionTools.h` —
   `getNParallelLoopsAttrs` + `getBroadcastAffineMap`。
5. `include/privateuse/Conversion/ArithToLinalg/ConversionPatterns.hpp` —
   trimmed 10 个 converter（Splat / DenseConstant / Bitcast / MakeRange / Broadcast /
   ExpandDims / Reshape / Reduce / Dot / Transpose）+ populate 函数。
6. `lib/Conversion/ArithToLinalg/ArithToLinalgPass.cpp` —
   pass 骨架（`GEN_PASS_DEF_ARITHTOLINALG` + `Passes.h.inc` 置于**全局作用域**，
   `::impl::ArithToLinalgBase`；factory 外置于匿名 namespace）；legality 判定集中在
   `isElementwiseMappableOnRankedTensors`。
7. `lib/Conversion/ArithToLinalg/CMakeLists.txt` — `add_triton_library(PrivateUseArithToLinalg ...)`。
8. `test/ArithToLinalg/*.mlir` — 专门测试。

### 4.2 修改文件

9. `include/privateuse/Conversion/CMakeLists.txt` — `add_subdirectory(ArithToLinalg)`。
10. `lib/Conversion/CMakeLists.txt` — `add_subdirectory(ArithToLinalg)`。
11. `tools/privateuse-opt/RegisterPrivateUseDialects.h` — include 新 Passes.h + `registerArithToLinalgPasses()`。
12. `privateuse.cc` — include 新 Passes.h + `add_arith_to_linalg(pm)` 绑定（无参 factory）。
13. `backend/compiler.py` — `_specialize_ir` 在 `add_ptr_to_memref` 后、canonicalize / cse 前
    插入 `privateuse.add_arith_to_linalg(pm)`。
14. `test/AccessLowering/lower.mlir` — RUN 行用 `-privateuse-arith-to-linalg` 替换
    `-convert-elementwise-to-linalg`，并移到 `-canonicalize` **之前**（与真实管线顺序一致）；CHECK 不变。

### 4.3 新库 LINK_LIBS

```
MLIRLinalgTransforms   # populate helper
MLIRMathDialect
MLIRDialectUtils       # getReassociationIndicesForReshape
MLIRArithDialect MLIRLinalgDialect MLIRTensorDialect MLIRAffineDialect
MLIRTransforms MLIRPass MLIRIR MLIRSupport
TritonIR PrivateUseUtils
DEPENDS PrivateUseArithToLinalgPassIncGen
```

注意：`ArithToLinalg` 库需自行链接 `MLIRLinalgTransforms / MLIRMathDialect / MLIRDialectUtils`
（其他 Conversion 子库不提供这些）。

### 4.4 链接机制

`add_triton_library` → GLOBAL `TRITON_LIBS` → libtriton；`privateuse-opt` 经 `${triton_libs}`
自动纳入，插件加载时解析符号（与其他 Conversion 子库同机制）。
**无需**显式改插件 `CMakeLists.txt` 的 LINK_LIBS。

---

## 5. 复用组件

- 上游 `linalg::populateElementwiseToLinalgConversionPatterns` +
  `isElementwiseMappableOpOnRankedTensors` 谓词（`ElementwiseToLinalg.cpp`）。
- `OpTrait::hasElementwiseMappableTraits`（`mlir/lib/IR/Operation.cpp:1399`）。
- 参考 `ConversionPatterns.hpp` 的 converter + `ConversionTools.h`（trimmed 移植）。
- `getReassociationIndicesForReshape`（`mlir/Dialect/Utils/ReshapeOpsUtils.h`）。
- 工作目录 pass 模板 `lib/Conversion/AccessLowering/LowerStructuredAccessPass.cpp`。

---

## 6. 验证方法（日后实现）

1. `make`（triton 目录，原生改动）。
2. `BUILD_DIR=$(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')`
3. `cd $BUILD_DIR; ninja privateuse-opt; lit -v test/ArithToLinalg test/AccessLowering/lower.mlir`
4. `lit -v test/AccessLowering`（确认测试通过，尤其 `unstructured.mlir` 的 `--implicit-check-not=tt.load/tt.store/tt.splat/tt.bitcast`）。
5. `lower.mlir`：make_range / splat / broadcast / expand_dims 已被访问降级折进 memref，
   `linalg.generic` 只在 @scale(mulf) / @loop_acc(addf)；@loop_acc 累加器初始化是 alloc + linalg.fill
   （dense 常量经 DenseConstantConverter 转 fill，不产 global）。

**新测试覆盖**：

- 各 converter：make_range（含 start!=0、size-1 fold 成 dense 后转 fill）、同 rank 1 维 broadcast（4x1→4x4）、
  **同 shape broadcast**（验证不再 fatal）、expand_dims（含 srcRank==0）、
  reshape（含**非 assoc 2x4→4x2** + allow_reorder）、splat（含 i1）、
  dense splat 常量（转 fill）、值 bitcast、elementwise arith。
- **unstructured 形态**：`to_buffer` 包裹的 make_range / subi / splat / cmpi 残留被正确转换（确认非 no-op）。
- memref + tensor + tt 混合 IR 下 partial conversion 成功。

---

## 7. 评审修订记录（triton-b2）

与参考仓库会话 triton-b2 评审一轮，采纳的修订：

- **谓词只看 operand，不查 result**：legality 必须 ⊆ 上游 pattern 覆盖。
- **`arith.constant` 特判门控**：保证 DenseConstantConverter 的 `cast<DenseElementsAttr>` 安全。
- **splat 产物取舍**：最终统一用 `linalg.fill`（SplatConverter / DenseConstantConverter）；曾考虑的 `tensor.splat` 因 bufferize 产 `linalg.map`、canonicalize 可逆折回 dense 而弃用，别对其后形态断言。
- **删 `getBroadcastDims`**：确认 test-pre 的 lib / include 里无 broadcastDims 消费方。
- **ExpandDims rank-0 不加 guard**：参考手写 reassoc 已正确。
- **MakeRange / Reshape 的 guard 与 hoist**：按参考形态。
- **lit 顺序对齐 compiler.py**：新 pass 移到 canonicalize 之前。
- **BitcastConverter + warning walk**：防静默漏转（最大风险点）。
- **BitcastOp 必须加入动态 legality 列表**：否则整个 TritonDialect legal 后 converter 恒为 no-op；
  按结果类型 `isPointerLike(op.getType())` 判定，覆盖指针 tensor 结果；pattern 侧对指针结果
  `notifyMatchFailure` 对齐。由此提炼出「converter ↔ legality 配对不变量」（见 3.2）。
- **warning 走 result 过滤即可**：`tt.func / tt.return` 无结果天然不触发；test-pre 无 `tt.assert`，
  不必扩展 operand 检查。
- **unstructured 非 no-op**：复核 `LowerUnstructuredAccessPass.cpp` 把索引/值张量包进 `ToBufferOp`，
  值计算原样保留；`unstructured.mlir` 的 make_range / dense<0> / subi / cmpi / tt.splat
  确实残留，真实管线里新 pass 转它们是期望行为，专门加 unstructured 形态测试是对的。

### 实现补记（落地时的实测细节）

- **`Passes.td` 末尾 `#endif` 必须有结尾换行符**：否则 mlir-tblgen 预处理器无法终结该指令，
  把 `#endif` 当成顶层 token，报 `Unexpected token at top level`，并连锁导致 `Passes.h.inc`
  未生成、`privateuse.cc` 编译失败。`.h / .cpp` 无此要求，仅 TableGen 预处理器敏感。
- **同 shape `tt.broadcast`（4x4→4x4）折叠为恒等**：转换后直接返回源（`tt.return %arg0`），
  不产生 `linalg.generic`。这满足「同 shape 不再 fatal」的设计意图，测试按恒等返回断言。
- **Triton 张量元素数须为 2 的幂**：reshape 测试用 `2x4→8`（collapse）与 `2x4→4x2`（非 assoc），
  不能用 `2x3→6 / 3x2`（6 非 2 的幂，`tt.reshape` verifier 直接拒绝）。
- **rank-0 `tt.expand_dims` 实测合法**：`tensor<f32> → tensor<1xf32>`（空 reassociation）被
  `tensor.expand_shape` verifier 接受（`inferCollapsedType([])` 得 rank-0、元素数 1==1 均匹配），
  与评审「无需额外 guard」的判断一致，专门用 `expand_dims_scalar.mlir` 覆盖。
- **warning walk 实测**：白名单外的 `tt.trans`（带 `order` 属性）保持 legal 但触发非致命 warning，
  用 `warning.mlir`（`2>&1 | FileCheck`）覆盖。
- **评审后改用 `linalg.fill`**：SplatConverter 与 DenseConstantConverter 的产物由
  `tensor.splat` 改为 `linalg.fill`（参考 tmp）。原因：真实管线 arith-to-linalg 紧跟 canonicalize，
  `tensor.splat(常量)` 会被 `SplatOp::fold` 折回 dense，converter 等于被撤销；`linalg.fill` 稳定。
  转换驱动默认 `foldingMode=BeforePatterns`，非法 op 先 fold 再跑 pattern——同 shape broadcast、
  size-1 make_range、常量源 splat/expand_dims/reshape 都在 pattern 前被 fold；指针 op 因标 legal 永不 fold。
- **Q3 标量结果盲点已修**：warning walk 显式 `isa<triton::ReduceOp>` 告警，覆盖 `tt.reduce → f32`
  全归约（标量结果被 tensor 过滤漏掉）；不放宽 result 谓词以免误报 tt.return。
- **size-1 make_range 路径已固定**：`{start=5,end=6}` 这类 size-1 make_range 先 fold 成 dense splat，
  再经 DenseConstantConverter → scalar + `linalg.fill`；`value_ops.mlir @make_range_size1` 覆盖。

### 后续精简与修订

按真实管线契约精简（lit 12/12、corpus 597 基线无新增验证通过）：

- **删指针保留守卫**：真实管线里指针链被 ptr-to-memref 塌缩、指针 bitcast 被访问降级穿过后 DCE，
  没有指针值算子到达本 pass。故删 `isPointerLike`、5 处 converter 指针 bail、legality 的指针判定；
  tt 值算子改 `addIllegalOp`。若竟有指针漏到，partial conversion 硬报错。
  随之删除人工测试 `pointer_preserve.mlir`（单独跑 arith-to-linalg、不经 ptr-to-memref）。
- **DenseConstantConverter 先删后恢复**：曾一度删除（dense 常量原样保留 → bufferize 产
  `memref.global` + get_global + copy）；后按需求恢复——dense splat 转 scalar + `linalg.fill`，
  bufferize 产 alloc + fill，不产 global。常量判定（RankedTensor + dense splat）并入
  `isElementwiseMappableOnRankedTensors`，legality lambda 收敛为一行；**splat 守卫保留**
  （放行 ReshapeConverter 生成的非 splat shape 常量，见 3.2）。
- **删末尾 warning walk**：白名单内 tt 值算子均 illegal，漏转硬报错，无需诊断；范围外算子
  有意留下。删 `warnUnconvertedValueOps` 及专门测试 `warning.mlir`（lit 13→12）。
- **ReshapeOp 恒 illegal**：去掉动态 shape 的动态 legality，并入 `addIllegalOp`；converter 仍对
  动态 shape bail（真实管线 / corpus 无动态 reshape 到达）。
- **测试 RUN 行对齐管线**：lower.mlir / unstructured.mlir 在 ptr-to-memref 与 arith-to-linalg 之间
  补 `-canonicalize`（与 compiler.py 一致），先 DCE dead 指针 bitcast。否则 dead bitcast 会喂给本 pass，
  删守卫后 BitcastConverter 对其生成非法 `arith.bitcast` 而失败。
- **形态**：常量源 splat 经中间 canonicalize fold 成 dense，再经 DenseConstantConverter 转 fill，
  bufferize 成 alloc + fill。lower.mlir @loop_acc、value_ops.mlir @dense_const / @make_range_size1、
  unstructured_form.mlir @index_form 均断言 linalg.fill。
- 上文 §3.2/§3.3/§3.4/§3.6/§6 已按此更新；§7 早期评审记录与「实现补记」保留作历史存档
  （其中 warning walk / DenseConstantConverter 删除相关条目已被本次修订取代）。

### 本轮修订：非法默认方言 + reduce/dot/trans + 目录重构（lit 12/12、corpus 零 legalize 失败）

- **合法性模型反转**：从「`addLegalDialect<triton>` + 逐个 `addIllegalOp`」改成
  `addIllegalDialect<triton>` + 白名单 `addLegalOp<func/return/get_program_id/get_num_programs>`。
  新 tt 算子默认非法、漏转即硬报错；不再需要「每加 converter 补 illegal 标记」——配对不变量
  反转为「pass-through 的算子要进白名单」（见 3.2）。
- **删 `ModuleOp / UnrealizedConversionCastOp` 显式 legal**：默认已 legal，冗余。
- **删 `tt.call` 整模块提前 return**：含 call 的模块现走转换，`CallOp` 无 converter → 到达即报错
  （当前测试 / corpus 无 call 到达本 pass，call.mlir 只跑访问降级）。
- **`tt.gather` 归值算子**：从 `AccessAnalysis`（`deferredAccess`）与 `AccessReportPass` 的
  `classifyOp` 移出，不再当访存；arith-to-linalg 侧标非法但 **converter 待补**。
- **新增 3 个 converter**（参考 tmp 裁剪移植）：`ReduceConverter`（`tt.reduce` →
  `linalg.generic` reduction iterator）、`DotConverter`（`tt.dot` → `linalg.matmul` /
  `linalg.batch_matmul` + 零累加器检测）、`TransposeConverter`（`tt.trans` → `linalg.transpose`）。
  三者都从 illegal-by-default 里获得覆盖，corpus 的 192/19/18 个 legalize 失败随之清零。
- **目录重构**：`Conversion` 下拆成自包含子目录——`AccessLowering`（3 个降级 pass，独立
  `--name AccessLowering` / `PrivateUseAccessLowering`）；`AnalyzeAccess`（`AccessReportPass.cpp`）
  移到 `Utils/AccessAnalysis`（`--name AccessAnalysis` / `PrivateUseAccessAnalysis`）；顶层
  `Conversion/Passes.{td,h}` 与 `PrivateUseConversions` 库删除；`RegisterPrivateUseDialects.h` /
  `privateuse.cc` 改到新路径与 `registerAccessLoweringPasses()/registerAccessAnalysisPasses()`。
  纯结构改动，无行为变化（详见 `docs/access-lowering.md`）。