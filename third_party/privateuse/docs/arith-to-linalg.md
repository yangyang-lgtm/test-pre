# 值算子降级：privateuse-arith-to-linalg Pass 设计方案

> **状态：已按本文实现并验证。** `privateuse-arith-to-linalg` pass（核心值算子范围）
> 已落地：新建 `include/privateuse/Conversion/ArithToLinalg/` 与 `lib/Conversion/ArithToLinalg/`，
> 接入 `privateuse-opt`、`privateuse.cc` 绑定与 `backend/compiler.py` 管线，新增
> `test/ArithToLinalg/` lit 测试并改 `lower.mlir` RUN 行。构建通过，privateuse 全套 lit
> 14/14 通过，Python 绑定运行时确认可用。下文「文件清单 / 复用 / 验证」即对应已实现的代码；
> 实现中的几处实测细节见第 7 节末尾「实现补记」。

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

### 1.4 范围界定：只移植核心值算子

逻辑参考 `tmp/triton/third_party/privateuse/lib/Conversion/ArithToLinalg`，但**只移植核心值算子**。
参考实现里的下列机制**全部不移植**：

| 不移植的机制 | 原因 |
|---|---|
| load / store / addptr 转换 | 工作目录已由访问降级三层覆盖 |
| dot / reduce / scan | 依赖缺失方言或超出核心值算子范围 |
| program-info（program_id 等） | 依赖缺失方言 |
| PtrAnalysis / MaskAnalysis / TritonStructured / TritonTilingExt 相关 | 工作目录无这些组件，参考头文件顶部即 include 它们 |

因此**不移植整个参考 `ConversionPatterns.hpp`**（它顶部 include 了上述缺失组件），
而是新建一个 trimmed 头，只含核心值算子 converter。

---

## 2. 关键约束

### 2.1 指针链必须保留

`PtrToMemrefPass` 转换所有函数：把 `!tt.ptr` 签名转成 memref，指针链
（`tt.splat / tt.addptr / tt.broadcast / tt.expand_dims`）塌缩成 base token。
lower-or-error 契约保证到达它的函数访存均已降级、无残留 `tt.load`/`tt.store`，
指针链只喂降级访存（后者用 base token），因此塌缩永远正确，无需门控
（旧的 `hasLoweredAccess` / keepChains 分支均已删除）。

`test/ArithToLinalg/pointer_preserve.mlir` 的 `@keep_chain / @keep_ptr_bitcast`
是只算指针、不访存的函数，**不经过 ptr-to-memref**（该测试只跑 arith-to-linalg），
指针链原样到达本 pass，由本 pass 的 isPointerLike 守卫保留。

本 pass 只转**非指针**值算子：

- 指针用途的算子保持 legal，不动；
- `unrealized_conversion_cast` 桥**保留**（不像参考实现那样 erase）；
- 判定指针与否按元素类型，参照 `PtrToMemrefPass.cpp` 的 `isPointerLike`。

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
- 含 `tt.call` 的 module **提前 return**（不处理跨函数调用情形）。
- pass 末尾做一次 warning walk（见 3.4）。

### 3.2 ConversionTarget

**合法方言与算子**

```cpp
target.addLegalDialect<arith::ArithDialect, math::MathDialect, linalg::LinalgDialect,
                       tensor::TensorDialect, bufferization::BufferizationDialect,
                       memref::MemRefDialect, scf::SCFDialect, func::FuncDialect,
                       cf::ControlFlowDialect, pumem::PUMemDialect, affine::AffineDialect,
                       triton::TritonDialect>();
target.addLegalOp<ModuleOp, UnrealizedConversionCastOp>();
```

- `ModuleOp` / `UnrealizedConversionCastOp` 属 builtin，不在方言列表里，需单独 `addLegalOp`；
  cast 桥必须保留（见 2.1）。
- 整个 `TritonDialect` 被设成 legal —— 这是 3.2 末尾「配对不变量」与 3.4「漏转防护」的根源。

**arith / math 动态 legal**

先特判 `arith.constant`：

- dense splat 的 Float / Integer 张量 → illegal（交给 DenseConstantConverter）；
- 其余 legal；非 splat / `dense_resource` 保持 legal。
  这条门控是 DenseConstantConverter 里 `cast<DenseElementsAttr>` 安全的前提。

其余算子逐字对齐上游 `isElementwiseMappableOpOnRankedTensors`
（`mlir/lib/Dialect/Linalg/Transforms/ElementwiseToLinalg.cpp:24-36`）：

```cpp
OpTrait::hasElementwiseMappableTraits(op)   // 注意：不是单一 trait，是 TraitList 的判定函数
  && 任一 operand 是 RankedTensorType
  && 所有 operand ∈ {RankedTensor, 标量 int/float/index/complex}   // noneInvalid
```

**只看 operand，不查 result**。legality 必须是上游 pattern 覆盖范围的子集（⊆）；
若用 result 驱动，会造出「被判 illegal 却没有 pattern 可用」的 op，使 partial conversion 失败。

**tt 值算子动态 legal**

```cpp
target.addDynamicallyLegalOp<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp,
                             triton::ReshapeOp, triton::MakeRangeOp, triton::BitcastOp>(...);
```

- illegal 条件 = 非指针元素类型；指针用途一律 legal（按 element type，参照 `isPointerLike`）。
- **ReshapeOp 额外要求静态输出 shape**：参考对动态 shape 返回 failure，若无条件判 illegal
  会使 partial conversion 失败。
- **MakeRangeOp 恒 illegal**：verifier 已保证 1-D / i32 / start < end。
- **BitcastOp 按结果类型判定**：

```cpp
target.addDynamicallyLegalOp<triton::BitcastOp>([](triton::BitcastOp op) {
  return isPointerLike(op.getType());   // 标量 ptr 或 tensor<...x!tt.ptr<...>> 都 legal
});
```

  reinterpret 成 ptr 的 `tt.bitcast` 会留在软跳过函数里；pattern 侧对指针结果
  `notifyMatchFailure`，与这条 legality 对齐。

**配对不变量（评审强调，务必遵守）**

整个 `TritonDialect` 被设成 legal，`applyPartialConversion` 对判定为 legal 的 op
**直接跳过、不尝试 pattern**。因此：

> **每加一个 converter，必须同步加一条对应的动态 legality；否则该 converter 恒为 no-op（死代码）。**

BitcastOp 就是典型例子：若只加 BitcastConverter 而忘了把 BitcastOp 加进动态 legality 列表，
converter 永远不会触发。后续新增任何 converter 都要照此办理。

### 3.3 Conversion Patterns

trimmed 移植，统一约定：用 `adaptor.getSrc()` 取操作数、`report_fatal_error` 改
`notifyMatchFailure`、指针元素 bail。

| Converter | 转换 | 要点 |
|---|---|---|
| 上游 helper | arith / math → `linalg.generic` | 收尾调 `linalg::populateElementwiseToLinalgConversionPatterns(patterns)`（`mlir/Dialect/Linalg/Passes.h`），核心 elementwise 全交给它 |
| SplatConverter | `tt.splat` → `linalg.fill`（非指针；指针 splat 保留） | 参考 tmp 实现；统一用 `linalg.fill`，canonicalize 稳定（见 3.6） |
| DenseConstantConverter | dense splat → scalar `arith.constant` + `linalg.fill` | 靠 3.2 的 arith.constant 门控保证 `cast<DenseElementsAttr>` 安全；`linalg.fill` canonicalize 稳定 |
| BitcastConverter | `tt.bitcast`（值）→ `arith.bitcast` | 指针 bail；`arith.bitcast` 本身 ElementwiseMappable，上游 helper 接力转 `linalg.generic`。**评审新增**，覆盖 `.to(bitcast=True)` 常见残留 |
| MakeRangeConverter | `tt.make_range` → `linalg.generic` + `linalg.index` + `index_cast` | start 常量用外层 rewriter hoist（参考形态） |
| BroadcastConverter | `tt.broadcast` → `linalg.generic` + broadcast affine map | **删掉 getBroadcastDims 及 broadcastDims attr**（下游无消费者，同 shape broadcast 的 fatal 退化点随之消失）；`tt.broadcast` verifier 要求同 rank，`getBroadcastAffineMap` 的前导维展开对合法 TTIR 是死代码 |
| ExpandDimsConverter | `tt.expand_dims` → `tensor.expand_shape` | 参考手写 reassoc 对 srcRank==0 已正确（空 reassociation 合法），无需额外 guard |
| ReshapeConverter | `tt.reshape` → `tensor.collapse/expand_shape` 否则 `tensor.reshape` | 用 `getReassociationIndicesForReshape`；动态 shape bail；allow_reorder / efficient_layout 转 row-major 语义成立，无需特判 |

**ConversionTools.h 只移植两个函数**：`getNParallelLoopsAttrs` + `getBroadcastAffineMap`
（删 `getBroadcastDims`）。

### 3.4 静默漏转防护（评审最大风险点）

整个 `TritonDialect` legal 后，白名单外的残留张量值 triton 算子
（`tt.fp_to_fp / clampf / precise_div / sqrt / trans / join / split / cat / extern_elementwise / assert`）
会**原样留下**，失败被推迟到 bufferize / `_emit_cpp` 才暴露。

pass 末尾 walk 一遍，对满足下列条件的 triton 算子发 **warning**（非 hard error，避免误伤软跳过函数）：

- 有**非指针 ranked-tensor 结果**，**或**是 `tt.reduce`（全归约结果是标量，tensor 过滤会漏）；
- 不在转换白名单内；
- 且非 load / store / addptr。

让缺口尽早可见。warning 以 **result 过滤**为主：`tt.func / tt.return` 无结果天然不触发；
工作目录里没有 `tt.assert`，不必为无结果的张量消费算子扩展 operand 检查。**唯一例外**是
显式 `isa<triton::ReduceOp>` 直接告警，覆盖标量结果盲点（不放宽 result 谓词，否则会误报 tt.return）。

### 3.5 getDependentDialects

```cpp
registry.insert<arith::ArithDialect, math::MathDialect, linalg::LinalgDialect,
                tensor::TensorDialect, affine::AffineDialect, triton::TritonDialect>();
```

### 3.6 已知交互与陷阱（写测试时注意）

- **本 pass 的 splat / dense 常量产物用 `linalg.fill`（不是 `tensor.splat`）**：
  `tensor.splat(常量)` 会被 canonicalize 折回 dense 常量（`SplatOp::fold`），而真实管线
  arith-to-linalg 紧跟 canonicalize，会把 converter 产物撤销（dense → bufferize 成
  `memref.global` + `get_global`）。改用 `linalg.fill`（参考 tmp 实现）后形态 canonicalize 稳定，
  bufferize 产物是 alloc + fill。这是评审后由 `tensor.splat` 改过来的取舍。
- 访问降级不再产 `tensor.splat`：结构化/非结构化 pass 都把值 splat 留作 `tt.splat`，
  由本 pass 的 SplatConverter 统一转成 `linalg.fill`。管线里不再有任何 `tensor.splat`。
- 上游 helper `getOrCreateOperandsMatchingResultTypes` 对「mixed scalar+tensor 且 scalar 在前」
  的写法有潜在崩溃路径；Triton 一般 tensor 在前，pass 前先 canonicalize 即可，不依赖 scalar-first。
- **unstructured 路径不是 no-op**：`lowerUnstructuredAccess` 把索引 / 值张量包进 `to_buffer`，
  但 `make_range / arith.subi / dense 常量 / 值 tt.splat / cmpi mask` 仍残留为张量算子，
  新 pass 会转它们（期望行为）。`unstructured.mlir` 的 RUN 行已加入 `-privateuse-arith-to-linalg`
  （ptr-to-memref 之后、canonicalize 之前），i32→i64 加宽的 CHECK 改为 `linalg.generic`
  （ins `tensor<4xi32>` outs `tensor<4xi64>`）；另加专门测试覆盖此形态。

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
   trimmed 7 converter（含 Bitcast）+ populate 函数。
6. `lib/Conversion/ArithToLinalg/ArithToLinalgPass.cpp` —
   pass 骨架（`GEN_PASS_DEF_ARITHTOLINALG` + `Passes.h.inc` 置于**全局作用域**，
   `::impl::ArithToLinalgBase`；factory 外置于匿名 namespace）+ 末尾 warning walk。
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

注意：现有 `PrivateUseConversions` 库并未链接 `MLIRLinalgTransforms / MLIRMathDialect / MLIRDialectUtils`，
新库必须自行链接。

### 4.4 链接机制

`add_triton_library` → GLOBAL `TRITON_LIBS` → libtriton；`privateuse-opt` 经 `${triton_libs}`
自动纳入，插件加载时解析符号（与 `PrivateUseConversions` 同机制）。
**无需**显式改插件 `CMakeLists.txt` 的 LINK_LIBS。

---

## 5. 复用组件

- 上游 `linalg::populateElementwiseToLinalgConversionPatterns` +
  `isElementwiseMappableOpOnRankedTensors` 谓词（`ElementwiseToLinalg.cpp`）。
- `OpTrait::hasElementwiseMappableTraits`（`mlir/lib/IR/Operation.cpp:1399`）。
- 参考 `ConversionPatterns.hpp` 的 converter + `ConversionTools.h`（trimmed 移植）。
- `getReassociationIndicesForReshape`（`mlir/Dialect/Utils/ReshapeOpsUtils.h`）。
- 工作目录 pass 模板 `lib/Conversion/LowerStructuredAccessPass.cpp`；
  指针判定 `PtrToMemrefPass.cpp` 的 `isPointerLike`。

---

## 6. 验证方法（日后实现）

1. `make`（triton 目录，原生改动）。
2. `BUILD_DIR=$(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')`
3. `cd $BUILD_DIR; ninja privateuse-opt; lit -v test/ArithToLinalg test/AccessLowering/lower.mlir`
4. `lit -v test/AccessLowering`（确认测试通过，尤其 `unstructured.mlir` 的 `--implicit-check-not=tt.load/tt.store/tt.splat/tt.bitcast`）。
5. `lower.mlir` 的 CHECK 应不变（同一上游 elementwise helper；make_range / splat / broadcast / expand_dims
   已被访问降级折进 memref，`linalg.generic` 只在 @scale(mulf) / @loop_acc(addf)）。

**新测试覆盖**：

- 各 converter：make_range（含 start!=0）、同 rank 1 维 broadcast（4x1→4x4）、
  **同 shape broadcast**（验证不再 fatal）、expand_dims（含 srcRank==0）、
  reshape（含**非 assoc 2x4→4x2** + allow_reorder）、splat（含 i1）、
  dense splat 常量、值 bitcast、elementwise arith。
- **指针保留**：纯指针函数（pointer_preserve，不经 ptr-to-memref）里 make_range → `linalg.generic`，
  而 `tt.splat(ptr) / addptr / cast` 桥保留（make_range 恒 illegal，是唯一会在这类函数改 IR 的值算子）。
- **unstructured 形态**：`to_buffer` 包裹的 make_range / subi / splat / cmpi 残留被正确转换（确认非 no-op）。
- memref + tensor + tt 混合 IR 下 partial conversion 成功；白名单外残留值算子触发 warning。

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