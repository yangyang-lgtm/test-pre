# 访存降级：LowerStructuredAccessPass 与 LowerUnstructuredAccessPass

本文逐函数、逐段解释这两个 pass 的实现与设计理由，以及它们依赖的组件
（`AccessAnalysis`、`ValueRange`、`Utils`、`PUMem` dialect、Pass 基础设施）。
阅读顺序建议：先看第 1 节的整体模型，再按需查第 2 节的组件细节，最后看
第 3、4 节的逐行讲解。

---

## 1. 宏观模型

### 1.1 在管线中的位置

`third_party/privateuse/backend/compiler.py` 中的 `_specialize_ir` 按以下顺序装配：

```python
privateuse.add_specialize_kernel(pm)
passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
pm.add_pass(passes.common.add_canonicalizer(pm))
privateuse.add_analyze_access(pm)          # 只诊断，不改 IR
privateuse.add_lower_structured_access(pm) # 本 pass：规整访问 → memref.copy
privateuse.add_lower_unstructured_access(pm) # 本 pass：逐元素访问 → pumem.gather/scatter
privateuse.add_ptr_to_memref(pm)          # 收尾：函数签名/地址链 → memref
```

三者是"分析 → 降级 → 收尾"的分层：分析把所有可证明的事实沉淀为数据
（`AccessInfo`），两个降级 pass 只消费数据、不再推理；`PtrToMemrefPass`
负责把 `!tt.ptr` 参数和残留地址链改成 memref，使后续 bufferize 阶段不再
看到 Triton 指针。

### 1.2 访存的一般形态

Triton 的 `tt.load / tt.store` 以"指针张量"寻址：

```
ptr[i0, i1, ...] = base + offset + Σ_d (start_d + stride_d * i_d)
```

`AccessAnalysis` 把任意地址表达式规约成这个规范形（`AccessView`）：

- `base`：入口块（entry block）的参数，类型 `!tt.ptr<T>`；
- `offset`：与循环/程序号相关的标量表达式（`ScalarExpr` DAG）；
- `dims[d] = {start, stride, size}`：每个张量维度的线性部分。

以及 mask 的规范形（`AccessMask`）：mask 被分解为"逐维线性不等式"的合取，
每个不等式记为一个 `LinearBound`。

### 1.3 两种降级策略

| | 结构化（structured） | 非结构化（unstructured） |
|---|---|---|
| 适用条件 | 规范形存在、mask 可证明或为盒形窗口、写地址单射（或窗口单射） | 地址链可分解为 `base + 逐元素索引张量` |
| 生成 IR | `memref.reinterpret_cast` + `memref.copy`（± `linalg.fill`/`memref.subview`） | `pumem.gather` / `pumem.scatter` |
| 性能假设 | 访存是连续/跨步的"整块"搬运，可向量化 | 下标任意，逐元素处理 |
| 失败策略 | 跳过该 op（软失败），留给 unstructured | 硬报错（lower-or-error）：无法降级即 `emitError` + `signalPassFailure`，成功结束后无 `tt.load`/`tt.store` 残留；白名单门只放行 `tt.load`/`tt.store`，descriptor/atomic 一律报错 |

两个 pass 依次运行，structured 先"捡走"所有规整访问；unstructured 处理
剩下的（典型是"下标来自内存"的间接访存，如 embedding lookup）。

### 1.4 降级后的 IR 形状

结构化 load（无 mask）：

```
%view = memref.reinterpret_cast %base_token to offset: [...] sizes: [...] strides: [...] : memref<?xT> to memref<...xT, strided<[...], offset: ?>>
%buf  = memref.alloc() {alignment = 64} : memref<...xT, 128>
memref.copy %view, %buf
%tensor = bufferization.to_tensor %buf restrict writable : memref<...xT, 128>
```

非结构化 load：

```
%base  = unrealized_conversion_cast %arg to memref<?xT>
%idx   = bufferization.to_buffer %offsets_tensor : tensor<...xI> to memref<...xI, 128>
%buf   = memref.alloc() {alignment = 64} : memref<...xT, 128>
pumem.gather %base, %idx, %buf [, %mask [, %other]] : ...
%tensor = bufferization.to_tensor %buf restrict writable : memref<...xT, 128>
```

两类结果都收敛到 `bufferization.to_tensor` 上，后续 `-one-shot-bufferize`
即可把 tensor 世界和 memref 世界接起来，不再需要 Triton 指针语义。

---

## 2. 依赖组件

### 2.1 `Utils/Constants.h`

```cpp
inline constexpr int64_t kSramSpace        = 128;  // scratchpad 地址空间编号
inline constexpr int64_t kDramSpace        = 0;    // 全局地址空间编号
inline constexpr int64_t kTrivialSpace     = -1;
inline constexpr int64_t kDefaultAlignment = 64;   // 局部 tile 对齐（字节）

inline constexpr llvm::StringLiteral kGridAttr = "privateuse.grid";
```

- `kSramSpace`：所有临时 tile buffer 都分配在 `memory_space = 128` 的
  scratchpad 上，代码生成阶段据此发 local 内存指令。两个 pass 的
  `sramBufferType()` 都用它。
- `kDefaultAlignment`：`memref.alloc` 的对齐属性。多数加速器的 DMA /
  vector load 要求 64 字节起步的对齐，统一给 64 避免下游再补。
- `kGridAttr`：模块属性 `privateuse.grid = array<i64: X, Y, Z>`，是
  `AccessAnalyzer` 构造时建立 program id 区间（`pid ∈ [0, grid-1]`）的唯一
  依据；没有它，所有依赖 pid 的常量折叠都无从谈起。例如
  `offset = pid.x * 128` 的区间估计就来自这里。
- 曾经的 `kLoweredAccessAttr`（`privateuse.lowered_access`）已删除，两个
  pass 不再给"降级过访存"的函数打 unit 属性。其后 `PtrToMemrefPass` 一度用
  结构化谓词 `hasLoweredAccess`（函数含 memref/bufferization/pumem 算子）门控，
  只改写降级过访存的函数；该谓词现在也已删除——lower-or-error 契约保证
  到达 `PtrToMemrefPass` 的函数访存均已降级（否则 unstructured 早已报错），
  故 pass 直接转换所有函数，无需再判定。

### 2.2 `ValueRange`（`Analysis/ValueRange.h`）

这是标量表达式的符号化表示，是整个分析的基础设施。

#### ScalarExpr

```cpp
struct ScalarExpr {
  enum class Kind : uint8_t { Const, Sym, Add, Sub, Mul, DivS, DivU, RemS,
                              RemU, Shl, ShrS, ShrU, MinS, MaxS, MinU, MaxU,
                              And, Or, Xor };
  Kind kind;
  int64_t value;       // Const 的值
  SymKind symKind;     // Sym 的种类
  unsigned symId;      // Sym 的编号（LoopIv 用）
  ExprPtr lhs, rhs;    // 二元子表达式
};
```

- 表达式是不可变 DAG，`ExprPtr = shared_ptr<const ScalarExpr>`，构造靠
  `cst / sym / bin / add / sub / mul / neg` 这些静态工厂。
- 为什么要符号化：地址是 `pid`、循环归纳变量（loop iv）、函数参数的
  线性组合。只有符号化后才能：
  1. 判断两个地址表达式是否只差一个常量（mask 比较 `i < bound` 的建模）；
  2. 做区间推理（见 `Interval`），从而静态证明 mask 恒真/恒假；
  3. 在降级时把同一个表达式重新物化成 SSA 值（`materializeIndex`）。
- `SymKind` 只有四类：`PidX/PidY/PidZ` 和 `LoopIv`。也就是说，目前被
  当作"符号"的来源只有 program id 和 scf.for 的归纳变量，其余（普通
  函数参数等）要么折叠成常量、要么直接导致分析失败（保守）。

#### Interval 与 SymbolBounds

```cpp
struct Interval {
  bool hasLo, hasHi;
  int64_t lo, hi;
  static Interval point(int64_t v);
  static Interval unknown();
};

struct SymbolBounds {
  std::array<std::optional<Interval>, 3> pids; // pid 区间，来自 privateuse.grid
  llvm::DenseMap<unsigned, Interval> loopIvs;  // loop iv 的区间，来自 scf.for 边界
};
```

- `Interval` 是"闭区间近似"：只知道 `lo..hi`，不知道具体值。`unknown()`
  表示完全不知道（任何基于它的证明都不成立）。
- `evaluateInterval(expr, bounds)` 在表达式树上做区间算术（加/乘/移位/
  min/max 等）；溢出时退化为 `unknown()`，宁可"证明不出来"也不产生
  错误结论。
- 这套区间推理被两类地方使用：
  1. `classifyMask`：判断 `bound` 在 `i ∈ [0, size)` 上是否恒成立/恒不成立；
  2. `buildCompareMask`：无下标依赖的比较直接折叠成常量真/假
     （`foldConstantComparison`）。

### 2.3 `Analysis/AccessAnalysis.h`：分析结果的数据结构

#### AccessDim / AccessView

```cpp
struct AccessDim { ExprPtr start; int64_t size; int64_t stride; };
struct AccessView {
  Value base;                                        // 入口块!tt.ptr参数
  ExprPtr offset;                                    // 标量偏移表达式
  SmallVector<AccessDim> dims;                       // 每维(start, size, stride)
  LogicalResult normalize(Type resultType, Location loc);
  bool isInjective() const;
};
```

- 语义：第 `(i0,...,in)` 个元素的地���偏移 =
  `offset + Σ_d (dims[d].start + dims[d].stride * i_d)`，单位是**元素**，
  不是字节。
- `normalize(resultType)`（`AccessAnalysis.cpp`）：
  - 对张量结果，要求 `dims.size() == rank`；`size == 0` 的维度用张量
    静态形状补齐，非零但不一致则报错；动态维度直接拒绝——降级路径只
    支持静态形状的 tile。
  - 对标量结果，要求 `dims` 为空。
- `isInjective()`：判断从逻辑下标到线性偏移的映射是否一一对应。做法是
  把 `size > 1` 的维度按 stride 升序排序，然后要求每个 stride **严格
  大于**此前所有维度覆盖的跨度 `span`（用 `Int128` 做 128 位累加以防
  溢出）：

  ```
  span = 0
  for dim in sorted_by_stride:
      if dim.stride <= span: return false
      span += (dim.size - 1) * dim.stride
  ```

  直觉：低 stride 维度铺满后，高 stride 维度的步长必须越过整段跨度，
  访存才不会互相覆盖。`stride == 0` 的维度（广播）不参与判断，因此
  "广播 load"是单射的（多个逻辑元素读同一个地址，读没有副作用），而
  "行距 14 < 16 的 16x16 store"不是（写会重叠）。

#### LinearBound / MaskKind / AccessMask

```cpp
struct LinearBound {
  ExprPtr offset;   // 不等式写成 offset + stride*i < 0 (或 <= 0)
  int64_t stride;
  bool strict;      // true: < 0；false: <= 0
  bool lower;       // 下界（反转方向），只会在判别式中使用
};

enum class MaskKind { None, ProvenTrue, ProvenFalse, ScalarDynamic, Box };

struct AccessMask {
  MaskKind kind = MaskKind::None;
  SmallVector<SmallVector<LinearBound>> dimBounds; // 每维的合取项
  static AccessMask uniform(bool truth);
  static AccessMask dynamicScalar();
};
```

- mask 的规范形是一个**合取**：mask 为真 ⟺ 每一维上的每个 `LinearBound`
  都满足。例如 `mask = (i0 < m) & (i1 < n)` 会得到 dim0/dim1 各一条 bound。
- `kind` 的单一定义（这是最近一次重构的核心）：
  - `None`：没有 mask；
  - `ProvenTrue/ProvenFalse`：构造期已确证（常量 splat、折叠出结果的
    比较、`andi` 中带 proven-false 分支）；
  - `ScalarDynamic`：rank-0 标量 mask，且值运行时才知道；
  - `Box`：有 `dimBounds`，但未证明恒真/恒假，表示 mask 覆盖一个
    "盒形前缀窗口"。
- 为什么要区分 `ProvenTrue` 和 `Box`：前者可以完全忽略 mask（整块
  copy），后者必须只搬运有效窗口（见 3.3）。
- `classifyMask` 只处理 `Box`：对每条 bound 在整段 `i ∈ [0, size)` 上做
  区间评估，全部 holds → `ProvenTrue`；任一 fails → `ProvenFalse`
  （合取里只要一个分量恒假，整体恒假）；否则保持 `Box`。早退保证已证
  事实不会被重算（例如 `uniform(false)` 没有 bound，若重算会被误翻成
  `ProvenTrue`）。

#### AccessInfo / DeferredReason

```cpp
enum class DeferredReason : uint8_t { None, NegativeStride, DataDependentOffset,
                                      UnsupportedDescriptor, Atomic };

struct AccessInfo {
  bool isLoad;
  int64_t elemBytes;      // 仅报告用；两个降级 pass 不消费
  AccessView view;
  AccessMask mask;
  DeferredReason deferred;
};
```

- `deferred != None` 表示"分析知道这个访问暂时降不了"，但原因分类稳定，
  报告里按原因计数。`NegativeStride` 只影响结构化路径；unstructured 按
  实际地址链改写，不依赖 stride 的正负。
- `elemBytes` 只给 `-privateuse-analyze-access` 的报表用。

### 2.4 `AccessAnalyzer` 的对外 API（两个 pass 实际调用的部分）

```cpp
explicit AccessAnalyzer(ModuleOp module);
FailureOr<AccessInfo> analyze(Operation* op);
FailureOr<Value> materializeIndex(const ExprPtr& expr, OpBuilder&, Location);
FailureOr<Value> materializeBoundCount(const LinearBound&, OpBuilder&, Location);
FailureOr<Value> materializeDimSize(ArrayRef<LinearBound> bounds, int64_t dimSize,
                                    OpBuilder&, Location);
FailureOr<SmallVector<Value>> materializeMaskSizes(const AccessInfo&, OpBuilder&, Location);
```

#### analyze(op)

对单个 `tt.load / tt.store` 做完整分析：

1. `buildValue(ptr)` 把地址 DAG 规约成 `AccessView`（失败时看 `deferred`
   决定返回 deferred 还是硬失败）；
2. 要求 `view.base` 是 entry block 的 `BlockArgument`、且是标量
   `!tt.ptr<T>`（基址必须是函数参数，后续 `PtrToMemrefPass` 只改参数）；
3. `normalize(ptr 类型)` 校验 rank/形状；
4. 若有 mask：`buildMask(mask, rank)` 分解成 `AccessMask`；然后对每条
   **下界** bound 做区间评估（不能证明恒成立 → `DataDependentOffset`；
   成立的从 `dimBounds` 移除，因为降级只支持上界窗口）；
5. `classifyMask` 把 `Box` 收紧；
6. 任一维 `stride < 0` → `NegativeStride`。

#### buildValue 的覆盖范围（决定"什么能被分析"）

`buildValue` 递归处理：`arith.constant`（整数/稠密常量）、`tt.splat`、
`tt.make_range`、`tt.expand_dims`、`tt.broadcast`、`tt.reshape`、
`tt.addptr`、`arith.addi/subi`、`arith.muli`（乘常量）、`tt.bitcast`
（指针↔指针透传）、整型 cast（统一转成有符号 `extsi` 视角）、循环
不变量（`scf.for` iter_arg 追到初值）、entry block 参数（base 或标量
地址）。从内存 load 出来的值 → `DataDependentOffset`；其他 op → 硬失败。

#### materializeIndex(expr, builder, loc)

把 `ScalarExpr` 重新物化为 `index` 类型的 SSA 值：

- `Const` → `arith.constant index`；
- `Sym` → `materializeSymbol`：pid 用分析时记录的 SSA 值
  （`pidValues_`），loop iv 用 `ivValues_`；不是 `index` 类型的加
  `arith.index_cast`；
- 二元节点 → 对应 arith 整数指令（加/减/乘/divsi/divui/remsi/.../minsi/maxsi...）。

为什么需要"物化"：分析只是"看"表达式，降级时必须把同一个表达式重新
变成 IR 里的值，才能作为 `reinterpret_cast` 的偏移或 gather 的索引。

#### materializeBoundCount(bound, builder, loc)

计算一条上界在 `i ∈ [0, size)` 内允许的**元素个数**：

```
count = strict ? ceil((-offset) / stride) : floor((-offset) / stride) + 1
count = max(0, count)
```

- 语义：`offset + stride*i < 0`（或 `<= 0`）即"该 lane 有效"。
- `strict` 用 `arith.ceildivsi`，非 strict 用 `arith.floordivsi`——
  必须用有符号向下/向上取整，因为 `-offset` 可能为负（offset 为正时
  整个窗口为空）。
- 非 strict 的 `+1` 有一个显式的溢出保护：`count+1` 若回绕（`sum < count`），
  结果取 `INT64_MAX`，随后 `maxsi(0, ...)` 与 `materializeDimSize` 里的
  `minsi(dimSize)` 会把它夹回范围。这是手写检查而不是依赖 UB 的算术。

注意：`-offset` 用 `subi 0, offset` 而不是取反指令，是因为 `index` 是
符号无关的 64 位整数，减法是唯一的通用写法。

#### materializeDimSize / materializeMaskSizes

```cpp
size = min(bound_count_0, bound_count_1, ..., dimSize)
```

一维可能有多条上界（`andi` 合并出来的），取最小值即取交集；最后再和
静态维度大小取 min 防止越界。`materializeMaskSizes` 对每维调用一次，
得到与张量 rank 等长的动态尺寸列表，供 `reinterpret_cast` / `subview`
使用。

### 2.5 `Utils/Utils.h`

```cpp
bool isPtrTypeLike(Type t);                 // 标量指针 / 指针元素张量
Type storageElementType(Type element);      // i1 的存储类型是 i8
Value maskedFillValue(LoadOp, Type, OpBuilder&, Location);
Value createBufferWindow(OpBuilder&, Location, Value buf,
                         ArrayRef<int64_t> shape, ArrayRef<Value> dimSizes);
```

- `isPtrTypeLike`：`!tt.ptr<T>` 或 `tensor<...x!tt.ptr<T>>`。unstructured
  的链回溯用它判断"这个 bitcast 是否只是换指针元素类型"。
- `storageElementType`：`i1 → i8`，其余原样。Triton 的 `i1` 张量在内存
  里占 1 字节，`!tt.ptr<i1>` 与 `!tt.ptr<i8>` 指向同一块存储。所有
  降级路径都以"存储元素类型"对齐 buffer/view 类型，避免出现
  `memref<...xi1>` 这种和物理布局不符的类型。
- `maskedFillValue`：取 masked load 的 `other`，要求它是 splat 形态
  （`tt.splat` 或稠密 splat 常量），返回其标量值；
  没有 `other` 时返回零；否则直接 emitError（"masked load with
  non-splat 'other' is unsupported"）。为什么限制 splat：tile buffer 是
  先整块 fill 再拷窗口，填充值必须是一个标量。
- `createBufferWindow`：对 tile buffer 生成 `memref.subview`
  `[0,0,...] [size_0,...,size_n] [1,...,1]`，即"有效窗口"。静态维度
  用 `getIndexAttr`，动态维度用分析物化出的 Value。
- `Int128` + `checkedAdd/Sub/Mul`：所有常量区间的算术语义都要"溢出即
  放弃"（返回 false，让分析退化为 unknown），手写 128 位检查而不用
  `__builtin_*_overflow`，保证不同编译器/目标下行为一致。

### 2.6 `PUMem` dialect 的 Gather / Scatter

```mlir
pumem.gather   %source, %indices, %destination [, %mask [, %other]]
pumem.scatter  %source, %indices, %destination [, %mask]
```

- `gather`：`destination[i] = mask[i] ? source[indices[i]] : other`。
  `source` 是 rank-1 的"扁平运行时 buffer"（元素寻址），`indices` 是
  与 `destination` 同 shape 的偏移 buffer。无 mask 时等价于全真；
  有 mask 无 `other` 时，被屏蔽的 lane 不写（保持原值）。
- `scatter`：`destination[indices[i]] = source[i]`，有 mask 时被屏蔽的
  lane 既不读也不写。
- 两个 op 都在原地修改目标 buffer（side effect 声明在 td 里），语义类似
  `memref.copy`，因此不需要返回值。
- verifier 保证：`source`/`destination` rank-1、indices 为整型、shape
  与 tile 对齐、元素类型一致、mask 为 i1 且 shape 匹配、`other` 必须
  伴随 mask 且类型与元素类型一致。

### 2.7 Pass 基础设施

- `Passes.td` 声明 `LowerStructuredAccess` / `LowerUnstructuredAccess`，
  构造函数分别是 `createLowerStructuredAccessPass()` /
  `createLowerUnstructuredAccessPass()`——两者都是 `ModuleOp` pass。
- `.cpp` 里用 `GEN_PASS_DEF_LOWER*` + `Passes.h.inc` 生成基类
  `::impl::Lower*Base`。`getDependentDialects` 声明本 pass 会创建的
  dialect（不在其中注册的 dialect 在 pass 运行时创建 op 会崩）。
- `privateuse.cc` 暴露 `add_lower_structured_access(pm)` 等 Python 绑定，
  供 `compiler.py` 装配管线。
- lit 测试组织：`test/AccessAnalysis/`（分析报表）、
  `test/AccessLowering/`（降级 + `-one-shot-bufferize` 端到端）。

---

## 3. LowerStructuredAccessPass 逐函数讲解

文件：`third_party/privateuse/lib/Conversion/LowerStructuredAccessPass.cpp`

### 3.1 匿名命名空间里的顶层工具函数

#### 函数过滤（isExternal）

结构化 pass 不再用 `isConvertibleFunction` 做 ABI 预筛，`runOnOperation`
里只跳过外部声明：

```cpp
for (triton::FuncOp func : functions) {
  if (func.isExternal()) continue;
  ...
}
```

- `isExternal()`（声明但无函数体）直接跳过，没有可改写的 IR。
- 旧 `isConvertibleFunction` 还会拒绝 tensor 类型的参数/返回值，这层
  静态 ABI 检查已删除；函数能否降级完全交给逐访问的 `isLowerable`。

#### getAccessPointerType

```cpp
Type getAccessPointerType(Operation* op)
{
  Value ptr;
  if (auto load = dyn_cast<LoadOp>(op)) {
    if (load.getIsVolatile()) return {};
    ptr = load.getPtr();
  } else if (auto store = dyn_cast<StoreOp>(op)) {
    if (store.getIgnoreCta()) return {};
    ptr = store.getPtr();
  } else {
    return {};
  }
  if (auto tensor = dyn_cast<RankedTensorType>(ptr.getType())) {
    if (tensor.getRank() == 0) return {};
    if (!isa<triton::PointerType>(tensor.getElementType())) return {};
    return tensor;
  }
  if (isa<triton::PointerType>(ptr.getType())) return ptr.getType();
  return {};
}
```

- 返回"可降级访问的指针类型"：`tensor<...x!tt.ptr<T>>`（rank ≥ 1）或标量
  `!tt.ptr<T>`；其他情况返回空。
- 拒绝 volatile load：volatile 的"每次访问都不可省略/不可重排"语义用
  `memref.copy`（可能被向量化/合并）无法保证。
- 拒绝 `ignore_cta` store：这需要 CTA 级语义（同 CTA 内只看一个 lane），
  当前后端没有对应支持，直接留给后续阶段。
- 为什么结构化 pass 也收标量访问：标量地址同样规约成 `base + offset`，
  直接生成 `memref.load/store` 比 gather/scatter 便宜。

#### floorDiv / ceilDiv

```cpp
int64_t floorDiv(int64_t lhs, int64_t rhs) {
  int64_t q = lhs / rhs;
  if (lhs % rhs != 0 && (lhs < 0) != (rhs < 0)) --q;
  return q;
}
int64_t ceilDiv(int64_t lhs, int64_t rhs) {
  int64_t q = lhs / rhs;
  if (lhs % rhs != 0 && (lhs < 0) == (rhs < 0)) ++q;
  return q;
}
```

- C++ 的 `/` 向零截断，而窗口计数公式需要真正的 floor/ceil。
  例：`limit=-3, stride=2`，截断除法得 `-1`（错），floor 才是 `-2`。
- 只在 `isMaskWindowInjective` 的**常量**窗口推导里使用；动态情形走
  `materializeBoundCount` 的 `arith.ceildivsi/floordivsi`。

#### isMaskWindowInjective

```cpp
bool isMaskWindowInjective(const AccessInfo& info)
{
  if (info.mask.kind == MaskKind::None || info.mask.kind == MaskKind::ProvenTrue) return false;
  AccessView window = info.view;
  for (auto [d, dim] : llvm::enumerate(window.dims)) {
    for (const LinearBound& bound : info.mask.dimBounds[d]) {
      if (bound.lower) return false;
      if (!bound.offset->isConst() || bound.offset->value == INT64_MIN) return false;
      int64_t limit = -bound.offset->value;
      if (!bound.strict && limit == INT64_MAX) return false;
      int64_t count = bound.strict ? ceilDiv(limit, bound.stride)
                                   : floorDiv(limit, bound.stride) + 1;
      dim.size = std::min(dim.size, std::max<int64_t>(count, 0));
    }
  }
  return window.isInjective();
}
```

- 用途：**store** 的整块 view 不单射时，判断"mask 实际覆盖的窗口"是否
  单射。`@overlap_box_store` 就是例子：16x16 tile 行距 14 会重叠，但
  mask 只写出 14x14 前缀，窗口内 stride 14 恰好单射，因此仍可降级。
- 只处理常量上界：动态窗口大小要在运行时才知道，静态单射证明做不了，
  保守拒绝。
- `offset == INT64_MIN` 拒绝：`-offset` 会溢出；非 strict 且
  `limit == INT64_MAX` 拒绝：`+1` 会溢出——与 `materializeBoundCount`
  的运行时保护对应，这里直接在编译期放弃。
- 计数公式与 `materializeBoundCount` 完全一致（strict → ceil，
  非 strict → floor+1），保证"编译期证明"和"运行期物化"是同一套语义。
- 注意开头的 `None/ProvenTrue` 早退：没有窗口概念时返回 false，让调用
  方回落到"整块 view 必须单射"的判断。

#### isLowerable

```cpp
bool isLowerable(const AccessInfo& info, Type pointerType, Operation* op, triton::FuncOp func)
{
  if (info.deferred != DeferredReason::None) return false;
  if (info.mask.kind == MaskKind::ScalarDynamic || info.mask.kind == MaskKind::ProvenFalse) return false;
  if (isa<StoreOp>(op) && !info.view.isInjective() && !isMaskWindowInjective(info)) return false;
  auto base = dyn_cast<BlockArgument>(info.view.base);
  if (!base || base.getOwner() != &func.getBody().front()) return false;
  auto baseType = dyn_cast<triton::PointerType>(base.getType());
  if (!baseType) return false;
  auto accessType = cast<triton::PointerType>(
    isa<RankedTensorType>(pointerType) ? cast<RankedTensorType>(pointerType).getElementType() : pointerType);
  auto pointee = accessType.getPointeeType();
  if (storageElementType(pointee) != pointee) return false;
  return storageElementType(baseType.getPointeeType()) == storageElementType(pointee);
}
```

逐条理由：

1. `deferred != None`：分析已经判定这个访问不该走结构化路径（负 stride、
   数据相关偏移、descriptor、atomic）。后续 pass 不再重复判断。
2. `ScalarDynamic`：运行时标量条件，无法物化窗口；`ProvenFalse`：整块
   被屏蔽，structured 的"窗口搬运"没有意义（而且常量 false 的窗口为
   空，还不如直接走 unstructured 让 mask 在运行时自然处理）。
3. store 单射检查：写重叠地址的最终结果依赖写的顺序，Triton 语义不保证；
   load 读重叠地址是安全的（广播读），所以不检查。
4. base 必须是**入口块参数**：`PtrToMemrefPass` 之后只会把函数参数换成
   memref；若 base 是循环 iter_arg / 函数返回值，签名改写后就找不到
   对应的 memref。
5. 元素类型必须"存储兼容"：访问指针的 pointee 必须是存储类型本身
   （例如 `i8`），不能是 `i1`；且与 base 的存储元素类型一致
   （`storageElementType(base) == storageElementType(pointee)`），保证
   `memref<?xT>` token 与 `memref<...xT>` view 的元素类型一致。
   这就是 `@bool_store` 要先 `tt.bitcast` 到 `!tt.ptr<i8>` 的原因。

#### sramBufferType

```cpp
MemRefType sramBufferType(RankedTensorType tensorType, OpBuilder& builder)
{
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                         MemRefLayoutAttrInterface(), builder.getI64IntegerAttr(kSramSpace));
}
```

- tile buffer 的类型：**静态形状** + 访问元素类型 + identity layout +
  memory space 128。静态形状是硬要求——tensor 世界只接受静态张量；
  "动态"部分只存在于全局侧的 view 上。

### 3.2 AccessLowering::run

```cpp
LogicalResult run()
{
  if (!collectInfos()) return success();
  for (const auto& [op, info] : infos_)
    if (failed(lowerAccess(op, info))) return failure();
  return success();
}
```

- 先**一次性收集**所有可降级访问（`collectInfos`），再统一改写。
  为什么不在 walk 里直接改：改写会 erase 当前 op、插入新 op，边遍历边
  改 IR 容易漏访问/重复访问；先快照 `(op, info)` 也保证 analyzer 看到
  的一直是原始 IR。
- `collectInfos` 返回 false（什么都没收集到）→ 直接成功，函数里不会产生
  memref/bufferization/pumem 算子。（在真实管线里，这种软跳过函数随后会被
  unstructured 降级或报错，不会以未降级形态到达 `PtrToMemrefPass`。）
- 注意：任一 `lowerAccess` 返回 failure 会让整个 pass 失败——因为此时
  部分 op 已经被改写，IR 处于半降级状态，继续下去语义不明；结构化路径
  的"预期内失败"都应该在 `collectInfos` 阶段被过滤掉。

### 3.3 collectInfos

```cpp
bool collectInfos()
{
  ScopedDiagnosticHandler handler(func_->getContext(), [](Diagnostic&) {});
  size_t count = infos_.size();
  func_.walk([&](Operation* op) {
    auto pointerType = getAccessPointerType(op);
    if (!pointerType) return;
    FailureOr<AccessInfo> info = analyzer_.analyze(op);
    if (failed(info) || !isLowerable(*info, pointerType, op, func_)) return;
    infos_.try_emplace(op, std::move(*info));
  });
  return infos_.size() != count;
}
```

- `ScopedDiagnosticHandler`：分析是**试探性**的（walk 到每个访存都试一次），
  失败是正常情况，不应该往 stderr 打噪声；真正的错误报告交给
  `-privateuse-analyze-access`。handler 在作用域结束时自动恢复。
- `walk` 顺序收集；`analyze` 内部维护 per-analyzer 的符号状态
  （pid/loop iv 映射），同一函数内复用。
- `try_emplace` 而不是 `insert`：避免对已存在的 key 做无谓的 move。
- `return infos_.size() != count`：告知 run 本次是否新增了条目。

### 3.4 materializeBase

```cpp
Value materializeBase(const AccessInfo& info, Type pointee, OpBuilder& builder, Location loc)
{
  auto tokenType = MemRefType::get({ShapedType::kDynamic}, pointee);
  return UnrealizedConversionCastOp::create(builder, loc, tokenType, info.view.base).getResult(0);
}
```

- 把一个 `!tt.ptr<T>` 值临时"伪装"成 `memref<?xT>`：`unrealized_conversion_cast`
  是 MLIR 的类型桥，不做任何实际运算，后续 `PtrToMemrefPass` /
  reconcile 阶段会把 base 真正换成 memref 参数，并把这些 cast 折叠掉。
- 为什么不一上来就改函数签名：base 的地址链（splat/addptr/...）还在
  别处被引用（比如同一函数里未降级的访存、mask 链），"就地改签名"
  会迫使所有使用者一起改；桥接 cast 让每个访问独立降级，链的清理
  留给专用 pass 统一处理。
- `memref<?xT>` 是最宽松的 rank-1 动态形状；`reinterpret_cast` 的
  result 类型才是真正使用的 view。

### 3.5 materializeView

```cpp
FailureOr<Value> materializeView(OpBuilder& builder, Location loc, const AccessInfo& info,
                                 Type pointee, ArrayRef<Value> dimSizes = {})
{
  Value base = materializeBase(info, pointee, builder, loc);

  ExprPtr total = info.view.offset;
  SmallVector<int64_t> sizes, strides;
  SmallVector<OpFoldResult> sizeAttrs, strideAttrs;
  for (auto [d, dim] : llvm::enumerate(info.view.dims)) {
    total = ScalarExpr::add(std::move(total), dim.start);
    Value dynamic = d < dimSizes.size() ? dimSizes[d] : Value();
    sizes.push_back(dynamic ? ShapedType::kDynamic : dim.size);
    strides.push_back(dim.stride);
    sizeAttrs.push_back(dynamic ? OpFoldResult(dynamic) : OpFoldResult(builder.getIndexAttr(dim.size)));
    strideAttrs.push_back(builder.getIndexAttr(dim.stride));
  }

  FailureOr<Value> offset = analyzer_.materializeIndex(total, builder, loc);
  if (failed(offset)) return failure();

  auto layout   = StridedLayoutAttr::get(builder.getContext(), ShapedType::kDynamic, strides);
  auto viewType = MemRefType::get(sizes, pointee, layout);
  return memref::ReinterpretCastOp::create(builder, loc, viewType, base, *offset, sizeAttrs, strideAttrs).getResult();
}
```

- 目的：把 `base + offset + Σ(start + stride*i)` 变成一条
  `memref.reinterpret_cast`，得到一个"全局内存上的 strided view"。
- **offset 吸收每维 start**：`total = offset + Σ start_d`。这样 view 的
  起始地址就是第 0 个元素，后面的 copy 不需要再算任何偏移。
- `dimSizes` 非空（masked 情形）时把该维 size 换成动态 Value：源 view
  只覆盖"有效窗口"，`memref.copy` 自然不会读写窗口之外。
- `StridedLayoutAttr` 的 offset 记 `ShapedType::kDynamic`：运行时才知道
  实际字节偏移，类型系统只需要知道"不是静态布局"。
- `stride` 用 `getIndexAttr` 写进类型和 `reinterpret_cast` 的操作数，
  两者必须一致（MLIR verifier 会检查）。
- 为什么用 `reinterpret_cast` 而不是 `memref.subview`：base 是一个
  不透明的 `memref<?xT>`（还没有真实形状），subview 要求源 memref 有
  匹配的 rank/形状；`reinterpret_cast` 可以从 rank-1 动态 memref 直接
  重新解释成任意 rank/stride 的 view。

### 3.6 lowerScalarLoad / lowerScalarStore

```cpp
LogicalResult lowerScalarLoad(LoadOp load, const AccessInfo& info)
{
  OpBuilder builder(load);
  Location loc            = load.getLoc();
  Value base              = materializeBase(info, load.getType(), builder, loc);
  FailureOr<Value> offset = analyzer_.materializeIndex(info.view.offset, builder, loc);
  if (failed(offset)) return failure();

  Value value = memref::LoadOp::create(builder, loc, base, *offset);
  load.replaceAllUsesWith(value);
  load->erase();
  return success();
}
```

- 标量访问没有 dims，直接 `memref.load %base[%offset]`。
- `load.getType()` 就是 `!tt.ptr<T>` 的 pointee 类型；`materializeBase`
  把它桥接成 `memref<?xT>`，`memref.load` 只需要 rank-1 索引。
- `replaceAllUsesWith` + `erase` 是标准替换；标量结果类型不变
  （`f32` 还是 `f32`），所以不需要任何 cast。
- store 对称：`memref.store value, %base[%offset]`，然后 erase。

### 3.7 lowerLoad（张量路径）

```cpp
LogicalResult lowerLoad(LoadOp load, const AccessInfo& info)
{
  auto tensorType = dyn_cast<RankedTensorType>(load.getType());
  if (!tensorType) return lowerScalarLoad(load, info);

  OpBuilder builder(load);
  Location loc = load.getLoc();
  bool masked  = info.mask.kind != MaskKind::None && info.mask.kind != MaskKind::ProvenTrue;
  SmallVector<Value> dimSizes;
  if (masked) {
    FailureOr<SmallVector<Value>> sizes = analyzer_.materializeMaskSizes(info, builder, loc);
    if (failed(sizes)) return failure();
    dimSizes = std::move(*sizes);
  }

  Type elementType      = tensorType.getElementType();
  FailureOr<Value> view = materializeView(builder, loc, info, elementType, dimSizes);
  if (failed(view)) return failure();

  Value buf = memref::AllocOp::create(
    builder, loc, sramBufferType(tensorType, builder), builder.getI64IntegerAttr(kDefaultAlignment));
  Value dst = buf;
  if (masked) {
    Value fillValue = triton::maskedFillValue(load, elementType, builder, loc);
    if (!fillValue) return failure();
    linalg::FillOp::create(builder, loc, ValueRange{fillValue}, ValueRange{buf});
    dst = createBufferWindow(builder, loc, buf, tensorType.getShape(), dimSizes);
  }
  memref::CopyOp::create(builder, loc, *view, dst);
  Value tensor = bufferization::ToTensorOp::create(builder, loc, tensorType, buf,
                                                   /*restrict=*/true, /*writable=*/true);
  load.getResult().replaceAllUsesWith(tensor);
  load->erase();
  return success();
}
```

分段解释：

1. **分派**：结果不是 ranked tensor → 标量路径。之所以用结果类型判断，
   是因为 `tt.load` 的类型直接对应"读几个元素"。
2. **masked 判定**：`None/ProvenTrue` 之外都算"需要 mask 语义"。走到这里
   的 `Box` 是唯一可能（`ScalarDynamic/ProvenFalse` 已被 `isLowerable`
   拒掉），这个写法保留了对将来新增 kind 的保守性——只要不是"确定无
   mask"，就按掩码处理。
3. **masked 时先算窗口大小**：`materializeMaskSizes` 生成每维 `count`
   （min 上界与维度大小），供后面两处使用：
   - 源 view 的动态 size（只读有效窗口，避免越界读）；
   - tile buffer 的 subview 窗口（只覆盖有效前缀）。
4. **建源 view**：`elementType = tensorType.getElementType()`。注意这里
   用的是张量元素类型（如 `f32`），与 `isLowerable` 里
   `storageElementType(pointee) == pointee` 的保证配套；对 `i1` 张量，
   降级根本不会发生，因此不会出现 `memref<...xi1>`。
5. **分配 tile**：`memref<shape x elt, 128>`，64 字节对齐。所有临时数据
   都落在 scratchpad。
6. **masked：先 fill 再 copy 窗口**：
   - `linalg.fill(other, buf)` 把整个 tile 填成 `other`（缺省 0）；
   - `dst = subview buf[0..count)`；
   - `memref.copy(view, dst)` 只覆盖有效前缀。
   这样"被 mask 掉的 lane 读到 `other`"与 Triton 语义一致，而且只读
   有效地址（不会因为越界读崩溃）。
7. **ToTensor**：把 buffer 的结果变成 SSA tensor，替换 load 的所有使用。
   `restrict = true` 告诉 bufferization"这个 tensor 与其他 tensor 不
   别名"，`writable = true` 允许后续 in-place 改写；这两个标注直接
   影响 `one-shot-bufferize` 是否插入拷贝。
8. erase 原 load。

### 3.8 lowerStore（张量路径）

```cpp
LogicalResult lowerStore(StoreOp store, const AccessInfo& info)
{
  auto tensorType = dyn_cast<RankedTensorType>(store.getValue().getType());
  if (!tensorType) return lowerScalarStore(store, info);

  OpBuilder builder(store);
  Location loc = store.getLoc();
  bool masked  = info.mask.kind != MaskKind::None && info.mask.kind != MaskKind::ProvenTrue;
  SmallVector<Value> dimSizes;
  if (masked) {
    FailureOr<SmallVector<Value>> sizes = analyzer_.materializeMaskSizes(info, builder, loc);
    if (failed(sizes)) return failure();
    dimSizes = std::move(*sizes);
  }

  FailureOr<Value> view = materializeView(builder, loc, info, tensorType.getElementType(), dimSizes);
  if (failed(view)) return failure();

  Value buf = bufferization::ToBufferOp::create(
    builder, loc, sramBufferType(tensorType, builder), store.getValue(), /*read_only=*/true);
  Value src = masked ? createBufferWindow(builder, loc, buf, tensorType.getShape(), dimSizes) : buf;
  memref::CopyOp::create(builder, loc, src, *view);
  store->erase();
  return success();
}
```

- store 是 load 的镜像，但数据方向反过来：
  - **源**是待写张量 → `bufferization.to_buffer` 把它变成 SRAM 形状的
    memref（`read_only = true`，表示这个 buffer 只被读）；
  - **目的**是全局 view：masked 时用 `materializeView(..., dimSizes)`
    的动态 size，只写有效窗口（越界写是灾难，所以目的侧裁剪同样重要）；
  - masked 时源侧取 `subview` 窗口，保证 copy 的 shape 与动态 view 一致
    （`memref.copy` 要求 rank 相同、逐维尺寸匹配）。
- unmasked：`src = buf`、view 是静态尺寸，整块 copy。
- 注意 store 的实现顺序与 load 略有不同：load 先建 view 再 alloc；
  store 先建 view 再 ToBuffer。顺序无所谓，都是"在 op 之前插入"。

### 3.9 lowerAccess

```cpp
LogicalResult lowerAccess(Operation* op, const AccessInfo& info)
{
  if (auto load = dyn_cast<LoadOp>(op)) return lowerLoad(load, info);
  return lowerStore(cast<StoreOp>(op), info);
}
```

- 只有 load/store 会进 `infos_`（`getAccessPointerType` 已过滤），所以
  这里的 cast 是安全的。
- 结构化 pass 不再改写任何值 splat（旧的 `lowerSplats` / `liveAddressOps`
  已删除）。值 `tt.splat` 原样保留，统一交给后续的 `privateuse-arith-to-linalg`
  pass 转成 `linalg.fill`（见 `docs/arith-to-linalg.md`）。

---

## 4. LowerUnstructuredAccessPass 逐函数讲解

文件：`third_party/privateuse/lib/Conversion/LowerUnstructuredAccessPass.cpp`

### 4.1 顶层工具函数

#### 函数过滤（isExternal）

与结构化 pass 完全相同：`runOnOperation` 里只 `if (func.isExternal()) continue;`
跳过外部声明。旧的 `isConvertibleFunction`（含 tensor 参数/返回值拒绝）已删除。

#### getAccessPointerType

```cpp
RankedTensorType getAccessPointerType(Operation* op)
{
  ...
  auto type = dyn_cast<RankedTensorType>(ptr.getType());
  if (!type || type.getRank() == 0) return {};
  if (!isa<triton::PointerType>(type.getElementType())) return {};
  return type;
}
```

- 与结构化版的区别：**只接受 rank ≥ 1 的指针张量**。标量访问由结构化
  pass 负责；gather/scatter 是为"逐 lane 不同地址"设计的，标量没有意义。
- 同样拒绝 volatile / ignore_cta。

#### getAccessMask

```cpp
Value getAccessMask(Operation* op)
{
  if (auto load = dyn_cast<LoadOp>(op)) return load.getMask();
  return cast<StoreOp>(op).getMask();
}
```

- 只取 mask 操作数本身（可能是 null），不做语义解释；语义解释交给
  `hasEffectiveMask` 里的 analyzer。

#### sramBufferType / tensorBufferType

```cpp
MemRefType tensorBufferType(Value tensor, OpBuilder& builder)
{
  auto type = cast<RankedTensorType>(tensor.getType());
  return MemRefType::get(type.getShape(), type.getElementType(),
                         MemRefLayoutAttrInterface(), builder.getI64IntegerAttr(kSramSpace));
}
```

- `sramBufferType` 同结构化版（tile buffer）。
- `tensorBufferType` 由任意 tensor 值推出同形状的 SRAM memref 类型，
  用于 `to_buffer` 的目标类型（indices / mask / store 源）。
- 为什么 indices/mask buffer 也放 128 空间：它们是临时数据，随后被
  gather/scatter 消费；放 scratchpad 避免占用全局内存带宽。

#### mergeOffsetType / castOffset / castScalarOffset

```cpp
Type mergeOffsetType(Type lhs, Type rhs, Builder& builder)
{
  if (lhs.isInteger(64) || rhs.isInteger(64)) return builder.getI64Type();
  return builder.getI32Type();
}

Value castOffset(Value value, Type target, OpBuilder& builder, Location loc)
{
  auto type = cast<RankedTensorType>(value.getType());
  if (type.getElementType() == target) return value;
  auto resultType = RankedTensorType::get(type.getShape(), target);
  // Triton offsets are signed; widening must sign-extend (extui corrupts negative offsets).
  return arith::ExtSIOp::create(builder, loc, resultType, value);
}
```

- `mergeOffsetType`：两段偏移相加时统一到 `i64`（任一侧是 i64）否则 `i32`。
  为什么不无条件 i64：i32 偏移的 tile 更小、访存更省；只有真正需要
  64 位时（map 里存 i64）才升级。
- `castOffset`：要求逐元素**符号扩展**（`extsi`）。偏移可能为负
  （`base - k` 形态），`extui` 会把负数变成巨大正数，直接打穿地址
  空间。代码里为此留了注释。
- `castScalarOffset`：标量版本的同一逻辑，直接比 `== target`。

#### PointerChain

```cpp
struct PointerChain {
  Value base;     // 入口块指针参数
  Value offsets;  // 逐元素"元素偏移"张量；可为空（等价于全 0）
};
```

- `offsets` 的单位是元素，不是字节——与 gather/scatter 的 indices 语义
  一致。
- 可以为空表示"链路里没有偏移"，物化时用零张量。

### 4.2 UnstructuredLowering::run / collectAccesses

```cpp
LogicalResult run()
{
  SmallVector<Operation*> accesses;
  if (failed(collectAccesses(accesses))) return failure();
  for (Operation* op : accesses)
    if (failed(lowerAccess(op))) return failure();
  return success();
}

LogicalResult collectAccesses(SmallVectorImpl<Operation*>& accesses)
{
  bool unsupported = false;
  func_.walk([&](Operation* op) {
    if (isa<LoadOp, StoreOp>(op)) {
      accesses.push_back(op);
      return;
    }
    if (isa<DescriptorLoadOp, DescriptorStoreOp, DescriptorReduceOp, DescriptorGatherOp, DescriptorScatterOp,
            AtomicRMWOp, AtomicCASOp, AtomicPollOp>(op)) {
      op->emitError("privateuse-lower-unstructured-access: unsupported access op '")
        << op->getName() << "' (only tt.load/tt.store are allowed)";
      unsupported = true;
    }
  });
  return failure(unsupported);
}
```

- 先快照再改写：`collectAccesses` 一次性收集 op 指针列表，改写阶段逐个
  erase，每个指针只处理一次。
- **白名单门（collectAccesses）**：访存 op 只放行 `tt.load`/`tt.store`。
  walk 同时检查是否存在 descriptor 类（`DescriptorLoadOp`/`DescriptorStoreOp`/
  `DescriptorReduceOp`/`DescriptorGatherOp`/`DescriptorScatterOp`）或 atomic 类
  （`AtomicRMWOp`/`AtomicCASOp`/`AtomicPollOp`）算子——命中即 `emitError`
  （"unsupported access op '<name>' (only tt.load/tt.store are allowed)"）并置
  `unsupported`。只要发现任一非白名单访存 op，`collectAccesses` 返回 failure，
  `run` 立即返回，**不再降级任何 load/store**。
  - 为什么用显式 op 列表而非 interface：与 `AccessAnalysis` 一致，白名单边界
    要精确可控；新增访存 op 必须显式评审后加入，避免 interface 悄悄放行未
    支持的形态。
  - 为什么 atomic/descriptor 在这里硬报错而非软跳过：lower-or-error 契约要求
    unstructured 之后不得残留任何无法降级的访存。atomic 的读改写、descriptor
    的 TMA 语义都无法用 gather/scatter 表达；放行只会把失败推迟到下游。
    descriptor 在管线里通常已被 `add_rewrite_tensor_descriptor_to_pointer`
    改写成指针访存，能走到这里的 descriptor/atomic 都是真正不支持的形态。
- 收集**所有** `tt.load`/`tt.store`（`isa<LoadOp, StoreOp>`），不用
  `getAccessPointerType` 预过滤——lower-or-error 契约要求每个访问都被处理：
  要么降级，要么报错，不允许静默残留。
- **首个失败即返回**：`run` 里任一 `lowerAccess` 失败立即 `return failure()`，
  不再继续降级后续访问（与白名单门一致：发现缺口即停）。函数间的累积报告
  由 `runOnOperation` 负责（见 4.11）。
- 不再打函数属性。下游 `PtrToMemrefPass` 直接转换所有函数：lower-or-error
  保证到达它的函数访存均已降级，无需再用谓词判定哪些函数"降级过访存"。

### 4.3 hasEffectiveMask

```cpp
bool hasEffectiveMask(Operation* op)
{
  if (!getAccessMask(op)) return false;
  ScopedDiagnosticHandler handler(func_->getContext(), [](Diagnostic&) {});
  FailureOr<AccessInfo> info = analyzer_.analyze(op);
  return failed(info) || info->mask.kind != MaskKind::ProvenTrue;
}
```

- 判断"这个 mask 是否需要在运行时展开":
  - 没有 mask → false；
  - 分析失败（mask 形态无法分类）→ **true**，即保守地按"有 mask"处理，
    用真实 mask 张量在运行时逐 lane 掩码。这是安全兜底：分析失败不等于
    必须放弃降级，只是不能利用结构化性质；
  - `ProvenTrue` → false（整块为真，等价于无 mask）；
  - `Box / ScalarDynamic / ProvenFalse` → true。
- `ScalarDynamic/ProvenFalse` 在这里是"正确性兜底"：常量 false 的 mask
  会走带 mask 的 gather/scatter，被屏蔽 lane 不读写，语义正确。

### 4.4 findBase

```cpp
BlockArgument findBase(Value ptr)
{
  if (auto arg = dyn_cast<BlockArgument>(ptr))
    return isa<triton::PointerType>(arg.getType()) ? arg : BlockArgument();

  if (auto bitcast = ptr.getDefiningOp<BitcastOp>()) {
    if (isPtrTypeLike(bitcast.getSrc().getType())) return findBase(bitcast.getSrc());
    return {};
  }
  if (auto addptr = ptr.getDefiningOp<AddPtrOp>()) return findBase(addptr.getPtr());
  if (auto splat = ptr.getDefiningOp<SplatOp>()) return findBase(splat.getSrc());
  if (auto broadcast = ptr.getDefiningOp<BroadcastOp>()) return findBase(broadcast.getSrc());
  if (auto expand = ptr.getDefiningOp<ExpandDimsOp>()) return findBase(expand.getSrc());
  return {};
}
```

- 沿地址链回溯到"入口块指针参数"。与 `buildChain` 的形状操作一一对应。
- BlockArgument 分支要求是 `!tt.ptr` 标量（不是指针张量）：链的尽头必须
  是"单个基址"，张量形态的 BlockArgument 意味着地址空间本身在传参，
  当前不支持。
- bitcast 只有"指针→指针"才透传（`isPtrTypeLike` 同时检查源和目标），
  指针↔整数的 bitcast 会改变地址语义，不能当成同一地址。
- `reshape` 不在链里：unstructured 不重建 reshape 的坐标映射；含
  reshape 的链会在 findBase/buildChain 里失败并**硬报错**。

### 4.5 lowerAccess

```cpp
LogicalResult lowerAccess(Operation* op)
{
  Value ptr;
  if (auto load = dyn_cast<LoadOp>(op)) {
    if (load.getIsVolatile()) {
      op->emitError("privateuse-lower-unstructured-access: volatile load cannot be lowered to gather/scatter");
      return failure();
    }
    ptr = load.getPtr();
  } else {
    auto store = cast<StoreOp>(op);
    if (store.getIgnoreCta()) {
      op->emitError("privateuse-lower-unstructured-access: ignore_cta store cannot be lowered to gather/scatter");
      return failure();
    }
    ptr = store.getPtr();
  }

  auto pointerType = dyn_cast<RankedTensorType>(ptr.getType());
  if (!pointerType || pointerType.getRank() == 0 || !isa<triton::PointerType>(pointerType.getElementType())) {
    op->emitError("privateuse-lower-unstructured-access: access is not a ranked pointer tensor (scalar or rank-0)");
    return failure();
  }
  auto pointee = cast<triton::PointerType>(pointerType.getElementType()).getPointeeType();
  if (storageElementType(pointee) != pointee) {
    op->emitError("privateuse-lower-unstructured-access: direct i1 access requires i8 storage packing");
    return failure();
  }

  BlockArgument base = findBase(ptr);
  if (!base || base.getOwner() != &func_.getBody().front()) {
    op->emitError("privateuse-lower-unstructured-access: pointer base is not a kernel entry-block argument");
    return failure();
  }
  auto baseType = dyn_cast<triton::PointerType>(base.getType());
  if (!baseType || storageElementType(baseType.getPointeeType()) != storageElementType(pointee)) {
    op->emitError("privateuse-lower-unstructured-access: base and access storage element types differ");
    return failure();
  }

  OpBuilder builder(op);
  FailureOr<PointerChain> chain = buildChain(ptr, base, builder, op->getLoc());
  if (failed(chain)) {
    op->emitError("privateuse-lower-unstructured-access: unsupported pointer chain (cannot rebuild offsets)");
    return failure();
  }

  bool masked = hasEffectiveMask(op);
  if (auto load = dyn_cast<LoadOp>(op)) {
    if (failed(lowerLoad(load, *chain, masked, pointee))) return failure();
  } else {
    if (failed(lowerStore(cast<StoreOp>(op), *chain, masked, pointee))) return failure();
  }
  return success();
}
```

逐段（每道门都是 lower-or-error：命中即 `emitError` + `return failure()`）：

1. volatile load / `ignore_cta` store → 报错（语义无法用 gather/scatter 表达）。
2. 取 ptr，要求是 rank>0 的指针张量；标量 / rank-0 / 非指针元素 → 报错
   （标量访问由 structured pass 负责，unstructured 不接手）。
3. `storageElementType(pointee) != pointee`（仅 i1）→ 报错：直接访问 `i1`
   的指针不支持（Triton 的 i1 存储是 i8，gather 的元素类型必须和真实
   存储一致；`bool_masked` 那样先 bitcast 到 i8 才支持）。
4. base 必须是入口块参数（循环携带 / call 派生 / 非法 bitcast 都回溯不到）
   → 报错；base 与访问的元素存储类型不一致 → 报错（后续签名改写只覆盖
   函数参数）。
5. `buildChain` 失败 → 报错（地址链形态超出能力，例如含 reshape、
   含标量+张量混合 addptr 等）。
6. 通过全部门后才 `hasEffectiveMask`（重新 analyze 一次，不缓存），
   分派 lowerLoad / lowerStore；其内部失败（如非 splat 的 `other`）同样硬失败。

**lower-or-error 契约**：unstructured 是访问降级的最后一道关。pass 成功
结束后 IR 中**不得残留任何 `tt.load`/`tt.store`**——每个访问要么降成
gather/scatter，要么硬报错。这取代了旧的"尽力而为、软跳过留残留"哲学：
残留只会把失败推迟到下游 bufferize/_emit_cpp，难以定位；现在在降级阶段
就把覆盖缺口显式暴露。structured pass 仍软跳过（把幸存者交给 unstructured），
unstructured 则不再放行任何无法降级的访问。

### 4.6 buildScalarOffsets

```cpp
FailureOr<Value> buildScalarOffsets(Value ptr, BlockArgument base, OpBuilder& builder, Location loc)
{
  if (ptr == base) return Value{};
  if (auto bitcast = ptr.getDefiningOp<BitcastOp>()) {
    if (!isPtrTypeLike(bitcast.getSrc().getType())) return failure();
    return buildScalarOffsets(bitcast.getSrc(), base, builder, loc);
  }
  auto addptr = ptr.getDefiningOp<AddPtrOp>();
  if (!addptr) return failure();
  FailureOr<Value> inner = buildScalarOffsets(addptr.getPtr(), base, builder, loc);
  if (failed(inner)) return failure();
  Value offsets = addptr.getOffset();
  if (!*inner) return offsets;
  Type type = mergeOffsetType((*inner).getType(), offsets.getType(), builder);
  Value lhs = castScalarOffset(*inner, type, builder, loc);
  Value rhs = castScalarOffset(offsets, type, builder, loc);
  return Value(arith::AddIOp::create(builder, loc, lhs, rhs));
}
```

- 用于 `tt.splat` 的**标量**指针源：`tt.splat %p` 里 `%p` 可能是
  `tt.addptr` 链叠出来的标量指针（`base + k1 + k2`）。
- 递归结构：`base` 返回空（无偏移）；bitcast 透传；addptr 把内层偏移与
  本层 offset 相加。返回空值 `Value{}` 表示"偏移为 0"，由调用方决定
  物化方式（splat 时整体变成零张量）。
- 每层合并都做类型提升和符号扩展，避免 i32/i64 混用导致 verifier 报错
  或负数截断。

### 4.7 buildChain

```cpp
FailureOr<PointerChain> buildChain(Value ptr, BlockArgument base, OpBuilder& builder, Location loc)
{
  if (auto addptr = ptr.getDefiningOp<AddPtrOp>()) {
    FailureOr<PointerChain> chain = buildChain(addptr.getPtr(), base, builder, loc);
    if (failed(chain)) return failure();
    Value offsets = addptr.getOffset();
    if (!chain->offsets) return PointerChain{base, offsets};

    Type chainType  = cast<RankedTensorType>(chain->offsets.getType()).getElementType();
    Type offsetType = cast<RankedTensorType>(offsets.getType()).getElementType();
    Type type       = mergeOffsetType(chainType, offsetType, builder);
    Value lhs       = castOffset(chain->offsets, type, builder, loc);
    Value rhs       = castOffset(offsets, type, builder, loc);
    return PointerChain{base, arith::AddIOp::create(builder, loc, lhs, rhs)};
  }
  if (ptr == base) return PointerChain{base, Value{}};
  if (auto splat = ptr.getDefiningOp<SplatOp>()) {
    FailureOr<Value> offsets = buildScalarOffsets(splat.getSrc(), base, builder, loc);
    if (failed(offsets)) return failure();
    if (!*offsets) return PointerChain{base, Value{}};
    auto type = RankedTensorType::get(cast<RankedTensorType>(splat.getType()).getShape(), (*offsets).getType());
    return PointerChain{base, SplatOp::create(builder, loc, *offsets, type)};
  }
  if (auto broadcast = ptr.getDefiningOp<BroadcastOp>()) {
    FailureOr<PointerChain> chain = buildChain(broadcast.getSrc(), base, builder, loc);
    if (failed(chain)) return failure();
    if (!chain->offsets) return PointerChain{base, Value{}};

    auto shape       = cast<RankedTensorType>(broadcast.getType()).getShape();
    auto elementType = cast<RankedTensorType>(chain->offsets.getType()).getElementType();
    auto type        = RankedTensorType::get(shape, elementType);
    return PointerChain{base, BroadcastOp::create(builder, loc, type, chain->offsets)};
  }
  if (auto bitcast = ptr.getDefiningOp<BitcastOp>()) {
    if (!isPtrTypeLike(bitcast.getSrc().getType())) return failure();
    return buildChain(bitcast.getSrc(), base, builder, loc);
  }
  if (auto expand = ptr.getDefiningOp<ExpandDimsOp>()) {
    FailureOr<PointerChain> chain = buildChain(expand.getSrc(), base, builder, loc);
    if (failed(chain)) return failure();
    if (!chain->offsets) return PointerChain{base, Value{}};

    auto shape       = cast<RankedTensorType>(expand.getType()).getShape();
    auto elementType = cast<RankedTensorType>(chain->offsets.getType()).getElementType();
    auto type        = RankedTensorType::get(shape, elementType);
    return PointerChain{base, ExpandDimsOp::create(builder, loc, type, chain->offsets, expand.getAxisAttr())};
  }
  return failure();
}
```

按分支解释（**注意：这里的最终形状是"指针张量"的形状，offsets 必须
逐元素对齐**）：

1. **addptr**（张量）：递归得到内层 `(base, offsets)`，把本层 offset
   张量与之相加。内层无偏移时直接拿本层 offset 当结果（省一次加法）。
   合并前两侧都做类型统一（`mergeOffsetType` + `castOffset`），因为
   Triton 允许 `addptr` 的指针和 offset 元素类型/位宽不同（如
   `!tt.ptr<i64>` 指针加 `i32` 偏移）。
2. **ptr == base**：链底，offsets 为空。
3. **splat**（标量指针 splat 成张量）：源侧走 `buildScalarOffsets`
   （标量链），得到标量偏移后用 `tt.splat`（`triton::SplatOp`）成张量。
   偏移张量是普通整型数据，这个值 `tt.splat` 会存活到 ptr-to-memref
   （非指针 splat 在那里合法），再由 `privateuse-arith-to-linalg` 的
   SplatConverter 统一转成 `linalg.fill`——splat 降级一律推迟到该 pass。
4. **broadcast**：先递归，再把 offsets 广播到结果形状。为什么偏移也要
   广播：gather/scatter 的 indices 与 tile 同 shape，指针被广播了几份，
   偏移就要复制几份。
5. **bitcast**：只允许指针→指针透传（i1 指针 bitcast 到 i8 指针这类
   别名转换），偏移不变。
6. **expand_dims**：递归后对 offsets 做同轴 `ExpandDimsOp`，保持
   形状与指针张量一致。
7. 其他（`make_range`、`arith.addi` 直接喂指针等）：失败。因为这些
   形态不是"地址链"，而是"下标计算"——下标本应来自 `addptr` 的 offset
   操作数；如果指针直接来自算术 op，说明地址本身被当成整数计算过，
   gather 语义无法安全重建。

### 4.8 materializeBase / materializeIndices / materializeMask

```cpp
Value materializeBase(const PointerChain& chain, Type pointee, OpBuilder& builder, Location loc)
{
  auto tokenType = MemRefType::get({ShapedType::kDynamic}, pointee);
  return UnrealizedConversionCastOp::create(builder, loc, tokenType, chain.base).getResult(0);
}
```

- 同结构化 pass：把 `!tt.ptr<T>` 桥接成 `memref<?xT>` token，交
  `PtrToMemrefPass` 收尾。gather/scatter 的 source/destination 声明为
  `AnyMemRef`，rank-1 动态 memref 正好匹配"扁平基址 buffer"的语义。

```cpp
Value materializeIndices(const PointerChain& chain, RankedTensorType accessType,
                         OpBuilder& builder, Location loc)
{
  Value offsets = chain.offsets;
  if (!offsets) {
    auto type  = RankedTensorType::get(accessType.getShape(), builder.getI32Type());
    Value zero = arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(0));
    offsets    = SplatOp::create(builder, loc, zero, type);
  }
  return bufferization::ToBufferOp::create(
    builder, loc, tensorBufferType(offsets, builder), offsets, /*read_only=*/true);
}
```

- 无偏移（`tt.splat %base` / `base + 0`）时构造零索引张量：gather 的
  indices 语义是"相对 base 的元素偏移"，全 0 就是"每个 lane 都取 base
  本身"。默认 `i32` 足够表达一个小 tile 内的偏移。
- `to_buffer(offsets)`：把逐元素索引从 SSA tensor 落到 SRAM buffer，
  供 gather/scatter 按元素读取。`read_only = true` 告诉 bufferize
  这个 buffer 只读。

```cpp
Value materializeMask(Value mask, RankedTensorType accessType, OpBuilder& builder, Location loc)
{
  if (!isa<RankedTensorType>(mask.getType())) {
    auto type = RankedTensorType::get(accessType.getShape(), builder.getI1Type());
    mask      = SplatOp::create(builder, loc, mask, type);
  }
  return bufferization::ToBufferOp::create(builder, loc, tensorBufferType(mask, builder),
                                           mask, /*read_only=*/true);
}
```

- Triton 的 mask 操作数可能已经是张量，也可能是标量 i1；标量时先
  splat 成访问形状。
- 同样落成 SRAM buffer，元素类型 `i1`（gather/scatter verifier 要求
  mask 是 i1 memref）。

### 4.9 lowerLoad

```cpp
LogicalResult lowerLoad(LoadOp load, const PointerChain& chain, bool masked, Type pointee)
{
  OpBuilder builder(load);
  Location loc    = load.getLoc();
  auto tensorType = cast<RankedTensorType>(load.getType());

  Value mask, other;
  if (masked) {
    mask  = materializeMask(load.getMask(), tensorType, builder, loc);
    other = triton::maskedFillValue(load, tensorType.getElementType(), builder, loc);
    if (!other) return failure();
  }

  Value baseToken = materializeBase(chain, pointee, builder, loc);
  Value indices   = materializeIndices(chain, tensorType, builder, loc);
  Value buf       = memref::AllocOp::create(
    builder, loc, sramBufferType(tensorType, builder), builder.getI64IntegerAttr(kDefaultAlignment));
  pumem::GatherOp::create(builder, loc, baseToken, indices, buf, mask, other);

  Value tensor = bufferization::ToTensorOp::create(builder, loc, tensorType, buf,
                                                   /*restrict=*/true, /*writable=*/true);
  load.getResult().replaceAllUsesWith(tensor);
  load->erase();
  return success();
}
```

- masked 时：mask buffer + `other` 标量。`other` 的取值规则复用
  `maskedFillValue`（缺省 0、必须是 splat）——与结构化 load 的 fill
  完全一致，保证两条路径语义相同。
- `pumem.gather` 的参数顺序是 `(source, indices, destination, mask?, other?)`：
  source 为 `memref<?xT>` 基址 token，indices/destination 同 shape，
  destination 是 128 空间的 tile。
- 无 mask 时 `mask/other` 为空 Value，builder 自动构造可选段
  （`AttrSizedOperandSegments` 保证空段编码正确）。
- 最后 `to_tensor` 把 tile 变成 SSA tensor，替换 load。

### 4.10 lowerStore

```cpp
LogicalResult lowerStore(StoreOp store, const PointerChain& chain, bool masked, Type pointee)
{
  OpBuilder builder(store);
  Location loc    = store.getLoc();
  auto tensorType = cast<RankedTensorType>(store.getValue().getType());

  Value mask;
  if (masked) mask = materializeMask(store.getMask(), tensorType, builder, loc);

  Value buf = bufferization::ToBufferOp::create(
    builder, loc, sramBufferType(tensorType, builder), store.getValue(), /*read_only=*/true);
  Value baseToken = materializeBase(chain, pointee, builder, loc);
  Value indices   = materializeIndices(chain, tensorType, builder, loc);
  pumem::ScatterOp::create(builder, loc, buf, indices, baseToken, mask);
  store->erase();
  return success();
}
```

- store 的值先 `to_buffer` 成 SRAM tile（`read_only = true`），再 scatte
  到全局 buffer。与 gather 的 "alloc + gather" 相比，这里不需要额外
  alloc：`to_buffer` 在 bufferize 阶段会被直接消解成值的 materialization，
  临时内存由 bufferizer 分配。
- `pumem.scatter` 参数顺序是 `(source, indices, destination, mask?)`，
  destination 是 `memref<?xT>` 基址 token——方向与 gather 相反。
- masked 时被屏蔽的 lane 既不读 tile 也不写全局。

### 4.11 Pass 外壳

```cpp
void runOnOperation() override
{
  ModuleOp module = getOperation();
  if (module.walk([](triton::CallOp) { return WalkResult::interrupt(); }).wasInterrupted()) {
    module.emitError("privateuse-lower-unstructured-access: cannot lower a module containing tt.call");
    signalPassFailure();
    return;
  }

  SmallVector<triton::FuncOp> functions;
  module.walk([&](triton::FuncOp func) { functions.push_back(func); });

  for (triton::FuncOp func : functions) {
    if (func.isExternal()) continue;
    if (failed(UnstructuredLowering(func, module).run())) {
      signalPassFailure();
    }
  }
}
```

- **含 `tt.call` 的模块直接报错**：两个降级 pass 之后，`PtrToMemrefPass`
  会改写函数签名（指针参数 → memref）。如果有 call，被调函数的签名变化
  必须同步改调用点，否则 IR 不一致；调用点改写不在本阶段支持范围内。
  旧实现保守地整体跳过（留残留）；lower-or-error 契约下改为：只要模块含
  `tt.call`（无论是否有 `tt.load`/`tt.store`）→ `module.emitError` +
  `signalPassFailure`，不再静默放行。`walk(...).wasInterrupted()` 是
  "存在即返回"惯用法（walk 不因中断而失败）。
- 先收集函数列表再遍历：`module.walk` 期间不允许修改 IR 结构，把
  `FuncOp` 收集出来后再逐个处理。
- 每个函数新建一个 `UnstructuredLowering`（内部新建 analyzer，读取
  module 的 grid 属性），函数间状态隔离。
- **逐函数累积报告**：任一 `run()` 失败即 `signalPassFailure`，但**不 return**，
  继续处理后续函数，从而一次报出所有函数的缺口。`signalPassFailure` 只是
  标记本 pass 失败、不中断 pass 内的循环；PassManager 会在本 pass 结束后
  中止整条管线。函数内部则是首个失败即返回（见 4.2）。

---

## 5. 两个 pass 的分工矩阵

| mask kind | structured | unstructured | 说明 |
|---|---|---|---|
| `None` | 整块 copy | （不会被本 pass 处理的情况不存在；无 mask 的间接访存也可 gather） | 无 mask |
| `ProvenTrue` | 整块 copy | 视作无 mask | 常量真/可证明恒真 |
| `Box` | fill + 窗口 copy；store 需窗口单射 | 带 mask gather/scatter | 盒形前缀 |
| `ProvenFalse` | 拒绝（`isLowerable`） | 带 mask gather/scatter（全部屏蔽） | 恒假 |
| `ScalarDynamic`（标量） | 拒绝 | 不收集（只收张量） | 当前标量动态 mask 两条路径都不降级 |
| 分析失败（mask/地址） | 跳过 | 带 mask gather/scatter（安全兜底） | mask 形态无法分类 |

补充边界（当前实现的已知限制）：

- volatile load、`ignore_cta` store：structured 跳过，unstructured **硬报错**。
- 直接对 `!tt.ptr<i1>` 的访问（不经过 `tt.bitcast` 到 i8）：structured 跳过，
  unstructured **硬报错**。
- 动态形状张量：`normalize` 直接拒绝，structured 不处理；unstructured 若
  无法重建地址链则**硬报错**。
- `reshape` 出现在地址链中：structured 可能仍能分析（`buildReshape`
  支持规整 reshape），unstructured 的 findBase/buildChain 不支持，**硬报错**。
- 标量访问只由 structured 处理；unstructured 收到标量 `tt.load`/`tt.store`
  会**硬报错**（管线里 structured 先降掉标量，正常不触发）。
- descriptor 类（`tt.descriptor_load/store/reduce/gather/scatter`）与 atomic 类
  （`tt.atomic_rmw/cas/poll`）访存 op：unstructured 白名单门（`collectAccesses`）
  **硬报错**，只放行 `tt.load`/`tt.store`（见 4.2）。

## 6. 示例走查

### 6.1 纯结构化整块搬运（`@copy`）

输入（`test/AccessLowering/lower.mlir`）：

```mlir
%baseS = tt.splat %base : i32 -> tensor<128xi32>
%r     = tt.make_range {start = 0, end = 128} : tensor<128xi32>
%idx   = arith.addi %baseS, %r : tensor<128xi32>
%pS    = tt.splat %src : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
%p     = tt.addptr %pS, %idx : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
%v     = tt.load %p : tensor<128x!tt.ptr<f32>>
```

分析结果：`base = %src`，`offset = pid.x * 128`，`dims[0] = {0, 1, 128}`，
`mask = None`。降级后：

```mlir
%view = memref.reinterpret_cast %arg0 to offset: [%off], sizes: [128], strides: [1]
        : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
%buf  = memref.alloc() {alignment = 64} : memref<128xf32, 128>
memref.copy %view, %buf
%t    = bufferization.to_tensor %buf restrict writable : memref<128xf32, 128>
```

### 6.2 盒形 mask load（`@tail_load`）

输入：`mask = (i < pid + 16)`，`other = -1.0` 的 splat。

- `classifyMask`：`i < pid+16` 在 `i ∈ [0,128)` 上无法恒真/恒假
  （pid 范围 0..3）→ `Box`。
- `materializeMaskSizes`：`count = ceil((-(0 - (pid+16)))/1) = pid+16`，
  再与 128 取 min。
- 生成的 IR：`reinterpret_cast ... sizes:[%n]`（源只到 %n）、
  `linalg.fill(-1.0)` 填满 128 tile、`subview` 取 `[0..%n)`、
  `memref.copy`，最后 `to_tensor` 整个 128 tile。
- 语义：有效 lane 从全局读，其他 lane 保持 `other`。

### 6.3 二维盒形 + 读写（`@box_2d`）

`mask = (row < m) & (col*2 <= n)`：

- 两个比较各产生一条上界 bound，`andi` 合并后 `dimBounds` 有两条；
- 每维的窗口大小 = min 两条 bound 的 count 与维度大小；
- store 需要窗口单射：窗口大小在运行时才知道，`isMaskWindowInjective`
  在编译期无法证明（offset 依赖 pid）——但整块 view（8x8，stride 8）
  本身单射，因此直接按整块 view 检查通过；
- 降级后 load 是 "fill + subview + copy"，store 是
  "to_buffer + subview + copy 到动态 view"。

### 6.4 间接访存（`@transform`）

输入：`%m = tt.load %mp` 得到下标张量，再用 `%m` 作为第二个 load 的
offset。分析第二个 load 时，`buildValue` 发现 offset 来自 load →
`DataDependentOffset`，structured 拒绝。unstructured：

- `findBase` 找到 `%src`；
- `buildChain`：`tt.addptr %pS, %m` → chain = `{src, %m}`；
- `lowerLoad`：indices buffer 就是 `%m` 的 `to_buffer`，
  `pumem.gather %src_token, %m_buf, %tile`；
- store 对称成 `pumem.scatter`。

### 6.5 64 位偏移（`@wide_offset`）

map 里存的是 i64：`%m: tensor<4xi64>`。`buildChain` 的
`mergeOffsetType` 看到 i64 就生成 `arith.addi` 的 i64 版本，
indices buffer 是 `memref<4xi64, 128>`，gather 的 indices 元素类型随之
是 i64。旧版把偏移强转 i32 在这里会截断地址，测试 `@wide_offset` 就是
回归护栏。

## 7. 术语表

| 术语 | 含义 |
|---|---|
| 规范形 / canonical view | `base + offset + Σ(start + stride*i)` 的地址表示 |
| 单射 / injective | 逻辑下标到线性偏移一一对应，写不重叠 |
| 盒形窗口 / box window | mask 可表示为逐维线性上界交集时，实际有效的"前缀方块" |
| 证明恒真/恒假 | 区间推理在整段 lane 范围内得出 holds / fails |
| 存储元素类型 | `storageElementType`：`i1` 记为 `i8`，其余原样 |
| token | 用 `unrealized_conversion_cast` 把 `!tt.ptr<T>` 临时伪装成的 `memref<?xT>` |
| 软跳过 | pass 不改写该 op、返回 success，把它留给后续阶段（仅 structured） |
| 硬失败 | pass 报错并 `signalPassFailure` |
| lower-or-error | unstructured 契约：每个 `tt.load`/`tt.store` 要么降成 gather/scatter，要么硬报错，成功结束后无残留 |
| 白名单门 | unstructured 的 `collectAccesses`：访存 op 只放行 `tt.load`/`tt.store`，descriptor/atomic 一律硬报错 |
| SRAM / scratchpad | memory space 128 的片上缓冲，tile 的落脚点 |
