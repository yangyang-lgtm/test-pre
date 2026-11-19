# pustream 实现方案(修订版)

> 状态:设计修订版,已按评审意见对齐;尚未动代码。
> 目标不变:把 bufferize / 分配 / 调度彻底解耦,让同一份异步中段 IR 既能
> lower 到 CPU 模拟后端(async→sync),又能 lower 到类 NPU 设备
> (async→DMA+semaphore)。
> 修订摘要:砍掉 IREE 的 resource 对象模型,中段 IR 直接使用 memref;新增
> 只有 `!pustream.timepoint` 与 5 个 op;依赖采用 memory-hazard 语义;
> 分配全静态;目标设备契约前置于 M0。
>
> **v2 更新(2026-11-19,见 §13)**:后端最终形态确定为"发射器 + 三个执行体"
> (load 队列 / store 队列 / compute 池)与 4 个 runtime 原语。据此把 pustream
> 重新抽象为**语句式发射序**:删除 `!pustream.timepoint`、`await`、`join`、
> `immediate` 与 `awaits`,只保留 `async.copy` / `async.dispatch` / `wait` /
> `wait_dispatch` 四个语句 op;等待由插入 pass 从 effect 状态重新推导。
> §4、§4.3、§4.5、§5-I1、§6 的 `max_stage_by_sem`、§7.2 的 P5/P6、§9 中与
> token 模型绑定的内容**由 §13 取代**;分层、分配、设备契约、发射章节仍然有效。

---

## 0. 与原方案的差异

| 原方案 | 修订 | 原因 |
|---|---|---|
| 无类型 `resource` 类型 | memref + memory space(DRAM=0 / SRAM=128) | 全静态、单设备下类型擦除只会把 layout 赶出 IR;layout 必须跨 executable 边界 |
| alloc/dealloc/subview/pack/lifetime 枚举 | 只用 `memref.alloc` / `memref.view`;单 slab + 常量 offset;无 dealloc | 全特化后端,单 arena 生命周期 = kernel;打包是 IR 重写,不是新抽象 |
| `async.transfer` 只带 src/dst/tp | copy 携带 box 几何(sizes / strides / elem_bytes) | 真实访问是 strided copy,几何必须随 op |
| transfer 链式等待前一个 transfer | `awaits` = 启动前提;transfer 之间不互等 | 链式等"完成"会废掉 double-buffer |
| timepoint 单态、无硬件映射 | timepoint→semaphore 池分配 + 活跃数上限 verifier | NPU 的 semaphore/event 数量有限 |
| Phase 3 在 outline 之前转换 dispatch | form-executables 提前,matmul 进 executable | 原顺序没有 dispatch 目标;matmul 会停留在 host 侧 |
| matmul 直接复用 runtime_matmul | P3 钉死**覆盖语义**并归一化 init | runtime_matmul 是覆盖,linalg.matmul 是累加,两分支会算出不同结果 |
| Phase 7 refcount / dealloc | 删除 | 静态分配不需要运行时释放 |
| ScheduleConcurrency / async.execute | 推迟 / 删除 | 当前没有需要并行的独立 async 链 |

## 1. 目标与范围

### 1.1 要解决的问题
- `_fuse_and_bufferize` 的 in-place 复用决策埋在 one-shot-bufferize 内部,
  后续无法全局优化。
- `dynamic_loop_kernel` post-bufferize 出现 reset copy(copy-before-write),
  无法被现有 `hoist_invariant_staging` 处理——该 pass 在 `_specialize` 里运行,
  严格早于 bufferize,看不到这些 copy。
- `_software_pipeline` 是空壳,`num_stages=2` 从未生效。
- matmul/transpose 在 outline 之前被换成 host 侧 `runtime_matmul`,导致最重要的
  compute 进不了 executable,NPU 路径上无解。
- 没有目标设备模型,调度器与打包器没有约束输入。

### 1.2 设计原则
1. **调度先于分配**:buffer 的活跃区间由 timepoint 依赖决定,打包必须在调度之后。
2. **bufferize 保守**:每个 tensor 独立 alloc,不做 in-place 复用;复用 / multi-buffer
   交给 `pustream` 的全局优化器。
3. **不擦除类型**:中段 IR 保持 memref;typed 访问不需要额外的 view op,设备 ABI
   只在最后 materialize 一次。
4. **依赖是 memory hazard,不是队列序**:无 hazard 即无依赖;RAW/WAR/WAW 才接
   timepoint。单队列的入队序只是满足序的一种实现,多队列下不成立。
5. **全静态**:所有 size / offset 在编译期折叠;单 arena,无运行时分配/释放。
6. **不变量可验证**:支配、拷贝几何、打包、容量、信号量数量都有 verifier,
   不依赖数值测试兜底。
7. **CPU / NPU 共用中段 IR**,末段在两个点分叉:设备接口与 executable 体。

### 1.3 非目标(本期不做)
- 不重写 `puarith` / `pumath` 的标量语义(它们活在 executable 内部)。
- 不实现 NPU 真实驱动(本期只定义 lowering 形状 + runtime 符号占位)。
- 不做跨 kernel 常量池 / 权重打包。
- gather/scatter 异步化(见 §12;先声明为不同步异步化)。
- 动态 shape、多设备、多 stream(这些是引入 resource 类型的前提,当前不存在)。

### 1.4 约束
- 所有现有方言(`pumem` / `puarith` / `pumath` / `puc`)均可修改。
- 遵循 `AGENTS.md`:native/compiler 改动测试前先 `make`;C++ `assert` 内不放
  副作用代码;`class` 成员尾下划线 `foo_`,`struct` 成员 plain。

## 2. 现状 lowering 链(真实路径)

```
ttir
 └─ _specialize            backend/compiler.py:175
     specialize_kernel / lower_structured_access / lower_unstructured_access
     ptr_to_memref / arith_to_linalg / pid_loop / licm
     hoist_invariant_staging / canonicalize / cse
 └─ _fuse_and_bufferize    backend/compiler.py:194   ← P1 替换
     collapse_shape / empty_tensor_to_alloc_tensor / bufferize / canonicalize
 └─ _software_pipeline     backend/compiler.py:205   ← P5 填上
 └─ _transform             backend/compiler.py:209   ← P3 拆分重排
     linalg_to_runtime_call / outline_linalg_ops / lower_linalg_reduce
     convert_linalg_to_loops / scf_to_pumem / arith_to_puarith / math_to_pumath
 └─ _lower_to_pumem        backend/compiler.py:224   ← P8 扩展
     expand_strided_metadata / memref_expand / lower_affine
     memref_to_pumem / sink_constant_call_args / arith_to_puarith
 └─ _emit_cpp              backend/compiler.py:238   ← P9 保留
     reconcile_unrealized_casts / to_puc / puc_localize_carried
     puc_simplify_assign / puc_void_ptr_args / puc_launch / translate_puc_to_cpp
```

关键既有资产(用途见各 Phase):
- `BufferizePass`:one-shot,SRAM space=128,align=64,`allowReturnAllocsFromLoops=true`。
- `OutlineLinalgOps`:把 pure-buffer linalg outline 成 private `func.func`,打
  `pumem.outlined` attr。注意它不处理已被换成 `pumem.runtime_call` 的 matmul。
- `MemrefToPumem`:memref→(base ptr, offset, sizes..., strides...) tuple。
- `cpu_backend_runtime.h`:`pumem_sync()` / `pumem_wait()` 是 **C runtime 函数**
  (no-op),不是 MLIR op;`copy_*` 同步执行;`alloc/dealloc/reset_alloc`
  (reset_alloc 为空,单 arena 的雏形)。
- 方言:`pumem`(ptr ABI + copy + runtime_call + for/if)、`puc`(C emit),
  `puarith`/`pumath`(标量)。

注册约定(新代码必须遵守):
- Pass:`include/privateuse/{Conversion,Transform}/<Name>/Passes.td` + `Passes.h`,
  `lib/.../<Name>Pass.cpp`,pass 名 `privateuse-<kebab>`,构造器
  `mlir::triton::create<Name>Pass()`。
- Python binding:`privateuse.cc` 里 `m.def("add_<snake>", ...)`。
- 方言注册:`privateuse.cc:70 load_dialects` 的 `registry.insert<...>`。
- 方言:`include/privateuse/Dialect/<Name>/IR/<Name>Dialect.td`,
  `lib/Dialect/<Name>/IR/<Name>Dialect.cpp`,namespace `::mlir::<name>`。

## 3. 目标架构分层

| IREE | privateuse | 职责 |
|---|---|---|
| Flow(tensor dispatch) | `tensor`/`linalg`(arith_to_linalg 后) | 计算表达,DPS 别名 |
| **Stream**(resource+timepoint) | **`pustream`(新)** | 异步依赖、调度、分配打包 |
| HAL executable | outlined `func.func` | 不透明 compute 单元 |
| HAL buffer / command | `pumem`(ptr+copy+runtime_call) | 设备指针 ABI、DMA/semaphore |
| HAL C emit | `puc` → C++ | 发射 |

内存模型:
- space 0 = DRAM:kernel 入参与其视图,不参与打包。
- space 128 = SRAM:所有 `memref.alloc` 出来的 scratch,参与打包。
- "lifetime"不再是 IR 概念:external / staging / transient 的区分由
  space + SSA 活跃区间 + 打包器自然处理,无信息损失。

## 4. 中段 IR:`pustream` 方言

目录:`include/privateuse/Dialect/PuStream/IR/PuStreamDialect.td`,
`lib/Dialect/PuStream/IR/PuStreamDialect.cpp`,namespace `::mlir::pustream`。

### 4.1 类型

```tablegen
def PuStream_TimepointType : TypeDef<PuStream_Dialect, "Timepoint"> {
  // 异步完成令牌,无参数。CPU lowering 全部擦除。
}
```

只有这一个新类型。没有 resource、没有 lifetime 枚举。

### 4.2 Ops(全集,共 5 个)

| op | 操作数 → 结果 | 语义 |
|---|---|---|
| `pustream.async.copy` | `(src: memref, dst: memref, box{sizes, dst_strides, src_strides, elem_bytes}, awaits...) -> tp` | 异步 block 拷贝;仅 DRAM↔SRAM(动态 offset 由 src/dst 的 subview 携带) |
| `pustream.async.dispatch` | `@exe(memrefs..., awaits...) {reads=[...], writes=[...]} -> tp` | 异步 compute;callee 为 outlined func |
| `pustream.await` | `(tp)` | 消费点等待完成 |
| `pustream.timepoint.join` | `(tp...) -> tp` | 多输入汇聚 |
| `pustream.timepoint.immediate` | `() -> tp` | 已完成 token;CPU 起点 / 无依赖起点 |

`async.copy` td 草图:

```tablegen
def PuStream_AsyncCopyOp : PuStream_Op<"async.copy", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let arguments = (ins
    AnyMemRef:$source,
    AnyMemRef:$target,
    DenseI64ArrayAttr:$sizes,        // 全静态
    DenseI64ArrayAttr:$dstStrides,   // 元素 stride
    DenseI64ArrayAttr:$srcStrides,
    I64Attr:$elemBytes,
    Variadic<PuStream_Timepoint>:$awaits);
  let results = (outs PuStream_Timepoint:$done);
  let hasVerifier = 1;   // 几何与 src/dst 类型一致;跨 space 只允许 dram<->sram
}
```

box 的 sizes/strides 是静态 attr;与 k 相关的动态偏移由 `memref.subview` 携带,
I2 校验的是 subview 之后的类型,而不是 base memref。

`async.dispatch` td 草图:

```tablegen
def PuStream_AsyncDispatchOp : PuStream_Op<"async.dispatch", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyMemRef>:$operands,
    Variadic<PuStream_Timepoint>:$awaits,
    DenseI64ArrayAttr:$reads,       // operands 下标
    DenseI64ArrayAttr:$writes);     // operands 下标
  let results = (outs PuStream_Timepoint:$done);
  let hasVerifier = 1;   // reads/writes 与 callee 实际效应一致(formation 时冻结)
}
```

### 4.3 语义规则

- **awaits 是启动前提,不是完成链**:op 可以开始的前提是所列 timepoint 完成。
  - transfer 之间**不互相等待**:DMA 队列(单队列)自然保序;多队列下由 hazard
    规则决定是否需要依赖(§5 I1)。
  - 只有 WAR(覆盖正在被 compute 读的 slot)或 WAW 才把 dispatch 的 tp 接到
    transfer 的 awaits 上。
- `await` 只插在真实消费点(compute 读之前);不允许"发出即等"。
- `join` 用于多生产者汇聚;`immediate` 表示"无依赖"。
- `async.execute` 不实现:outlined executable 本身就是 region,dispatch 已覆盖。
- **SRAM→SRAM copy 不允许跨过 P4**:P2 能证明安全的改 in-place;不能证明的改写成
  `linalg.generic`(box/置换 copy),由 P3 outline 进 executable;P4 之后 verifier
  拒绝残留的 SRAM→SRAM `memref.copy`。不设"同步屏障"这种模糊态。
- CPU lowering:copy→同步 copy,await→no-op,timepoint 全部擦除。
- NPU lowering:copy→DMA 描述符 + signal,await→semaphore wait,
  dispatch→命令入队 + event。

### 4.4 executable 的 ABI 与效应

- executable = outlined `func.func`,参数是 memref(保持 scf/linalg 原样)。
- effects 在 form-executables 时从 callee 体计算并**冻结**在 dispatch 的
  reads/writes 上;之后 executable 视为不透明,分析只依赖该声明。
- 设备 ABI 只在 P8 一处 materialize:memref → (base ptr, offset, sizes, strides)。
- 内置 runtime 函数(如 `runtime_matmul`)按"外部 executable"处理:
  签名与读写效应在 formation 时人工声明,CPU 分支直接调用,NPU 分支替换为命令。

**matmul 语义钉死为覆盖**:`runtime_matmul` 是 `C = A*B`(局部累加器从 0 起,
`cpu_backend_runtime.h:203-205`),而 `linalg.matmul` 是 `C = A*B + C`。P3 必须
归一化 matmul 的 init:
- init 是死值(空 tensor / 新 alloc)→ 保持,dispatch 声明 `writes = [C]`;
- 源程序确实需要初始化 → 显式插 `linalg.fill`,其效应计入 `reads`/`writes`;
- NPU 分支按覆盖语义实现(清零后累加,或覆盖指令)。

I3 以声明为准;不允许两个分支数值语义不同。

### 4.5 PUMem 扩展(设备接口侧新增)

现有 pumem 没有异步/事件能力(`PUMemDialect.td` 只有 copy/for/if/runtime_call 等),
需要新增(形状与 `pumem.copy` 对齐,`PUMemDialect.td:213-219`):

| op / 类型 | 签名 | 用途 |
|---|---|---|
| `!pumem.semaphore` | — | 信号量/事件句柄 |
| `pumem.async_copy` | ptr + sizes/dst_strides/src_strides/byte_size/dir → `!pumem.semaphore` | DMA 拷贝 |
| `pumem.dispatch` | `@callee(operands...) -> !pumem.semaphore` | 命令入队 + 完成事件 |
| `pumem.wait` | `(sem)` | 等待信号量 |
| `pumem.signal` | `(sem)` | 显式置位(设备需要时) |

- CPU 分支**不需要**新增 op:pustream 在 memref 阶段就被擦除——`async.copy` 降为
  `memref.copy`(复用现有 CopyOpLowering),`async.dispatch` 降为 `func.call`,
  timepoint/semaphore 全部丢弃。
- NPU 分支:先做 memref→ptr 的 ABI 展开,再生成 `pumem.async_copy` /
  `pumem.dispatch` / `pumem.wait`;两者一一对应,几何复用 `pumem.copy` 的字段。
- §2 提到的 `pumem_sync()` / `pumem_wait()` 只是 C runtime 函数,不是 MLIR op。

## 5. 不变量与验证

全部做成真实 verifier(或 debug pass),在 lit 与 pytest 中常开。

**I1 — hazard 支配(dispatch 粒度、iteration-aware)**:
- 检查粒度是 op 的操作数:dispatch 按其 `reads`/`writes` 声明,copy 按 src/dst;
  executable 内部不透明,不进入分析。
- RAW:任何 memref 的读,其 `awaits` 传递闭包必须包含最后一次写它的 op 的 tp;
- WAR:任何写(dispatch 写或 copy 目标)必须包含最后一次读它的 dispatch 的 tp;
- WAW:写必须包含最后一次写;
- **init**:循环携带 buffer 的第 0 轮消费者必须 await 其 pre-loop 生产者的 tp
  (初始化 copy/fill);没有生产者则用 `immediate`。
- **drain**:把最终结果写回 DRAM 的 copy 必须 await 最后一轮 dispatch 的 tp
  (该 tp 必须是 `scf.for` 的结果);kernel 退出前 join+await 所有未消费的在飞 tp
  ——arena 生命周期 = kernel,退出时还有 DMA 在写 slab 就是 use-after-free。
- 传递闭包经 `join`;循环内按 **slot 分别跟踪 tp**(经 `scf.if` 的每个 case),
  不做泛型闭包。M0 先覆盖 2-slot 轮转,M3 扩展。

**I2 — 拷贝几何**:sizes/strides/elem_bytes 与 src/dst 的 memref 类型一致;
跨 space 只允许 DRAM↔SRAM。

**I3 — dispatch 效应**:reads/writes 与 callee 实际读写一致;executable 不透明后
以声明为准。

**I4 — 打包**:同时活跃的 buffer 字节区间不重叠;每个起始 offset 满足 DMA 对齐;
slab 总字节 ≤ 目标 SRAM 容量。

**I5 — 信号量**:P8 分配后,活跃 timepoint 数 ≤ 目标设备 semaphore/event 上限。

辅助手段(可选 debug):模拟 runtime 记录 enqueue 顺序,`await` 时检查被等事件
已经入队,把顺序错误在 CPU 上暴露出来,而不是只靠数值回归。

## 6. 目标设备契约(M0 前置输入文档)

调度器与打包器的约束来源,必须在动代码前定稿:

- DMA:队列数量;描述符支持的维度 / stride 形态;对齐要求;单次传输上限。
  → 决定 copy 是否需要拆分、多队列下哪些 transfer 可并发。
- semaphore/event:类型与数量上限。→ I5 与 `num_stages` 上限。
- SRAM:容量、bank、最小对齐。→ 打包容量断言与 `num_stages` 推导。
- dispatch:是否可并发、命令队列深度。→ 是否需要依赖调度之外的并发调度。
- 内存注册:是否有 (I)IOMMU 类 per-buffer 注册。若有,在 slab 子区间上注册,
  不破坏单 slab 模型。

`num_stages` 的推导:

```
max_stage_by_sram  = (SRAM 容量 - 非 staged scratch) / 每 stage 的 staging 字节
max_stage_by_sem   = semaphore 上限 / 每 stage 活跃 timepoint 数
num_stages = min(option 上限, max_stage_by_sram, max_stage_by_sem)
```

超限时编译期报错并给出建议(缩 tile 或降 stage),不允许静默溢出。

## 7. 修订后的 Pass pipeline

### 7.1 总览

```
P0 specialize            不变
P1 bufferize             copyBeforeWrite,每 tensor 独立 alloc
P2 cleanup-copies        新增:post-bufferize 的 staging 提升 / 死 copy / copy→fill
P3 form-executables      提前:outline + 效应标注(含 matmul / transpose)
P4 to-stream             compute 单元→async.dispatch;copy→async.copy;起点 immediate
P5 schedule-pipeline     k 循环 multi-buffer,阶段数由 SRAM 预算推导
P6 simplify-timepoints   join/await 化简 + I1 验证
P7 plan-allocation       async-aware liveness → 常量 offset 单 slab + I4 断言
P8 lower(两个分叉)       设备接口:CPU async→sync / NPU DMA+semaphore
                         executable 体:CPU 标量化/runtime_matmul / NPU 保持 linalg
P9 emit                  指针元组 ABI materialize 一次 → puc → C++
```

### 7.2 各阶段职责

**P0 — Specialize(不变)**
保持现有 `_specialize` 内容;`pid_loop` 仍在最外层。

**P1 — Conservative Bufferize(替换)**
- `OneShotBufferizationOptions`:设 `copyBeforeWrite = true`(跳过 in-place 分析,
  每次写分配新 buffer)。
- 保留 SRAM space=128、align=64、`allowReturnAllocsFromLoops=true`。
- binding 不变(`add_bufferize`)。
- 预期:copy 数先上升,由 P2 回收;**不要**在这里做任何复用。

**P2 — Cleanup Copies(新)**
`Transform/CleanupCopies/`,pass `privateuse-cleanup-copies`,
`createCleanupCopiesPass()`,binding `add_cleanup_copies`。
- 循环不变 staging 提升:修好现有 hoist 的"DPS init 拒绝"逻辑(或在这里重写),
  允许把 alloc+copy+bridge 移出循环。
- 死 copy 消除:copy 的源值除本次 copy 外无其他读者,且 copy 目标被完整覆盖
  → 删除 copy(必要时改为 fill)。
- copy→in-place:elementwise 更新(读源、写同形 buffer、无跨元素依赖)可证明安全
  → 把 `ins(%acc) outs(%n)` 改写为 `ins(%acc) outs(%acc)`,删掉 reset copy。
- SRAM→SRAM copy:能证明安全的按上一条改 in-place;其余改写为 `linalg.generic`
  (box/置换 copy),交给 P3 outline 进 executable;P4 verifier 拒绝残留(§4.3)。
- 本 pass 只做**可证明安全**的复用;跨 buffer 的全局复用全部留给 P7。

**P3 — Form Executables(提前,原 Phase 8 拆分)**
`Transform/FormExecutables/`(或在现有 OutlineLinalgOps 上加效应标注),
pass `privateuse-form-executables`。
- 统一 outline 所有 compute linalg(matmul / generic / reduce / transpose),
  **matmul 必须进 executable**;`outline_linalg_ops` 只认 pure-buffer linalg、
  跳过嵌套 linalg,可直接复用,只是挪位置。
- 计算并冻结每个 executable 的读/写效应,写回后续 dispatch。
- matmul 按 §4.4 归一化为**覆盖语义**(必要时显式插 `linalg.fill`)。
- `linalg_to_runtime_call` 是 module 级 partial conversion(会进入嵌套 func),
  从顶层移除,挪到 **P8b 在 outline 之后**跑,只在 CPU 分支生效。
- 默认**不融合** matmul+elementwise(M1 先保正确);融合策略不在本期。

**P4 — Convert To PuStream(新,原 Phase 3)**
`Conversion/ToPuStream/`,pass `privateuse-to-pustream`。
- `func.call @outlined`(或外部 executable)→ `pustream.async.dispatch`;
- `memref.copy`(跨 space)→ `pustream.async.copy`,几何从 memref 类型取常量;
- SRAM→SRAM copy 不留在 stream:kernel 内同步 copy 在设备上同样要发命令且没有
  hazard 跟踪;能删的 P2 删,其余 P3 变成 `linalg.generic` 进 executable,
  P4 verifier 拒绝残留(§4.3);
- 起点统一为 `timepoint.immediate`;
- kernel 入参 memref 保持 space 0,不做任何转换。
- pass 自带 verifier:拒绝残留的 SRAM→SRAM `memref.copy`(§4.3)。

**P5 — Schedule + Pipeline(新,原 Phase 4)**
`Transform/SchedulePipeline/`,pass `privateuse-schedule-pipeline`,
binding `add_schedule_pipeline(num_stages)`。
- 只做 matmul/reduce 的 k 循环(元素级算子无 tile 复用,不需要 multi-buffer)。
- 优先复用 MLIR 上游 `scf::pipelineForLoop`:用自定义 `PipeliningOption` 的
  `getScheduleFn` 把 `async.copy` 排早 stage、`async.dispatch` 排在 compute stage;
  验证上游能否正确建模 **timepoint iter_args 与 slot 轮转**。
  若不能,退化为薄的 prologue/epilogue peeler(自研范围限制在 k 循环模式)。
- 调度方案(以 num_stages=2 为例):prologue 发 `num_stages-1` 轮 transfer;
  循环第 k 轮:await 当前 slot 的 tp → dispatch(awaits 当前 tp;若写 acc,还要
  awaits 上一轮 dispatch 的 tp,处理 WAW)→ 预取 `k+num_stages-1` 到**另一个 slot**,
  其 WAR 依赖是**上一轮 dispatch 的 tp**(该 slot 的最后一个读者)→ yield 新的
  slot tp 与本轮 dispatch 的 tp。N>2 时 iter_args 携带每 slot 的 tp。
- 越界预取用 `scf.if` 屏蔽(`k+num_stages-1 < K`);epilogue 不再预取。
- slot 选择用 **`scf.if`**:`ScfToPumem` 把整个 scf 方言标 illegal 且只有
  For/If/Yield pattern,`scf.index_switch` 会让转换直接失败;`arith.select`
  虽在 op 层合法,但 `ArithToPuarith` 是 OneToOne 转换,memref 的 one-to-N
  元组塞不进去。>2 slot 用嵌套 `scf.if`,或给 ScfToPumem 补 IndexSwitch pattern。
- `num_stages` 按 §6 公式推导,option 只作上限。

**P6 — Simplify Timepoints(新,原 Phase 5)**
`Transform/SimplifyTimepoints/`,pass `privateuse-simplify-timepoints`,
binding `add_simplify_timepoints`。
- join/await 等价化简:`await join(a,b)` → `await a; await b`;
  删除已完成 token 的等待;合并重复 await。
- 运行 I1 支配 verifier(含循环轮转情形),失败即报错。

**P7 — Plan Allocation(新,原 Phase 6)**
`Transform/PlanAllocation/`,pass `privateuse-plan-allocation`,
binding `add_plan_allocation`。
- liveness:buffer 从第一次写到最后一次 hazard 消费者 **await 完成**;
  循环内分配按跨迭代活跃处理。
- 线性扫描打包:每个 space 一个 slab;同时活跃区间不重叠;每个 buffer 起始
  offset 按 DMA 对齐;offset 全部折叠为常量。
- 重写:删掉原 `memref.alloc`,改为每个 space 一个
  `memref.alloc : memref<SLAB_BYTESxi8, space>` + 每个逻辑 buffer 一个
  `memref.view %slab[%c_offset][] : memref<...xi8> -> memref<..., space>`。
- **必须新增 ViewOpLowering**:`memref.reinterpret_cast` 不允许改变元素类型
  (`ReinterpretCastOp::verify`,`MemRefOps.cpp:2092-2094`),不能做 i8→f32;
  `expand-strided-metadata` 也不覆盖 `memref.view`(`ExpandStridedMetadata.cpp:899`
  只认 subview/reinterpret_cast)。
- `ViewOp::verify`(`MemRefOps.cpp:3893-3915`)允许 base/result 元素类型不同,但要求
  **两者都是 identity layout、同一 memory space**——正合单 slab + 常量 offset 模型;
  约束:打包出的逻辑 buffer 必须是**连续(identity)子区间**,不能带 strided 子布局。
- lowering 形状:`point_cast(addptr(slab_i8_base, byte_offset))` 作 base,
  offset=0,sizes/strides 取结果类型常量。
- 断言 I4:slab 总量 ≤ SRAM 容量;失败给出缩 tile / 降 stage 建议。
- 无 `pustream.dealloc`;arena 生命周期 = kernel。

**P8 — Lower(保留 + 扩展,两个分叉)**
CPU 分支顺序:**先 8b**(executable 内 compute lowering),**再 8a**(设备接口 +
ABI 展开,module 级);NPU 分支只跑 8a。

8a 设备接口 `Conversion/PuStreamToPumem/`:
- CPU 删除 `pustream.await` / `join` / `immediate`;NPU 映射为 `pumem.wait`
  (其中 `join` 展开为对每个输入分别 wait);
- `async.copy`:CPU → 同步 `pumem.copy`;NPU → `pumem.async_copy`(返回
  semaphore;硬件需要显式置位时再发 `pumem.signal`);
- `async.dispatch`:CPU → `func.call` / `pumem.runtime_call`;NPU →
  `pumem.dispatch`(命令入队 + 完成事件);
- NPU 分支配 semaphore 池:按活跃区间分配,验证 I5;
- executable 签名 → (base ptr, offset, sizes, strides) tuple(复用现有
  expand_strided_metadata / memref_expand / lower_affine / memref_to_pumem 链路)。

8b executable 体:
- CPU:`linalg_to_runtime_call`(matmul/transpose → `runtime_matmul`)/
  `lower_linalg_reduce` / `convert_linalg_to_loops` / `scf_to_pumem` /
  `arith_to_puarith` / `math_to_pumath`(现有 pass 从顶层移入 executable 作用域);
- NPU:保持 linalg(或降为目标 compute IR),交给 NPU 编译器;本期只保证
  IR 形状与符号占位。

**P9 — Emission(保留 + 扩展)**
现有 `to_puc` / `localize_carried` / `simplify_assign` / `void_ptr_args` /
`puc_launch` / `translate_puc_to_cpp` 保留;translate 按 `runtime_backend`
选择同步/异步 runtime 符号;semaphore/timepoint 句柄用 `puc.local` / `puc.assign`。

## 8. compiler.py 改动

```python
def __call__(self, kernel_ir, grid, kernel_name, args):
    constants = self._kernel_constants(kernel_ir, kernel_name, args)
    module = self._parse_ir(kernel_ir)
    module = self._specialize(module, grid, constants)       # P0
    module = self._bufferize(module)                         # P1 copyBeforeWrite
    module = self._cleanup_copies(module)                    # P2 新
    module = self._form_executables(module)                  # P3 新(提前)
    module = self._to_stream(module)                         # P4 新
    module = self._schedule_pipeline(module)                 # P5 新(num_stages)
    module = self._simplify_timepoints(module)               # P6 新 + I1
    module = self._plan_allocation(module)                   # P7 新 + I4
    module = self._lower(module)                             # P8 分叉
    cpp = self._emit_cpp(module)                             # P9
    _check_entry_symbol(cpp, kernel_name)
    return self._compile_object(cpp)
```

参数透传与 cache key(精确到改动点):
- `PrivateUseOptions` 增加 `runtime_backend`(默认 "cpu");`num_stages` 保留为上限。
- `DeviceInfo.from_options(target)` **只接收 target**,需要改签名接收 options
  (或在 `add_stages` 里把两个值塞进 fields);`DeviceInfo.pipeline` 不能再返回
  默认 `LoweringPipeline()`,要用 fields 构造。
- `LoweringPipeline` 已有 `runtime_backend` 类注解(digest 用 `__annotations__`),
  **需补 `num_stages` 注解**。
- **driver.py:334 的 `LoweringPipeline.runtime_backend` 读的是类属性**,用于决定
  launcher 是否做 CPU tensor 拷贝;改成 per-kernel option 后必须改读
  metadata/device_info,否则 NPU 模式仍走 CPU 拷贝。
- 只改 `digest()` 而不改 fields 不够:值不生效,缓存也不会失效。

## 9. Worked example:k 循环 double-buffer

以 matmul 累加循环为例(示意,SSA 从简)。

**P1 后(保守 memref):**

```mlir
scf.for %k = %c0 to %K step %c1 iter_args(%acc = %acc_init) -> memref<MxNxf32, 128> {
  %a = memref.alloc() : memref<MxKxf32, 128>
  %b = memref.alloc() : memref<KxNxf32, 128>
  memref.copy %dram_a[%k], %a             // DRAM→SRAM staging
  memref.copy %dram_b[%k], %b
  %p = memref.alloc() : memref<MxNxf32, 128>
  linalg.matmul ins(%a, %b) outs(%p)
  %n = memref.alloc() : memref<MxNxf32, 128>
  memref.copy %acc, %n                    // reset copy(copyBeforeWrite 产物)
  linalg.generic addf ins(%acc, %p) outs(%n)
  scf.yield %n
}
```

**P2 后(reset copy 消除,%n 与 %acc 合并且 in-place):**

```mlir
scf.for %k = ... iter_args(%acc = %acc_init) -> memref<MxNxf32, 128> {
  %a, %b : staging copies(留下,待 P5 multi-buffer)
  %p = memref.alloc() : memref<MxNxf32, 128>
  linalg.matmul ins(%a, %b) outs(%p)
  linalg.generic addf ins(%acc, %p) outs(%acc)   // 无 reset copy
  scf.yield %acc
}
```

**P4 后(async IR,未流水线化;依赖按 hazard 建立):**

```mlir
%imm = pustream.timepoint.immediate : !pustream.timepoint
// %acc_init 若由 fill/copy 生产,记其 tp 为 %t_init;否则 %t_init = %imm
%acc_f, %td_f = scf.for %k = %c0 to %K step %c1
    iter_args(%acc = %acc_init, %td = %t_init)
      -> (memref<MxNxf32, 128>, !pustream.timepoint) {
  // 覆盖 a/b 前必须等上一轮 dispatch 读完(WAR)。这一等也是未流水线时
  // load 与 compute 无法重叠的原因;P5 用双 buffer 把它甩到 k-1 轮。
  %ta = pustream.async.copy %dram_a[%k], %a {box = ...} awaits(%td)
  %tb = pustream.async.copy %dram_b[%k], %b {box = ...} awaits(%td)
  pustream.await %ta
  pustream.await %tb
  %td_new = pustream.async.dispatch @k_step(%a, %b, %acc)
              {reads = [0, 1, 2], writes = [2]}
              awaits(%ta, %tb, %td)     // %td:%acc 的 WAW(上一轮 dispatch)
  scf.yield %acc, %td_new
}
// drain:最终写回 DRAM 必须等最后一轮 dispatch
pustream.await %td_f
pustream.async.copy %acc_f, %dram_out {box = ...} awaits(%td_f)
```

第一轮 `%t_init` 若是 `immediate`,`awaits(%td)` 退化为空等待;若 `%acc_init`
有生产者,则第一轮 copy/dispatch 都会等它(I1 的 init 规则)。

**P5 后(double-buffer,num_stages=2,方案 B:prologue 1 轮 / 预取 k+1):**

```mlir
%imm = pustream.timepoint.immediate : !pustream.timepoint
// 若 %acc_init 有生产者则 %t_init = 其 tp,否则 %t_init = %imm
%ta0 = pustream.async.copy %dram_a[%c0], %a0 {box = ...} awaits(%imm)
%tb0 = pustream.async.copy %dram_b[%c0], %b0 {box = ...} awaits(%imm)
%acc_f, %ta_f, %tb_f, %td_f = scf.for %k = %c0 to %K step %c1
    iter_args(%acc = %acc_init, %ta = %ta0, %tb = %tb0, %td = %t_init) {
  pustream.await %ta
  pustream.await %tb
  %td_new = pustream.async.dispatch @k_step(%a_cur, %b_cur, %acc)
              {reads = [0, 1, 2], writes = [2]} awaits(%ta, %tb, %td)
  // 预取 k+1 到另一个 slot;该 slot 的最后一个读者是上一轮 dispatch(%td)
  %k1 = arith.addi %k, %c1
  %in_range = arith.cmpi slt, %k1, %K
  %ta_next, %tb_next = scf.if %in_range
      -> (!pustream.timepoint, !pustream.timepoint) {
    %na = pustream.async.copy %dram_a[%k1], %a_other {box = ...} awaits(%td)
    %nb = pustream.async.copy %dram_b[%k1], %b_other {box = ...} awaits(%td)
    scf.yield %na, %nb
  } else {
    scf.yield %td, %td              // 无预取,给下一轮一个已完成的 tp
  }
  scf.yield %acc, %ta_next, %tb_next, %td_new
}
pustream.await %td_f
pustream.async.copy %acc_f, %dram_out {box = ...} awaits(%td_f)
```

要点:
- 初始化 copy 的 tp 作为 iter_args 种子;`%t_init` 按 I1 的 init 规则取生产者
  的 tp 或 `immediate`。
- 循环体内没有 `memref.alloc`;`%a_cur`/`%a_other` 是 slot 选择的结果
  (2 slot 时即交替;实现用 `scf.if`,N>2 用嵌套 `scf.if`,见 P5)。
- 越界预取必须 `scf.if` 屏蔽,否则最后一轮会拷 `%dram_a[%K]`(越界)。
- 预取的 WAR 只等**上一轮 dispatch 的 tp**(该 slot 的最后一个读者),
  所以第 k 轮 load 与第 k-1 轮 compute 可以重叠。
- drain copy 等的是循环结果 tp(`%td_f`);kernel 退出前对残留 tp 做
  join+await(I1 drain 规则)。

**P7 后(静态打包):**

```mlir
%slab = memref.alloc() : memref<SLAB_BYTESxi8, 128>
%a0 = memref.view %slab[%c_off_a0][] : memref<SLAB_BYTESxi8, 128> -> memref<MxKxf32, 128>
// ... 其余 buffer 同理,offset 全部是常量;无 dealloc
```

(`memref.view` 需要新增 ViewOpLowering,见 §7 P7。)

**P8 CPU:**

```mlir
pumem.copy dram_to_sram ...        // 同步执行
pumem.runtime_call @k_step(...)    // 或 func.call
```

**P8 NPU:**

```mlir
%sem = pumem.async_copy dram_to_sram ... -> !pumem.semaphore
pumem.wait %sem
pumem.dispatch @k_step(...) -> %ev     // 命令入队 + event
```

## 10. 里程碑

- **M0 — 同步等价**
  - 前置:§6 设备契约文档完成。
  - 方言骨架:`timepoint` + 5 op + verifier(I1 含 init/drain 与循环情形起步、
    I2、I3)。
  - P1 + P2 + P4 仅把 copy 转成 async(compute 暂留同步,dispatch 在 M1 启用)
    + CPU 同步 lowering。
  - 验收:现有 kernel pytest 全绿;数值与当前输出一致;lit 覆盖 stream IR 结构
    与支配 verifier(含跨迭代 case)。
- **M1 — Executables**
  - P3;matmul/transpose 进 executable;`linalg_to_runtime_call` 移入 CPU 分支;
    可执行体两个分叉。
  - 验收:matmul / reduce kernel 数值正确;NPU 分支 IR 中 executable 保持 linalg。
- **M2 — 静态分配**
  - P7:async-aware liveness、常量 offset、单 slab、I4 断言;新增
    `memref.view` 的 ViewOpLowering。
  - 验收:IR 检查 offset 无重叠、slab ≤ SRAM;数值不变(数值在 M0 已锁定)。
- **M3 — 流水线 + NPU 形状**
  - P5(k 循环 multi-buffer,优先上游 `scf::pipelineForLoop`,必要时薄 peeler)
    + P6 + I5 semaphore 分配。
  - 验收:lit 检查 prologue/epilogue 与 tp 轮转结构、I1 循环情形、I5;
    CPU 数值回归;NPU IR 含 async_copy/wait/signal。

## 11. 测试策略

- **lit**(无需 GPU):每个新 pass 加 `third_party/privateuse/test/Transform/`
  或 `test/Conversion/` 下的 `.mlir`(沿用现有 `test/lit.cfg.py`),CHECK 关键结构:
  timepoint 接线、offset、两个分叉的 IR 形状、verifier 负例。
  privateuse 的 lit 用 **`privateuse-opt`**(不是 `triton-opt`);`lit.cfg.py`
  自动发现 `build/cmake.*/third_party/privateuse/tools/privateuse-opt`。
  跑法:`ninja privateuse-opt` 后
  `lit -v third_party/privateuse/test/Transform/<x>.mlir`。
- **pytest**(模拟后端数值回归):privateuse 的 kernel 测试在
  `third_party/privateuse/test/flaggems` 与 `test/cases`(不是 `python/test/`)。
  现有测试全绿是每个里程碑的门槛;新增 `test_<feature>_<condition>` 覆盖
  double-buffer、packing 边界(非对齐元素数、大 grid pid 循环、动态 K)。
  `pytest -s --tb=short`。
- **verifier 测试**:I1 的正反例(含 scf.for 轮转)、I2 几何错配、I4 重叠/超限。
- native/compiler 改动测试前先 `make`;仅 Python 改动不需要。

## 12. 风险与开放问题

1. **上游 pipelining 能力**:`scf::pipelineForLoop` 能否用自定义 schedule 正确
   建模 timepoint iter_args 与 slot 轮转,需在 M3 前做原型;不能则退薄 peeler。
2. **跨迭代支配**:I1 的循环轮转检查是 M3 最复杂部分;M0 先覆盖简单循环,
   M3 扩展到多 slot。
3. **DMA 描述符能力**:维度/stride/对齐限制决定 copy 拆分;多队列硬件下
   hazard 规则必须严格,不能依赖队列序。
4. **gather/scatter**:现有 `pumem.gather/tile_gather` 访问没有异步路径;
   本期声明为不同步异步化,后续要么拆多次 box 拷贝,要么引入描述符扩展。
5. **SRAM 预算与 stage 回退**:超限时的降级策略(缩 tile / 降 stage / 拆分循环)
   需要产品化,不能只报错。
6. **参数透传**:`num_stages` / `runtime_backend` 必须经 `DeviceInfo.fields` 生效,
   且 driver 不能再读 `LoweringPipeline` 的类属性(driver.py:334);否则 option
   改了既不生效,也不会让缓存失效(见 §8)。
7. **executable 体的 NPU 契约**:NPU 编译器接受什么形态的 compute IR,决定
   P8b 是否保持 linalg;需要在 M1 与设备侧对齐。
8. **matmul 覆盖语义的设备一致性**:§4.4 已在 IR 层钉死,但 NPU 编译器必须实现
   同样的覆盖语义(或统一为 fill+累加);M1 与设备侧确认,否则两分支数值不同。
9. **`scf.index_switch` 不支持**:ScfToPumem 只覆盖 For/If/Yield;若 slot 选择
   不想嵌套 if,需要补 IndexSwitch pattern。
10. **`memref.view` 无 lowering**:必须新增 ViewOpLowering(M2),否则 P7 的
    slab 方案无法落地;`reinterpret_cast` 不能改变元素类型,不是替代品。
    逻辑 buffer 必须是 identity 连续子区间(view 的 verify 要求双端 identity
    layout + 同 memory space)。

---

## 13. 重新抽象:语句式 pustream(v2)

> **这一节取代** §4 / §4.3(类型、5 个 op、awaits 语义)、§4.5(pumem semaphore)、
> §5-I1(token 支配 verifier)、§6 中按活跃 timepoint 推导的 `max_stage_by_sem`、
> §7.2 的 P5/P6、§9 的示例。§3 分层、§6 设备契约框架、§7.2 的 P1-P4/P7-P9、
> §10 里程碑框架仍然有效。

### 13.1 动机:抽象必须与后端契约同构

后端最终形态是"发射器 + 三个执行体",只有两种"序":

| 执行体 | 排序保证 | 等待原语 |
|---|---|---|
| load 队列(DRAM→SRAM) | 全局 FIFO(按发射顺序) | `pumem_wait(handle)` |
| store 队列(SRAM→DRAM) | 全局 FIFO | `pumem_wait(handle)` |
| compute 池(dispatch) | 无序 | 只有全局屏障 `pumem_wait_dispatch()` |

runtime 的完整原语集(4 + 1):

1. `async_copy_{dram,sram}_to_{dram,sram}<Rank>(handle, meta, dst, src, elem_size, sizes..., dst_strides..., src_strides...)`
2. handle / meta(buffer slot 的序列点与在途装箱槽位,在途复用时 abort)
3. `pumem_wait(handle)`:等该 handle 的 last_read / last_write
4. `pumem_wait_dispatch()`:等所有已发 dispatch
5. `pumem_sync()`:exit 排空

原 timepoint 模型发明了运行时不具备的语义:

- **token 级精度是假的**:运行时只能等 handle 的"最后一次拷贝",任意 token 的
  等待最终都退化成 handle 等待;
- **compute 的 per-op token 也是假的**:没有 per-dispatch 完成事件,只有全局屏障;
- 于是 ScheduleTimepoints 不得不把 DMA token 与 dispatch token 混进 `awaits` /
  `join`,再靠 region 末尾 join+await 全排空兜底——两套粒度并存的机制是
  "实现逻辑奇怪"的根因。

**设计原则:IR 原语数 = runtime 原语数。** pustream 只描述发射序,等待是资源
栅栏(语句),依赖不再是 SSA 值。

### 13.2 新的 op 集(4 个语句,无 timepoint)

| op | 操作数 | 语义 |
|---|---|---|
| `pustream.async.copy` | `src`, `dst`(memref slice)+ 静态 box 几何 | 按 space 方向入 load/store 队列;**无结果** |
| `pustream.async.dispatch` | `@callee(args...)`,`reads`/`writes` 声明 | 入 compute 池;**无结果** |
| `pustream.wait` | `buffer`(memref slice) | 冲刷该 buffer slot 的 DMA(两个方向)→ `pumem_wait(handle)` |
| `pustream.wait_dispatch` | — | 全局 compute 屏障 → `pumem_wait_dispatch()` |

```tablegen
def PuStream_WaitOp : PuStream_Op<"wait", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let arguments = (ins AnyMemRef:$buffer);
  let hasVerifier = 1;   // buffer 必须是 P7 分配过 handle 的 DMA 参与者
}
// wait_dispatch 无操作数;两个 op 都声明副作用,防止被 DCE。
```

语义规则:

1. **发射序即程序序**。所有排序只来自三种手段:程序序、同方向队列 FIFO、
   两类栅栏语句。不存在 token 传递。
2. **同方向 copy 由全局 FIFO 严格保序**,IR 不为其插排序 wait;只有跨方向 /
   跨域 hazard 需要语句。这条写进方言语义,不是 runtime 实现细节。
3. **wait 粒度 = handle 粒度 = 逻辑 buffer(P7 决定)**;多缓冲时是 slot。
   `wait %slice` 只冲刷该 slice 所属 slot。
4. **dispatch 不暴露完成值**;compute 依赖只能表达为 `wait_dispatch`(全局)。
   同一批无 hazard 的 dispatch 在屏障之前可并发。
5. 不再有 `!pustream.timepoint`、`await`、`join`、`immediate`、`awaits`。
6. kernel/launcher 退出必须排空全部队列与 dispatch,实现上可折叠为
   `pumem_sync()`。

**为什么可以不要 token**:运行时精度上限就是 handle 序列点,IR 表达更细的东西
lowering 也只能降级成同一组 wait。把"谁等谁"从值图改成发射点上的栅栏语句,
IR 语义反而与 runtime 完全对齐。

### 13.3 等待插入 pass(替代 ScheduleTimepoints)

输入:语句式 IR(§13.2),可含已流水化的多缓冲循环。输出:同 IR + wait 语句。

状态:每个 base memref → `{dma, compute}` 两个"可能仍有在飞 op"的 may 标志。
分析**不看 SSA 依赖**,只看 effect 契约(`reads`/`writes` 与 copy 的 src/dst)。
方向不进状态:wait 规则不区分方向,同方向 FIFO 只让发出的 wait 变成 no-op。

规则(按发射序扫描;`if` 分支出口取并集;`for` 头状态取不动点):

| 语句 | 前置检查 | 插入 | 状态更新 |
|---|---|---|---|
| `async.copy` | `src`/`dst` 的 compute 标志 | 至多一次 `wait_dispatch` | 清全部 compute |
| | `src`/`dst` 的 dma 标志 | **对该访问带 handle 的 slice** 各插 `wait %slice` | dma **不清** |
| | — | — | 置 dma 于 src、dst |
| `async.dispatch` | 任一 read/write 的 compute 标志 | 一次 `wait_dispatch` | 清全部 compute |
| | 任一 read/write 的 dma 标志 | 对相应操作数 slice 各插 `wait %slice` | dma **不清** |
| | — | — | 置 compute 于全部 reads∪writes |
| `wait %b` | — | — | **不清任何标志**(见下) |
| `wait_dispatch` | — | — | 清全部 compute |
| 同步访问(同 space `memref.copy` / gather / scatter) | 读/写 base 的 `compute`/`dma` 标志 | 同消费点规则:屏障 / 操作数 slice wait | **不置在飞标志**(host 线程同步完成) |

**为什么 slice wait 不清 dma 标志**:base 级标志表达"该 base 的任一 slot 可能
有在飞 DMA";`wait %slice` 只冲刷一个 slot 的 handle,证明不了同 base 的其他
slot 已完成。若清了,多缓冲场景会漏掉对另一 slot 的依赖,分析不 sound。保持
不清的代价是:一个 base 被访问过之后,后续每次访问都会带一条 wait;但这条
wait 落在消费者自己的 slot 上,空 handle 立即返回——只有调用开销,没有阻塞。
compute 标志不同:`wait_dispatch` 是全局屏障,可以安全地清全部。

**wait 只对带 handle 的操作数发射**:handle 归属于 SRAM staging slot(§13.4),
DRAM 侧操作数不产生 wait;同一 DRAM 区间在 kernel 内既被 store 写又被 load 读
的形态由 verifier 先拒绝(§13.10 开放问题)。

**同步访问必须纳入本表**:P4 会保留同 space 的 `memref.copy` 同步语义
(gather/scatter 也没有异步路径,旧 `ScheduleTimepointsPass` 有专门的
`scheduleSyncCopy` 处理)。它们在 host 线程上执行,完成即落地,所以**不置**
在飞标志;但若读到未完成的 DMA / compute 就是真错误,因此必须按消费点规则
插 wait(compute 标志 → 屏障;dma 标志 → 操作数 slice wait)。漏掉这条,P6
之后的同步 copy 会裸读正在被 dispatch 写的 SRAM。

**三层的职责边界**:pipeliner 拥有 slot 身份(`remui` 下标与 per-slot
handle);分析只拥有 hazard 的**存在性**(base 级 may 标志);runtime 拥有
slot 精度(wait 只冲该 handle)。所以分析不做模 N 推理,也不需要证明"第 i 轮
与第 i+N 轮是同一 slot"——这是 base 粒度而非 subview 表达式粒度的原因。

**meta 复用的守卫**:runtime 在**入队时**检查 `meta.busy`,同 meta 背靠背
入队即使同队列 FIFO 也会 abort,所以每次 meta 复用都必须发生在同 handle 的
wait 之后。上面"dma 标志不清 + copy 前按需插 wait"恰好自动满足:一个 base
被访问过之后,每次 copy 入队前都有一条落在自己 slice 上的 wait(prologue 的
首个 copy 无标志,但 meta 全新)。该性质写成不变量交给 verifier(§13.8 I7),
不需要为 meta 单独记状态。

要点:

- **wait 的操作数永远是消费者自己用的 slice**,不是 base。分析在 base 粒度
  保守(分不清多缓冲的 slot),只会多插;多插的 wait 落到消费者自己的 slot 上,
  runtime 只冲该 handle,不会误伤别的 stage——**stage 精度不需要分析知道 N**。
- 多插的 wait 在运行时多数是 no-op(该 slot 的上一笔已被更早的 wait 冲刷过,
  或同方向 FIFO 已完成),换来的是规则统一、没有 slot 级推理。
- `for` 的头状态 = pre-header ∪ back-edge 的不动点(插入单调,迭代收敛)。
  不再有"region 末尾全排空"的兜底。
- 退出排空:函数 / kernel 返回前,对残留标志插 `wait` + `wait_dispatch`;
  lowering 可整体折成 `pumem_sync()`。
- 相邻重复的 `wait %b` / `wait_dispatch` 可去重(可选优化)。

正确性论证(简):对任意两条冲突访问 K1→K2,若 K1 可能仍在飞,K2 处必命中
上表某个分支——同 buffer 的 dma 标志触发 slice wait;compute 标志触发全局屏障;
同方向跨 slot 的 DMA 顺序由队列 FIFO 保证。verifier(I1')用同一套数据流复算,
断言每个消费点的 wait 覆盖冲突生产者。

### 13.4 pumem / ABI / runtime 映射

新增(形状对齐现有 `pumem.copy`):

| op / 类型 | 签名 | 用途 |
|---|---|---|
| `!pumem.dma_handle` | — | buffer slot 的 DMA 序列点 |
| `!pumem.dma_meta` | — | 在途装箱槽位 |
| `pumem.dma_slot` | `(buffer) -> (handle, meta)` | 单缓冲 slot(P7 插入) |
| `pumem.dma_slot_array` | `(buffer, N) -> (handles[N], metas[N])` | 多缓冲 slot |
| `pumem.dma_slot_at` | `(handles[N], idx) -> (handle, meta)` | 动态 stage 索引 |
| `pumem.async_copy` | `(handle, meta, dst, src, 几何..., dir)` | → `async_copy_*` |
| `pumem.wait` | `(handle)` | → `pumem_wait` |
| `pumem.wait_dispatch` | `()` | → `pumem_wait_dispatch` |
| `pumem.dispatch` | `@callee(args...)` | → `pumem_async_dispatch` |

映射表:

| pustream | pumem | C++ |
|---|---|---|
| `async.copy %src, %dst` | `pumem.async_copy %h[%s], %m[%s], ...` | `async_copy_dram_to_sram<1>(h_in[s], m_in[s], ...)` |
| `async.dispatch @f(...)` | `pumem.dispatch` / `func.call` | `pumem_async_dispatch(f, ...)` |
| `wait %slice` | `pumem.wait %h[%s]` | `pumem_wait(h_in[s]);` |
| `wait_dispatch` | `pumem.wait_dispatch` | `pumem_wait_dispatch();` |
| exit 排空 | — | `pumem_sync();` |

变化点:

- **CPU 分支不再把 async 降成同步 copy**:CPU 后端本身就是异步模拟器;
  `pumem.copy`(同步)只保留给 gather/scatter 等无异步路径的场景。
- **handle 归属**:一个 handle/meta 对绑定一个 SRAM staging slot(多缓冲时成
  数组);copy 的 handle 取它读写的 SRAM 侧 slot,load 记 `last_read`、store
  记 `last_write`。因此 `wait %slice`(SRAM 侧)→ `h[stage]` 一条就能同时覆盖
  "load 进该 slot"和"从该 slot 存出"两条序列。
- **ToPumem 的"timepoint 消解"整段删除**:没有 token 需要翻译。
- **P7 分配 handle/meta**:每个参与 DMA 的逻辑 buffer 一个 handle+meta;
  多缓冲 alloc(`pustream.stages = N`)分配数组,动态 stage 下标用
  `pumem.dma_slot_at` 取值。多缓冲维度的动态 subview 走现有
  expand-strided-metadata,不需要 `scf.index_switch`(旧 §7.2 P5 的顾虑消失)。
- **NPU 分支**:handle → semaphore/event 池;`wait` → semaphore wait;
  `wait_dispatch` → join 所有已发 event。§6 设备契约需增加 event 能力项。

### 13.5 N-stage

N-stage 不是 pustream 的新概念,拆成两个正交步骤:**pipeliner 只做"多缓冲 +
错位发射"的结构变换,不插 wait;等待由 §13.3 在最终循环形状上重新推导。**
pustream 里只有两个载体:多缓冲 memref 的 stage 下标、循环 IV 的预取偏移。

#### 13.5.1 pipeliner 的四步改写

前置条件:循环体是"copy 写 tile → dispatch 读 tile → copy 存结果"的直线结构,
tile 不跨迭代携带;load 的 DRAM 偏移是 IV 的仿射函数。

1. **识别**:copy 的目标 tile 在本轮被 dispatch 读(load);dispatch 的写被
   copy 读(store)。
2. **多缓冲**:这些 SRAM tile 加 stage 维(in 取 N,out 取 S),alloc 打
   `pustream.stages` 标记。N/S 由 §13.5.3 分析推导。
3. **stage 算术**:所有访问改 `alloc[stage]` subview;预取偏移改 `off(i+N-1)`。
4. **前导 + 谓词主循环**:前导编译期展开 `L(0..N-2)`,每条按 `s < K` 谓词化;
   主循环内预取 `L(i+N-1)` 按 `i+N-1 < K` 谓词化。不需要尾声,trip count 任意
   (`K < N` 也正确)。

```mlir
// 变换前
scf.for %i = 0 to %K step 1 {
  %pin  = memref.subview %in[]   ...
  %pout = memref.subview %out[]  ...
  pustream.async.copy %dram_in[%off(%i)], %pin          // L(i)
  pustream.async.dispatch @body(%pin, %pout) reads(%pin) writes(%pout)
  pustream.async.copy %pout, %dram_out[%off(%i)]        // S(i)
}
```

```mlir
// pipeliner 产出(pustream.wait 尚未插入)
%in  = memref.alloc() : memref<N x TILE, 3> {pustream.stages = N}
%out = memref.alloc() : memref<S x TILE, 3> {pustream.stages = S}

// 前导:L(0) .. L(N-2),每条谓词化;pipeliner 不插 wait
// (插入 pass 会对 s>0 的 slot 补一条保守 wait,运行时为 no-op,见 §13.5.2)
%ok0 = arith.cmpi slt, %c0, %K
scf.if %ok0 {
  %p0 = memref.subview %in[0]
  pustream.async.copy %dram_in[%off(0)], %p0
}
// ... s = 1 .. N-2 同模式

scf.for %i = 0 to %K step 1 {
  %sc    = arith.remui %i, N
  %sp    = arith.remui %i + N - 1, N
  %inext = arith.addi %i, N - 1
  %has   = arith.cmpi slt, %inext, %K
  scf.if %has {
    %pp = memref.subview %in[%sp]
    pustream.async.copy %dram_in[%off(%inext)], %pp      // L(i+N-1)
  }
  %pin  = memref.subview %in[%sc]
  %pout = memref.subview %out[%i mod S]
  pustream.async.dispatch @body(%pin, %pout) reads(%pin) writes(%pout)
  pustream.async.copy %pout, %dram_out[%off(%i)]         // S(i)
}
```

`N=2` 的 slot 演算:前导把 L(0) 放进 slot 0;主循环第 i 轮预取 L(i+1) 进
`(i+1)%2`,消费 L(i) 于 `i%2`,互不踩。

#### 13.5.2 插入 wait 后的稳态循环

| 位置 | wait | 理由 | 运行时代价 |
|---|---|---|---|
| 预取写 slot 前 | `wait %pp` | 同 slot 上一份拷贝 + meta 复用 | 多为 no-op |
| dispatch 前 | `wait %pin` | RAW:L(i) | 等该 slot 的 last load |
| dispatch 前 | `wait %pout` | WAR:out slot 上一轮 store | S≥2 时多为 no-op |
| dispatch 后 | `wait_dispatch` | store 读 C(i) 结果 | 计算完成点,不可避免 |
| store 入队前 | `wait %pout` | 保守:out base 的 dma 标志在 base 粒度不清 | 恒为 no-op(该 slot 的上一笔 store 已被本轮 dispatch 前等过) |
| (无 store 的归约) | 预取前多一条 `wait_dispatch` | 覆盖 slot 的读者是上一轮 C(i-1) | 屏障原本就要等 C(i-1),不额外串行 |

```mlir
scf.if %has {
  %pp = memref.subview %in[%sp]
  pustream.wait %pp
  pustream.async.copy %dram_in[%off(%inext)], %pp
}
%pin  = memref.subview %in[%sc]
pustream.wait %pin
%pout = memref.subview %out[%i mod S]
pustream.wait %pout
pustream.async.dispatch @body(%pin, %pout) reads(%pin) writes(%pout)
pustream.wait_dispatch
pustream.wait %pout            // 保守补插;该 slot 已被上面等过,runtime no-op
pustream.async.copy %pout, %dram_out[%off(%i)]
```

关键性质:**这几条 wait 全部由 §13.3 的通用规则推出**,pipeliner 不参与;
分析不知道 N、不做取模推理——精度来自"wait 操作数是消费者自己的 slice +
handle per slot"。

#### 13.5.3 N 与 S 的推导(分析,取代 §6 公式)

N 不来自前端 hint,`tl.range(num_stages=…)` / 编译选项只作为**可选上界或
覆盖**(调参入口),主路径由 P5 对每个流水循环做静态推导:

```
T_load    = 每轮 copy 搬运字节 / DMA 有效带宽          (盒几何 + 设备契约)
T_store   = 每轮 store 字节   / DMA 有效带宽
T_compute = 每轮 dispatch 计算量 / compute 吞吐        (linalg 形态 + 设备契约)

N_min = 1 + ceil(T_load  / T_compute)      // 预取要在 N-1 轮计算内完成
S_min = 1 + ceil(T_store / T_compute)      // store 槽要撑过 S-1 轮计算

若 K < 2:不流水化(N = S = 1,无前导);下式仅对 K >= 2 定义
N = clamp(N_min, 2, min(N_sram, N_event, K))
S = clamp(S_min, 2, min(S_sram, N_event, K))
```

- `N_sram`:SRAM 预算 / 每 stage 输入 staging 字节(含对齐);`S_sram` 同理。
  该预算与 §7.2 P7 的 slab 打包是**同一个池**,P5 推导时必须先扣除非 staged
  scratch,再把剩余额度在多个流水循环之间按收益分配。
- `N_event`:仅 NPU——每 stage 活跃 handle/event 数 × N ≤ 设备上限;
  CPU 后端 handle/meta 是普通结构体,不设此项。
- `K`:trip count。深度超过 `K-1` 只会浪费 SRAM。`K < 2` 直接跳过流水化,
  避免 clamp 上界小于下界的未定义形态。
- **公式假设 DMA 与 compute 是互不相交的资源**:对带真实 DMA 引擎的 NPU 成立;
  CPU 模拟器里 load/store/compute 共享同一个 2–16 worker 池,`T_*` 的标定必须
  在并发 compute 负载下测,否则会过度 provision,CPU 上推导值只作为上限。
  三者同时重叠至少要 3 个 worker(ThreadGuard 下限为 2),2 核机器上物理不可能,
  这是性能兑现问题,不影响正确性。
- 参数估计:CPU 后端的带宽 / 计算吞吐来自设备契约(§6)的常量或一次微基准
  (roofline);strided copy 用"搬运字节 / 有效带宽",有效带宽可按 stride
  形态打折。所有估计只影响**性能**,不影响正确性。
- 缺失设备参数时退化为 `N=2, S=2` 并给出编译期 remark(正确性不变,只是
  不再保证隐藏);不允许静默取一个很大的 N。
- 总 SRAM 超限时,按"回退 N 损失的隐藏时间"从小到大逐个回退,而不是一刀切。
- 可选调参:同一 kernel 编译 N ∈ {推导值-1, 推导值, +1} 三档由 runtime 实测
  选择(本期不做,留作 P5 的扩展点)。

推导的物理含义:steady state 下 host 每轮被 `wait_dispatch` 串行在 compute 上,
所以一轮的时间 ≈ `T_compute`;预取深度 N-1 决定 load 可以提前几轮发出,
S-1 决定 store 可以滞后几轮完成。当 `T_load ≤ T_compute` 时 N=2 即足够;
当 kernel 是 DMA-bound 时,`N_min` 会超过 SRAM 上限,此时分析应同时给出
"kernel 受限于 DMA"的编译期诊断(建议缩 tile),而不是继续加 N。

#### 13.5.4 边界情况

- **任意 trip count**:前导 / 预取全部谓词化,`K < N`、非整除都正确,无
  fallback 循环。
- **归约(compute 跨迭代)**:每轮 compute 依赖上一轮(acc 的 WAW/RAW),屏障
  每轮一次;tile load 仍可预取 N-1 深,load 延迟仍被隐藏(§13.5.2 表末行)。
- **多条独立 load 链**:各自多缓冲、各自 slot;barrier 在汇合点耦合。
- **launcher 退出**:最后一个 store 仍在队列,exit 排空 → `pumem_sync()`;
  外层 pid 循环的跨迭代复用由同一个 wait 插入 pass 在 pid 层处理。

#### 13.5.5 对 §9 示例的对照改写(matmul k 循环,N=2)

```mlir
%a = memref.alloc() : memref<2 x MxKxf32, 128> {pustream.stages = 2}
%b = memref.alloc() : memref<2 x KxNxf32, 128> {pustream.stages = 2}

%pa0 = memref.subview %a[0];  %pb0 = memref.subview %b[0]
pustream.async.copy %dram_a[%c0], %pa0      // 前导
pustream.async.copy %dram_b[%c0], %pb0

scf.for %k = %c0 to %K step %c1 {
  %sc = arith.remui %k, 2
  %sn = arith.remui %k + 1, 2
  %k1 = arith.addi %k, 1
  %in = arith.cmpi slt, %k1, %K
  scf.if %in {
    %pn = memref.subview %a[%sn];  %qn = memref.subview %b[%sn]
    // 插入 pass:compute 标志 → wait_dispatch(等 C(k-1) 释放 slot);
    // 同 slot 拷贝 → wait %pn / %qn(no-op)
    pustream.wait %pn
    pustream.wait %qn
    pustream.wait_dispatch
    pustream.async.copy %dram_a[%k1], %pn
    pustream.async.copy %dram_b[%k1], %qn
  }
  %pc = memref.subview %a[%sc];  %qc = memref.subview %b[%sc]
  pustream.wait %pc
  pustream.wait %qc
  pustream.wait_dispatch                     // acc 的 RAW/WAR:等 C(k-1)
  pustream.async.dispatch @k_step(%pc, %qc, %acc) reads([0,1,2]) writes([2])
}
pustream.wait_dispatch                        // drain:等最后一轮 C(K-1)
pustream.async.copy %acc, %dram_out           // 写回
// launcher 退出:pumem_sync()
```

与 §9 的 P5 版本对比:没有 timepoint iter_args、没有 join、没有 `scf.if` 里的
tp 选择;slot 旋转只是 `remui` 下标 + 多缓冲 alloc。

### 13.6 overlap 能力与边界

稳态时间线:

```
host:   [wait C(i)] [发 S(i)、L(i+N)、C(i+1)] [wait C(i+1)] ...
load:        [ L(i+N-1) ]                       [ L(i+N) ]
store:  [ S(i-1) ]            [ S(i) ]
compute:     [ C(i) ..........................][ C(i+1) ..............]
```

| 重叠 | 结论 | 说明 |
|---|---|---|
| load ↔ compute | 完全 | N-stage 预取;`wait %cin` 只等 L(i),已被 N-1 轮隐藏 |
| store ↔ compute | 半 | store 的**执行**与下一轮 compute 重叠;但入队必须等 `wait_dispatch`(store 读 compute 输出,无跨域依赖边) |
| load ↔ store | 完全 | 两条独立 FIFO 队列 |
| compute ↔ compute | 跨迭代不行 | 全局屏障;同一轮内无 hazard 的 dispatch 并发 |

- 背压点只有两个:load 吞吐跟不上 N-1 的预取深度(`wait %cin` 阻塞)、store
  吞吐跟不上 S 的宽限(`wait %pout` 等 S(i-S) 未完)。这是 §13.5.3 推导 N/S
  的直接依据。
- **overlap 的真实载体是同一个线程池**(2–16 worker),不是独立 DMA 引擎;
  runtime 是数据流调度,排队等前驱的 copy 任务不占 worker。并发度受 worker
  数与 DMA/compute 各自占用时长约束;load + store + compute 三者同时重叠至少
  需要 3 个 worker(ThreadGuard 下限为 2)。
- runtime 的 TaskPool / 两条队列 / dispatch group 目前是**进程全局单例**:
  launcher 顺序执行时没问题;若并行跑多个 kernel,全局 FIFO 会跨 kernel 串行、
  `wait_dispatch` 会跨 kernel 互等。正确性不受影响,是已知的性能耦合(§13.10)。
- 进一步的空间:runtime 的 `create_task` 本来就支持 `awaits`(copy 的 FIFO
  链就是用它实现的),`enqueue_dispatch` 只是没传依赖。将来让 dispatch 暴露
  完成 task、copy 可 await 一个 compute task,store 就能在 compute 完成前入队、
  compute 之间也能流水——那是小的 runtime 扩展而非新机制。§6 设备契约应把
  "无 per-dispatch 完成事件"记为**当前设备限制**,而不是普适模型。

### 13.7 Pass pipeline 变化

| 阶段 | v2 变化 |
|---|---|
| P0 specialize | 不变 |
| P1 bufferize | 不变 |
| P2 cleanup-copies | 不变 |
| P3 form-executables | 不变(reads/writes 契约不变) |
| P4 to-pustream | 产出语句式 IR:copy/dispatch 无结果;删除 awaits/join/immediate |
| **P5 pipeline** | N-stage **纯结构变换**(§13.5);N/S 由 §13.5.3 分析推导;取代原 timepoint iter_args 方案 |
| **P6 insert-waits** | 新增 §13.3;替代 `ScheduleTimepointsPass` 与 `SimplifyTimepointsPass` |
| P7 plan-allocation | 增加 handle/meta 分配(含多缓冲数组);slab 打包不变 |
| P8 lower | 删除 timepoint 消解;新增 `pumem.async_copy` / `pumem.wait` / `pumem.wait_dispatch` / `pumem.dispatch` lowering |
| P9 emit | 新增 dispatch/wait 调用发射;其余不变 |

```python
module = self._specialize(module)          # P0
module = self._bufferize(module)           # P1
module = self._cleanup_copies(module)      # P2
module = self._form_executables(module)    # P3
module = self._to_stream(module)           # P4(语句式)
module = self._pipeline(module)            # P5(结构流水化 + N/S 推导)
module = self._insert_waits(module)        # P6(新)
module = self._plan_allocation(module)     # P7
module = self._lower(module)               # P8
cpp = self._emit_cpp(module)               # P9
```

删除/重写:`ScheduleTimepointsPass`(重写为 InsertWaits,含同步 op 规则)、
`SimplifyTimepointsPass`(无 join,删除;保留相邻 wait 去重可选项)。
`pumem-async-codegen-example.cpp` 的**原语集合**不变,但 wait 落点按 v2 重排:
现有示例在每条 store 之后 `pumem_wait(h_out)`、退出用 `wait_dispatch`、单缓冲
alloc 在循环内,都不是 v2 的输出,需重生成。里程碑映射:M0 = 语句 op +
InsertWaits + CPU lowering;M2 增加 handle 数组;M3 只剩 pipeliner 与 overlap
验证,不再有 timepoint 轮转验证。

### 13.8 不变量变化

| 编号 | v2 状态 |
|---|---|
| I1(hazard 支配,基于 token) | **改为 I1'(wait coverage)**:verifier 用同一套前向数据流、额外携带每 base 的 last-producer 身份,检查每个消费点到 producer 的路径上存在覆盖 wait;它防的是后续 pass 挪/删 wait,独立语义论证在 §13.3,不做模 N 推理 |
| I2(拷贝几何) | 不变 |
| I3(dispatch 效应) | 不变 |
| I4(打包) | 不变;扩展:handle/meta 数组长度与 `pustream.stages` 一致 |
| I5(信号量) | 仅 NPU:按"每 stage 活跃 event/handle 数"计;CPU 后端不设限;N 推导见 §13.5.3 |
| 新增 I6(queue 语义) | verifier:copy 方向与 space 一致;DRAM↔SRAM 之外禁止;同方向全局 FIFO 是语义,不检查 |
| 新增 I7(meta 复用支配) | verifier:每次 copy 的 (handle, meta) 复用时,同 slice 的 wait 必须支配该入队点(§13.3 规则自动保证);runtime 是入队即 abort、不是 race,该检查常开 |

### 13.9 迁移清单

删除:`!pustream.timepoint`、`pustream.await`、`pustream.timepoint.join`、
`pustream.timepoint.immediate`、copy/dispatch 的 `awaits`;`ScheduleTimepoints`
的 PendingWaits / token / region drain 逻辑;`SimplifyTimepoints`;ToPumem 的
timepoint 消解。

保留/改写:`async.copy` / `async.dispatch`(去掉结果与 awaits,加 verifier);
`pumem.copy`(同步,留给 gather/scatter 等同 space 场景;InsertWaits 对它只
消费 hazard、不置标志);slab 打包(P7)。

测试:lit 的 CHECK 从 token 接线改为 wait 结构 + I1' 负例;现有
`test/flaggems`、`test/cases` 数值回归不变;新增 N-stage 结构测试(前导 /
谓词 / remui 下标)与归约预取 barrier 测试。

### 13.10 开放问题

1. **pipeliner 实现**:上游 `scf::pipelineForLoop` 面向有结果的 load,且不做
   加 stage 维的多缓冲与下标重写,其 slot 轮转会产出 `arith.select` /
   `scf.index_switch`(下游不支持)。因此自研薄 peeler,范围限 tile 循环直线体,
   接口按 `PipeliningOption` 形状保留可换性。**优先原型的是下游动态下标折叠
   (第 2 条),不是 peeler 本身。**
2. **多缓冲动态下标的下游(命门)**:`memref.subview %alloc[%s]` 经
   expand-strided-metadata / memref-to-pumem 能否把运行时 stage offset 折进
   (base, offset, sizes, strides);M3 前用 N=3 动态下标 lit 验穿。若不能 fold,
   回退是常量 offset subview + 选择,N>2 会撞 `scf.index_switch`,等于放弃深流水。
3. **`wait_dispatch` 的 NPU 映射**:全局屏障映射为"join 所有已发 event";
   NPU 是否需要 per-dispatch event 恢复 compute 流水,与设备侧确认。
4. **pid 循环嵌套**:流水循环在外层 pid 循环内时,不动点与 exit drain 的位置。
5. **handle 数组与 slab offset**:多缓冲 slot 的内存布局(连续 slot 区间)与
   handle 数组索引的一致性,I4 扩展断言。
6. **N/S 推导的常数标定**:CPU 后端的有效带宽 / 计算吞吐需要一次微基准标定,
   且必须在并发 compute 负载下测(共享 worker 池),结果固化进设备契约;
   否则 §13.5.3 只能退化到 N=2。
7. **示例与测试更新范围**:`pumem-async-codegen-example.cpp` 需按 v2 的 wait
   布局重生成(原语集合不变,落点变);IR lit 测试与 §9 示例需重写。
8. **DRAM 侧同 kernel 读写复用**:若同一 DRAM 区间先被 store 写、后被 load 读,
   SRAM slot 上的 handle 覆盖不到该冲突。verifier 先按 **base 级**保守拒绝
   (会连带拒绝 in-place elementwise 这类每轮 tile 不重叠的合法形态,登记为未来
   case)。扩展路径:P7 给"既读又写"的 DRAM base 也分配 handle,IR 与 wait 规则
   不变(规则本来就对 src/dst 带 handle 的操作数发射);代价是该 handle 上的
   store 可能被保守 wait 串行化,属 workspace 场景的已知成本。
9. **runtime 进程全局状态**:TaskPool / 两条队列 / dispatch group 是单例,
   并行 launcher 会跨 kernel 串行互等;将来按 launcher 持有 runtime 实例,
   §6 设备契约先记录该限制。
