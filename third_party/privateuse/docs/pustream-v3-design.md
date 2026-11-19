# pustream v3.1 设计方案:语句式 + 三层等待架构

> 状态:设计提案(v3.1,经与 v2 作者评审收敛),与 `pustream-implementation.md`
> §13(v2)并列。相对 v3 的实质修订:等待产出从"pipeliner 精确 + 单缓冲局部"
> 二分,改为**三层**(pipeliner 私有 per-slot DMA wait / body 局部 compute 屏障 /
> 单缓冲通用兜底);补 masked-load 的 OOB-live 判据与 H2 决策树;I8 重定位为结构
> 断言;N/S 加每后端成本模型 gate;新增 barrier 摊销与实测证据。
>
> 目标:在**固定硬件契约**下实现 func 级(整段 outlined compute 与整段 copy)的
> compute↔copy overlap,覆盖所有 tile 循环(含 pid 循环),并用可验证的编译器
> 机器精确摸到硬件 overlap 天花板。
>
> 与 v2 的关系:**op 集与硬件 1:1 不变**;分歧只在等待的产出架构。v2 的通用
> base 粒度分析不被删除,而是**降级为第三层兜底 + 独立校验器**。
>
> 本文自包含。引用现有资产给出文件路径;与 v2 的逐条对照见 §17。

---

## 0. 硬件契约(不可变前提)

后端最终形态是"一个发射 core + 三个执行体"。发射 core 把任务分别发给三个
执行体,执行体之间**没有设备侧依赖边**:

| 执行体 | 方向 | 排序保证 | 等待原语 |
|---|---|---|---|
| load 队列 | DRAM→SRAM | 全局 FIFO(按发射序) | `pumem_wait(handle)` |
| store 队列 | SRAM→DRAM | 全局 FIFO | `pumem_wait(handle)` |
| compute 池 | dispatch | 无序 | `pumem_wait_dispatch()`(全局屏障) |

runtime 完整原语集(4 个,不可扩展):

1. `async_copy_{dram,sram}_to_{dram,sram}<Rank>(handle, meta, dst, src, elem_size, sizes..., dst_strides..., src_strides...)`
   按方向入 load/store 队列;在 handle 上记 `last_read`(store)/`last_write`(load)。
2. `pumem_wait(handle)`:host 阻塞,等该 handle 的 `last_read` 与 `last_write` 两个 task。
3. `pumem_wait_dispatch()`:host 阻塞,等所有已发 dispatch。
4. `pumem_sync()`:host 阻塞,排空全部 task(退出用)。

handle / meta 的语义(`backend/runtime/pumem_async_runtime.h`):

- `pumem_dma_handle_t`:`last_read` / `last_write` 两个 `atomic<pumem_task*>`,
  分别记该 buffer slot 上最后一次 store / load 的 task。
- `pumem_dma_meta_t`:`atomic<bool> busy`。**入队时** `busy.exchange(true)` 若已 busy
  直接 `abort`(`pumem_async_runtime.cpp:246`)。busy 由 copy task 执行完清。
  注意:abort 发生在**入队**而非执行,所以即便同队列 FIFO 会串行执行,背靠背
  复用同一 meta 仍会 abort——meta 复用必须被一次 `pumem_wait` 隔开。

**地基推论**:跨执行器的排序(load→compute、compute→store、compute→load 的 WAR)
没有任何设备侧依赖边可走,**只能由发射 core 在发射依赖任务之前阻塞自己来中介**。
因此:

- IR 的排序原语就是 **host wait**,不是设备侧 awaits,更不是完成 token。
- func 级 overlap 的天花板由硬件决定:稳态下 C(i) ∥ L(i+N-1) ∥ S(i-1);
  store 只能半重叠(入队必须等 C(i),执行可与 C(i+1) 重叠);compute 跨迭代
  不能并发(只有全局屏障)。**这些是硬件必然,不是设计缺陷。**

> 关于 CPU 模拟器:执行模型与上表一一对应——**1 个 load 执行体线程(FIFO)、
> 1 个 store 执行体线程(FIFO)、1 个 compute 执行体**(内部分 W 条 lane,默认 2,
> `PUMEM_COMPUTE_LANES` 可调),host 线程只发射并阻塞在 wait 上。同方向 copy 的
> 串行来自**执行体只有一个消费者**,不是来自任务依赖边,所以模拟器里根本不存在
> "copy 带依赖边"的能力。compute 是**一个整体结构**:dispatch 只进它一个队列、
> 不暴露完成值、不接依赖,`pumem_wait_dispatch` 只能整体排空;但这个整体内部有多条
> lane,彼此无 hazard 的 dispatch 会并发(§7.2 的 cmpf∥mulf、§9 的 U 组都要靠它)。
> 模拟器忠实镜像硬件,**不是被削弱的表面**。任何要求 dispatch 暴露 task、copy 带
> 依赖边、meta 改 DAG 排序的方案都超出硬件契约,本方案不采用。

设计目标因而不是突破天花板,而是:**用可验证的编译器机器精确摸到天花板,并覆盖
所有 tile 循环(含 pid 循环)。** overlap 的余量只在三处:N/S 深度(SRAM 约束)、
流水化范围、以及 host 调用次数(barrier 摊销,§9)。

## 1. 设计原则

1. **IR 原语数 = runtime 原语数。** pustream 只有 4 个语句 op,无 token、无结果值。
   依赖是发射点上的栅栏语句,不是 SSA 值。
2. **等待产出三分,各层不需要对方的知识。** 多缓冲只来自 pipeliner,故 per-slot
   DMA wait 由 pipeliner 私有发出(它知道 slot 映射);compute 屏障由 body 局部
   规则发出(只用 reads/writes,不需要 N);非流水代码由单缓冲通用分析兜底。
   **通用分析不被删除,而是降级为兜底 + 独立校验器**(§4)。
3. **单缓冲代码 handle-wait 本身精确。** 一个 buffer 一个 handle,wait 该 handle
   就是精确等待,局部规则恒 sound,无需保守。
4. **发射序即程序序。** 排序只来自三种手段:程序序、同方向队列 FIFO、两类栅栏
   语句。同方向 copy 由 FIFO 保序,IR 不为其插排序 wait(写进方言语义)。
5. **精确性的价值是可验证/简单,不是吞吐。** 实测(§15)表明 v2 保守 wait 集与
   v3 零 no-op 集的吞吐差在噪声内(符号都会翻);真实吞吐杠杆是 overlap 机制、
   barrier 分组/摊销、每后端成本 gate。设计据此把"零 no-op"当代码质量目标,
   不当性能论据。
6. **不变量可验证,但认清各检查的边界。** 几何/效应/打包/meta 复用支配有硬
   verifier;wait coverage 对流水区域是**结构断言**、对兜底是 base 级检查;
   **slot 映射正确性靠数值回归 + remui lit + enqueue-order 调试器,不靠 coverage**
   (§14)。
7. CPU / NPU 共用中段 IR,末段在设备接口与 executable 体两处分叉。

## 2. 方言与 op 集

### 2.1 pustream(memref 级中段 IR)

目录沿用 `include/privateuse/Dialect/PuStream/IR/`、`lib/Dialect/PuStream/IR/`,
namespace `::mlir::pustream`。**无新类型**(用 memref + memory space:DRAM=0 / SRAM=128)。

| op | 操作数 | 语义 |
|---|---|---|
| `pustream.async.copy` | `%src, %dst`(memref slice)+ 静态 box 几何 | 按 space 方向入 load/store 队列;**无结果** |
| `pustream.async.dispatch` | `@callee(%args...) {reads, writes}` | 入 compute 池;**无结果** |
| `pustream.wait` | `%slice`(memref slice) | host 阻塞,冲该 slice 所属 slot 的 handle(两方向) |
| `pustream.wait_dispatch` | — | 全局 compute 屏障 |

多缓冲 alloc 打 `pustream.stages = N` 标记。

```tablegen
def PuStream_AsyncCopyOp : PUSTREAM_Op<"async.copy", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let arguments = (ins AnyMemRef:$source, AnyMemRef:$target,
                       DenseI64ArrayAttr:$sizes,
                       DenseI64ArrayAttr:$dstStrides,
                       DenseI64ArrayAttr:$srcStrides,
                       I64Attr:$elemBytes);
  let hasVerifier = 1;   // 几何与类型一致;跨 space 只允许 DRAM<->SRAM(I2/I6)
}

def PuStream_AsyncDispatchOp : PUSTREAM_Op<"async.dispatch", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       OptionalAttr<ArrayAttr>:$templateArgs,
                       Variadic<AnyType>:$args,
                       DenseI64ArrayAttr:$reads,    // args 下标
                       DenseI64ArrayAttr:$writes);
  let hasVerifier = 1;   // reads/writes 与 callee 冻结效应一致(I3)
}

def PuStream_WaitOp : PUSTREAM_Op<"wait", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let arguments = (ins AnyMemRef:$buffer);
  let hasVerifier = 1;   // buffer 必须是 P6 分配过 handle 的 DMA 参与者
}

def PuStream_WaitDispatchOp : PUSTREAM_Op<"wait_dispatch", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]>;
// wait / wait_dispatch 声明副作用,防止被 DCE。
```

语义规则:

1. 发射序即程序序;同方向 copy 由全局 FIFO 严格保序,IR 不为其插排序 wait。
2. wait 粒度 = handle 粒度 = 逻辑 buffer slot(P6 决定);多缓冲时是 slot。
   `wait %slice` 只冲该 slice 所属 slot 的 handle。
3. dispatch 不暴露完成值;compute 依赖只能表达为 `wait_dispatch`(全局)。
   同一批无 hazard 的 dispatch 在屏障之前可并发。
4. 不再有 `!pustream.timepoint`、`await`、`join`、`immediate`、`awaits`。
5. kernel/launcher 退出必须排空全部队列与 dispatch,实现上可折叠为 `pumem_sync()`。

### 2.2 pumem(设备接口,ptr ABI)

现有 pumem 只有同步 copy/for/if/runtime_call,需新增异步能力(形状对齐 `pumem.copy`):

| op / 类型 | 签名 | 用途 |
|---|---|---|
| `!pumem.dma_handle` | — | buffer slot 的 DMA 序列点 |
| `!pumem.dma_meta` | — | 在途装箱槽位 |
| `pumem.dma_slot` | `(buffer) -> (handle, meta)` | 单缓冲 slot(P6 插入) |
| `pumem.dma_slot_array` | `(buffer, N) -> (handles[N], metas[N])` | 多缓冲 slot |
| `pumem.dma_slot_at` | `(handles[N], idx) -> (handle, meta)` | 动态 stage 索引 |
| `pumem.async_copy` | `(handle, meta, dst, src, 几何..., dir)` | → `async_copy_*` |
| `pumem.dispatch` | `@callee(args...) {reads, writes}` | → `pumem_async_dispatch` |
| `pumem.wait` | `(handle)` | → `pumem_wait` |
| `pumem.wait_dispatch` | `()` | → `pumem_wait_dispatch` |
| `pumem.sync` | `()` | → `pumem_sync` |

保留同步 `pumem.copy`(gather/scatter 等无异步路径的场景)。

### 2.3 puc

沿用现有 C emit 方言与 pass(`to_puc` / `localize_carried` / `simplify_assign` /
`void_ptr_args` / `puc_launch` / `translate_puc_to_cpp`)。

## 3. Pass pipeline

```
P0 specialize          不变(pid_loop 仍在最外层)
P1 bufferize           conservative, copyBeforeWrite, SRAM space=128, align=64
P2 cleanup-copies      staging 提升 / 死 copy / copy→in-place / SRAM→SRAM 改写
P3 form-executables    outline + 冻结 reads/writes;matmul 归一化为覆盖语义
P4 to-pustream         cross-space memref.copy→async.copy;compute→async.dispatch;【不插 wait】
P5 pipeline            ★L1:多缓冲 + per-slot DMA wait + N/S;覆盖 k 循环与 pid 循环;【不发 wait_dispatch】
P5a body-barriers      ★L2:body 局部 compute 屏障(compute-may 前向扫描,无 N)+ 入口屏障
P5b baseline-waits     ★L3:非流水代码的单缓冲通用 wait + 退出排空(精确清除)
P6 plan-allocation     handle/meta 分配 + slab 打包 + meta 复用支配验证
P7 lower-to-pumem      memref→ptr ABI;pustream op→pumem op;wait→pumem.wait(h[slot])
P8 executable-body     CPU: linalg→runtime_call/loops;NPU: 保持 linalg
P9 emit                puc→C++
```

```python
module = self._specialize(module)          # P0
module = self._bufferize(module)           # P1
module = self._cleanup_copies(module)      # P2
module = self._form_executables(module)    # P3
module = self._to_stream(module)           # P4(语句式,无 wait)
module = self._pipeline(module)            # P5  L1:结构流水化 + per-slot DMA wait + N/S
module = self._body_barriers(module)       # P5a L2:compute 屏障 + 入口屏障
module = self._baseline_waits(module)      # P5b L3:单缓冲通用兜底 + drain
module = self._plan_allocation(module)     # P6
module = self._lower(module)               # P7 + P8
cpp = self._emit_cpp(module)               # P9
```

**关键:没有"一个全局分析包办所有 wait"。** 等待按三层产出(§4):L1 发 per-slot
DMA wait(多缓冲,知道 N),L2 发 compute 屏障(不需要 N),L3 对非流水单缓冲代码
通用兜底。v2 的通用分析机器保留在 L3 与独立校验器里,不再面对多缓冲。

P0–P3 沿用 v2 / 现有设计,本文不重复。下文聚焦 P5 起。

## 4. 三层等待架构(总览)

### 4.1 三层职责

| 层 | pass | 管什么 | 需要的知识 | 发的 wait | 可重排? |
|---|---|---|---|---|---|
| **L1** | P5 pipeliner | 跨迭代 **DMA↔DMA** hazard:预取 RAW、out slot WAR、meta 复用 | slot 映射(N、remui) | `wait %slice`(per-slot) | **否**(精确命中生产者 slot) |
| **L2** | P5a body-barriers | **涉及 compute** 的 hazard:compute→compute、DMA↔compute、入口屏障 | reads/writes(base 粒度),**不需要 N** | `wait_dispatch`(全局) | **是**(幂等) |
| **L3** | P5b baseline-waits | 非流水代码的全部 hazard(单缓冲) | base 粒度 may 标志 | `wait` + `wait_dispatch` | per-slot 否 / 屏障是 |

**为什么三分而不是二分(v3 原方案)**:v3 把 wait 全交给 pipeliner + 单缓冲局部
规则,但 pipeliner 的固定 wait 集对**多 dispatch body**(leaky_relu 的
cmpf∥mulf→select)和**循环外 compute 生产者**(slope 常量 fill)覆盖不到。若用
"通用分析兜底"二分,兜底分析会重新面对多缓冲,退回 v2 的保守。三分把通用分析
**限制在单缓冲**(L3),多缓冲区域的 compute 屏障交给不需要 N 的 L2,清楚得多。

### 4.2 ③ 的归属与双重职责(硬契约)

streaming 每轮末尾的 `wait_dispatch`(下称 ③)**归 L2**,且身兼两职:

1. **store 依赖**:S(i) 读 C(i) 写过的 out slot,host 必须先等 C(i);
2. **跨迭代 compute 屏障**:它是 C(i) 与 C(i+1) 之间唯一的全局屏障,是
   "compute 跨迭代不并发"的语义锚点,也覆盖共享 scratch 的 RAW/WAW。

**契约三条**(写进 L1/L2 接口):

- **(a)** L1 只发 per-slot DMA wait(①②),**永不发 `wait_dispatch`**。
- **(b)** L2 的规则不是"body 末尾补一条",而是:**任何消费 compute 写过(或读过)
  的 SRAM 的 DMA/sync/dispatch op 之前,若 compute-may 置位则插 `wait_dispatch`**。
  store 是该规则的一个实例;这样 store 不在 body 末尾时也不会漏,且两层不会重复发。
- **(c)** `wait_dispatch` **幂等且可重排**(全局屏障,等所有已发 dispatch),所以
  L1/L2 的先后、循环展开顺序都不影响正确性;**per-slot wait 不可重排**(必须精确
  命中生产者 slot)。这是三分能落地的关键性质。

**推论**:任何"优化掉 ③"的改动都会连带破坏跨迭代 compute 排序。③ 不是冗余,
是硬件全局屏障语义的体现。退出时的 `wait_dispatch` 才是冗余(末轮 ③ 已等过
C(K-1)),由 `pumem_sync()` drain 覆盖。

### 4.3 独立校验器的真实边界(故障隔离的诚实评估)

保留一个**永不插入、只校验**的 coverage verifier(带 per-base last-producer),
但要认清它查得到什么、查不到什么:

- **查不出 slot 映射错误**:wait 落在消费者自己的 slice 上,handle 按构造永远正确;
  即使数据被预取进了**错误 slot**,coverage 也通过。slot 映射正确性只能靠
  **数值回归 + remui 结构 lit + enqueue-order 调试器**。
- **查得出**:wait 被后续 pass 挪走/删掉;L3 兜底代码的遗漏。
- 对流水区域,**结构断言**(①②③ 的精确位置匹配 canonical 模板)比通用复算更强
  也更便宜。

所以"独立分析交叉校验 pipeliner slot bug"的故障隔离收益**有限**(v3 曾高估);
校验器的真实价值是 wait 完整性 + 兜底覆盖,不是 slot 正确性。

## 5. L1:P5 pipeliner(多缓冲 + per-slot DMA wait)

### 5.1 识别 canonical tile 循环

```mlir
scf.for %i = 0 to %K step 1 {
  %pin  = memref.subview %in[%off(%i)]  ...
  %pout = memref.subview %out[%off(%i)] ...
  pustream.async.copy %dram_in[%off(%i)], %pin          // L(i)
  pustream.async.dispatch @body(%pin, %pout, %acc?)
                          reads(...) writes(...)         // C(i)
  pustream.async.copy %pout, %dram_out[%off(%i)]        // S(i),可选
}
```

前置条件(v3.1 收紧):

- tile 不跨迭代携带(acc 归约除外);
- load 的 DRAM 偏移是 IV 的仿射函数;
- dispatch 读 load 写入的 tile(load 链);dispatch 的写被 store 读(store 链);
- **输入 staging slot 对 body 只读**:body 的 dispatch 不得写 load 目标 slot
  (输出 slot 可写、carried acc 是独立 slot)。违反者(如 masked-load 的 tail fill
  写输入 slot)先经 §6 的 H2 变换,否则不流水化。
- **DMA 骨架 canonical**:copy/dispatch/store 的发射结构是直线或统一谓词化的;
  body 内部的 compute 分支由 L2 并集处理(§7),不要求 body 分支自由。

分两型:

- **streaming**:有 S(i)(elementwise、带写回的 tile 循环)。
- **reduction**:无 S(i),acc 跨迭代携带(matmul k 循环、reduce)。

**pid 循环同形**:i=pid,K=num_pids,独立 tile(无 acc),属 streaming。
v2 只处理 pid 循环的跨迭代复用、不流水化它;本方案把 pid 循环当 tile 循环
流水化,使 elementwise 获得跨 pid 的 func 级 overlap(§5.6)。

不匹配 canonical 形状的循环:不流水化,保持单缓冲,由 L3 插 wait(§8),
正确性不变,只是没有 overlap。

### 5.2 N 与 S 的推导(含每后端成本 gate)

```
T_load    = 每轮 copy 搬运字节 / DMA 有效带宽 + 每 task 固定开销
T_store   = 每轮 store 字节   / DMA 有效带宽 + 每 task 固定开销
T_compute = 每轮 dispatch 计算量 / compute 吞吐 + 每 dispatch 固定开销

N_min = 1 + ceil(T_load  / T_compute)     // 预取要在 N-1 轮计算内完成
S_min = 1 + ceil(T_store / T_compute)     // store 槽要撑过 S-1 轮计算

N = clamp(N_min, 2, min(N_sram, N_event, K))
S = clamp(S_min, 2, min(S_sram, N_event, K))
```

- **每 task 固定开销必须计入**(v3 遗漏):enqueue + 阻塞 wait 有固定成本。
  小 tile 下它主导 T_load/T_store,纯搬运公式会给出没有收益的 N/S。
- **小 tile / 是否用 async 的 gate 是每后端运行时成本模型决策,不是编译期硬规则**:
  比较"async 流水估计成本"与"host 同步成本",低于阈值直接走同步路径(连 async
  都不用)。**CPU 上 host 自己就是执行器**(sync 模式直接 box_copy+compute,无
  task/mutex),小 tile sync 比 async serial 快两个数量级(§15);**这个结论不能
  平移 NPU**(NPU host 不是执行器,async_copy 是硬件 doorbell,固定开销小得多)。
  gate 的阈值与开销项必须按后端标定。
- `N_sram` / `S_sram`:SRAM 预算 / 每 stage staging 字节(含对齐)。与 P6 slab
  打包是同一个池:先扣除非 staged scratch,再把剩余额度在多个流水循环间按收益分配。
  **matmul 的 a/b 两条 load 链可各自推导 N**(T_load 不同),省 SRAM。
- `N_event`:仅 NPU——每 stage 活跃 handle/event 数 × N ≤ 设备上限;CPU 不设此项。
- `K`:trip count。**K<2 时不流水化**(N=S=1,无前导),避免 `clamp(x,2,hi)` 在
  hi<2 时上界小于下界的非法情形。
- **流水深度必须显著小于每线程块数 `K/thread`**(v3.1 新增 gate):否则前导
  L(0..N-2) 与入口屏障摊不掉,流水化反而更慢。实测(§15)4 block/线程配 N=8 得
  **0.85×(慢于 serial)**,单块 0.93×。故 N 上界还要加 `min(..., K/thread 的保守
  分数)`,与"是否用 async"同属每后端运行时成本模型,不是编译期硬规则。
- 前端 `num_stages` 降级为可选上界 / 覆盖(调参入口),主路径由分析推导。
- 缺设备参数 → 退化 N=S=2 + 编译期 remark(正确性不变,只是不保证隐藏);
  不允许静默取很大的 N。
- **资源模型 caveat**:公式假设 DMA 与 compute 是独立资源——对带真实 DMA 引擎的
  NPU 成立;对 CPU 模拟器**不成立**(load/store/compute 抢同一个 2–16 worker 池)。
  CPU 的"有效带宽"必须在并发 compute 负载下标定。所有估计只影响性能,不影响正确性。
- DMA-bound 时 `N_min` 会超 SRAM 上限,分析应同时给出"kernel 受限于 DMA"的
  编译期诊断(建议缩 tile),而不是继续加 N。

### 5.3 结构变换(四步)

1. **多缓冲**:SRAM tile 加 stage 维(in 取 N,out 取 S),alloc 打 `pustream.stages`。
   masked-load 经 §6 变换后可能 +1 专用 tail slot。
2. **stage 算术**:所有访问改 `alloc[stage]` subview;预取偏移改 `off(i+N-1)`。
3. **前导**:编译期展开 L(0..N-2),每条按 `s < K` 谓词化,入 slot 0..N-2。
   **前导不插 wait**(slot 全新,无在飞生产者;P6 保证初始 meta 全新,§10)。
4. **谓词主循环**:`sc=i%N, sp=(i+N-1)%N, sn=i%S`;预取 L(i+N-1) 入 in[sp],
   按 `i+N-1 < K` 谓词化。**无 drain 尾声**,trip count 任意(K<N、非整除都正确)。
   (masked-load 的 partial tail 处理见 §6,不是这里的 drain 尾声。)

```mlir
// pipeliner 产出骨架(per-slot DMA wait 见 §5.4;compute 屏障由 L2 后插)
%in  = memref.alloc() : memref<N x TILE, 128> {pustream.stages = N}
%out = memref.alloc() : memref<S x TILE, 128> {pustream.stages = S}

// 前导 L(0)..L(N-2),谓词化,无 wait
%ok0 = arith.cmpi slt, %c0, %K
scf.if %ok0 {
  %p0 = memref.subview %in[0] ...
  pustream.async.copy %dram_in[%off(0)], %p0
}
// ... s = 1 .. N-2 同模式

scf.for %i = 0 to %K step 1 {
  %sc    = arith.remui %i, N
  %sp    = arith.remui %i + N - 1, N
  %inext = arith.addi %i, N - 1
  %has   = arith.cmpi slt, %inext, %K
  scf.if %has {
    %pp = memref.subview %in[%sp] ...
    pustream.async.copy %dram_in[%off(%inext)], %pp      // L(i+N-1)
  }
  %pin  = memref.subview %in[%sc] ...
  %pout = memref.subview %out[%i mod S] ...
  pustream.async.dispatch @body(%pin, %pout) reads(...) writes(...)
  pustream.async.copy %pout, %dram_out[%off(%i)]         // S(i)
}
```

slot 旋转只是 `remui` 下标 + 多缓冲 alloc,**不用 `scf.index_switch`**
(ScfToPumem 只认 For/If/Yield),也不用 `arith.select`(memref one-to-N 塞不进
ArithToPuarith)。动态 stage 下标走 expand-strided-metadata(§11)。
(§6 专用 tail slot 的谓词化下标是例外,见该节。)

### 5.4 per-slot DMA wait 发射(L1 只发这些)

pipeliner 按它已知的 slot 映射,在结构变换时发出 **per-slot DMA wait**。
**L1 不发 `wait_dispatch`**(那是 L2 的事,§4.2 契约 a)。

**streaming(有 store):**

```mlir
scf.if %has {                                  // 预取 L(i+N-1)
  %pp = memref.subview %in[%sp] ...
  pustream.async.copy %dram_in[%off(%inext)], %pp        // 无 wait
}
%pin = memref.subview %in[%sc] ...
pustream.wait %pin                             // ① RAW:等 L(i)
%pout = memref.subview %out[%sn] ...
pustream.wait %pout                            // ② WAR:等 S(i-S) 读完 out slot
pustream.async.dispatch @body(%pin, %pout) reads(...) writes(...)
// L2 在此插 wait_dispatch(③,store 消费 compute 写过的 out)
pustream.async.copy %pout, %dram_out[%off(%i)] // 无 wait
```

**reduction(无 store,acc 携带):**

```mlir
// L2 在此插 wait_dispatch(① 跨迭代 compute 屏障:预取 WAR + acc RAW)
scf.if %has {
  %pp = memref.subview %in[%sp] ...
  pustream.async.copy %dram_in[%off(%inext)], %pp        // 无 wait
}
%pin = memref.subview %in[%sc] ...
pustream.wait %pin                             // ② RAW:等 L(i)
pustream.async.dispatch @body(%pin, %acc) reads(...) writes(...)
```

**为什么预取与 store 前不需要 DMA wait——pipeliner 知道 slot 映射,故精确:**

- **streaming 预取** L(i+N-1) 覆盖 in[(i-1)%N]:
  - meta 复用生产者 L(i-1)(该 slot 的上一笔 load)已被**上一轮 ① wait %pin** 等过;
  - WAR 生产者 C(i-1)(该 slot 的最后一个 compute 读者)已被**上一轮 ③(L2)** 等过。
  - 故预取前无需任何 DMA wait(compute WAR 由 L2 的 compute-may 规则覆盖,而
    streaming 上一轮 ③ 已清 compute-may,本轮预取时屏障不需要——§7)。
- **reduction 的跨迭代屏障提到预取之前**:因为无 store、无 ③,C(i-1) 在上一轮
  barrier **之后**才发射,本轮预取时仍在飞。L2 在预取前插 `wait_dispatch`,同时
  覆盖预取 WAR(等 C(i-1) 读完 in[(i-1)%N])与 acc RAW(等 C(i-1) 写完 acc)。
- **store S(i) 前**:meta_out[sn] 的上一笔 S(i-S) 已被**本轮 ② wait %pout** 等过,
  故 S(i) 前无需 DMA wait(store 的 compute RAW 由 L2 ③ 覆盖)。
- **dispatch C(i) 写 out[sn] 的 WAW**(上一轮写者 C(i-S)):C(i-S) 已被第 i-S 轮
  ③(L2)等过,故无需额外 wait。

每轮 wait 计数:streaming **2 条 per-slot DMA wait(①②)+ L2 的 ③**;
reduction **1 条 per-slot DMA wait(②)+ L2 的 ①**。多 dispatch body 的 L2 屏障
另计(§7)。零 no-op DMA wait。

**退出排空**:末轮 ③ 已等 C(K-1),退出 `wait_dispatch` 冗余;残留末轮 store 由
`wait %out[(K-1)%S]` + `pumem_sync()` 覆盖,实现上整体折成 `pumem_sync()`。
reduction 循环后 acc 写回 DRAM 的 store 由 L2/L3 在其前插 `wait_dispatch`(确保
acc 终值),退出 `pumem_sync()`。

### 5.5 为什么 per-slot DMA wait 不需要独立全局分析

v2 的 base 粒度 may 标志 + 不动点 + dma 永不清,是为了让一个**独立分析**在
**不知道 N** 的前提下 sound 地插 DMA wait。代价是每次访问补一条 no-op wait。

本方案把这部分交给 pipeliner,理由是一个结构性事实:

> **多缓冲只可能来自 pipeliner。** 单缓冲代码一个 buffer 一个 handle,handle-wait
> 本身精确(L3)。多缓冲代码的 slot 映射只有 pipeliner 知道——而 pipeliner 正是
> 制造多缓冲的人,它在算 remui 时就已经知道每个消费点对应哪个生产者 slot。让它
> 顺手发出精确 per-slot DMA wait,比让另一个分析去反推 slot 身份更简单、更精确。

v2 担心的"prologue 用常量下标 `subview %in[0]`、主循环用 `subview %in[%sp]`,
表达式级分析会漏同一 runtime slot"——在这里不存在:两段都是 pipeliner 自己写的,
它知道是同一 slot,无需任何表达式等价推理。**"DMA wait 分析不需要知道 N"
被平凡满足,因为根本没有独立的 DMA wait 分析。** compute 屏障则确实不需要 N
(L2 用全局 compute-may,§7),通用分析降级到单缓冲 L3。

### 5.6 pid 循环流水化(elementwise 的 overlap)

elementwise 没有 k 循环,它的 tile 循环就是 pid 循环。要让它有 func 级 overlap,
必须把 pid 循环当 canonical streaming 循环流水化(多缓冲跨 pid、预取 L(pid+N-1))。

现有 pid 循环是步长=num_threads 的 strided 循环:

```mlir
scf.for %pid = %start to %num_pids step %num_threads { ... }
```

pipeliner 先改写成计数循环,slot 用迭代序号而非 pid:

```mlir
%K = ceildiv(%num_pids - %start, %num_threads)
scf.for %i = 0 to %K step 1 {
  %pid = %start + %i * %num_threads
  %sc  = arith.remui %i, N          // 用 i,不是 pid
  ...
  // 预取迭代 i+N-1 → pid_next = %start + (%i+N-1)*%num_threads
  %has = arith.cmpi slt, %pid_next, %num_pids
  ...
}
```

其余与 §5.3/§5.4 streaming 完全一致。这样 elementwise 稳态下 C(pid) 与
L(pid+N-1)、S(pid-1) 并发,获得跨 pid 的 func 级 overlap。

> **非整除验证**:`K=ceildiv(...)` 的非整除 num_pids 必须用现有 large-grid pid
> 测试套件(`test/cases` 大 grid 用例)验一遍,确保计数改写 + 谓词化边界正确。
>
> 跨 pid 并发的前提是缓冲不冲突。多缓冲本身保证 in/out slot 跨迭代不踩;若 P6
> slab 打包让不同 pid 的 buffer 复用同一物理区间,需保证活跃区间不重叠(I4),
> 否则引入真实 WAR——那是打包问题,不是流水化问题。

### 5.7 overlap 稳态时间线

```
host:  [发 L(i+N-1)] [wait L(i)] [wait S(i-S)] [发 C(i)] [wait C(i)] [发 S(i)]
load:        [ L(i+N-1) .................. ]
store:  [ S(i-1) ...... ]        [ S(i) ...... ]
compute:     [ C(i) ........................ ][ C(i+1) ... ]
```

host 阻塞在 C(i) 期间,worker 并发跑 C(i)、L(i+N-1)、S(i-1) → func 级
compute↔copy overlap。背压点两个:

- load 吞吐跟不上 N-1 预取深度 → `wait %pin` 阻塞;
- store 吞吐跟不上 S 宽限 → `wait %pout` 等 S(i-S) 未完。

这正是 §5.2 推导 N/S 的直接依据。overlap 的真实载体是同一个线程池(2–16 worker),
排队等前驱的 copy task 不占 worker;三者完全并发需 ≥3 worker。

## 6. H2:masked-load 与 OOB-live 判据

### 6.1 问题

corpus 的点核几乎全是 masked load(relu/leaky_relu/elu/sigmoid…,BLOCK=256):
partial tile 先 `fill(slot, pad)` 再 `copy(valid prefix)`,fill→load 是语义顺序。
一旦预取把 L(i) 提前 N-1 轮入队,body 内第 i 轮的 fill 要么清零已加载数据、要么
与在飞 load 竞争——**无法在保持预取的同时保证每轮 fill→load 顺序**。这不只是
wait 问题,是**流水化资格**问题。v3 的 canonical 前置太松(没排除 body 写输入
slot),v3.1 收紧(§5.1)并给出变换。

### 6.2 通用判据:OOB 是否 live

**hoist fill 到 pre-header 覆盖全部输入槽**能否成立,取决于:

> **没有任何被 store 的 lane 依赖 load 区之外的元素。**

- **elementwise** 是它退化成 `store_mask ⊆ load_mask` 的特例:store 只写 load 过
  的 lane,body 逐元素无跨 lane 归约 → OOB 输入值在 store 处**死亡**。
- **方向不能反**:若 store 写了未 load 的 lane,填充值变成可见输出,而复用槽尾是
  stale 数据 → hoist 直接错,必须 per-tile fill。

**两个反例**(OOB live,hoist-to-N 不 sound):

1. **归约:matmul masked-K**。zero pad 循环不变,但 OOB K 元素被**求和**(非死亡),
   hoist 把上一个 full tile 的 stale 数据求和进结果 → 错。
2. **邻域:stencil/halo**。pad 区经邻域影响 in-bounds 输出,fill 值 live → 错。

**关键纠正(v3 评审中发现)**:hoist-to-N 下"槽尾恒为填充值"**不成立**——full tile
的 load(ext=B)写满整槽,覆盖 prologue 的零;tail 复用该槽时 OOB 区是上一个 full
tile 的残留。elementwise bench err=0 的真正原因是 **store extent == load extent,
OOB lane 在 store 处死亡**,不是槽尾为零。

### 6.3 H2 决策树

| 情形 | 变换 | 说明 |
|---|---|---|
| **OOB 死**(elementwise,store⊆load) | hoist fill 到 N 槽(冗余无害),或直接消除 fill | fill 其实是死代码;hoist 比消除简单(无需证明 OOB 死)。每迭代零成本,尾块仍在流水里 |
| **OOB 活 + 后缀 remainder**(单尾块:masked-K 归约 / 1D stencil) | **专用 tail slot**(§6.4) | full tile 从不写它 → OOB 恒为 pad → 归约/邻域正确;async 尾处理复用 body |
| **非后缀**(2D 每块 masked) | per-tile fill,或按维不流水化 | 没有单一"尾块",专用 slot 不适用 |

### 6.4 专用 tail slot(让"槽尾恒为 pad"真正成立)

给尾块一个**不进 N 轮转的专用 slot**(下标 N),入口零化一次:

- full tile 只用旋转槽 0..N-1,**从不写专用 tail slot** → 它的 OOB 区恒为入口
  填充值(这才是"槽尾恒为 pad"成立的条件,hoist-to-N 做不到)。
- tail load 只写有效前缀,OOB 保持 pad → masked-K 归约 OOB 贡献 0,正确;
  1D stencil 的 halo pad 同理。
- 入口 fill 是 compute dispatch,由 **L2 的入口屏障**(§7,H1)顺带 drain,tail load
  入队时 fill 早已完成,无 fill→load 竞争。
- slot 下标谓词化:`slot = select(i == last, N, i % N)`(这是 §5.3 "不用 select"
  的例外——此处 select 的是**整数下标**不是 memref,塞得进 ArithToPuarith)。
- 代价:**+1 输入槽** + 谓词化下标;tail handle/meta 独立(P6 一 buffer 一 handle),
  只用一次无复用。
- 尾处理可复用同一 async body(L3 单缓冲精确 wait),不必退化成标量实现。

**适用边界**:仅后缀 remainder(单尾块)。2D 每块 masked 不适用,那类只能 per-tile
fill 或按维不流水化。

### 6.5 入口屏障(H1)

body 读取的常量 SRAM(如 leaky_relu 的 slope broadcast zero/hundredth)由**循环外
dispatch** 产生。streaming 的 per-slot wait 不含入口屏障,C(0) 无屏障开跑会读到
在飞 fill 输出。修:**pre-header 一道 `wait_dispatch`**(fill 之后、loop 之前),
一次屏障零每轮成本。本质是 L2 把 compute-may 从 pre-loop 播种(§7)。reduction 的
跨迭代 ① 在 i=0 恰好覆盖,无此问题。

## 7. L2:P5a body 局部 compute 屏障

### 7.1 规则(compute-may 前向扫描,不需要 N)

L2 在 body(及 pre-header)上维护一个 **compute-may 标志**:

- `async.dispatch` → 置 compute-may(记 reads/writes 的 base)。
- `wait_dispatch` → 清 compute-may。
- **规则 (b)**:任何 op 若与在飞 compute 有 hazard——
  - DMA/sync **读** compute 写过的 SRAM(RAW),
  - DMA **写** compute 读过的 SRAM(WAR),
  - dispatch 读/写 compute 读/写过的 SRAM(RAW/WAR/WAW)——
  且 compute-may 置位,则在其前插 `wait_dispatch`(并清标志)。

这是 base 粒度 may 标志 + 全局清除,**不需要 N、不需要 slot 映射**。它统一解释:

- **streaming store 前的 ③**:store 读 compute 写过的 out → RAW → 屏障。
- **streaming 预取前无屏障**:预取写 in[(i-1)%N](C(i-1) 读过,WAR),但上一轮 ③
  已清 compute-may → 本轮预取时标志清,不插。
- **reduction 预取前的 ①**:无 store 无 ③,C(i-1) 仍在飞 → compute-may 置位 →
  预取(WAR)前插屏障,同时覆盖 acc RAW。
- **入口屏障(H1)**:pre-loop fill 置 compute-may,body 首个读常量 SRAM 的 op
  前插屏障;实现上 hoist 到 pre-header 一道。

### 7.2 多 dispatch body 分组(红利)

L2 不只串行化,还把**独立 dispatch 分组并发**。leaky_relu body:

```mlir
pustream.async.dispatch @cmpf(%pin, %zero, %mask)   // 写 mask
pustream.async.dispatch @mulf(%pin, %hundredth, %scaled)  // 写 scaled,与 cmpf 独立
pustream.wait_dispatch                              // L2:select 消费 mask/scaled
pustream.async.dispatch @select(%mask, %pin, %scaled, %pout)  // 写 out
pustream.wait_dispatch                              // L2 ③:store 消费 out
```

cmpf∥mulf 同组(无共享写)→ 一道屏障;select 后 ③。共 2 道 compute 屏障。
实测(§15)leaky_relu 的吞吐主要由**屏障数**决定(serial ~4 道/轮 vs overlap 2 道),
L2 分组是真实收益。

### 7.3 幂等可重排(契约 c)

`wait_dispatch` 等所有已发 dispatch,幂等;多插一道只是冗余 no-op,不影响正确性。
所以 L1(per-slot)与 L2(屏障)的发射先后、循环展开顺序都无关紧要。per-slot
wait 则必须精确命中生产者 slot,不可重排。这是三层能独立实现、组合仍 sound 的
关键性质。

## 8. L3:P5b 单缓冲通用兜底(非流水代码)

对 P5 未流水化的代码(非 canonical 循环、直线 staging copy、循环后写回、§6 的
per-tile fill 尾块),用**单缓冲通用分析**:一个 buffer 一个 handle,handle-wait
精确。前向扫描,每个 op 前:

| op | 前置 wait | 状态更新 |
|---|---|---|
| `async.copy` | 读/写 buffer 有在飞 DMA 生产者 → `wait` 其 handle;有在飞 compute 生产者 → `wait_dispatch` | 置 DMA 于 src/dst |
| `async.dispatch` | reads/writes 有在飞 compute 生产者 → `wait_dispatch`;有在飞 DMA 生产者 → `wait` 相应 handle | 置 compute 于 reads∪writes |
| 同步 op(gather/scatter/残留 memref.copy) | 同上消费 hazard | 自身同步执行,**不置在飞标志** |
| `wait` / `wait_dispatch` | — | 清相应精确标志 |

单缓冲下"每次访问前 wait 所触 handle"恒 sound(handle 粒度精确),**wait 可精确
清除标志**(不像 v2 多缓冲 base 粒度"dma 永不清")。这是 v2 通用分析的降级保留:
机器还在,但只面对单缓冲,不再有多缓冲保守。

退出:对残留标志插 `wait` + `wait_dispatch`,可折成 `pumem_sync()`。

> 同步 op 规则是 v2 §13.3 遗漏的:ToPuStream 实际把同 space 的 memref.copy 留为
> 合法(`ToPuStreamPass.cpp:164-168`),gather/scatter 也无异步路径。它们消费
> hazard(需 wait)但自身同步执行、不产生在飞 op,必须显式定规则。
>
> §6 专用 tail slot 的尾处理走 L3:单缓冲精确 wait,复用 async body。尾声若复用
> pipelined 的 handle/meta 必须先 wait(否则撞 I7 busy abort);简单做法是尾声用
> 新 handle 或 host box_copy。

## 9. barrier 摊销(streaming 可选优化)

每轮多次 host 调用(enqueue + 阻塞 wait)在 compute 只有几十 µs 时占比很高。
按 U 个 tile 展开:**U 次 dispatch + 一次 `wait_dispatch` + U 次 store**,比抠掉
一条 no-op wait 收益大得多(§15:精确性是噪声,屏障数才是杠杆)。

**安全条件(两条都要)**:

1. **out slot 互不重叠、store 间无 WAR**:S ≥ U(或 U 个 store 直接读独立 slot),
   否则仍需逐槽 ②。
2. **U 个 dispatch/store 共享的所有可变 scratch/carried 必须 ×U 多缓冲或不存在**:
   - relu(无 scratch、无 carried)→ 可自由摊销;
   - leaky_relu 的 mask/scaled 是单缓冲 scratch,U 组之间 cmpf 写 mask vs 前组
     select 读 mask 会 WAR → 摊销需 **scratch×U**;
   - matmul 的 acc carried 不可多缓冲(归约依赖)→ **不可摊销**。

所以摊销不是"streaming-only"这么粗,而是按共享可变状态判定。它是 N/S/流水化
之外的一个独立选项,进成本模型权衡(SRAM ×U vs host 调用数)。

## 10. P6 handle/meta 分配 + 打包

- 每个参与 DMA 的 SRAM buffer 一对 handle+meta;`stages=N` → 数组 `h[N], m[N]`。
  handle 数组按 **base alloc** 键控,slot index 对应数组下标。
- **一 buffer 一 handle 一 meta,初始全新,按方向配**(硬前置):
  - 初始 meta 全新(busy=false)→ 前导/首轮预取无 wait 才成立(§5.3);
  - 输入/输出 slot 分离(本方案默认),或同 buffer 按方向配 meta——**共用一个 meta
    会让 load 入队 busy=true 后 store 入队直接 abort**。`pumem_wait` 两方向等待,
    共用 handle 保守但 sound,共用 meta 则 abort。
  - 违反"一 buffer 一 handle"会让 L3 的精确清除退化成 v2 那个不 sound 的清标志。
- compute-only buffer(acc)平时不需 DMA handle;作为 store src 时分配。
- §6 专用 tail slot 分配独立 handle/meta(下标 N),只用一次。
- DRAM 侧暂不分配 handle。同 kernel 内同一 DRAM 区间先被 store 写、后被 load 读
  的形态,由 verifier **base 级**拒绝(会连带拒绝合法 in-place,列为未来 case)。
  扩展路径:给既读又写的 DRAM base 也分配 handle,store 记 last_write、load 记
  last_read,wait 操作数规则放开到"两侧带 handle 就发 wait"——分析天然覆盖,
  只是 P6 分配策略缺口,不动 IR 语义。
- **meta 复用支配不变量(I7,硬约束)**:每次 meta 复用必须被其 handle 上的
  `wait` 支配。L1 精确 wait 已保证(§5.4 论证);verifier 用 per-handle 精确
  "meta 在飞"标志(wait 可清)复算。这与 hazard 排序是**两个独立关注点**:
  hazard 排序关心"冲突生产者是否完成",meta 复用关心"装箱槽位是否空闲";
  前者由 wait 覆盖,后者由 wait 精确清除。runtime 入队 abort 是硬正确性约束,
  此 verifier 常开。
- slab 打包沿用 v2 P7:每 space 单 slab、常量 offset、identity-layout `memref.view`、
  I4 断言;需新增 ViewOpLowering(`memref.view` 无现成 lowering)。多缓冲 slot 是
  连续子区间,handle 数组下标 ↔ slot offset 一致(I4 扩展)。

## 11. P7 pustream → pumem

memref → ptr ABI 元组 (base, offset, sizes, strides),复用现有
expand-strided-metadata / memref_expand / lower_affine / memref_to_pumem 链路。

| pustream | pumem | 说明 |
|---|---|---|
| `async.copy %src,%dst` | `async_copy h[slot], m[slot], dst_tuple, src_tuple, geom, dir` | slot 由 remui → `dma_slot_at` |
| `async.dispatch @f(%a)` | `dispatch @f(tuple_a) {reads,writes}` | 签名 → ptr tuple |
| `wait %slice` | `wait h[slot]` | slice → base alloc + index → handle 数组下标 |
| `wait_dispatch` | `wait_dispatch()` | |
| 退出排空 | `sync()` | 折叠残留 wait |

**关键 lowering:`wait %slice` → `wait h[slot]`。** slice 是 `subview %alloc[%idx]`,
经 viewBase 取 base alloc + 提取 index,映射到该 base 的 handle 数组下标。
动态 index 走 `pumem.dma_slot_at(h_array, %idx)`。

**动态 stage offset 的下游**(remui 方案的命门):`memref.subview %alloc[%dyn]`
经 expand-strided-metadata 应把运行时 stage offset 折进 offset 元组;现有链路
已处理 k 相关 DRAM 动态偏移,大概率通。**M3 前先拿 N=3 动态下标 lit 验穿这条
路径**;万一不 fold,回退是 N 个常量 offset subview + scf.if 选择(N>2 又撞
index_switch,需给 ScfToPumem 补 pattern)。§6 专用 tail slot 的 `select(i==last,
N, i%N)` 整数下标同样走这条路。

executable 签名 → (base ptr, offset, sizes, strides) tuple,只在 P7 一处 materialize。

## 12. P8/P9 pumem → puc → C++

P8 executable 体:CPU 分支 `linalg_to_runtime_call`(matmul/transpose →
runtime_matmul)/ lower_linalg_reduce / convert_linalg_to_loops / scf_to_pumem /
arith_to_puarith / math_to_pumath;NPU 分支保持 linalg。

P9 设备接口发射:

| pumem | C++ |
|---|---|
| `async_copy` | `async_copy_dram_to_sram<Rank>(h, m, dst, src, elem, sizes..., strides...)` |
| `dispatch` | `pumem_async_dispatch(callee, args...)` |
| `wait` | `pumem_wait(h)` |
| `wait_dispatch` | `pumem_wait_dispatch()` |
| `sync` | `pumem_sync()` |
| `dma_slot_array` / `dma_slot_at` | `pumem_dma_handle_t h_in[N];` / `h_in[sc]` |

handle/meta 用 `puc.local` / `puc.assign` 落成 kernel 局部变量;
`translate_puc_to_cpp` 按 runtime_backend 选符号。编译链接需带
`pumem_async_runtime.cpp` 与 `-pthread`(见 `backend/compiler.py` 现状)。

## 13. Worked example

### 13.1 matmul k 循环(reduction,N=2)

P5(L1)+ P5a(L2)后 pustream IR:

```mlir
%a = memref.alloc() : memref<2 x MxKxf32, 128> {pustream.stages = 2}
%b = memref.alloc() : memref<2 x KxNxf32, 128> {pustream.stages = 2}
// acc 单缓冲,compute-only(写回时分配 handle)

%pa0 = memref.subview %a[0] ...;  %pb0 = memref.subview %b[0] ...
pustream.async.copy %dram_a[%c0], %pa0          // 前导 L(0),无 wait
pustream.async.copy %dram_b[%c0], %pb0

scf.for %k = %c0 to %K step %c1 {
  %sc = arith.remui %k, 2
  %sp = arith.remui %k + 1, 2
  %k1 = arith.addi %k, %c1
  %has = arith.cmpi slt, %k1, %K
  pustream.wait_dispatch                        // L2 ① C(k-1):预取 WAR + acc RAW
  scf.if %has {
    %pn = memref.subview %a[%sp] ...;  %qn = memref.subview %b[%sp] ...
    pustream.async.copy %dram_a[%k1], %pn       // L(k+1),无 wait
    pustream.async.copy %dram_b[%k1], %qn
  }
  %pc = memref.subview %a[%sc] ...;  %qc = memref.subview %b[%sc] ...
  pustream.wait %pc                             // L1 ② L_a(k) RAW
  pustream.wait %qc                             // L1 ② L_b(k) RAW
  pustream.async.dispatch @k_step(%pc, %qc, %acc) reads([0,1,2]) writes([2])
}
pustream.async.copy %acc, %dram_out             // 写回(L2/L3 前置 wait_dispatch)
// 退出 pumem_sync()(末轮 wait_dispatch 冗余,折 sync)
```

P9 后 C++:

```cpp
pumem_dma_handle_t h_a[2], h_b[2], h_acc;
pumem_dma_meta_t   m_a[2], m_b[2], m_acc;
// a[2][M*K], b[2][K*N], acc[M*N]  SRAM
async_copy_dram_to_sram<2>(h_a[0], m_a[0], a[0], dram_a + k0, ...);   // 前导
async_copy_dram_to_sram<2>(h_b[0], m_b[0], b[0], dram_b + k0, ...);
for (int64_t k = 0; k < K; ++k) {
  int64_t sc = k % 2, sp = (k + 1) % 2;
  pumem_wait_dispatch();                        // L2 ① C(k-1)
  if (k + 1 < K) {
    async_copy_dram_to_sram<2>(h_a[sp], m_a[sp], a[sp], dram_a + k1, ...);
    async_copy_dram_to_sram<2>(h_b[sp], m_b[sp], b[sp], dram_b + k1, ...);
  }
  pumem_wait(h_a[sc]);                          // L1 ② L_a(k)
  pumem_wait(h_b[sc]);                          // L1 ② L_b(k)
  pumem_async_dispatch(k_step, a[sc], b[sc], acc);   // C(k): acc += a*b
}
async_copy_sram_to_dram<2>(h_acc, m_acc, dram_out, acc, ...);  // 前置 wait_dispatch
pumem_sync();
```

slot 演算(N=2):前导 L(0) 入 slot 0;k=0 预取 L(1) 入 slot 1、消费 slot 0;
k=1 预取 L(2) 入 slot 0(WAR 的 C(0) 已被 k=0 的 wait_dispatch 等过)、消费 slot 1。
互不踩,meta 复用均被前序 wait 隔开。masked-K 时按 §6.4 给 a/b 各加专用 tail slot。

### 13.2 leaky_relu pid 循环(streaming,N=S=2,含 fill)

corpus kernel(`test/flaggems/test_activation.py:24`):masked load →
`where(a>0, a, a*slope)`(cmpf∥mulf→select)→ masked store。slope broadcast 成
常量 SRAM(zero/hundredth),循环外 fill 产生。OOB 死(store⊆load),用 §6.3
hoist fill。

P5+P5a 后 pustream IR(计数循环改写后):

```mlir
%in  = memref.alloc() : memref<2 x TILExf32, 128> {pustream.stages = 2}
%out = memref.alloc() : memref<2 x TILExf32, 128> {pustream.stages = 2}
%zero = memref.alloc() : memref<TILExf32, 128>      // slope 常量,单缓冲
%hundredth = memref.alloc() : memref<TILExf32, 128>
%mask = memref.alloc() : memref<TILExi8, 128>       // scratch,单缓冲
%scaled = memref.alloc() : memref<TILExf32, 128>

pustream.async.dispatch @fill(%zero, 0.0)           // 循环外常量
pustream.async.dispatch @fill(%hundredth, 0.01)
pustream.async.dispatch @fill(%in, 0.0)             // hoisted tail padding(OOB 死,冗余无害)
pustream.wait_dispatch                              // L2 入口屏障(H1):drain 全部 fill

scf.if %start_lt_num_pids {                         // 前导 L(start),无 wait
  %p0 = memref.subview %in[0] ...
  pustream.async.copy %dram_in[%off(%start)], %p0
}
scf.for %i = 0 to %K step 1 {                       // pid = start + i*step
  %sc  = arith.remui %i, 2
  %sp  = arith.remui %i + 1, 2
  %pid = ...;  %pnext = %pid + %step
  %has = arith.cmpi slt, %pnext, %num_pids
  scf.if %has {
    %pp = memref.subview %in[%sp] ...
    pustream.async.copy %dram_in[%off(%pnext)], %pp  // 预取,无 wait
  }
  %pin = memref.subview %in[%sc] ...
  pustream.wait %pin                                // L1 ① RAW L(i)
  %pout = memref.subview %out[%sc] ...
  pustream.wait %pout                               // L1 ② WAR S(i-2)
  pustream.async.dispatch @cmpf(%pin, %zero, %mask)
  pustream.async.dispatch @mulf(%pin, %hundredth, %scaled)
  pustream.wait_dispatch                            // L2:select 消费 mask/scaled
  pustream.async.dispatch @select(%mask, %pin, %scaled, %pout)
  pustream.wait_dispatch                            // L2 ③:store 消费 out
  pustream.async.copy %pout, %dram_out[%off(%pid)]  // S(i),ext=有效前缀,无 wait
}
// 退出 pumem_sync()(末轮 ③ 已等 C(K-1),残留 store 折 sync)
```

P9 后 C++(稳态循环体):

```cpp
pumem_dma_handle_t h_in[2], h_out[2];
pumem_dma_meta_t   m_in[2], m_out[2];
// in[2][TILE], out[2][TILE], zero/hundredth/mask/scaled  SRAM
pumem_async_dispatch(fill_kernel, zero, 0.0f, TILE);
pumem_async_dispatch(fill_kernel, hundredth, 0.01f, TILE);
pumem_async_dispatch(fill_kernel, in_all, 0.0f, 2 * TILE);  // hoisted padding
pumem_wait_dispatch();                            // 入口屏障
if (start < num_pids)
  async_copy_dram_to_sram<1>(h_in[0], m_in[0], in[0], dram_in + off(start), ...);
for (int64_t i = 0; i < K; ++i) {
  int64_t sc = i % 2, sp = (i + 1) % 2;
  int64_t pid = start + i * step;
  int64_t ext = std::min(TILE, n - pid * TILE);
  if (pid + step < num_pids)
    async_copy_dram_to_sram<1>(h_in[sp], m_in[sp], in[sp], dram_in + off(pid + step), ...);
  pumem_wait(h_in[sc]);                           // L1 ① L(i)
  pumem_wait(h_out[sc]);                          // L1 ② S(i-2) WAR
  pumem_async_dispatch(cmpf_kernel, in[sc], zero, mask, TILE);
  pumem_async_dispatch(mulf_kernel, in[sc], hundredth, scaled, TILE);
  pumem_wait_dispatch();                          // L2 select 依赖
  pumem_async_dispatch(select_kernel, mask, in[sc], scaled, out[sc], TILE);
  pumem_wait_dispatch();                          // L2 ③ store 依赖
  async_copy_sram_to_dram<1>(h_out[sc], m_out[sc], dram_out + off(pid), out[sc], ..., ext, ...);
}
pumem_sync();
```

OOB lane 被 masked store(ext)丢弃,故 hoisted fill 冗余但无害;err=0 的真实原因
是 store extent == load extent,不是槽尾恒零(§6.2)。稳态下 host 阻塞 C(i) 期间,
worker 并发跑 C(i)、L(i+1)、S(i-1),elementwise 由此获得跨 pid 的 func 级 overlap。
此形状与 `test/overlap-bench/leaky_relu_bench.cpp` 一致。

## 14. 不变量与验证

| 编号 | 内容 |
|---|---|
| I2 | 拷贝几何:sizes/strides/elem_bytes 与 src/dst 类型一致 |
| I3 | dispatch 效应:reads/writes 与 callee 冻结效应一致 |
| I4 | 打包:活跃区间不重叠、DMA 对齐、slab ≤ SRAM;**扩展:handle/meta 数组长度 = stages,slot 下标 ↔ offset 一致** |
| I5 | 仅 NPU:每 stage 活跃 event/handle 数 × N ≤ 设备上限;CPU 不设限 |
| I6 | queue 语义:copy 方向与 space 一致;DRAM↔SRAM 之外禁止;同方向全局 FIFO 是语义,不检查 |
| **I7** | **meta 复用支配:每次 meta 复用被其 handle 上的 wait 支配**(per-handle 精确标志,wait 可清;硬约束,常开) |
| **I8(重定位)** | **wait 完整性**:流水区域用**结构断言**(①②③ 精确位置匹配 canonical 模板);兜底区域用 **base 级 coverage**(带 last-producer)。**不是 slot 映射正确性证明** |

**I8 的诚实定位**(v3 曾高估):coverage 查不出 slot 映射错误——wait 落在消费者
自己的 slice 上,handle 按构造正确,数据进错 slot 也通过。slot 映射正确性靠
**数值回归 + remui 结构 lit + enqueue-order 调试器**。I8 的真实价值是查 wait 被
后续 pass 挪/删 + 兜底遗漏。v2 I1' 用 base 粒度 may 标志复算、不记 producer 身份,
退化成同义反复;本方案流水区域用结构断言(更强更便宜),兜底用带 last-producer
的 base coverage(可独立验)。

辅助手段(可选 debug):模拟 runtime 记录 enqueue 顺序,wait 时检查被等事件
已入队,把顺序错误在 CPU 上暴露,而不是只靠数值回归。

## 15. 实测证据(CPU 模拟器)

> **模拟器模型(已按 §0 重构)**:runtime 由"2–16 worker 共享池 + 任务 DAG"改为
> **1 load 执行体 + 1 store 执行体 + 1 compute 执行体(内部 W 条 lane,默认 W=2)**,
> host 线程只负责发射与 wait。同方向 copy 由执行体自身串行,不再用依赖边模拟 FIFO。
> 这与 §0 的"一个发射 core + 三个执行体、执行体之间无设备侧依赖边"一一对应,但前提
> 变了,旧数字不可直接比较(以下为 3–5 轮取中位数):
>
> - **copy 不再与 compute 抢 worker** → 摊销(§9)的收益变大:leaky_relu
>   U=1/2/4 = 1.77×/3.10×/4.39×(旧共享池 1.71/2.65/2.51),relu U=1/2/4 = 2.40×/4.35×/4.39×。
> - **compute lane 数影响的是"同组 dispatch 能否并发"**:W=1 时 leaky_relu 的
>   cmpf/mulf 退化成串行,u=1 为 1.53×(W=2 时 1.77×);摊销到 u≥2 后 W=1 与 W=2
>   基本重合(4.19× vs 4.39×),因为那时每组的 barrier 已经很少。corpus 形状
>   (block=256)W=2 为 1.66×、block=4096 为 1.54×、reduce_sum 为 1.97×。
>   W 是设备 compute 执行体内部的 lane 数,**不是编译期假设**,标定时贴设备。
> - 结论 1/3/4 的数字均在旧共享池下测得;结论 5/6 已按新模型重测。

资产:`test/overlap-bench/overlap_bench.cpp`(单 dispatch elementwise)、
`test/overlap-bench/leaky_relu_bench.cpp`(多 dispatch + fill,镜像 corpus)、
`test/overlap-bench/reduce_sum_bench.cpp`(masked-K 归约,镜像
`test/cases/test_reduction_fill.py`,含专用 tail slot 与错误用法 `--naive-hoist`)、
`test/overlap-bench/amortize_bench.cpp`(§9 barrier 摊销:U 展开 / scratch×U /
`--shared-scratch` 错误用法)。
`run.sh` / `run-leaky-relu.sh` / `run-reduce-sum.sh` / `run-amortize.sh` 构建运行。

**结论 1:overlap 机制真实有效。** 大 tile(128×64K,inner=4)overlap 对 serial
~3×;leaky_relu(≥4 block/线程)overlap 对 serial 1.7–2.3×。这是 func 级
compute↔copy overlap 的功能收益,v2/v3 都拿到。

**结论 2:精确性(v2 保守集 vs v3 零 no-op)是噪声。** 大 tile v2=10.67ms /
v3=12.09ms(本轮 v3 略慢),另一轮 v2=12.40 / v3=11.82(v3 略快)——**符号相反,
差在噪声内**。leaky_relu 同样 v3 略慢。所以"零 no-op wait"的价值是代码简单/
可验证,**不是吞吐**;不当性能论据。

**结论 3:固定开销主导小 tile,且是 CPU 模拟器产物。** 小 tile(256×128,inner=4)
sync=0.08ms / serial=18.29ms(~229×)/ overlap=7.40ms(~93×)。每次阻塞 wait
~10–25µs(std::thread/mutex/condvar)。**corpus BLOCK=256=1KB 正在灾难区**。
但这是 CPU 模拟器(host 即执行器,sync 直接 box_copy+compute)的产物,**不能平移
NPU**(host 不是执行器,async_copy 是硬件 doorbell)。→ N/S 开销项与小 tile gate
必须按后端标定(§5.2)。

**结论 4:多 dispatch body 的吞吐由屏障数决定。** leaky_relu serial ~4 道屏障/轮
vs overlap 2 道(L2 分组 cmpf∥mulf);N=1/S=1 仍 ~1.3×,纯靠少屏障/少等待。
→ L2 分组与 barrier 摊销(§9)是真实杠杆,比抠 no-op wait 大得多。

**结论 5:masked-K 归约验证专用 tail slot 与反例。** `reduce_sum_bench` 正确模式
(专用 tail slot,入口零化一次、full tile 从不写):非整除用例 1048699/4096/32
(tail ext=123)err=0,overlap 对 serial 2.0–2.4×;边界(513/256 tail ext=1、单块、
K<N、N=1/S=1)全 err=0。**错误用法 `--naive-hoist`(零化到轮转槽)在同一非整除
用例 err=3.98e3**——tail 槽 OOB 是上一个 full tile 残留、被全块求和,**正是 §6.2
的 masked-K 反例的实测复现**。整除用例(1M/1024)无尾块,naive 在该例 err=0,
印证"错误只在有 partial tile 且该槽此前被 full tile 用过(K>N)时显现"。
> 复现:bench 已支持任意 flag 顺序,`--naive-hoist` / `--stages` 放在 `--case`
> 前后等价;非整除反例用 `--case 1048699:4096:32 --naive-hoist` 即可复现 err=3.98e3。

**结论 6:barrier 摊销(§9)的收益是真实的,但很快饱和在 U=2–4。**
`amortize_bench`(leaky_relu body,4M/4096/32,N=8/S=4,三执行体,W=2,中位数):
serial 110.7ms、U=1 62.7ms(1.77×)、U=2 35.7ms(3.10×)、U=4 25.2ms(4.39×),err=0;
relu body(单 barrier)U=1/2/4 = 2.40×/4.35×/4.39×。**leaky_relu 到 U=4 仍在涨、
relu 在 U=2 已饱和**,因为前者的屏障数多一倍,饱和点随 body 的屏障数移动。
(旧共享池下 leaky_relu 在 U=4 就停在 2.5×——池争用把摊销收益盖住了,说明
"摊销杠杆有多大"本身取决于 copy 与 compute 是否真的分资源。)
**机理不是"少几次 host 调用"**:
同一 group 内 U 个 dispatch 之间无 hazard,在 skip 屏障的前提下**并发执行**,
所以一次屏障的实际阻塞代价被摊到 U 个 tile 上,同时屏障的固定唤醒成本也只付一次。
这与"compute 跨迭代不并发"不矛盾——摊销正是**在 group 内**绕过那条限制,也是
§9 安全条件必须卡死的原因。饱和点是每后端成本模型该标定的量(CPU 上是 U≤2–4)。
**反例**:`--shared-scratch`(mask/scaled 单缓冲却仍摊销)在 U=1 时 err=0、U=2 时
err=2——后一组的 cmpf 覆盖了前一组 select 仍要读的 mask,**即 §9 安全条件 2 的
实测复现**;非整除用例(1048699/4096/32)在 U≤8 全 err=0。

## 16. 测试策略

- **lit**(无需 GPU,用 `privateuse-opt`):每个 pass 加
  `third_party/privateuse/test/Transform/` 或 `test/Conversion/` 下 `.mlir`。
  CHECK 关键结构:前导/谓词/remui 下标、**per-slot DMA wait 集**(L1 ①②)、
  **L2 屏障位置**(③ / 入口 / 多 dispatch 分组)、handle 数组、offset、两个分叉
  IR 形状、verifier 负例。新增:N=3 动态下标 lowering 验穿、归约预取 barrier、
  pid 流水化、K<N 边界、meta 复用支配负例、I8 结构断言负例、**§6 专用 tail slot
  谓词化下标 + hoisted fill**。
- **pytest**(模拟后端数值回归):`test/flaggems` 与 `test/cases` 全绿是每个里程碑
  门槛;新增 `test_<feature>_<condition>` 覆盖 double-buffer、N-stage、pid 流水化、
  打包边界(非对齐元素数、大 grid、动态 K)、**masked-load tail(hoist / 专用 slot /
  per-tile fill 三路径)**、**matmul masked-K 归约 tail**。`pytest -s --tb=short`。
- **overlap-bench**:作为 overlap 机制与屏障数的回归基准(非绝对性能模型,§15)。
- native/compiler 改动测试前先 `make`;仅 Python 改动不需要。

## 17. 与 v2 的差异 / 迁移

### 17.1 差异总结

| 维度 | v2(§13) | v3.1 |
|---|---|---|
| op 集 | 4 语句 | 相同(不动) |
| wait 来源 | 全局 base 粒度 may 标志 + if 并集 + for 不动点 | **三层**:L1 pipeliner per-slot DMA / L2 body compute 屏障 / L3 单缓冲通用兜底 |
| 通用分析 | 主路径,面对多缓冲 | **降级**到 L3(单缓冲)+ 独立校验器 |
| no-op wait | 每次访问补一条 | L1 零;L2 屏障幂等(冗余无害) |
| 等待插入 pass | 独立 InsertWaits | P5(L1)+ P5a(L2)+ P5b(L3) |
| coverage | 全局数据流复算(同义反复) | I8 结构断言(流水)+ base coverage(兜底);slot 靠数值 |
| pid 循环 | 只处理跨迭代复用 | 流水化,elementwise 获 overlap |
| masked-load tail | 未处理 | §6 OOB-live 判据 + 决策树(hoist / 专用 tail slot / per-tile) |
| 同步 op 规则 | §13.3 遗漏 | L3 显式定义 |
| N/S | 纯搬运公式 | + 每 task 固定开销 + 每后端成本 gate |
| barrier 摊销 | 无 | §9 可选(共享 scratch/carried ×U 判定) |
| 编译器复杂度 | 高(通用分析包办) | 中(三层各简,通用分析降级) |
| overlap 天花板 | 硬件决定 | 相同(已摸到) |

### 17.2 迁移清单

删除:`!pustream.timepoint`、`pustream.await`、`pustream.timepoint.join`、
`pustream.timepoint.immediate`、copy/dispatch 的结果与 `awaits`;
`ScheduleTimepoints` 的 PendingWaits / token / region drain;`SimplifyTimepoints`;
ToPumem 的 timepoint 消解。

改写:`async.copy` / `async.dispatch`(去结果与 awaits,加 verifier);
`ScheduleTimepoints` → P5(L1 per-slot DMA wait)+ P5a(L2 compute 屏障)+
P5b(L3 单缓冲通用)。

新增:pumem async op + handle/meta 类型 + slot 数组;P5 的 pid 循环流水化;
§6 H2 变换(hoist fill / 专用 tail slot / per-tile fill);I7 / I8 verifier;
`memref.view` 的 ViewOpLowering;barrier 摊销选项。

保留:slab 打包(P6)、`pumem.copy`(同步,留给 gather/scatter)、puc 发射链、
**v2 通用分析机器(降级到 L3 + 校验器)**。

示例:`pumem-async-codegen-example.cpp` 按本方案 wait 布局重生成——去掉 store
**之后**的 `pumem_wait(h_out)`(旧模型残留),改为前导/预取无 wait、消费前精确
wait、退出 sync。现有示例是旧模型输出,不是本方案目标输出。

### 17.3 里程碑映射

- **M0**:语句 op + L3 单缓冲 wait + CPU lowering;现有 kernel 数值全绿。
- **M1**:P3 executables;matmul/transpose 进 executable;L2 body 屏障(多 dispatch
  分组 + 入口屏障)。
- **M2**:P6 handle 分配(含数组)+ slab 打包 + I4/I7。
- **M3**:P5 pipeliner(k 循环 + pid 循环)+ N/S 推导 + §6 H2 变换 + I8 + overlap
  验证;先验穿动态 subview 下游(§11)。
- **M4(可选)**:barrier 摊销 + 每后端成本模型标定。

## 18. 风险与开放问题

1. **动态 subview 下游**(§11):remui 动态 stage offset 经 expand-strided-metadata
   → memref_to_pumem 能否 fold 进 offset 元组,M3 前必须 lit 验穿。这是 remui
   方案的命门,优先级高于 peeler 本体。§6 专用 tail slot 的谓词化下标同验。
2. **pipeliner 实现**:canonical 形状识别 + 多缓冲 + remui 重写 + 谓词化 + per-slot
   DMA wait + §6 H2 变换,范围限 tile 循环。上游 `scf::pipelineForLoop` 面向有 SSA
   结果的 load,不做多缓冲 stage 维插入与 remui 轮转,还会吐 ScfToPumem 不支持的
   index_switch/select——倾向自研薄 peeler,接口做成 PipeliningOption 形状留可换性。
3. **H2 资格判定**:OOB-live 判据(§6.2)需要 store/load mask 关系 + body 跨 lane
   依赖分析;误判会把归约/stencil 错误流水化。专用 tail slot 仅适用后缀 remainder,
   2D 每块 masked 必须 per-tile fill 或不流水化。判定保守(不确定就不流水化)。
4. **pid 循环改写的下游**:strided → 计数循环 + 显式 pid 计算,需验证
   pid_loop / licm 既有资产与新形状的兼容;非整除 num_pids 用 large-grid 测试验。
5. **N/S 常数标定 + 成本 gate**:CPU 有效带宽/计算吞吐/每 task 固定开销需并发负载
   下微基准标定,固化为设备契约;gate 是每后端运行时成本模型,非编译期硬规则。
   CPU 结论不能平移 NPU(§15)。否则退化 N=2。
6. **DRAM 同 kernel 读写复用**:base 级 verifier 拒绝(连带 in-place);待实际
   kernel 需要时给 DRAM base 分配 handle 扩展(§10)。
7. **wait_dispatch 的 NPU 映射**:全局屏障 → join 所有已发 event;NPU 是否需要
   per-dispatch event 恢复 compute 流水,与设备侧确认(硬件契约本期不可扩展,
   仅记录为未来能力项)。
8. **非 canonical 循环**:不流水化、L3 单缓冲 wait,正确但无 overlap;若目标
   kernel 出现复杂控制流 tile 循环,需扩展 pipeliner 识别范围。
9. **多缓冲 slot 与 slab offset 一致性**:handle 数组下标 ↔ 连续 slot 区间布局,
   I4 扩展断言。
10. **三层组合的耦合点**:③ 归属 L2(§4.2 契约)是唯一耦合;L1 永不发 wait_dispatch、
    L2 屏障幂等可重排,保证组合 sound。需 lit 验"L1+L2 组合后无重复/遗漏屏障"。

## 19. 一句话总结

硬件是"一个 core 向三个执行体发任务、只有 host wait",所以 4 个 host-wait 语句
op 是正确且最小的抽象,func 级 overlap 天花板由硬件定死、稳态已摸到。v3.1 不动
op 集,把等待产出**三分**:制造多缓冲的 pipeliner 发精确 per-slot DMA wait(L1),
body 局部规则发 compute 屏障(L2,不需要 N、幂等可重排),v2 通用分析降级为单缓冲
兜底 + 独立校验器(L3);流水化扩到 pid 循环,masked-load 用 OOB-live 判据 + 专用
tail slot 解决。**精确性的价值是可验证/简单而非吞吐(实测噪声级),真实杠杆是
overlap 机制、barrier 分组/摊销、每后端成本 gate——完全不碰硬件契约。**