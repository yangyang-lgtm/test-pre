# pustream v5：以时点为一等值的异步设计

## 0. 为什么推翻 v4

### 0.1 一个真实故障

clamp 内核在 v4 上跑出全 NaN。循环体里有两个 dispatch：

    dispatch @maxsi(输入 tile -> 中间 tile)
    dispatch @minsi(中间 tile -> 输出 tile)

流水化 pass 只认识"读输入 tile"的那一个，于是给整个循环打了 `pustream.pipelined`；
串行化 pass 看到标记就整块跳过。结果 `minsi` 既不在流水化 pass 的调度范围内
（它不认识），也不在串行化 pass 的调度范围内（整块跳过）——读到了 `maxsi` 还在飞的输出。

### 0.2 三个根因

1. **依赖不是值**。谁依赖谁只存在于"语句在 block 里的位置"和 `reads/writes` 这两个整数下标表里。
   两个 pass 都只能靠位置猜测，猜错就是 NaN。
2. **slot 是后加的**。缓冲区本来是 memref，多缓冲只能靠"事后给 alloc 加一维、再造一个
   `!pustream.slot` 去索引、把 remui 硬塞进循环"。多缓冲本质上是**缓冲区生命周期**问题，
   被做成了循环重写问题。
3. **所有权是区域级的**。`pustream.pipelined` 标记说"这个 body 我全包了"，但流水化 pass
   实际只调度它认识的几条语句。区域级标记无法表达"部分语句已调度"。

### 0.3 结论

不是给两个 pass 划一条更细的边界，而是换模型：**让异步性成为值**，让调度从"猜"变成"读图"。

## 1. 后端约束（一切设计的起点）

后端对编译器可见的只有这些（`backend/runtime/pumem_async_runtime.h`）：

| 原语 | 语义 |
|---|---|
| `pumem_dma_handle_t { id, last_read, last_write }` | 每个缓冲区一个序号点 |
| `pumem_dma_meta_t { busy }` | 每个在飞描述符一个；未完成就复用会直接 abort |
| `pumem_wait(handle)` | 等该缓冲区的 `last_read` 和 `last_write` |
| `pumem_wait_dispatch()` | 等计算队列排空到"当前已发出"的位置 |
| `pumem_sync()` | 等所有队列 `outstanding == 0` |
| `pumem_task_t*` | 运行时内部，**不对编译器暴露** |

### 1.1 四条推论

- **C1 时点不能是 per-task**。IR 看不到 task，所以任何"等第 k 个任务"的语义都无法表达。
- **C2 时点只有两种可表达形态**：某缓冲区安静（handle），或计算队列安静（全局屏障）。
- **C3 同一缓冲区上的多次传输天然聚合成一个等待点**——handle 只有一个 `last_read` / `last_write`。
  这意味着"同一 buffer 的多时点"不需要 join，等一次就够。
- **C4 meta 的 busy 决定了"同一缓冲区的连续传输必须先等待"**。多缓冲要解决的正是这一点。

## 2. 设计目标

| | 目标 | 判定方式 |
|---|---|---|
| G1 | 依赖由 SSA 边表达，不靠扫描推断 | 删掉所有"插 wait"的策略代码 |
| G2 | "用了未就绪的 buffer"能被静态发现 | verifier 报错，而不是 NaN |
| G3 | 多缓冲由值旋转表达，不改 buffer 的 shape | 没有 stage 维、remui、slot |
| G4 | 流水化之外不需要任何所有权标记 | 没有 `pustream.pipelined` 的区域跳过 |
| G5 | 降级只用到 §1 的原语 | 运行时零改动 |

## 3. 类型

### 3.1 `!pustream.buffer<T>`

一个 SRAM 缓冲区句柄，`T` 是它承载的 memref 类型：

    !pustream.buffer<memref<256xf32, 128>>
    !pustream.buffer<memref<?xf32, strided<[1]>, 128>>   // slice 出来的视图

- 它不是 memref：没有地址运算、没有 load/store，只能被 pustream 的语句消费。
- 带 memref 类型参数，是因为传输需要 (elem type, rank, sizes, strides)，dispatch 的
  outlined 签名也是 memref 形状。放在类型里，lowering 不必另造一份几何信息。
- 与 v4 的 `!pustream.slot<T>` 的区别：**没有 stage index**，也就没有 `pustream.stages`。

### 3.2 `!pustream.timepoint`

不透明。**降级后不产生任何值**——它的唯一职责是让依赖成为 SSA 边。

    %tp = pustream.transfer %dram, %buf        // buffer 时点
    %tp = pustream.fill %buf, %value           // buffer 时点
    %tp = pustream.dispatch @f(%ins)(%outs)    // compute 时点

## 4. op 集

### 4.1 总览

| op | 签名 | 产出时点 |
|---|---|---|
| `pustream.alloc` | `() -> !pustream.buffer<T>` | 无（新鲜，就绪） |
| `pustream.dealloc` | `(%buf)` | 无 |
| `pustream.slice` | `(%buf, off, size) -> !pustream.buffer<T'>` | 继承源（就绪才可就绪） |
| `pustream.transfer` | `(%dram, %dst) -> %tp` | buffer(dst) |
| `pustream.fill` | `(%dst, %value) -> %tp` | buffer(dst) |
| `pustream.dispatch` | `@callee(%ins)(%outs) -> %tp` | compute |
| `pustream.copy` | `(%src, %dst)` | 无（同步 host copy） |
| `pustream.await` | `(%tp, %buf) -> %buf` / `(%tp)` | 无（消费时点） |
| `pustream.sync` | `()` | 无（排空） |

### 4.2 alloc / dealloc

    %buf = pustream.alloc() : !pustream.buffer<memref<256xf32, 128>>

alloc 出来的值是**就绪**的（刚分配，没有在飞传输）。dealloc 要求缓冲区已就绪（由 C4 推出：
dealloc 时若还有在飞传输，meta 仍在忙）。

### 4.3 slice

    %tile = pustream.slice %buf [%off] [%size]
      : !pustream.buffer<memref<256xf32, 128>> -> !pustream.buffer<memref<128xf32, 128>>

- 取一段连续区间，结果是新的 buffer 值，指向同一块分配；
- 就绪状态继承自源 buffer（源就绪则结果就绪）；
- offset/size 可以是动态值；
- 与 v4 的 subview 相比：slice 不参与地址运算，所以不会出现"subview 丢给 dispatch 导致
  baseOf 判断"这一套。

### 4.4 transfer

    %tp = pustream.transfer %dram_slice, %dst
      : memref<?xf32, strided<[1], offset: ?>>, !pustream.buffer<memref<256xf32, 128>>

- 方向由 space 推出（DRAM↔SRAM），不需要 `dir` 属性；
- 两个操作数都必须是就绪值（DRAM 侧是普通 memref，S2S 不走这条）；
- 不接收 await 操作数：目标缓冲区由 SSA 保证空闲（见 §6.1）。

### 4.5 fill

    %tp = pustream.fill %dst, %value : !pustream.buffer<memref<256xf32, 128>>, f32

masked tail 的逐 tile 填充。独立成 op 而不是塞进 dispatch，是因为它的语义（写满整个 buffer）
和 dispatch（按 ins/outs 读改写）不同，tail-slot 方案需要单独识别它。

### 4.6 dispatch

    %tp = pustream.dispatch @callee(%in0, %in1)(%out)
      {template_args = ...} : (!pustream.buffer<...>, ...) -> ()

- `(%ins)(%outs)` 是**分段操作数**，取代 v4 的 `args + reads/writes 下标表`：
  读写集合由操作数位置结构性给出，不需要额外的整数表去"近似"效果。
- 所有操作数必须是就绪值。
- 产出 compute 时点。

### 4.7 copy（同步）

    pustream.copy %src, %dst : !pustream.buffer<...>, !pustream.buffer<...>

SRAM→SRAM 的阻塞 host copy。两个操作数都必须就绪；它不产出时点（调用返回时已完成）。

### 4.8 await

两种形态：

    // 带 buffer：等这个缓冲区安静，得到就绪版本
    %ready = pustream.await %tp = %buf : !pustream.buffer<memref<256xf32, 128>>

    // 裸：等计算队列（写回前、或 host 需要读结果前）
    pustream.await %tp : !pustream.timepoint

`%tp` 只证明依赖存在，不产生值；**降级成哪条原语由 `%buf` 决定**（§5.2）。

### 4.9 sync

    pustream.sync

kernel 退出前的总排空。等价于"所有时点都 await 过"，但作为一条语句存在，便于入口统一插入。

## 5. 时点语义

### 5.1 三种降级形态

    !pustream.timepoint  ∈  { buffer(b), compute }

`buffer(b)`：降级成 `pumem_wait(handle(b))`。
`compute`：降级成 `pumem_wait_dispatch()`。

没有第三种。`pumem_sync()` 只由 `pustream.sync` 产生，不由时点产生。

### 5.2 await 的降级由 buffer 操作数决定

| IR | 时点来源 | 降级 |
|---|---|---|
| `%r = await %tp = %buf` | transfer / fill 写 %buf | `pumem_wait(handle(%buf))` |
| `%r = await %tp = %buf` | dispatch | `pumem_wait_dispatch()` |
| `await %tp` | dispatch | `pumem_wait_dispatch()` |
| `pustream.sync` | — | `pumem_sync()` |

因为降级只看 `%buf` 和时点的**来源类别**，时点本身不需要携带任何运行时状态。

### 5.3 为什么没有 join

- 时点不是运行时对象，没有"聚合两个时点"可落地的东西（§1 C2）。
- 同一缓冲区上的多个时点本来就被 handle 聚合（§1 C3）：等一次就覆盖全部。
- 跨缓冲区的聚合并无原语，只能展开成多次 wait；那就应当在 IR 里写成多次 await，
  而不是用一个 `join` 值把"要等哪几个 handle"藏起来。

因此 v5 不提供 `join`。需要等 N 个缓冲区时写 N 个 await（或旋转时携带 N 个时点）。

### 5.4 粒度上限（诚实定位）

- **缓冲区级**：`await %tp = %buf` 等的是 `%buf` 上的**所有**传输（last_read + last_write）。
  两个独立传输写同一 buffer 的不同区域，无法分别等待。要更细就必须拆成两个 buffer。
- **计算全局级**：`pumem_wait_dispatch` 等的是整条计算队列，不是某一次 dispatch。
  等任何一个 compute 时点都等于排空计算队列。

这两条不是实现缺陷，是后端能力的如实反映；写在这里，免得 pass 假设不存在的精度。

## 6. 依赖与验证

### 6.1 依赖是 SSA 边，不是分析结果

- 每个异步 op 的 buffer 操作数必须是**就绪值**：由 `alloc`、`await`、`copy`（同步）、
  或由就绪 buffer `slice` 出来。
- 因此"用未就绪的 buffer"在 IR 里根本写不出来（除非绕开 verifier）。
- 旋转把这条规则推广到循环：迭代 i 用的 buffer 是迭代 i-1 传下来的**未就绪值**，
  消费前必须先 `await`；预取的 buffer 在这一轮没人 await，它自然并发。

### 6.2 verify-async 规则

放在流水化之后、降级之前，块内检查：

| 规则 | 内容 | 能抓到的错误 |
|---|---|---|
| R1 | 异步 op 的每个 buffer 操作数必须是就绪值（定义 op 属于 {alloc, await, copy, slice(就绪)}） | 用了未 await 的 buffer |
| R2 | `await %tp = %buf` 的 `%tp` 必须由写该 buffer 的 transfer/fill，或由 dispatch 产生 | await 了不相干的时点 |
| R3 | dispatch 的 ins 与 outs 不得指向同一块分配 | 原地读写 |
| R4 | 函数返回前，所有在飞时点必须被 await 覆盖，或存在 `pustream.sync` | 漏排空 |

R1 正是"数据流依赖分析"应该出现的位置——**只用来报错，不用来决定插什么 wait**。

### 6.3 clamp 在新模型下长什么样

    %r1 = pustream.await %tp_in = %in                  // 输入就绪
    %t1 = pustream.dispatch @maxsi(%r1)(%mid)          // 写中间 tile
    %r2 = pustream.await %t1 = %mid                    // ← 缺失就违反 R1
    %t2 = pustream.dispatch @minsi(%r2)(%out)

如果漏了第二个 await，`minsi` 的 `%mid` 不是就绪值 → R1 直接报错。v4 里这个错误
只能表现为 NaN。

## 7. pass pipeline

### 7.1 流水线

| # | pass | 职责 |
|---|---|---|
| P1 | bufferize | 现有 |
| P2 | to-pustream | memref.copy / linalg → transfer / fill / dispatch / copy，**产出时点** |
| P3 | outline | 现有 |
| P4 | pipeline | 旋转 + 预取（只改 buffer 与时点，不发 wait） |
| P5 | verify-async | §6.2 的四条规则 |
| P6 | memref-to-pumem | buffer → 描述符 + handle/meta；await → wait |
| P7 | pumem → puc → C++ | 现有 |

**被删除的**：`ScheduleTimepointsPass` 整套（1:1 规则、区域跳过、`ownedByPipeliner`）、
`pustream.pipelined` 标记、`!pustream.slot`、`pustream.stages`、`slotHandleIndex`。

### 7.2 P4 的职责（旋转）

    %b0 = pustream.alloc() : !pustream.buffer<memref<256xf32,128>>
    %b1 = pustream.alloc() : !pustream.buffer<memref<256xf32,128>>
    %t0 = pustream.timepoint.immediate

    // 前导：把第 0..N-2 次迭代的预取发出去
    %t_pre0 = pustream.transfer %src@0, %b0

    scf.for %i = lo to hi step 1
        iter_args(%cur = %b0, %cur_tp = %t_pre0, %nxt = %b1)
        -> (!pustream.buffer<...>, !pustream.timepoint, !pustream.buffer<...>) {
      %next_src = pustream.slice %dram[%i+1][...]        // 地址平移
      %t_pre    = pustream.transfer %next_src, %nxt      // 预取，无人 await
      %ready    = pustream.await %cur_tp = %cur          // 等上一块
      %t_out    = pustream.dispatch @f(%ready)(%out)
      %r_out    = pustream.await %t_out = %out           // 等计算
      pustream.transfer %r_out, %dst_slice               // 写回
      scf.yield %nxt, %t_pre, %cur
    }

- "预取不等待"不需要标记：`%t_pre` 在本轮没有被任何 `await` 消费；
- 没有 stage 维、没有 remui、没有 slot；
- 多缓冲 = `iter_args` 里的多个 buffer 值，分配器以后要复用也只需要看这些值的生命周期。

## 8. 降级到后端

### 8.1 buffer → 描述符 + handle/meta

`!pustream.buffer<memref<T, space>>` 转换成：

    (ptr, offset, sizes..., strides..., handle, meta)

- 每个 `pustream.alloc` 在入口处生成一对 `pumem.dma_slot`（handle + meta）；
- `slice` 只改 ptr/offset/sizes/strides，**不新建 handle**（同一块分配一个序号点，见 C3）。

### 8.2 逐 op 映射

| IR | 后端 |
|---|---|
| `pustream.alloc` | `pumem.alloc` + `pumem.dma_slot` |
| `pustream.slice` | `pumem.addptr` + 描述符切片 |
| `pustream.transfer` | `async_copy_{dram,sram}_to_{sram,dram}<rank>` |
| `pustream.fill` | 见 §11（outlined fill dispatch，或 host memset） |
| `pustream.dispatch` | `pumem_async_dispatch` |
| `pustream.copy` | `copy_sram_to_sram<rank>` |
| `await %tp = %buf`（buffer 时点） | `pumem_wait(handle(%buf))` |
| `await %tp`（compute 时点） | `pumem_wait_dispatch()` |
| `pustream.sync` | `pumem_sync()` |

**运行时不需要任何改动。**

### 8.3 worked example（leaky_relu 稳态）

    %in, %out : !pustream.buffer<memref<256xf32,128>>
    %t_pre  = transfer %dram[i*256 ...], %in          → async_copy_dram_to_sram(h_in, m_in, ...)
    %rd     = await %t_prev = %in                     → pumem_wait(h_in)
    %t_c    = dispatch @outlined_leaky(%rd)(%out)     → pumem_async_dispatch(kernel, ...)
    %rw     = await %t_c = %out                       → pumem_wait_dispatch()
    transfer %rw, %dram[i*256 ...]                    → async_copy_sram_to_dram(h_out, m_out, ...)

与 v4 生成的代码相比：少了"每条语句后无脑 wait"，多了每个缓冲区一次语义明确的 wait。

## 9. 与 v4 的差异

| | v4 | v5 |
|---|---|---|
| 缓冲区 | `!pustream.slot<T>` = (alloc, stage index) | `!pustream.buffer<T>`，无 stage |
| 异步表示 | 副作用 + Ordering resource | 时点值 |
| 依赖 | 位置 + `reads/writes` 下标表 | SSA use-def |
| 等待 | 两套 pass 各自插入 | 只有 `await` |
| 多缓冲 | 加 stage 维 + remui + slot 下标 | buffer 值旋转 |
| 调度所有权 | 循环级标记 | 不需要（没有"插 wait"的 pass） |
| 错误表现 | NaN | verifier 报错 |
| `join` | 无 | 无（§5.3） |
| execute 区域 | 无 | 无（暂不做） |

## 10. 迁移计划

| 步 | 内容 | 可删掉的 | 验收 |
|---|---|---|---|
| 1 | 引入 `!pustream.buffer` / `!pustream.timepoint`；`transfer`/`fill`/`dispatch` 产出时点；`await` 替代 `wait/wait_dispatch`；删除 `ScheduleTimepointsPass` | 串行化 pass、Ordering resource、`slot` | 300 case 全绿 |
| 2 | P7：SlotType → BufferType，描述符/handle/meta 路径保留；`await` 的两种降级 | `slotHandleIndex`、dispatch 的 slot 裁剪特判 | 300 case 全绿 |
| 3 | P4 改成旋转（iter_args）；`match()` 只保留"body 里都是本 pass 认识的语句" | stage 化代码、`stages` 属性 | 300 case 全绿 + 流水化 lit |
| 4 | 加 `verify-async`（§6.2） | — | 手写一个漏 await 的用例，验证会报错 |

每一步先加锁定新语义的 lit，再改实现；任何一步不绿就回退该步。

## 11. 开放问题

1. **`fill` 怎么降级**：走 outlined fill dispatch（复用现有路径，代价是一次 dispatch），
   还是单独的 host memset（更快，但要在 runtime 加接口）？建议先走 dispatch。
2. **`pustream.dealloc` 要不要时点**：现在设想的约束是"dealloc 要求就绪"（C4 推出）。
   若以后要在传输还在飞时释放，需要 runtime 提供"释放后完成"的语义。
3. **`slice` 与 `baseOf`**：v5 里 slice 有自己的 op，P6 可以直接取 base 而不必再猜
   subview 链；但 slice 的 offset 是动态值，P6 需要把 offset 累加进描述符。
4. **计算时点是否可以更细**：若将来 runtime 提供 per-dispatch 的完成事件，`compute` 时点
   可以从全局降为 per-dispatch，IR 不用改（只是 §5.2 的降级表多一行）。
5. **SRAM 统计**：多缓冲后 alloc 数量翻倍，I4 的 SRAM 打包需要按"同时存活"而不是"总和"计算，
   这个和 v5 无关，但会在第 3 步暴露出来。

## 12. P6（memref-to-pumem）的改造清单

v5 之后 P6 的输入方言变了，但它的**职责没有变**：把"抽象的 buffer/时点"降成"指针 + handle/meta"，
顺带完成 1:N 的函数 ABI 转换。下面按"必须改 / 会变简单 / 可以删"分类。

### 12.1 必须改

| 项 | 现状 | v5 |
|---|---|---|
| 类型转换 | `SlotType` → (ptr, offset, sizes, strides, handle, meta) | `BufferType` → 同样的六元组；**去掉 stage 偏移那段**（`stageDescriptor` 删除） |
| 语句 lowering | `slot / async.copy / async.dispatch / wait / wait_dispatch / sync` | `alloc / slice / transfer / fill / dispatch / copy / await / sync` |
| dispatch 实参 | slot 参数 `take_front(2+2*rank)` | buffer 参数同样取描述符前缀；语义不变，判据从 SlotType 换成 BufferType |
| 时点 | 不存在 | `!pustream.timepoint` 降成"无值"；`await` 按 §5.2 决定发 `pumem_wait` 还是 `pumem_wait_dispatch` |
| 新 op | — | `pustream.alloc` → `pumem.alloc` + `pumem.dma_slot`；`pustream.slice` → `pumem.addptr` + 描述符切片 |

### 12.2 会变简单

- **几何不再需要推断**。现在 `AsyncCopyOpLowering` 要"从 DRAM 侧推 box、给 slot 侧合成 dense stride"；
  v5 里 `slice` 自带 offset/sizes/strides，copy 直接用两侧描述符。
- **不再需要 `baseOf` / `viewBase` 的猜测**。buffer 只有 `alloc` 和 `slice` 两个来源，
  基址就是沿 slice 链回溯，不需要在 memref 的 subview 里绕。
- **不再有 slot 的身份问题**。"同一 buffer 的多个 slot 是否同一个 handle"这类问题消失——
  一个 alloc 一对 handle/meta，slice 不新建。

### 12.3 可以删

- `pumem.copy` 及其 ToPuc 转换：`transfer` 与 `pustream.copy` 直接降成 runtime call，
  中间不需要一个 pumem 层的 copy op。
- `slotHandleIndex`、`DmaSlotOp` 的 slot 语义（改成 buffer 语义后可保留 op、改名字）。
- 入口折叠不受影响：`!pustream.buffer` 只出现在函数内部，kernel 签名仍然是 DRAM memref，
  所以 `collectEntryConstants` / `foldEntryConstants` 原样保留。

### 12.4 结构性建议

1. **先不要拆 pass**。P6 仍然一个 pass，内部按 "type conversion / per-statement lowering /
   entry folding" 三节组织；等语句集稳定（迁移第 3 步之后）再考虑拆成
   `pustream-to-pumem` + 一个小 `memref-to-pumem`。
2. **改名可以晚一步**。现在名字里的 memref 已经只覆盖一半职责；等拆 pass 时一起改，
   避免中间态一个文件两个名字。
3. **每条语句一个 lowering 函数**，不要再把多条语句塞进一个 pattern。现在的
   `AsyncCopyOpLowering` 里混了"推几何 + 组 operand + 定 rank"，v5 里这三件事分别由
   slice、copy lowering、类型系统承担。
