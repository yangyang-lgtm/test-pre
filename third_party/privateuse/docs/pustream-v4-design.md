# pustream v4:语句式方言与 lowering 设计

> 状态:设计提案。相对 v3.1 的修订范围是**方言本身**(类型、op 集、效果模型)
> 和 **lowering**,以及由此反推出的不变量检查方式。v3.1 关于 pass 分层、H2 判据、
> N/S 推导、barrier 摊销的分析仍然有效,本文不重复,只在受影响处引用并修订。
>
> 本文的每一条结论都从 **0 节的后端语义**推出来,而不是从"通常的异步 IR 长什么样"
> 推出来。凡是后端做不到的事,IR 里就不该能写出来。
>
> 一句话:把 pustream 从"一张带完成 token 的依赖图"改成"**一段发射序列**"。

---

## 0. 起点:后端语义

CPU 模拟后端(`backend/runtime/pumem_async_runtime.cpp`)已经按硬件契约重构为
"一个发射 core + 三个执行体":1 个 load 执行体线程(DRAM->SRAM,FIFO)、1 个 store
执行体线程(SRAM->DRAM,FIFO)、1 个 compute 执行体(内部 W 条 lane,默认 W=2),
host 线程只负责发射与阻塞等待。从这份实现可以读出五条公理:

**A1. 发射序即程序序,host 是唯一的排序者。**
执行体之间没有任何设备侧依赖边。跨执行体的顺序(load->compute、compute->store、
compute->load 的 WAR)只能由 host 在发射下一个任务之前阻塞自己来中介。

**A2. 每个方向一条全局 FIFO,单消费者。**
同一方向的 copy 之间永远不可能并发(单消费者 ⇒ 队内全序);跨方向可以。
这条是契约本身("全局 FIFO,按发射序"),不是模拟器的折衷。

**A3. compute 是一个整体,完成只能整体观测。**
dispatch 进一个队列、在 W 条 lane 上无序执行、**没有 per-task 完成值**。唯一能观测到
的完成是 `pumem_wait_dispatch`:一个全局计数屏障(issued - completed == 0),
O(1)、幂等、排空整个执行体。

**A4. 资源可用性在入队时判定。**
`meta.busy.exchange(true)` 发生在 enqueue,不是 execute;已 busy 直接 abort。
所以"能不能复用 meta"是**入队序**性质,不是数据依赖性质。

**A5. wait 的粒度是 slot,不是操作。**
`pumem_wait(handle)` 等的是 handle 上记录的**两个方向各自的最后入队者**,缺的
方向跳过。因为同方向全局 FIFO,等链尾蕴含等整条前缀,所以一个指针就等于"该方向
上的所有操作"。

### 从 A1–A5 直接推出四件事

1. **操作级完成值(token)在契约内不可 lowering**(A5 + A3):handle 命名的是 slot
   不是操作,同一个 slot 上先后两笔 copy 的完成在后端是同一个事件,SSA 区分不出来;
   dispatch 连完成值都没有。要真表达 token 需要第 5 个原语 `wait(token)`,契约不提供。
2. **方向性 wait 不可表达**(A5):handle 只有两个链尾指针,"只等 store 不等 load"
   在实现上没有可靠语义。
3. **compute 依赖不可细分**(A3):只能说"全排空",不能说"等这几条"。
4. **任何异步 op 的重排都是非法的**(A1 + A2):整个异步序列本身就是一条链。

---

## 1. 设计目标

1. 语句式:IR 原语与 runtime 语句一一对应,无结果值、无 token。
2. **slot 身份在 SSA 里可见**,使 handle / meta / 地址的映射是结构性的,而不是靠
   反推 offset。
3. **非法的程序写不出来**:方向错误、DRAM 侧 wait、slot-to-slot copy 在类型/verifier
   层面被拒,而不是靠数值回归兜底。
4. **顺序不可被编译器改动**:把 A1/A2 写进 effect 模型,而不是靠 pass 纪律。

---

## 2. 类型:`!pustream.slot<T>`

### 2.1 定义

```mlir
!pustream.slot<memref<256xf32, 128>>
```

一个 stage 限定的 SRAM slot 描述符。**值**层面的内容是 `(base alloc, 运行期 index)`;
**类型**层面只记 tile 的形状、元素类型和 space。

参数是 tile(单 stage)的 memref 类型;index 是运行期值,所以只能放在
`pustream.slot` op 的操作数上,不能进类型。

### 2.2 为什么需要这个类型

后端要从一个 slot 同时导出三样东西:

| 派生物 | 来源 |
|---|---|
| 地址 | `base + idx * tile_stride` |
| DMA handle | `h[idx]` |
| meta 槽位 | `m[idx]` |

如果 slot 身份藏在 `memref.subview %alloc[%dyn]` 里,P7 只能靠
expand-strided-metadata 把 `%dyn` 折进 offset、再除 stride 反推回来——这正是
v3.1 §11 标为"命门"的那条路径。有了 slot,三样都是结构性的,不需要反推。

第二个理由是**类型上禁错**:`pustream.wait` 的操作数只能是 slot,于是
`pustream.wait %dram_buffer` 这类没有 handle 可等的程序写不出来。

第三个理由是**统一形状**:单缓冲就是 `pustream.slot %alloc[%c0]`。流水的
多缓冲与非流水的单缓冲在 IR 里是同一种 op、同一种形状,差别只有索引表达式和
`stages` 属性。兜底路径不需要另一套分析。

### 2.3 与"直接用 subview"的对比

| | subview(+ 隐式索引) | `!pustream.slot` |
|---|---|---|
| wait 到 handle 的映射 | 反推 base + 除 stride | 结构性(`h[idx]`) |
| 动态下标 | 依赖 expand-strided-metadata fold | 就是一个数组下标 |
| 非法程序 | 只能 verifier 兜 | 类型上写不出来 |
| 整数 select(尾块) | 记为"例外"(memref 塞不进 ArithToPuarith) | 索引本来就是整数,无需例外 |
| 成本 | 0 | 一个类型 + 一个 Pure op |

---

## 3. op 集

### 3.1 总览

| op | 对应 runtime | 结果 | 说明 |
|---|---|---|---|
| `pustream.slot` | — | `slot` | 纯结构 op,取 stage 视图 |
| `pustream.async.copy` | `async_copy_{dram,sram}_to_{dram,sram}` | 无 | 按 slot 所在侧定方向 |
| `pustream.async.dispatch` | `pumem_async_dispatch` | 无 | 入 compute 队列 |
| `pustream.wait` | `pumem_wait` | 无 | 等一个 slot 静止 |
| `pustream.wait_dispatch` | `pumem_wait_dispatch` | 无 | 全局 compute 屏障 |
| `pustream.sync` | `pumem_sync` | 无 | 排空全部,kernel 退出 |

五个语句 op 与后端语句一一对应,加上一个纯结构 op。没有 `!pustream.timepoint`,
没有 `await` / `join` / `immediate`,没有任何结果值。

### 3.2 `pustream.slot`

```mlir
%s = pustream.slot %alloc[%i]
     : memref<4x256xf32, 128> -> !pustream.slot<memref<256xf32, 128>>
```

| 项 | 内容 |
|---|---|
| 操作数 | `AnyMemRef:$buffer`(SRAM staged buffer)、`Index:$index` |
| 结果 | `!pustream.slot<tile>` |
| traits | `Pure`, `AlwaysSpeculatable` |
| 效果 | 无 |
| verifier | buffer 在 SRAM space;tile 类型 == buffer 去掉第 0 维(元素类型与 space 一致);若 buffer 有 `stages` 属性则 == dim 0 |

`Pure` 很关键:它让 CSE 把同一 `(alloc, idx)` 的多份 slot 合并、让 LICM 可以
把它提出循环。**异步语句不能动,但 slot 可以随便动**——这是安全的,因为 slot 只是
描述符,不产生任何后端动作。

### 3.3 `pustream.async.copy`

```mlir
pustream.async.copy %dram_slice, %s
    : memref<256xf32, strided<[1], offset: ?>>, !pustream.slot<memref<256xf32, 128>>
```

| 项 | 内容 |
|---|---|
| 操作数 | `AnyType:$source`, `AnyType:$target`,**恰好一个**是 slot |
| 结果 | 无 |
| 属性 | **无** |
| 效果 | `MemRead<source>`, `MemWrite<target>`, `MemWrite<Ordering>` |
| verifier | 另一侧是 DRAM space 的 memref;元素类型一致;box rank 一致;slot-to-slot 拒绝;memref-to-memref 拒绝;**box 取自 DRAM 侧类型(extent 可为动态),slot 只提供基址;静态 extent 时要求 <= tile 容量** |
| 方向 | 静态:dst 是 slot 则 `dram_to_sram`;src 是 slot 则 `sram_to_dram` |

四点说明:

1. **没有几何属性**。v3.1 的 op 带 `sizes/dstStrides/srcStrides/elemBytes` 六个
   属性,那是在重复 memref 类型已经说过的信息,代价是 I2 变成一条手写一致性检查。
   现在几何取自两侧类型(DRAM 的 box + slot 的 tile 容量),I2 从"六个属性一致性"
   退化成"rank/元素类型一致 + 静态容量检查"。
2. **I6 消失**。方向不是"检查出来的",是"写不出来的"。
3. DRAM 侧仍是普通 memref(可以是带动态 offset 的 subview)——它没有 handle,所以
   不需要 slot。
4. **部分 tile 的几何不能说成"类型相等"**。尾块的 DRAM subview 是
   `memref<?xf32, strided<[1], offset: ?>>`(extent = ext < tile),与 `memref<256xf32,128>`
   本来就不相等。box 完全由 DRAM 侧决定,slot 只贡献基址;容量约束在 extent 静态时
   可查,动态时由构造保证(`ext = min(tile, n - off)`)或由 debug 断言兜。

### 3.4 `pustream.async.dispatch`

```mlir
pustream.async.dispatch @body(%pin, %pout, %zero) {reads = [0, 2], writes = [1]}
    : (!pustream.slot<memref<256xf32,128>>, !pustream.slot<memref<256xf32,128>>,
       memref<256xf32,128>)
```

| 项 | 内容 |
|---|---|
| 操作数 | `FlatSymbolRefAttr:$callee`、`Variadic<AnyType>:$args`(memref / slot / 标量) |
| 属性 | `DenseI64ArrayAttr:$reads`, `$writes`(下标进 args);可选 `templateArgs` |
| 结果 | 无 |
| 效果 | 每个 reads/writes 下标 => `MemRead/MemWrite<args[i]>`;加 `MemWrite<Ordering>` |
| verifier | 下标范围合法且不指向标量;与 callee 的冻结效应一致(I3) |

参数可以是 slot。P7 把 slot 参数展开成 `(base ptr, offset, sizes, strides)`,
与普通 memref 参数在 ABI 上是同一个元组。

### 3.5 `pustream.wait`

```mlir
pustream.wait %s
```

| 项 | 内容 |
|---|---|
| 操作数 | 一个 `!pustream.slot` |
| 结果 | 无 |
| 效果 | `MemRead<slot>`, `MemWrite<Ordering>` |
| verifier | slot 的 base alloc 必须是 DMA 参与者(出现在某条 copy 里) |

**语义(按 A5 的精确说法)**:等这个 slot 上**两个方向各自最后入队的那一笔**,缺的
方向跳过。等价说法:**该 slot 上没有任何在飞 DMA**。

三个直接后果:

- 一个 op 同时覆盖 ①RAW 和 ②WAR。原因不是 op 编码了方向,而是**一个 slot 通常
  只有一个方向在用**(输入槽只有 load,输出槽只有 store),另一方向本来就是空的。
- in-place 槽两个方向都有会两个都等,安全(多等的是更早的事件,在同一 FIFO 前缀内)。
- **IR 无法表达"只等 store 不等 load"**,这是刻意限制,写进 op 文档。否则以后
  加方向选择,在"一 handle 两指针"的实现上会得到不可靠语义。

### 3.6 `pustream.wait_dispatch`

```mlir
pustream.wait_dispatch
```

无操作数、无结果、`MemWrite<Ordering>`。语义:等 compute 执行体上所有已发 dispatch
退完(全局计数屏障,幂等,O(1))。

不带操作数是有意的:后端**没有任何办法**说"只等这几条 dispatch"。IR 若允许带操作数,
就是在承诺后端做不到的事。

### 3.7 `pustream.sync`

无操作数、无结果、`MemWrite<Ordering>`。语义:排空全部(copy + dispatch),映射
`pumem_sync()`。只允许作为 kernel 函数的最后一条语句——这条写成 verifier,就是
"退出排空"不变量的落地点。

### 3.8 效果与 traits 汇总

| op | traits | 效果 |
|---|---|---|
| `slot` | Pure, AlwaysSpeculatable | — |
| `async.copy` | 不可 speculate | MemRead(src), MemWrite(dst), MemWrite(Ordering) |
| `async.dispatch` | — | MemRead/MemWrite(按 reads/writes), MemWrite(Ordering) |
| `wait` | — | MemRead(slot), MemWrite(Ordering) |
| `wait_dispatch` | — | MemWrite(Ordering) |
| `sync` | — | MemWrite(Ordering) |

### 3.9 同步消费者(非 DMA op)

gather / scatter / 残留的 `memref.copy` 这类**同步 op 不吃 slot**:它们没有 handle 和
meta,执行就在 host 线程上,顺序由程序序本身保证。它们继续用
`memref.subview %alloc[%i]` 直接寻址,L3 在其前插 `wait`(它消费的 buffer 若有在飞 DMA)。

风险是同一块 stage 出现两种命名(subview 给同步 op、slot 给异步 op),P7 可能算出不同
地址。约束:同步 op 的 subview 必须形如 `subview %base[%i]`,与同索引的 slot 同源。
这条在 P7 做一次结构检查(它跨 op,不是单 op 的 verifier)。同步 op 的数量很少
(gather/scatter 与 P2 漏网的 copy),所以这条检查成本可以忽略。

---

## 4. 排序模型:一条 Ordering resource

```tablegen
def PuStream_Ordering : SideEffectResource<"::mlir::pustream::Ordering">;
```

五个语句 op 全部 `MemWrite<Ordering>`。这不是保守,是**忠实**:A1 说 host 是唯一
发射者,A2 说同方向全局 FIFO 全序,所以整个异步序列本身就是一条链,**不存在任何
合法的重排**。把这条约束写进 effect,而不是留给 pass 纪律。

它替代掉了三样东西:

| 原本需要 | 现在 |
|---|---|
| I8 检查"wait 有没有被后续 pass 挪走/删掉" | "被挪走"在 IR 层不可表达,只查完整性 |
| pass 之间的"别动我的 wait"君子协定 | 由 effect 强制 |
| A4 的 meta busy abort(重排把 copy 提到 wait 之前) | 在 canonicalize 阶段就被挡住 |

代价是编译器不能发明重排——但那是本来就不允许的。

**粒度选择**:一条全局资源,而不是 load/store/compute 三条。更细只会给 pass 更多
"合法"的重排空间,而那些重排在 A2 下全都是非法的。

**已知牺牲(写在这里,不再当开放问题)**:跨方向的**独立** op 互换(例如一条 load 和
一条与之无关的 store 交换先后)在语义上合法——load 队列和 store 队列本来就互不相干——
但会被同一条 Ordering 一起冻住。接受这个代价,理由有两条:一是这种互换对流水化没有
收益(两个队列本来就是并行推进的,发射先后不影响稳态);二是要放开它就得把资源拆成
load/store 两条,而那会给 pass 更多"看起来合法"的重排空间,正好是 A1/A2 要禁止的。

---

## 5. 不变量

| 编号 | 内容 | 检查成本 |
|---|---|---|
| I2 | 拷贝几何:box 取自 DRAM 侧(sizes/strides 可动态);元素类型/rank 与 slot 一致;静态 extent <= tile 容量 | 小(不再是类型相等) |
| I3 | dispatch 效应:reads/writes 与 callee 冻结效应一致 | 小 |
| I4 | 打包:活跃区间不重叠、DMA 对齐、slab <= SRAM;`stages == dim0`;slot 下标 < stages | 小(slot 与 offset 已结构性成立) |
| I5 | 仅 NPU:每 stage 活跃 event/handle 数 x N <= 设备上限 | 小 |
| I6 | 方向 | **不可表达,删除** |
| I7 | meta 复用支配:每次 meta 复用被其 slot 上的 wait 支配 | 中(见下) |
| I8 | wait 完整性 | 小(见下) |

### I7 的诚实定位

"slot 身份 SSA 可见就变成支配检查"只在**索引为常量**时严格成立。remui 场景下
"同一个 slot"是 `i%N == j%N` 这种运行期等价,静态不可判定。实际能力是:

- 常量索引(单缓冲、prologue、8.3 的专用 tail slot)=> 精确支配检查;
- 动态索引 => 退化成**"同一 base alloc 上任意两条 copy 之间必须有一条作用在该
  base 上的 wait"**的区间检查。

后者比精确弱,但它抓的正好是那个会 abort 的 bug("prefetch 前完全没有任何 wait"),
且不需要知道 N、不需要复算 in-flight 标志。仍然比 v3.1 的 per-handle 精确复算便宜。

### I8:两条规则,且必须双边

有了 Ordering resource,"wait 被挪走"不可表达,所以 I8 只剩**完整性**。但只查 writes
侧是不够的,必须双边:

> **R1(RAW,防静默垃圾)**:对每条 `async.dispatch` 的每个 read base b,从最后一条
> 写 b 的 DMA 语句到该 dispatch 之间,必须有一条作用在 **b** 上的 `wait`。
>
> **R2(屏障完整,防 compute 结果被提前消费)**:从一条 `async.dispatch` 到它的下一条
> `wait_dispatch` 之间,任何触及该 dispatch 的 `reads ∪ writes` 任一 base 的 DMA
> 语句都是 bug。

两条都是 per-base、纯局部、无不动点、不依赖规范形状。R2 合并了原来的 writes 规则和
prefetch 的 WAR 情形(预取写的是 dispatch 读过的 base,落在 R2 里)。

**为什么 R1 必须有**:A5 让"wait 一个从未入队的 handle"立即返回。所以漏掉 prologue
的 load **不会挂起、不会报错**,直接让 dispatch 读未初始化 SRAM——这是流水化最容易
犯的错(§8.1/§8.2 的第一版草稿就犯了这个错),而只查 writes 侧的完整性检查完全看不见。

**R1/R2 的能力上限**:它们是 per-base 的,所以一条作用在 base b 上的 `wait` 可以被用来
满足 b 上的 RAW 要求,即使生产者写的是**另一个 slot**。也就是说它们查不出 slot 映射
错误,只查"有没有等过这个 buffer"。slot 映射正确性仍然只能靠数值回归 + remui 结构
lit + enqueue-order 调试器。

**运行期兜底要小心**:直觉上应该让 runtime 在 debug 模式对"wait 一个两指针全空且该
slot 参与过 DMA"的 handle 报错,把静默失败变成显式失败。但它会**误报**:streaming 的
前 S 轮 `wait %pout` 本来就没有 producer(② 等的是 S(i-S),i<S 时不存在)。所以这个
检查只能是计数/告警而非常开断言,或者由 P5 给"稳态必然有 producer"的 wait 打标记后再开。
眼下可靠的是 R1(静态)+ 数值回归,不要指望 runtime 断言全覆盖。

---

## 6. pass pipeline

### 6.1 pipeline

```
P0 specialize          不变
P1 bufferize           不变
P2 cleanup-copies      不变(SRAM->SRAM 必须在这里消掉,否则 P4 产不出合法 copy)
P3 form-executables    outline;**按 tile 融合 body**(见 6.3)
P4 to-pustream         建 slot + async.copy + async.dispatch;不插 wait
P5 pipeline            L1:多缓冲 + per-slot wait + N/S;覆盖 k 循环与 pid 循环
P5a body-barriers      L2:compute-may 扫描 + wait_dispatch + 入口屏障
P5b baseline-waits     L3:单缓冲兜底 + 退出 sync
P6 plan-allocation     handle/meta 数组 + slab 打包 + I4/I7
P7 lower-to-pumem      slot 展开;语句 op -> pumem op
P8 executable-body     CPU: linalg->runtime_call/loops;NPU: 保持 linalg
P9 emit                puc -> C++
```

### 6.2 每个 pass 的产出与职责

| pass | 产出 | 必须维持 |
|---|---|---|
| P4 | 为每个 SRAM buffer 建 slot(单缓冲 `%c0`),copy、dispatch | 形状正确;此阶段仍 racy(内部阶段) |
| P5 | 计数循环改写、staged alloc、remui 索引 slot、prologue 谓词化 copy、per-slot wait | 每个消费点 wait 的是它该消费的那一代 slot;prologue 无 wait |
| P5a | `wait_dispatch` | compute-may 标志 + compute 读过/写过的 base 集合;**不做 hazard 分类** |
| P5b | 非流水区 wait + wait_dispatch + 退出 sync | 单缓冲下一 buffer 一 slot,handle-wait 精确 |
| P6 | `stages` 变 handle 数组长度 | I4 / I7 |
| P7 | slot 展开为 `(h[idx], m[idx], base+idx*stride)` | 纯机械展开 |

P5a 为什么不需要 hazard 分类:屏障是全局计数的,插一条就把所有在飞 compute 一起排空,
所以 per-buffer 的 RAW/WAR/WAW 分类**不能减少任何一次排空**。它唯一的作用是决定
"前面那条不 hazard 的 DMA 能不能抢在屏障之前发出去"。因此只需要
"这条 DMA 碰不碰 compute 状态"一个布尔判断。

### 6.3 P3 的 outline 粒度是最大的性能杠杆

对 elementwise body,把 `cmpf / mulf / select` 拆成三条 dispatch 会带来**两条额外的
全局屏障**——而屏障排空的是整个 compute 执行体(多 logical thread launch 时还包括别人)。
这三段之间没有任何 DMA,拆分买不到任何 overlap。

实测(`test/overlap-bench/amortize_bench.cpp`,三执行体后端 W=2,4M/4096/32,
N=8/S=4,中位数):

| 屏障/block | 相对 serial |
|---|---|
| 2.00 (u=1) | 1.77x |
| 1.00 (u=2) | 3.10x |
| 0.50 (u=4) | 4.39x |

单 barrier 的 body(amortize_bench 的 --relu)在 1.00 -> 0.50 -> 0.25 时为
2.40x -> 4.35x -> 4.39x,饱和点比 leaky_relu 来得更早,因为它的屏障数少一倍。

所以优化顺序应该是:

1. **每个 tile 一条 dispatch**(P3 融合);
2. **每 U 个 tile 排空一次**(v3.1 §9 摊销);
3. 插屏障的精度(L2/L3)——第三位。

而且 (1) 与 (2) 是耦合的:body 融合后 `mask`/`scaled` 这类跨 dispatch scratch 直接
消失(变成 executable 内的临时值),v3.1 §9 那条"共享可变 scratch 必须 xU"的安全条件
自动满足,摊销变成免费的。

---

## 7. lowering

### 7.1 pustream -> pumem

| pustream | pumem |
|---|---|
| `slot %buf[%i]` | 展开成三个值,见 7.2 |
| `async.copy %src, %dst` | `async_copy(h[idx], m[idx], dst_tuple, src_tuple, geom, dir)` |
| `async.dispatch @f(%a)` | `dispatch @f(tuple_a)` |
| `wait %slot` | `wait h[idx]` |
| `wait_dispatch` | `wait_dispatch()` |
| `sync` | `sync()` |

pumem 侧只需要两个索引形式的提取,不需要 v3.1 的三个 slot op:

```mlir
%h = pumem.dma_handle_at %h_in[%i] : !pumem.dma_handle
%m = pumem.dma_meta_at   %m_in[%i] : !pumem.dma_meta
```

v3.1 的 `dma_slot` / `dma_slot_array` / `dma_slot_at` 三个 op,语义内容加起来就是
"数组下标",压成上面两个即可。

### 7.2 slot 的展开(P7 到底做什么)

一个 slot 值被三处消费,而 pumem 没有 pair 类型,所以 P7 就地拆成三个值,
三个消费者各自索引同一个 `%sc`:

```mlir
// 源
%s = pustream.slot %in[%sc]
pustream.wait %s
pustream.async.copy %dram, %s
pustream.async.dispatch @body(%s) {reads = [0]}

// P7 之后
%h = pumem.dma_handle_at %h_in[%sc]
%m = pumem.dma_meta_at   %m_in[%sc]
%p = pumem.slice_ptr     %in, %sc          // base + %sc * tile_stride
pumem.wait %h
pumem.async_copy %h, %m, %p, %dram_tuple, geom, dir
pumem.dispatch @body(%p_tuple) {reads = [0]}
```

`%sc` 只计算一次、被三处复用,**不需要任何 offset 折返**。这就是把索引做成操作数
而不是藏在 subview 里的全部收益。`%sc` 本身通常就是 `arith.remui %i, %cN`,
P7 不需要理解它。

### 7.3 pumem -> puc -> C++

| pumem | C++ |
|---|---|
| `async_copy` | `async_copy_{dram,sram}_to_{dram,sram}<Rank>(h, m, dst, src, elem, sizes..., strides...)` |
| `dispatch` | `pumem_async_dispatch(callee, args...)` |
| `wait` | `pumem_wait(h)` |
| `wait_dispatch` | `pumem_wait_dispatch()` |
| `sync` | `pumem_sync()` |
| `dma_handle_at` / `dma_meta_at` | `h_in[i]` / `m_in[i]` |

handle/meta 数组用 `puc.local` 落成 kernel 局部变量;`translate_puc_to_cpp` 按
runtime_backend 选符号。

---

## 8. worked example

### 8.1 streaming 稳态(leaky_relu,body 已融合)

```mlir
%in  = memref.alloc() : memref<4x256xf32, 128> {pustream.stages = 4}
%out = memref.alloc() : memref<2x256xf32, 128> {pustream.stages = 2}

pustream.wait_dispatch                      // H1: drain 循环外的 fill

// prologue: L(0..N-2),无 wait(slot 全新,meta 全新)。pipeliner 会展开成 N-1 份谓词化
// copy,这里用循环示意。%ext_s = min(256, n - off(s))。
scf.for %s = %c0 to %c3 step %c1 {
  %ok = arith.cmpi slt, %s, %K
  scf.if %ok {
    %ps  = pustream.slot %in[%s]
    %src = memref.subview %dram_in[%off(%s)] [%ext_s] [1]
    pustream.async.copy %src, %ps
  }
}

scf.for %i = %c0 to %K step %c1 {
  %sc  = arith.remui %i, %c4               // input 槽: i % N
  %so  = arith.remui %i, %c2               // output 槽: i % S(不是 % N!)
  %sp  = arith.remui %i + %c3, %c4         // 预取槽: (i+N-1) % N
  %inext = arith.addi %i, %c3
  %has = arith.cmpi slt %inext, %K
  scf.if %has {
    %pf = pustream.slot %in[%sp]
    %nx = memref.subview %dram_in[%off(%inext)] [%ext_next] [1]
    pustream.async.copy %nx, %pf            // 预取 L(i+3):无 wait
  }
  %pin  = pustream.slot %in[%sc]
  %pout = pustream.slot %out[%so]
  pustream.wait %pin                        // 等 L(i)
  pustream.wait %pout                       // 等 S(i-2),i<S 时为空,立即返回
  pustream.async.dispatch @body(%pin, %pout, %zero) {reads = [0, 2], writes = [1]}
  pustream.wait_dispatch                    // ③ store 依赖 + 跨迭代 compute 屏障
  %dv = memref.subview %dram_out[%off(%i)] [%ext_i] [1]
  pustream.async.copy %pout, %dv
}
pustream.sync
```

两个容易写错的点,都是第一版草稿踩过的:

- **output 槽的索引必须是 `i % S`**,不能沿用 input 的 `i % N`。N=4/S=2 时用 `%sc`
  会让 slot 下标越过 `out` 的 2 个 stage(I4 违规),而且 `wait %pout` 等到的是一块
  已经被 S(i-1) 用过的槽。
- **prologue 不能省**。没有它,L(0..N-2) 从未入队,`wait %pin` 在 A5 下立即返回,
  dispatch 直接读未初始化 SRAM——不挂起、不报错,只能靠数值回归发现。这类错误正是
  §5 的 R1 要拦的。

每轮 2 条 per-slot wait + 1 道 compute 屏障,零 no-op(没有独立分析在插)。

### 8.2 reduction(matmul k 循环,屏障在预取之前)

```mlir
// prologue: L(0) 入 slot 0,无 wait(与 8.1 同理,省略会静默读垃圾)
%pa = pustream.slot %a[%c0]
%qb = pustream.slot %b[%c0]
pustream.async.copy %dram_a0, %pa
pustream.async.copy %dram_b0, %qb

scf.for %k = %c0 to %K step %c1 {
  %sc = arith.remui %k, %c2
  %sp = arith.remui %k + %c1, %c2
  %k1 = arith.addi %k, %c1
  %has = arith.cmpi slt %k1, %K
  pustream.wait_dispatch                    // ① C(k-1) 仍可能在飞(k=0 时为空,立即返回)
  scf.if %has {
    %pn = pustream.slot %a[%sp]
    %qn = pustream.slot %b[%sp]
    pustream.async.copy %a_next, %pn        // 写在 C(k-1) 读过的 slot 上
    pustream.async.copy %b_next, %qn
  }
  %pc = pustream.slot %a[%sc]
  %qc = pustream.slot %b[%sc]
  pustream.wait %pc
  pustream.wait %qc
  pustream.async.dispatch @k_step(%pc, %qc, %acc) {reads = [0, 1, 2], writes = [2]}
}
```

`%acc` 是普通 memref 不是 slot(compute-only,写回时 P6 才给它 handle)。屏障位置由
P5a 的 compute-may 规则自然产生:上一轮 dispatch 之后没有 ③,标志还立着,本轮预取是
DMA,于是插屏障——**不需要 reduction 特例**。

### 8.3 masked-K 专用 tail slot

```mlir
%is_last = arith.cmpi eq, %i, %last
%tail    = arith.select %is_last, %cN, %sc   // %cN 即常量 N;select 的整数性
%pcur    = pustream.slot %in[%tail]
```

v3.1 §5.3 说"不用 `arith.select`(memref one-to-N 塞不进 ArithToPuarith)",§6.4 又必须
用 select。有了显式索引操作数,**select 的整数性就是 IR 的常态**,不需要记成例外。

### 8.4 非流水兜底

```mlir
%buf = memref.alloc() : memref<256xf32, 128>
%b   = pustream.slot %buf[%c0]
pustream.async.copy %dram, %b
pustream.wait %b
pustream.async.dispatch @f(%b) {reads = [0], writes = [0]}
pustream.wait_dispatch
pustream.async.copy %b, %dram_out
pustream.sync
```

与 8.1 同一种 op、同一种形状。L3 就是"索引为常量 0 的流水",不需要独立的一套分析。

---

## 9. 刻意不支持的能力

| 不支持 | 理由 |
|---|---|
| 操作级完成值(token) | handle 命名 slot 不命名操作;同 slot 两笔 copy 的完成是同一事件 |
| 方向性 wait(只等 store) | handle 只有两个链尾指针,方向选择不可靠;所有 schedule 都不需要 |
| 带操作数的 wait_dispatch | 后端只能说"全排空" |
| slot-to-slot / DRAM-to-DRAM copy | P2 必须消掉;留着等于给 P7 出无解 |
| 在 IR 里编码 N | N 是性能旋钮不是语义;P6 只把它变成数组长度 |
| 独立分析插入的 DMA wait | 多缓冲只来自 pipeliner,它算 remui 时就知道 slot 身份 |
| DRAM 侧 wait | DRAM 不分配 handle(v3.1 §10 的边界,未来可扩展) |

---

## 10. 迁移清单(相对当前 PuStreamDialect.td)

**删**:`PuStream_TimepointType`、`pustream.timepoint.immediate`、`pustream.await`、
`pustream.timepoint.join`;`async.copy` / `async.dispatch` 的结果与 `awaits`
操作数;六个几何属性及相关的 verifier。

**加**:`!pustream.slot<T>` 类型(含 parser/printer)+ `pustream.slot` op;
`pustream.wait`、`pustream.wait_dispatch`、`pustream.sync`;
`PuStream_Ordering` 资源;六个 verifier。

**改**:`ScheduleTimepoints` 拆成 P5 / P5a / P5b;`ToPuStream` 产出 slot 而非
subview;`ToPumem` 的 wait/slot lowering;`form-executables` 改成按 tile 融合。

**文档**:dialect description 现在是 "explicit completion tokens (timepoints) and
hazard-based awaits",与本设计相反,需一并改写。

改动量估计:`.td` 全量重写(约 300 行);`PuStreamDialect.cpp` 加 slot 类型 +
6 个 verifier(约 400 行);三个 pass 的产出形状调整。

---

## 11. 与 v3.1 的差异

| 维度 | v3.1 | v4 |
|---|---|---|
| op 集 | 4 语句 | 5 语句 + 1 纯结构 op |
| slot 身份 | 隐式(subview 的 index,P7 反推) | `!pustream.slot`,SSA 可见 |
| wait 操作数 | 任意 memref slice | 只能是 slot |
| 几何 | 6 个属性 | 取自两侧类型(I2 退化成 rank/元素类型/容量检查) |
| 方向 | 由 space 推导 + verifier | 由"哪侧是 slot"结构性决定(I6 消失) |
| 重排 | 靠 pass 纪律 | 一条 Ordering resource,不可重排 |
| I7 | per-handle 精确 in-flight 复算 | per-base 区间(常量索引时精确) |
| I8 | 结构断言 + base coverage | 只查完整性 |
| wait 语义 | 等生产者 | slot 静止(RAW/WAR 合一) |
| 动态 subview | 风险 1("命门") | 退化成数组下标 |
| 退出排空 | 折成 `pumem_sync()` | 显式 `pustream.sync` |
| 最大性能杠杆 | L2 分组 / §9 摊销 | **P3 按 tile 融合**,其次摊销 |

---

## 12. 开放问题

1. **slot 类型参数化的实现成本**。`TypeDef` 带 `MemRefType` 参数 + 自定义
   parser/printer,约 100 行。若嫌重,替代方案是让 `async.copy` / `wait` 直接吃
   `(base, index)` 两个操作数:op 数不变,但每个 op 多一个操作数,且"哪一侧是
   staged"要靠位置约定。
2. **`pustream.sync` 该不该是 op**。它永远在 kernel 末尾、永远无操作数,理论上
   可以让 P7 直接追加 `pumem.sync`。保留的理由是可验证("每个 kernel 必须以 sync
   结尾")。
3. **I7 的 remui 精度**。per-base 区间检查是否足够,取决于实际 kernel 里同一 base 上
   是否会出现多组互不相关的 copy 链。若有,需要给 slot 一个 generation 标注。
4. **Ordering 粒度**。一条全局资源最忠实,代价及其理由已写进 §4 的"已知牺牲",
   不再作为开放问题。仅剩一个后续问题:将来若真需要跨方向交错(例如两个独立 kernel
   共用一套执行体),要不要拆成 load/store 两条资源。
5. **P3 融合的可行性**。需要在 outlining 之前做 linalg 融合,且要保持 P8 的
   `linalg_to_runtime_call` 路径可用(融合后元素级 body 会变成单一 generic)。
   这是本文收益最大但实现最不确定的一条。
6. **last_read / last_write 的命名**。当前 header 用 `last_read/last_write`,含义在
   不同文档里按"DRAM 视角"和"slot 视角"被写成相反的配对;因为 `pumem_wait` 无条件
   等两个,功能上无害,但做方向选择性优化时必踩。建议改名 `last_load` / `last_store`。
7. **多 host 线程**。三个执行体和 compute 屏障都是进程级单例。多 host 线程 launch
   时 `wait_dispatch` 会变成跨线程屏障。当前不在范围内,但要在 0 节写清。
