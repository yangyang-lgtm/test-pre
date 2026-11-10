# Triton 编译说明

## 快速开始

```bash
# 进入 Docker 容器
docker exec -it my_ubuntu bash

# 运行编译脚本
cd /home
./build_triton.sh

# 或者清理后重新编译
./build_triton.sh --clean
```

## 环境变量

可以通过环境变量自定义编译参数：

```bash
# 自定义并行任务数（默认 16）
export MAX_JOBS=8

# 自定义路径
export LLVM_BUILD_DIR=/path/to/llvm/build
export TRITON_SRC_DIR=/path/to/triton
export TRITON_SHARED_DIR=/path/to/triton-shared

# 运行编译
./build_triton.sh
```

## 编译选项

- 默认编译：`./build_triton.sh`
- 清理后编译：`./build_triton.sh --clean`

## 编译时间

- 首次编译：约 10-20 分钟（取决于 CPU 核心数）
- 增量编译：约 2-5 分钟

## 已包含的后端

编译完成后，Triton 将包含以下后端：

1. **nvidia** - NVIDIA GPU 后端
2. **amd** - AMD GPU 后端
3. **triton_shared** - 共享中间层后端（Microsoft triton-shared）

## 验证安装

```bash
python3 -c "
import triton
from triton.backends import backends
print('Triton version:', triton.__version__)
print('Available backends:', list(backends.keys()))
"
```

## 故障排除

### 1. 内存不足（OOM）

如果编译过程中出现 `Killed signal terminated program cc1plus` 错误，减少并行任务数：

```bash
export MAX_JOBS=4  # 或更小的值
./build_triton.sh
```

### 2. googletest 下载缓慢

脚本已通过 `TRITON_BUILD_UT=OFF` 禁用单元测试，避免下载 googletest。

如果仍然需要 googletest，可以手动拷贝：

```bash
# 在宿主机上
git clone https://github.com/google/googletest.git
cd googletest && git checkout v1.17.0

# 拷贝到容器
docker cp googletest my_ubuntu:/home/triton/build/cmake.linux-x86_64-cpython-3.10/_deps/googletest-src
```

### 3. triton-shared 编译错误

脚本会自动修复 `UseAnalysis.h` 的编译错误。如果手动修复：

```bash
cd /home/triton/third_party/triton-shared
# 编辑 include/triton-shared/Analysis/UseAnalysis.h
# 在 meetUseType 函数的 switch 语句后添加：
# llvm_unreachable("unknown UseType");
```

## 编译优化

### 使用 ccache 加速

```bash
apt-get install ccache
export TRITON_BUILD_WITH_CCACHE=true
./build_triton.sh
```

### 使用 Clang 编译

```bash
apt-get install clang lld
export TRITON_BUILD_WITH_CLANG_LLD=true
./build_triton.sh
```

## 脚本说明

`build_triton.sh` 脚本会自动：

1. 检查必要的目录和依赖
2. 修复 triton-shared 的编译错误
3. 设置正确的环境变量
4. 禁用单元测试和文档生成（加速编译）
5. 使用 pip install 编译并安装
6. 验证安装结果
