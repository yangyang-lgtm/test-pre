#ifndef PUMEM_ASYNC_RUNTIME_H
#define PUMEM_ASYNC_RUNTIME_H

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <tuple>
#include <utility>

struct pumem_task;

enum class pumem_copy_dir {
  dram_to_dram,
  dram_to_sram,
  sram_to_dram,
  sram_to_sram,
};

struct pumem_dma_handle_t
{
  int64_t id = 0;

  mutable std::atomic<pumem_task*> last_read{nullptr};
  mutable std::atomic<pumem_task*> last_write{nullptr};
};

struct pumem_dma_meta_t
{
  pumem_dma_meta_t()                                   = default;
  pumem_dma_meta_t(const pumem_dma_meta_t&)            = delete;
  pumem_dma_meta_t& operator=(const pumem_dma_meta_t&) = delete;

  std::atomic<bool> busy{false};
};

namespace pumem_detail {
void enqueue_copy(
  pumem_copy_dir dir, const pumem_dma_handle_t& handle, pumem_dma_meta_t& meta, void* dst, void* src, int64_t elem_size,
  const int64_t* box, size_t box_size
);
void enqueue_dispatch(std::function<void()> fn);

template <typename Tuple, size_t... Is>
std::array<int64_t, sizeof...(Is)> box_from(const Tuple& values, std::index_sequence<Is...>)
{
  return {static_cast<int64_t>(std::get<Is>(values))...};
}

template <int Rank, typename... Args>
void copy_task(
  pumem_copy_dir dir, const pumem_dma_handle_t& handle, pumem_dma_meta_t& meta, void* dst, void* src, int64_t elem_size,
  Args... args
)
{
  constexpr size_t kBox = size_t(3 * Rank);
  static_assert(sizeof...(Args) == kBox, "async copy takes sizes, dst_strides and src_strides only");
  auto values = std::tuple<Args...>(args...);
  auto box    = box_from(values, std::make_index_sequence<kBox>{});
  enqueue_copy(dir, handle, meta, dst, src, elem_size, box.data(), box.size());
}
} // namespace pumem_detail

template <typename Kernel, typename... Args> void pumem_async_dispatch(Kernel kernel, Args... args)
{
  pumem_detail::enqueue_dispatch([kernel, args...] { kernel(args...); });
}

template <int Rank, typename... Args>
void async_copy_dram_to_dram(
  const pumem_dma_handle_t& handle, pumem_dma_meta_t& meta, void* dst, void* src, int64_t elem_size, Args... args
)
{
  pumem_detail::copy_task<Rank>(pumem_copy_dir::dram_to_dram, handle, meta, dst, src, elem_size, args...);
}

template <int Rank, typename... Args>
void async_copy_dram_to_sram(
  const pumem_dma_handle_t& handle, pumem_dma_meta_t& meta, void* dst, void* src, int64_t elem_size, Args... args
)
{
  pumem_detail::copy_task<Rank>(pumem_copy_dir::dram_to_sram, handle, meta, dst, src, elem_size, args...);
}

template <int Rank, typename... Args>
void async_copy_sram_to_dram(
  const pumem_dma_handle_t& handle, pumem_dma_meta_t& meta, void* dst, void* src, int64_t elem_size, Args... args
)
{
  pumem_detail::copy_task<Rank>(pumem_copy_dir::sram_to_dram, handle, meta, dst, src, elem_size, args...);
}

template <int Rank, typename... Args>
void async_copy_sram_to_sram(
  const pumem_dma_handle_t& handle, pumem_dma_meta_t& meta, void* dst, void* src, int64_t elem_size, Args... args
)
{
  pumem_detail::copy_task<Rank>(pumem_copy_dir::sram_to_sram, handle, meta, dst, src, elem_size, args...);
}

void pumem_wait(const pumem_dma_handle_t& handle);
void pumem_wait_dispatch();
void pumem_sync();

#endif // PUMEM_ASYNC_RUNTIME_H
