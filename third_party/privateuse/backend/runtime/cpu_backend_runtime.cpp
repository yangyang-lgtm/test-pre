#include "cpu_backend_runtime.h"

#include <chrono>
#include <cstdlib>
#include <iostream>

namespace {
thread_local int64_t g_thread_id = 0;
} // namespace

void set_physical_threads_id(int64_t id)
{
  g_thread_id = id;
}

int64_t get_physical_id()
{
  return g_thread_id;
}

int64_t pumem_timer_read()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
    .count();
}


PumemScopeTimer::PumemScopeTimer(const char* name, int64_t iterations)
  : name_(name), iterations_(iterations), start_(pumem_timer_read())
{
}

PumemScopeTimer::~PumemScopeTimer()
{
  const int64_t elapsed = pumem_timer_read() - start_;
  const int64_t average = iterations_ > 0 ? elapsed / iterations_ : elapsed;
  std::cout << "pumem: " << name_ << " " << iterations_ << " iters, avg " << average << " ns, total " << elapsed
            << " ns\n";
}

void* alloc(int64_t bytes, int64_t alignment)
{
  if (alignment <= 0) alignment = alignof(std::max_align_t);
  void* ptr = nullptr;
  if (posix_memalign(&ptr, size_t(alignment), size_t(bytes)) != 0) return nullptr;
  return ptr;
}

void dealloc(void* ptr)
{
  std::free(ptr);
}

void reset_alloc() {}

void detail::box_copy(
  void* dst, const void* src, size_t rank, const size_t* sizes, const size_t* dst_strides, const size_t* src_strides,
  size_t elem_size
)
{
  if (rank == 0) {
    std::memcpy(dst, src, elem_size);
    return;
  }

  char* d       = static_cast<char*>(dst);
  const char* s = static_cast<const char*>(src);
  size_t count  = sizes[0];
  if (rank == 1) {
    for (size_t i = 0; i < count; ++i)
      std::memcpy(d + i * dst_strides[0] * elem_size, s + i * src_strides[0] * elem_size, elem_size);
    return;
  }
  for (size_t i = 0; i < count; ++i)
    box_copy(
      d + i * dst_strides[0] * elem_size, s + i * src_strides[0] * elem_size, rank - 1, sizes + 1, dst_strides + 1,
      src_strides + 1, elem_size
    );
}
