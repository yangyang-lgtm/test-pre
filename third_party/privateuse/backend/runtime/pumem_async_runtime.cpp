#include "pumem_async_runtime.h"

#include "cpu_backend_runtime.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

struct pumem_task
{
  std::mutex mutex;
  std::condition_variable cv;
  std::function<void()> fn;
  bool done = false;
  std::atomic<int> refs{1};
};

namespace {

struct Lane
{
  std::mutex mutex;
  std::condition_variable cv;
  std::deque<pumem_task*> queue;
  bool stopping = false;

  void push(pumem_task* task)
  {
    {
      std::lock_guard<std::mutex> lock(mutex);
      queue.push_back(task);
    }
    cv.notify_one();
  }

  pumem_task* pop()
  {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return stopping || !queue.empty(); });
    if (queue.empty()) return nullptr;
    pumem_task* task = queue.front();
    queue.pop_front();
    return task;
  }

  void stop()
  {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
    }
    cv.notify_all();
  }
};

int compute_lanes()
{
  static const int lanes = [] {
    const char* env     = std::getenv("PUMEM_COMPUTE_LANES");
    const int requested = env ? std::atoi(env) : 2;
    return std::max(1, requested);
  }();
  return lanes;
}

void run_lane(Lane& lane, bool is_compute);

class ThreadGuard
{
public:
  ThreadGuard(Lane& load, Lane& store, Lane& compute) : load_(load), store_(store), compute_(compute) {}
  ThreadGuard(const ThreadGuard&)            = delete;
  ThreadGuard& operator=(const ThreadGuard&) = delete;
  ~ThreadGuard() { stop(); }

  void start();
  void stop();

private:
  Lane& load_;
  Lane& store_;
  Lane& compute_;
  std::atomic<bool> started_{false};
  std::mutex mutex_;
  std::vector<std::thread> threads_;
};

struct Runtime
{
  Lane load_lane;
  Lane store_lane;
  Lane compute_lane;

  std::atomic<size_t> outstanding{0};
  std::condition_variable idle_cv;
  std::mutex idle_mutex;

  std::atomic<size_t> compute_issued{0};
  std::atomic<size_t> compute_completed{0};
  std::mutex compute_mutex;
  std::condition_variable compute_cv;

  std::mutex tasks_mutex;
  std::unordered_map<pumem_task*, std::unique_ptr<pumem_task>> tasks;
  ThreadGuard guard{load_lane, store_lane, compute_lane};
};

Runtime& runtime()
{
  static auto instance = std::make_unique<Runtime>();
  return *instance;
}

void account_retired()
{
  if (runtime().outstanding.fetch_sub(1) == 1) {
    std::lock_guard<std::mutex> lock(runtime().idle_mutex);
    runtime().idle_cv.notify_all();
  }
}

namespace {

void erase_if_unreachable(pumem_task* task)
{
  Runtime& state = runtime();
  std::lock_guard<std::mutex> guard(state.tasks_mutex);
  auto it = state.tasks.find(task);
  if (it == state.tasks.end()) return;
  bool done = false;
  {
    std::lock_guard<std::mutex> taskLock(task->mutex);
    done = task->done;
  }
  if (!done || task->refs.load() != 1) return;
  state.tasks.erase(it);
}

void task_release(pumem_task* task)
{
  if (!task) return;
  task->refs.fetch_sub(1);
  erase_if_unreachable(task);
}

void task_retain(pumem_task* task)
{
  task->refs.fetch_add(1);
}

} // namespace

void retire_dma(pumem_task* task)
{
  {
    std::lock_guard<std::mutex> lock(task->mutex);
    task->done = true;
    task->cv.notify_all();
  }
  account_retired();
  erase_if_unreachable(task);
}

void retire_compute(pumem_task* task)
{
  {
    std::lock_guard<std::mutex> lock(task->mutex);
    task->done = true;
    task->cv.notify_all();
  }
  {
    std::lock_guard<std::mutex> lock(runtime().compute_mutex);
    runtime().compute_completed.fetch_add(1);
  }
  runtime().compute_cv.notify_all();
  account_retired();
  erase_if_unreachable(task);
}

void run_lane(Lane& lane, bool is_compute)
{
  for (;;) {
    pumem_task* task = lane.pop();
    if (!task) return;
    task->fn();
    if (is_compute)
      retire_compute(task);
    else
      retire_dma(task);
  }
}

pumem_task* create_task(std::function<void()> fn)
{
  auto owned        = std::make_unique<pumem_task>();
  pumem_task* task  = owned.get();
  const int64_t tid = get_physical_id();
  task->fn          = [tid, fn = std::move(fn)] {
    set_physical_threads_id(tid);
    fn();
  };

  runtime().outstanding.fetch_add(1);
  {
    std::lock_guard<std::mutex> lock(runtime().tasks_mutex);
    runtime().tasks.emplace(task, std::move(owned));
  }
  return task;
}

void ThreadGuard::start()
{
  if (started_.load(std::memory_order_acquire)) return;
  std::lock_guard<std::mutex> guard(mutex_);
  if (started_.load(std::memory_order_relaxed)) return;
  threads_.emplace_back(run_lane, std::ref(load_), false);
  threads_.emplace_back(run_lane, std::ref(store_), false);
  const int lanes = compute_lanes();
  for (int i = 0; i < lanes; ++i) threads_.emplace_back(run_lane, std::ref(compute_), true);
  started_.store(true, std::memory_order_release);
}

void ThreadGuard::stop()
{
  std::lock_guard<std::mutex> guard(mutex_);
  if (threads_.empty()) return;
  load_.stop();
  store_.stop();
  compute_.stop();
  for (std::thread& thread : threads_) thread.join();
  threads_.clear();
}

void ensure_started()
{
  runtime().guard.start();
}

void wait_task(pumem_task* task)
{
  if (!task) return;
  std::unique_lock<std::mutex> lock(task->mutex);
  task->cv.wait(lock, [&] { return task->done; });
}

bool is_load(pumem_copy_dir dir)
{
  switch (dir) {
    case pumem_copy_dir::dram_to_sram:
    case pumem_copy_dir::sram_to_sram: return true;
    case pumem_copy_dir::dram_to_dram:
    case pumem_copy_dir::sram_to_dram: return false;
  }
  return true;
}

} // namespace

namespace {

struct ShadowEntry
{
  int readers = 0;
  int writers = 0;
};

struct ShadowState
{
  std::mutex mutex;
  std::unordered_map<void*, ShadowEntry> entries;
  bool enabled = true;
  bool checked = false;
};

ShadowState& shadow()
{
  static auto state = std::make_unique<ShadowState>();
  if (!state->checked) {
    const char* env = std::getenv("PUMEM_SHADOW_QUEUE");
    state->enabled  = !env || std::atoi(env) != 0;
    state->checked  = true;
  }
  return *state;
}

void shadow_abort(const char* what, void* buffer, int readers, int writers)
{
  std::fprintf(stderr, "pumem: %s on %p (%d readers, %d writers still in flight)\n", what, buffer, readers, writers);
  std::abort();
}

void shadow_enqueue(void* dst, void* src)
{
  if (!shadow().enabled) return;
  std::lock_guard<std::mutex> lock(shadow().mutex);
  ShadowEntry& target = shadow().entries[dst];
  ShadowEntry& source = shadow().entries[src];
  if (target.readers > 0) shadow_abort("write-after-read", dst, target.readers, target.writers);
  if (target.writers > 0) shadow_abort("write-after-write", dst, target.readers, target.writers);
  if (source.writers > 0) shadow_abort("read-after-write", src, source.readers, source.writers);
  ++target.writers;
  ++source.readers;
}

void shadow_retire(void* dst, void* src)
{
  if (!shadow().enabled) return;
  std::lock_guard<std::mutex> lock(shadow().mutex);
  ShadowEntry& target = shadow().entries[dst];
  ShadowEntry& source = shadow().entries[src];
  if (target.writers > 0) --target.writers;
  if (source.readers > 0) --source.readers;
}

} // namespace

pumem_dma_handle_t::~pumem_dma_handle_t()
{
  pumem_task* read  = last_read.exchange(nullptr);
  pumem_task* write = last_write.exchange(nullptr);
  if (read) task_release(read);
  if (write) task_release(write);
}

void pumem_detail::enqueue_dispatch(std::function<void()> fn)
{
  ensure_started();
  runtime().compute_issued.fetch_add(1, std::memory_order_release);
  runtime().compute_lane.push(create_task(std::move(fn)));
}

void pumem_detail::enqueue_copy(
  pumem_copy_dir dir, const pumem_dma_handle_t& handle, pumem_dma_meta_t& meta, void* dst, void* src, int64_t elem_size,
  const int64_t* box, size_t box_size
)
{
  if (meta.busy.exchange(true)) {
    std::fprintf(stderr, "pumem: asynchronous copy reuses a DMA metadata slot that is still in flight\n");
    std::abort();
  }

  std::vector<size_t> dims;
  dims.reserve(box_size);
  for (size_t i = 0; i < box_size; ++i) dims.push_back(static_cast<size_t>(box[i]));

  auto fn = [metaPtr = &meta, dims = std::move(dims), dst, src, elem_size] {
    size_t rank = dims.size() / 3;
    detail::box_copy(
      dst, src, rank, dims.data(), dims.data() + rank, dims.data() + 2 * rank, static_cast<size_t>(elem_size)
    );
    shadow_retire(dst, src);
    metaPtr->busy.store(false, std::memory_order_release);
  };
  shadow_enqueue(dst, src);

  ensure_started();
  const bool load  = is_load(dir);
  pumem_task* task = create_task(std::move(fn));
  task_retain(task);
  (load ? runtime().load_lane : runtime().store_lane).push(task);
  pumem_task* previous = (load ? handle.last_read : handle.last_write).exchange(task, std::memory_order_acq_rel);
  if (previous) task_release(previous);
}

void pumem_wait(const pumem_dma_handle_t& handle)
{
  wait_task(handle.last_read.load(std::memory_order_acquire));
  wait_task(handle.last_write.load(std::memory_order_acquire));
}

void pumem_wait_dispatch()
{
  Runtime& state      = runtime();
  const size_t target = state.compute_issued.load(std::memory_order_acquire);
  if (state.compute_completed.load(std::memory_order_acquire) >= target) return;
  std::unique_lock<std::mutex> lock(state.compute_mutex);
  state.compute_cv.wait(lock, [&] { return state.compute_completed.load() >= target; });
}

void release_retired_tasks()
{
  Runtime& state = runtime();
  std::lock_guard<std::mutex> guard(state.tasks_mutex);
  for (auto it = state.tasks.begin(); it != state.tasks.end();) {
    pumem_task* task = it->first;
    bool done        = false;
    {
      std::lock_guard<std::mutex> taskLock(task->mutex);
      done = task->done;
    }
    if (!done || task->refs.load() != 1) {
      ++it;
      continue;
    }
    it = state.tasks.erase(it);
  }
  if (std::getenv("PUMEM_TASK_STATS")) std::fprintf(stderr, "pumem: %zu tasks alive\n", state.tasks.size());
}

void pumem_sync()
{
  {
    std::unique_lock<std::mutex> lock(runtime().idle_mutex);
    runtime().idle_cv.wait(lock, [] { return runtime().outstanding.load() == 0; });
  }
  release_retired_tasks();
}
