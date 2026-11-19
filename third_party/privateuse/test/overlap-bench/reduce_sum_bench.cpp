// Reduction-sum tile pipeline benchmark for the CPU async runtime.
//
// Mirrors test/cases/test_reduction_fill.py: one program per BLOCK tile,
// masked load with other=0.0, tl.sum over the full block, one float partial
// per pid, host combines the partials. In SRAM terms `other=0.0` is a zero
// fill of the tile, and the reduce reads the full block -- unlike leaky_relu,
// an OOB lane is summed, not discarded.
//
// Modes:
//   sync    : host per-tile sum over the valid extent + double combine
//   serial  : generated shape: conditional fill(tile, 0) + load + wait +
//             full-block reduce + wait + store the 4-byte partial + wait
//   overlap : N input stages, S partial stages, prefetched loads. The partial
//             tile gets a dedicated tail slot that is zeroed once at entry and
//             never used by a full tile, so the full-block reduce still sees
//             zeros while the body never writes a load target. Hoisting the
//             fill over rotating slots is wrong: a reused slot holds lanes of
//             the previous full tile, which the reduce would sum. Pass
//             --naive-hoist to reproduce that error.
//
// Build/run: ./run-reduce-sum.sh [--n N] [--block B] [--threads T]
//                                [--stages N] [--out-stages S] [--repeats R]
//                                [--v3] [--naive-hoist] [--matrix]
//                                [--case <n>:<block>:<threads>]

#include "cpu_backend_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Params
{
  int64_t n          = 48256;
  int64_t block      = 256;
  int64_t threads    = 32;
  int64_t stages_in  = 2;
  int64_t stages_out = 2;
  int     repeats    = 3;
  bool    v3_waits   = false;
  bool    naive_hoist= false;
};

double now_ms()
{
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

void fill_kernel(float* dst, float value, int64_t n)
{
  for (int64_t j = 0; j < n; ++j) dst[j] = value;
}

void reduce_kernel(const float* in, float* out, int64_t n)
{
  float s = 0.0f;
  for (int64_t j = 0; j < n; ++j) s += in[j];
  out[0] = s;
}

int64_t num_blocks(const Params& p)
{
  return (p.n + p.block - 1) / p.block;
}

void fill_input(std::vector<float>& in)
{
  for (size_t j = 0; j < in.size(); ++j)
    in[j] = std::sin(0.001f * static_cast<float>(j)) * 2.0f - 1.0f;
}

double combine(const std::vector<float>& partials)
{
  double total = 0.0;
  for (float v : partials) total += static_cast<double>(v);
  return total;
}

double reference(const Params& p, const std::vector<float>& in)
{
  const int64_t nb = num_blocks(p);
  std::vector<float> partials(static_cast<size_t>(nb), 0.0f);
  for (int64_t pid = 0; pid < nb; ++pid) {
    const int64_t off = pid * p.block;
    const int64_t ext = std::min(p.block, p.n - off);
    reduce_kernel(in.data() + off, &partials[static_cast<size_t>(pid)], ext);
  }
  return combine(partials);
}

struct Result
{
  double ms    = 0.0;
  double total = 0.0;
};

double run_sync(const Params& p, std::vector<float>& in, std::vector<float>& partials)
{
  const int64_t nb = num_blocks(p);
  const double start = now_ms();
  for (int64_t pid = 0; pid < nb; ++pid) {
    const int64_t off = pid * p.block;
    const int64_t ext = std::min(p.block, p.n - off);
    reduce_kernel(in.data() + off, &partials[static_cast<size_t>(pid)], ext);
  }
  return now_ms() - start;
}

double run_serial(const Params& p, std::vector<float>& in, std::vector<float>& partials)
{
  const int64_t B  = p.block;
  const int64_t nb = num_blocks(p);

  float* in_sram  = static_cast<float*>(alloc(B * 4, 64));
  float* part_sram= static_cast<float*>(alloc(4, 64));

  const double start = now_ms();
  for (int64_t tid = 0; tid < p.threads; ++tid) {
    set_physical_threads_id(tid);
    pumem_dma_handle_t h_in;
    pumem_dma_meta_t   m_in;
    pumem_dma_handle_t h_out;
    pumem_dma_meta_t   m_out;
    for (int64_t pid = tid; pid < nb; pid += p.threads) {
      const int64_t off = pid * B;
      const int64_t ext = std::min(B, p.n - off);
      if (ext < B) {
        pumem_async_dispatch(fill_kernel, in_sram, 0.0f, B);
        pumem_wait_dispatch();
      }
      async_copy_dram_to_sram<1>(
        h_in, m_in, in_sram, in.data() + off, sizeof(float), ext, int64_t{1}, int64_t{1});
      pumem_wait(h_in);
      pumem_wait_dispatch();
      pumem_async_dispatch(reduce_kernel, in_sram, part_sram, B);
      pumem_wait_dispatch();
      async_copy_sram_to_dram<1>(
        h_out, m_out, partials.data() + pid, part_sram, sizeof(float), int64_t{1}, int64_t{1},
        int64_t{1});
      pumem_wait(h_out);
    }
  }
  pumem_sync();
  const double ms = now_ms() - start;

  dealloc(in_sram);
  dealloc(part_sram);
  return ms;
}

double run_overlap(const Params& p, std::vector<float>& in, std::vector<float>& partials)
{
  const int64_t B  = p.block;
  const int64_t nb = num_blocks(p);
  const int64_t N  = p.stages_in;
  const int64_t S  = p.stages_out;
  const int64_t tail_slot = N;

  float* in_all = static_cast<float*>(alloc((N + 1) * B * 4, 64));
  float* par_all= static_cast<float*>(alloc(S * 4, 64));

  const double start = now_ms();
  if (p.naive_hoist) {
    // Deliberately wrong for a reduce: with rotating slots a full tile has
    // already overwritten the tail lane region before the partial tile reuses
    // the slot, so the full-block reduce sums stale lanes.
    pumem_async_dispatch(fill_kernel, in_all, 0.0f, N * B);
  } else {
    // Dedicated tail slot: never touched by a full tile, so one zeroing at
    // entry keeps lanes [ext, B) at 0 for the partial tile.
    pumem_async_dispatch(fill_kernel, in_all + tail_slot * B, 0.0f, B);
  }
  pumem_wait_dispatch();

  for (int64_t tid = 0; tid < p.threads; ++tid) {
    set_physical_threads_id(tid);
    std::vector<pumem_dma_handle_t> h_in(static_cast<size_t>(N + 1));
    std::vector<pumem_dma_handle_t> h_out(static_cast<size_t>(S));
    std::vector<pumem_dma_meta_t>   m_in(static_cast<size_t>(N + 1));
    std::vector<pumem_dma_meta_t>   m_out(static_cast<size_t>(S));

    const int64_t K = nb > tid ? (nb - tid + p.threads - 1) / p.threads : 0;
    auto pid_at = [&](int64_t i) { return tid + i * p.threads; };
    auto ext_at = [&](int64_t i) { return std::min(B, p.n - pid_at(i) * B); };
    auto slot_of = [&](int64_t i) -> int64_t {
      const bool tail = i == K - 1 && ext_at(i) < B;
      return tail && !p.naive_hoist ? tail_slot : i % N;
    };
    auto load_tile = [&](int64_t i, pumem_dma_handle_t& h, pumem_dma_meta_t& m) {
      const int64_t s = slot_of(i);
      async_copy_dram_to_sram<1>(
        h, m, in_all + s * B, in.data() + pid_at(i) * B, sizeof(float), ext_at(i), int64_t{1},
        int64_t{1});
    };

    for (int64_t i = 0; i < std::min(N - 1, K); ++i) {
      const int64_t s = slot_of(i);
      load_tile(i, h_in[s], m_in[s]);
    }

    for (int64_t i = 0; i < K; ++i) {
      const int64_t pf = i + N - 1;
      if (pf < K) {
        const int64_t s = slot_of(pf);
        if (!p.v3_waits) pumem_wait(h_in[s]);
        load_tile(pf, h_in[s], m_in[s]);
      }
      const int64_t sc = slot_of(i);
      const int64_t so = i % S;
      pumem_wait(h_in[sc]);
      pumem_wait(h_out[so]);
      pumem_async_dispatch(reduce_kernel, in_all + sc * B, par_all + so, B);
      pumem_wait_dispatch();
      async_copy_sram_to_dram<1>(
        h_out[so], m_out[so], partials.data() + pid_at(i), par_all + so, sizeof(float),
        int64_t{1}, int64_t{1}, int64_t{1});
    }

    if (K > 0) {
      pumem_wait_dispatch();
      pumem_wait(h_out[(K - 1) % S]);
    }
  }
  pumem_sync();
  const double ms = now_ms() - start;

  dealloc(in_all);
  dealloc(par_all);
  return ms;
}

Result measure(int mode, const Params& p)
{
  std::vector<float> in(static_cast<size_t>(p.n));
  std::vector<float> partials(static_cast<size_t>(num_blocks(p)), 0.0f);
  fill_input(in);

  Result best;
  best.ms = 1e30;
  for (int r = 0; r < p.repeats; ++r) {
    std::fill(partials.begin(), partials.end(), 0.0f);
    double ms = 0.0;
    switch (mode) {
      case 0: ms = run_sync(p, in, partials); break;
      case 1: ms = run_serial(p, in, partials); break;
      default: ms = run_overlap(p, in, partials); break;
    }
    // Recover the total from the run (runs return ms only; combine the partials
    // after the run so every mode is checked against the same reference).
    const double total = combine(partials);
    if (ms < best.ms) {
      best.ms    = ms;
      best.total = total;
    }
  }
  return best;
}

double gbps(const Params& p, double ms)
{
  const double bytes = static_cast<double>(p.n) * sizeof(float);
  return ms > 0.0 ? bytes / ms / 1e6 : 0.0;
}

void print_row(const char* label, const Params& p, const Result& r, double ref_total, double serial_ms)
{
  const double err = std::fabs(r.total - ref_total);
  std::printf(
    "  %-13s stages=%lld/%lld  %9.3f ms  %7.2f GB/s  %6.1f us/block  err=%.3g", label,
    static_cast<long long>(p.stages_in), static_cast<long long>(p.stages_out), r.ms, gbps(p, r.ms),
    r.ms * 1000.0 / static_cast<double>(num_blocks(p)), err);
  if (serial_ms > 0.0 && r.ms > 0.0)
    std::printf("  speedup-vs-serial=%.2fx", serial_ms / r.ms);
  std::printf("\n");
}

void run_case(Params p)
{
  std::vector<float> in(static_cast<size_t>(p.n));
  fill_input(in);
  const double ref_total = reference(p, in);

  std::printf(
    "case n=%lld block=%lld threads=%lld blocks=%lld repeats=%d wait=%s%s\n",
    static_cast<long long>(p.n), static_cast<long long>(p.block), static_cast<long long>(p.threads),
    static_cast<long long>(num_blocks(p)), p.repeats, p.v3_waits ? "v3" : "v2",
    p.naive_hoist ? " naive-hoist" : "");

  const Result sync   = measure(0, p);
  const Result serial = measure(1, p);
  print_row("sync", p, sync, ref_total, 0.0);
  print_row("serial", p, serial, ref_total, 0.0);
  const Result overlap = measure(2, p);
  print_row(p.naive_hoist ? "overlap-naive" : "overlap", p, overlap, ref_total, serial.ms);
  std::printf("\n");
}

bool parse(int argc, char** argv, Params& p, bool& matrix, std::vector<Params>& cases)
{
  std::vector<std::string> case_specs;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> int64_t {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", arg.c_str());
        std::exit(2);
      }
      return std::strtoll(argv[++i], nullptr, 10);
    };
    if (arg == "--n") p.n = next();
    else if (arg == "--block") p.block = next();
    else if (arg == "--threads") p.threads = next();
    else if (arg == "--stages") p.stages_in = next();
    else if (arg == "--out-stages") p.stages_out = next();
    else if (arg == "--repeats") p.repeats = static_cast<int>(next());
    else if (arg == "--v3") p.v3_waits = true;
    else if (arg == "--naive-hoist") p.naive_hoist = true;
    else if (arg == "--matrix") matrix = true;
    else if (arg == "--case") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for --case\n");
        return false;
      }
      case_specs.emplace_back(argv[++i]);
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return false;
    }
  }
  // Expand after all options are known so flag order does not matter.
  for (const std::string& spec : case_specs) {
    Params q = p;
    long long qn = 0, qb = 0, qt = 0;
    if (std::sscanf(spec.c_str(), "%lld:%lld:%lld", &qn, &qb, &qt) != 3) {
      std::fprintf(stderr, "bad --case spec: %s\n", spec.c_str());
      return false;
    }
    q.n = qn;
    q.block = qb;
    q.threads = qt;
    cases.push_back(q);
  }
  return true;
}

} // namespace

int main(int argc, char** argv)
{
  Params p;
  bool matrix = false;
  std::vector<Params> cases;
  if (!parse(argc, argv, p, matrix, cases)) return 2;

  std::printf("hw threads=%u  pool workers=%u\n", std::thread::hardware_concurrency(),
              std::max(2u, std::min(std::thread::hardware_concurrency(), 16u)));

  if (!cases.empty()) {
    for (const Params& q : cases) run_case(q);
  } else if (matrix) {
    run_case(p);
    Params q = p;
    q.n = 1048576;
    q.block = 1024;
    run_case(q);
    q.block = 4096;
    run_case(q);
    q.block = 16384;
    run_case(q);
  } else {
    run_case(p);
  }
  return 0;
}