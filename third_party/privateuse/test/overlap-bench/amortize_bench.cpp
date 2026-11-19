// Barrier amortization benchmark for the CPU async runtime.
//
// Mirrors docs/pustream-v3-design.md section 9: U tiles are expanded into one
// group so that U dispatches are followed by a single wait_dispatch instead of
// one per tile. Three conditions make that sound, and this bench encodes all of
// them (they are exactly the section 9 decision table):
//
//   1. output slots must be disjoint:              out_stages >= U
//   2. input slots must not be recycled in-group:  stages     >= 2*U
//      (a group consumes U slots and prefetches the next U, so one unroll step
//      holds 2*U tiles alive; this is the extra cost amortization pays)
//   3. shared mutable scratch must be replicated xU: mask/scaled are per-u
//      fields of the group. --shared-scratch deliberately uses one copy and
//      amortizes anyway: that is the section 9 failure mode (a later cmpf
//      overwrites the mask an earlier select still reads) and must report
//      err != 0. It is the barrier-amortization analogue of --naive-hoist in
//      reduce_sum_bench.cpp.
//
// Reductions are not amortizable: the carried acc cannot be replicated, so the
// per-tile barrier has to stay. That is why this bench is elementwise only.
//
// Modes:
//   sync    : host-side loop, no runtime at all
//   serial  : single buffer, every op waited -- the currently emitted shape
//   u=1     : the design's steady state, one barrier pair per tile
//   u=2/4/8 : amortized, one barrier pair per U tiles
//
// Build/run: ./run-amortize.sh [--n N] [--block B] [--threads T]
//                              [--stages N] [--out-stages S] [--u 1,2,4]
//                              [--repeats R] [--relu] [--shared-scratch]
//                              [--case <n>:<block>:<threads>]

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
  int64_t n           = 4194304;
  int64_t block       = 4096;
  int64_t threads     = 32;
  int64_t stages_in   = 8;
  int64_t stages_out  = 4;
  std::vector<int64_t> unroll{1, 2, 4};
  int     repeats     = 3;
  bool    leaky       = true; // leaky_relu body (2 barriers/tile) vs relu (1)
  bool    shared_scratch = false;
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

void cmpf_kernel(const float* x, const float* zero, int8_t* mask, int64_t n)
{
  for (int64_t j = 0; j < n; ++j) mask[j] = x[j] > zero[j] ? int8_t{1} : int8_t{0};
}

void mulf_kernel(const float* x, const float* c, float* dst, int64_t n)
{
  for (int64_t j = 0; j < n; ++j) dst[j] = x[j] * c[j];
}

void select_kernel(const int8_t* mask, const float* x, const float* y, float* dst, int64_t n)
{
  for (int64_t j = 0; j < n; ++j) dst[j] = mask[j] ? x[j] : y[j];
}

// relu: single-dispatch elementwise body, no scratch, no carried state
void relu_kernel(const float* x, float* dst, int64_t n)
{
  for (int64_t j = 0; j < n; ++j) dst[j] = x[j] > 0.0f ? x[j] : 0.0f;
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

void reference(const Params& p, const std::vector<float>& in, std::vector<float>& out)
{
  out.assign(static_cast<size_t>(p.n), 0.0f);
  for (int64_t j = 0; j < p.n; ++j) {
    const float x = in[static_cast<size_t>(j)];
    if (p.leaky)
      out[static_cast<size_t>(j)] = x > 0.0f ? x : 0.01f * x;
    else
      out[static_cast<size_t>(j)] = x > 0.0f ? x : 0.0f;
  }
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b)
{
  double worst = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    worst = std::max(worst, static_cast<double>(std::fabs(a[i] - b[i])));
  return worst;
}

double run_sync(const Params& p, std::vector<float>& in, std::vector<float>& out)
{
  const double start = now_ms();
  for (int64_t j = 0; j < p.n; ++j) {
    const float x = in[static_cast<size_t>(j)];
    if (p.leaky)
      out[static_cast<size_t>(j)] = x > 0.0f ? x : 0.01f * x;
    else
      out[static_cast<size_t>(j)] = x > 0.0f ? x : 0.0f;
  }
  return now_ms() - start;
}

// Single buffer: every producer is waited before its consumer. This is the
// shape the current schedule emits, i.e. the baseline amortization has to beat.
double run_serial(const Params& p, std::vector<float>& in, std::vector<float>& out)
{
  const int64_t B  = p.block;
  const int64_t nb = num_blocks(p);

  float* in_sram   = static_cast<float*>(alloc(B * 4, 64));
  float* out_sram  = static_cast<float*>(alloc(B * 4, 64));
  float* zero      = static_cast<float*>(alloc(B * 4, 64));
  float* hundredth = static_cast<float*>(alloc(B * 4, 64));
  int8_t* mask     = static_cast<int8_t*>(alloc(B, 64));
  float* scaled    = static_cast<float*>(alloc(B * 4, 64));
  fill_kernel(zero, 0.0f, B);
  fill_kernel(hundredth, 0.01f, B);
  fill_kernel(in_sram, 0.0f, B); // hoisted padding, elementwise only (see 6.3)

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
      async_copy_dram_to_sram<1>(
        h_in, m_in, in_sram, in.data() + off, sizeof(float), ext, int64_t{1}, int64_t{1});
      pumem_wait(h_in);
      if (p.leaky) {
        pumem_async_dispatch(cmpf_kernel, in_sram, zero, mask, B);
        pumem_async_dispatch(mulf_kernel, in_sram, hundredth, scaled, B);
        pumem_wait_dispatch();
        pumem_async_dispatch(select_kernel, mask, in_sram, scaled, out_sram, B);
      } else {
        pumem_async_dispatch(relu_kernel, in_sram, out_sram, B);
      }
      pumem_wait_dispatch();
      async_copy_sram_to_dram<1>(
        h_out, m_out, out.data() + off, out_sram, sizeof(float), ext, int64_t{1}, int64_t{1});
      pumem_wait(h_out);
    }
  }
  pumem_sync();
  const double ms = now_ms() - start;

  dealloc(in_sram);
  dealloc(out_sram);
  dealloc(zero);
  dealloc(hundredth);
  dealloc(mask);
  dealloc(scaled);
  return ms;
}

// U tiles per group. U=1 is the design's steady state:
//   prefetch L(i+N-1), wait L(i), wait S(i-S), dispatch body, barrier, store.
// U>1 hoists the barriers out of the per-tile loop:
//   for u: dispatch body(u) -> one barrier -> for u: dispatch tail(u)
//   -> one barrier -> for u: store(u)
// The prefetch lead becomes one whole group (U tiles), which is the price of
// amortizing: stages_in >= 2*U keeps the consumed and prefetched slots disjoint.
double run_pipeline(const Params& p, int64_t U, std::vector<float>& in, std::vector<float>& out)
{
  const int64_t B  = p.block;
  const int64_t nb = num_blocks(p);
  const int64_t N  = p.stages_in;
  const int64_t S  = p.stages_out;
  const int64_t SU = p.shared_scratch ? 1 : U;

  float* in_all   = static_cast<float*>(alloc(N * B * 4, 64));
  float* out_all  = static_cast<float*>(alloc(S * B * 4, 64));
  float* zero     = static_cast<float*>(alloc(B * 4, 64));
  float* hundredth = static_cast<float*>(alloc(B * 4, 64));
  int8_t* mask    = p.leaky ? static_cast<int8_t*>(alloc(SU * B, 64)) : nullptr;
  float* scaled   = p.leaky ? static_cast<float*>(alloc(SU * B * 4, 64)) : nullptr;
  fill_kernel(zero, 0.0f, B);
  fill_kernel(hundredth, 0.01f, B);

  const double start = now_ms();
  // Hoisted padding over every input slot once. Safe for elementwise because the
  // store extent equals the load extent, so OOB lanes die at the masked store;
  // reductions must keep the per-tile fill or a dedicated tail slot instead.
  pumem_async_dispatch(fill_kernel, in_all, 0.0f, N * B);
  pumem_wait_dispatch();

  for (int64_t tid = 0; tid < p.threads; ++tid) {
    set_physical_threads_id(tid);
    std::vector<pumem_dma_handle_t> h_in(static_cast<size_t>(N));
    std::vector<pumem_dma_meta_t>   m_in(static_cast<size_t>(N));
    std::vector<pumem_dma_handle_t> h_out(static_cast<size_t>(S));
    std::vector<pumem_dma_meta_t>   m_out(static_cast<size_t>(S));

    const int64_t K = nb > tid ? (nb - tid + p.threads - 1) / p.threads : 0;
    auto pid_at = [&](int64_t i) { return tid + i * p.threads; };
    auto ext_at = [&](int64_t i) { return std::min(B, p.n - pid_at(i) * B); };
    auto load   = [&](int64_t tile) {
      const int64_t slot = tile % N;
      async_copy_dram_to_sram<1>(
        h_in[slot], m_in[slot], in_all + slot * B, in.data() + pid_at(tile) * B, sizeof(float),
        ext_at(tile), int64_t{1}, int64_t{1});
    };

    // Prologue: first group has no producer to wait for, so no wait is emitted.
    for (int64_t t = 0; t < std::min(U, K); ++t) load(t);

    for (int64_t i = 0; i < K; i += U) {
      const int64_t G = std::min(U, K - i);

      // Prefetch the next group. No wait: slot (i+U+u) % N last held tile
      // (i+U+u) - N, which was consumed in an earlier group and waited there.
      for (int64_t u = 0; u < G; ++u) {
        const int64_t t = i + U + u;
        if (t < K) load(t);
      }

      for (int64_t u = 0; u < G; ++u) {
        pumem_wait(h_in[(i + u) % N]);   // L1 (1) RAW on the input slot
        pumem_wait(h_out[(i + u) % S]);  // L1 (2) WAR on the output slot
      }

      if (p.leaky) {
        for (int64_t u = 0; u < G; ++u) {
          const int64_t  si = p.shared_scratch ? 0 : u;
          const int64_t  sc = (i + u) % N;
          pumem_async_dispatch(cmpf_kernel, in_all + sc * B, zero, mask + si * B, B);
          pumem_async_dispatch(mulf_kernel, in_all + sc * B, hundredth, scaled + si * B, B);
        }
        pumem_wait_dispatch(); // ONE barrier for the whole group's select inputs
        for (int64_t u = 0; u < G; ++u) {
          const int64_t si = p.shared_scratch ? 0 : u;
          const int64_t sc = (i + u) % N;
          const int64_t so = (i + u) % S;
          pumem_async_dispatch(
            select_kernel, mask + si * B, in_all + sc * B, scaled + si * B, out_all + so * B, B);
        }
      } else {
        for (int64_t u = 0; u < G; ++u) {
          const int64_t sc = (i + u) % N;
          const int64_t so = (i + u) % S;
          pumem_async_dispatch(relu_kernel, in_all + sc * B, out_all + so * B, B);
        }
      }

      pumem_wait_dispatch(); // ONE barrier for the whole group's stores

      for (int64_t u = 0; u < G; ++u) {
        const int64_t so = (i + u) % S;
        async_copy_sram_to_dram<1>(
          h_out[so], m_out[so], out.data() + pid_at(i + u) * B, out_all + so * B, sizeof(float),
          ext_at(i + u), int64_t{1}, int64_t{1});
      }
    }

    if (K > 0) {
      pumem_wait_dispatch();
      pumem_wait(h_out[(K - 1) % S]);
    }
  }
  pumem_sync();
  const double ms = now_ms() - start;

  dealloc(in_all);
  dealloc(out_all);
  dealloc(zero);
  dealloc(hundredth);
  if (mask) dealloc(mask);
  if (scaled) dealloc(scaled);
  return ms;
}

struct Result
{
  double ms      = 0.0;
  double max_err = 0.0;
};

Result measure(int mode, const Params& p, int64_t U, const std::vector<float>& ref)
{
  std::vector<float> in(static_cast<size_t>(p.n));
  std::vector<float> out(static_cast<size_t>(p.n), 0.0f);
  fill_input(in);

  Result best;
  best.ms = 1e30;
  for (int r = 0; r < p.repeats; ++r) {
    std::fill(out.begin(), out.end(), 0.0f);
    double ms = 0.0;
    switch (mode) {
      case 0: ms = run_sync(p, in, out); break;
      case 1: ms = run_serial(p, in, out); break;
      default: ms = run_pipeline(p, U, in, out); break;
    }
    best.ms      = std::min(best.ms, ms);
    best.max_err = std::max(best.max_err, max_abs_diff(out, ref));
  }
  return best;
}

double gbps(const Params& p, double ms)
{
  const double bytes = 2.0 * static_cast<double>(p.n) * sizeof(float);
  return ms > 0.0 ? bytes / ms / 1e6 : 0.0;
}

void print_row(
  const char* label, const Params& p, const Result& r, double serial_ms, double barriers_per_tile
)
{
  std::printf(
    "  %-14s %9.3f ms  %7.2f GB/s  %6.1f us/block  barrier/block=%.2f  err=%.1g", label, r.ms,
    gbps(p, r.ms), r.ms * 1000.0 / static_cast<double>(num_blocks(p)), barriers_per_tile, r.max_err);
  if (serial_ms > 0.0 && r.ms > 0.0) std::printf("  speedup-vs-serial=%.2fx", serial_ms / r.ms);
  std::printf("\n");
}

void run_case(Params p)
{
  const int64_t max_u = std::max<int64_t>(1, std::min(p.stages_out, p.stages_in / 2));
  std::vector<float> in(static_cast<size_t>(p.n));
  std::vector<float> ref;
  fill_input(in);
  reference(p, in, ref);

  std::printf(
    "case n=%lld block=%lld threads=%lld blocks=%lld body=%s stages=%lld/%lld repeats=%d%s\n",
    static_cast<long long>(p.n), static_cast<long long>(p.block), static_cast<long long>(p.threads),
    static_cast<long long>(num_blocks(p)), p.leaky ? "leaky_relu" : "relu",
    static_cast<long long>(p.stages_in), static_cast<long long>(p.stages_out), p.repeats,
    p.shared_scratch ? " shared-scratch" : "");

  const double base_barriers = p.leaky ? 2.0 : 1.0;
  const Result sync   = measure(0, p, 0, ref);
  const Result serial = measure(1, p, 0, ref);
  print_row("sync", p, sync, 0.0, 0.0);
  print_row("serial", p, serial, 0.0, base_barriers);

  int64_t last = 0;
  for (int64_t u : p.unroll) {
    const int64_t U = std::max<int64_t>(1, std::min<int64_t>(u, max_u));
    if (U == last) continue; // two requested factors can clamp onto the same U
    last = U;
    char label[32];
    std::snprintf(label, sizeof(label), "u=%lld", static_cast<long long>(U));
    const Result r = measure(2, p, U, ref);
    print_row(label, p, r, serial.ms, base_barriers / static_cast<double>(U));
    if (U != u)
      std::printf(
        "                 (u=%lld clamped to %lld: needs out_stages >= u and stages >= 2u)\n",
        static_cast<long long>(u), static_cast<long long>(U));
  }
  std::printf("\n");
}

bool parse(int argc, char** argv, Params& p, std::vector<Params>& cases)
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
    else if (arg == "--relu") p.leaky = false;
    else if (arg == "--shared-scratch") p.shared_scratch = true;
    else if (arg == "--u") {
      // comma separated unroll factors
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for --u\n");
        return false;
      }
      const std::string list = argv[++i];
      p.unroll.clear();
      size_t pos = 0;
      while (pos <= list.size()) {
        const size_t comma = list.find(',', pos);
        const std::string item = list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!item.empty()) p.unroll.push_back(std::strtoll(item.c_str(), nullptr, 10));
        if (comma == std::string::npos) break;
        pos = comma + 1;
      }
      if (p.unroll.empty()) p.unroll.push_back(1);
    } else if (arg == "--case") {
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
  std::vector<Params> cases;
  if (!parse(argc, argv, p, cases)) return 2;

  std::printf(
    "hw threads=%u  pool workers=%u\n", std::thread::hardware_concurrency(),
    std::max(2u, std::min(std::thread::hardware_concurrency(), 16u)));

  if (!cases.empty()) {
    for (const Params& q : cases) run_case(q);
  } else {
    run_case(p);
  }
  return 0;
}
