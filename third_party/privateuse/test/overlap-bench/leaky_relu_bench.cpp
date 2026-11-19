// Leaky-relu tile pipeline benchmark for the CPU async runtime.
//
// The compute chain mirrors docs/pumem-async-codegen-example.cpp exactly:
//   partial tile: fill(tile, 0) -> load(prefix) -> cmpf(x > 0, mask)
//                 -> mulf(x, 0.01) -> select(mask, x, x*0.01) -> store(prefix)
// with block = 256 and 32 physical ids in the original kernel.
//
// Modes:
//   sync    : host-side elementwise leaky-relu, no runtime
//   serial  : the currently emitted shape: one buffer, every op waited
//   overlap : N input stages / S output stages; the per-tile zero fill is
//             hoisted out of the loop (redundant here: store extent == load
//             extent and the body is elementwise, so OOB lanes are discarded
//             by the masked store; do not hoist it for reductions), loads are
//             prefetched N-1 iterations ahead, stores run S-deep
//
// Build/run: ./run-leaky-relu.sh [--n N] [--block B] [--threads T]
//                               [--stages N] [--out-stages S] [--repeats R]
//                               [--v3] [--case original|<n>:<block>:<threads>]

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
  int64_t n         = 48256;
  int64_t block     = 256;
  int64_t threads   = 32;
  int64_t stages_in = 2;
  int64_t stages_out= 2;
  int     repeats   = 3;
  bool    v3_waits  = false; // pipeliner-emitted rule: no wait before prefetch
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
    out[static_cast<size_t>(j)] = x > 0.0f ? x : 0.01f * x;
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
    out[static_cast<size_t>(j)] = x > 0.0f ? x : 0.01f * x;
  }
  return now_ms() - start;
}

double run_serial(const Params& p, std::vector<float>& in, std::vector<float>& out)
{
  const int64_t B  = p.block;
  const int64_t nb = num_blocks(p);

  float* in_sram  = static_cast<float*>(alloc(B * 4, 64));
  float* zero     = static_cast<float*>(alloc(B * 4, 64));
  float* hundredth= static_cast<float*>(alloc(B * 4, 64));
  int8_t* mask    = static_cast<int8_t*>(alloc(B, 64));
  float* scaled   = static_cast<float*>(alloc(B * 4, 64));
  float* out_sram = static_cast<float*>(alloc(B * 4, 64));
  fill_kernel(zero, 0.0f, B);
  fill_kernel(hundredth, 0.01f, B);

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
      pumem_async_dispatch(cmpf_kernel, in_sram, zero, mask, B);
      pumem_async_dispatch(mulf_kernel, in_sram, hundredth, scaled, B);
      pumem_wait_dispatch();
      pumem_async_dispatch(select_kernel, mask, in_sram, scaled, out_sram, B);
      pumem_wait_dispatch();
      async_copy_sram_to_dram<1>(
        h_out, m_out, out.data() + off, out_sram, sizeof(float), ext, int64_t{1}, int64_t{1});
      pumem_wait(h_out);
    }
  }
  pumem_sync();
  const double ms = now_ms() - start;

  dealloc(in_sram);
  dealloc(zero);
  dealloc(hundredth);
  dealloc(mask);
  dealloc(scaled);
  dealloc(out_sram);
  return ms;
}

double run_overlap(const Params& p, std::vector<float>& in, std::vector<float>& out)
{
  const int64_t B  = p.block;
  const int64_t nb = num_blocks(p);
  const int64_t N  = p.stages_in;
  const int64_t S  = p.stages_out;

  float* in_all   = static_cast<float*>(alloc(N * B * 4, 64));
  float* out_all  = static_cast<float*>(alloc(S * B * 4, 64));
  float* zero     = static_cast<float*>(alloc(B * 4, 64));
  float* hundredth= static_cast<float*>(alloc(B * 4, 64));
  int8_t* mask    = static_cast<int8_t*>(alloc(B, 64));
  float* scaled   = static_cast<float*>(alloc(B * 4, 64));
  fill_kernel(zero, 0.0f, B);
  fill_kernel(hundredth, 0.01f, B);

  const double start = now_ms();
  // Hoisted tail padding: zero every input slot once, before the loop. This is
  // not what makes the partial tile correct: a reused slot still holds lanes of
  // a previous full tile past the loaded prefix. It is safe here because the
  // store extent equals the load extent and the body is elementwise, so lanes
  // beyond the prefix die at the masked store. Reductions over the masked
  // dimension (e.g. matmul masked-K) must keep the per-tile fill instead.
  pumem_async_dispatch(fill_kernel, in_all, 0.0f, N * B);
  pumem_wait_dispatch();

  for (int64_t tid = 0; tid < p.threads; ++tid) {
    set_physical_threads_id(tid);
    std::vector<pumem_dma_handle_t> h_in(static_cast<size_t>(N));
    std::vector<pumem_dma_handle_t> h_out(static_cast<size_t>(S));
    std::vector<pumem_dma_meta_t>   m_in(static_cast<size_t>(N));
    std::vector<pumem_dma_meta_t>   m_out(static_cast<size_t>(S));

    const int64_t K = nb > tid ? (nb - tid + p.threads - 1) / p.threads : 0;
    auto pid_at = [&](int64_t i) { return tid + i * p.threads; };

    for (int64_t s = 0; s < std::min(N - 1, K); ++s) {
      const int64_t pid = pid_at(s);
      const int64_t off = pid * B;
      const int64_t ext = std::min(B, p.n - off);
      async_copy_dram_to_sram<1>(
        h_in[s], m_in[s], in_all + s * B, in.data() + off, sizeof(float), ext, int64_t{1}, int64_t{1});
    }

    for (int64_t i = 0; i < K; ++i) {
      const int64_t pf = i + N - 1;
      if (pf < K) {
        const int64_t s   = pf % N;
        const int64_t pid = pid_at(pf);
        const int64_t off = pid * B;
        const int64_t ext = std::min(B, p.n - off);
        if (!p.v3_waits) pumem_wait(h_in[s]);
        async_copy_dram_to_sram<1>(
          h_in[s], m_in[s], in_all + s * B, in.data() + off, sizeof(float), ext, int64_t{1}, int64_t{1});
      }
      const int64_t sc = i % N;
      const int64_t so = i % S;
      pumem_wait(h_in[sc]);
      pumem_wait(h_out[so]);
      pumem_async_dispatch(cmpf_kernel, in_all + sc * B, zero, mask, B);
      pumem_async_dispatch(mulf_kernel, in_all + sc * B, hundredth, scaled, B);
      pumem_wait_dispatch();
      pumem_async_dispatch(select_kernel, mask, in_all + sc * B, scaled, out_all + so * B, B);
      pumem_wait_dispatch();
      const int64_t pid = pid_at(i);
      const int64_t off = pid * B;
      const int64_t ext = std::min(B, p.n - off);
      async_copy_sram_to_dram<1>(
        h_out[so], m_out[so], out.data() + off, out_all + so * B, sizeof(float), ext, int64_t{1},
        int64_t{1});
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
  dealloc(mask);
  dealloc(scaled);
  return ms;
}

struct Result
{
  double ms      = 0.0;
  double max_err = 0.0;
};

Result measure(int mode, const Params& p, const std::vector<float>& ref)
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
      default: ms = run_overlap(p, in, out); break;
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

void print_row(const char* label, const Params& p, const Result& r, double serial_ms)
{
  std::printf(
    "  %-8s stages=%lld/%lld  %9.3f ms  %7.2f GB/s  %6.1f us/block  err=%.1g", label,
    static_cast<long long>(p.stages_in), static_cast<long long>(p.stages_out), r.ms, gbps(p, r.ms),
    r.ms * 1000.0 / static_cast<double>(num_blocks(p)), r.max_err);
  if (serial_ms > 0.0 && r.ms > 0.0)
    std::printf("  speedup-vs-serial=%.2fx", serial_ms / r.ms);
  std::printf("\n");
}

void run_case(Params p)
{
  std::vector<float> in(static_cast<size_t>(p.n));
  std::vector<float> ref;
  fill_input(in);
  reference(p, in, ref);

  std::printf(
    "case n=%lld block=%lld threads=%lld blocks=%lld repeats=%d wait=%s\n", static_cast<long long>(p.n),
    static_cast<long long>(p.block), static_cast<long long>(p.threads),
    static_cast<long long>(num_blocks(p)), p.repeats, p.v3_waits ? "v3" : "v2");

  const Result sync   = measure(0, p, ref);
  const Result serial = measure(1, p, ref);
  print_row("sync", p, sync, 0.0);
  print_row("serial", p, serial, 0.0);
  const Result overlap = measure(2, p, ref);
  print_row("overlap", p, overlap, serial.ms);
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
    else if (arg == "--matrix") matrix = true;
    else if (arg == "--case") {
      // <n>:<block>:<threads>
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
    run_case(p); // --n/--block as given
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