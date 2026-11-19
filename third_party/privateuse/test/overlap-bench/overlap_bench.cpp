// Compare the CPU async runtime's throughput with and without copy/compute
// overlap on a tile loop (load tile -> compute tile -> store tile).
//
// Modes:
//   sync    : host-side box_copy + compute, no runtime pool at all
//   serial  : async copies, but every copy/dispatch is waited before the next
//             step (the shape currently emitted by puc-translate)
//   overlap : N input stages / S output stages, prefetch issued N-1 iterations
//             ahead, waits only on the consumer's own slot (v2 statement form)
//
// Every mode writes the same output; results are checked bit-exact against a
// host reference computed with the same kernel.
//
// Build/run: ./run.sh [--tiles K] [--tile T] [--inner I] [--sweep]

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
  int64_t tiles     = 256;
  int64_t tile      = 16384; // floats per tile
  int64_t inner     = 32;    // FMA rounds per element in the compute kernel
  int64_t stages_in = 2;
  int64_t stages_out= 2;
  int     repeats   = 3;
  bool    single    = false; // run one overlap config instead of 2/3/4 sweep
  bool    v3_waits  = false; // skip the pre-prefetch wait (pipeliner-emitted rule)
};

double now_ms()
{
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

void compute_tile(float* out, const float* in, int64_t n, int64_t inner)
{
  for (int64_t j = 0; j < n; ++j) {
    float acc = in[j];
    for (int64_t r = 0; r < inner; ++r)
      acc = acc * 1.0000001f + 0.5f;
    out[j] = acc > 0.0f ? acc : 0.01f * acc;
  }
}

void reference(const Params& p, const std::vector<float>& in, std::vector<float>& out)
{
  for (int64_t i = 0; i < p.tiles; ++i)
    compute_tile(out.data() + i * p.tile, in.data() + i * p.tile, p.tile, p.inner);
}

void fill_input(std::vector<float>& in)
{
  for (size_t j = 0; j < in.size(); ++j)
    in[j] = std::sin(0.001f * static_cast<float>(j)) * 2.0f - 1.0f;
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
  std::vector<float> s_in(static_cast<size_t>(p.tile));
  std::vector<float> s_out(static_cast<size_t>(p.tile));
  const size_t n       = static_cast<size_t>(p.tile);
  const size_t one     = 1;
  const size_t elem    = sizeof(float);
  const double start   = now_ms();
  for (int64_t i = 0; i < p.tiles; ++i) {
    detail::box_copy(s_in.data(), in.data() + i * p.tile, 1, &n, &one, &one, elem);
    compute_tile(s_out.data(), s_in.data(), p.tile, p.inner);
    detail::box_copy(out.data() + i * p.tile, s_out.data(), 1, &n, &one, &one, elem);
  }
  return now_ms() - start;
}

double run_serial(const Params& p, std::vector<float>& in, std::vector<float>& out)
{
  std::vector<float> s_in(static_cast<size_t>(p.tile));
  std::vector<float> s_out(static_cast<size_t>(p.tile));
  pumem_dma_handle_t h_in;
  pumem_dma_meta_t   m_in;
  pumem_dma_handle_t h_out;
  pumem_dma_meta_t   m_out;
  const double start = now_ms();
  for (int64_t i = 0; i < p.tiles; ++i) {
    async_copy_dram_to_sram<1>(
      h_in, m_in, s_in.data(), in.data() + i * p.tile, sizeof(float), p.tile, int64_t{1}, int64_t{1});
    pumem_wait(h_in);
    pumem_async_dispatch(compute_tile, s_out.data(), s_in.data(), p.tile, p.inner);
    pumem_wait_dispatch();
    async_copy_sram_to_dram<1>(
      h_out, m_out, out.data() + i * p.tile, s_out.data(), sizeof(float), p.tile, int64_t{1}, int64_t{1});
    pumem_wait(h_out);
  }
  pumem_sync();
  return now_ms() - start;
}

double run_overlap(const Params& p, std::vector<float>& in, std::vector<float>& out)
{
  const int64_t N = p.stages_in;
  const int64_t S = p.stages_out;
  std::vector<float> s_in(static_cast<size_t>(N * p.tile));
  std::vector<float> s_out(static_cast<size_t>(S * p.tile));
  std::vector<pumem_dma_handle_t> h_in(static_cast<size_t>(N));
  std::vector<pumem_dma_handle_t> h_out(static_cast<size_t>(S));
  std::vector<pumem_dma_meta_t>   m_in(static_cast<size_t>(N));
  std::vector<pumem_dma_meta_t>   m_out(static_cast<size_t>(S));

  const double start = now_ms();
  const int64_t pro  = std::min(N - 1, p.tiles);
  for (int64_t s = 0; s < pro; ++s) {
    async_copy_dram_to_sram<1>(
      h_in[s], m_in[s], s_in.data() + s * p.tile, in.data() + s * p.tile, sizeof(float), p.tile, int64_t{1},
      int64_t{1});
  }
  for (int64_t i = 0; i < p.tiles; ++i) {
    const int64_t pf = i + N - 1;
    if (pf < p.tiles) {
      const int64_t s = pf % N;
      if (!p.v3_waits)
        pumem_wait(h_in[s]);
      async_copy_dram_to_sram<1>(
        h_in[s], m_in[s], s_in.data() + s * p.tile, in.data() + pf * p.tile, sizeof(float), p.tile, int64_t{1},
        int64_t{1});
    }
    const int64_t c = i % N;
    pumem_wait(h_in[c]);
    const int64_t so = i % S;
    pumem_wait(h_out[so]);
    pumem_async_dispatch(compute_tile, s_out.data() + so * p.tile, s_in.data() + c * p.tile, p.tile, p.inner);
    pumem_wait_dispatch();
    async_copy_sram_to_dram<1>(
      h_out[so], m_out[so], out.data() + i * p.tile, s_out.data() + so * p.tile, sizeof(float), p.tile, int64_t{1},
      int64_t{1});
  }
  pumem_sync();
  return now_ms() - start;
}

struct Result
{
  double ms      = 0.0;
  double max_err = 0.0;
};

Result measure(int mode, const Params& p, const std::vector<float>& ref)
{
  std::vector<float> in(static_cast<size_t>(p.tiles * p.tile));
  std::vector<float> out(static_cast<size_t>(p.tiles * p.tile), 0.0f);
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
  const double bytes = 2.0 * static_cast<double>(p.tiles) * static_cast<double>(p.tile) * sizeof(float);
  return ms > 0.0 ? bytes / ms / 1e6 : 0.0;
}

void print_row(const char* label, const Params& p, const Result& r, double serial_ms)
{
  std::printf(
    "  %-8s stages=%lld/%lld  %8.2f ms  %7.1f GB/s  %6.1f us/tile  err=%.1g", label,
    static_cast<long long>(p.stages_in), static_cast<long long>(p.stages_out), r.ms, gbps(p, r.ms),
    r.ms * 1000.0 / static_cast<double>(p.tiles), r.max_err);
  if (serial_ms > 0.0 && r.ms > 0.0)
    std::printf("  speedup-vs-serial=%.2fx", serial_ms / r.ms);
  std::printf("\n");
}

void run_case(const Params& p)
{
  std::vector<float> in(static_cast<size_t>(p.tiles * p.tile));
  std::vector<float> ref(static_cast<size_t>(p.tiles * p.tile));
  fill_input(in);
  reference(p, in, ref);

  std::printf(
    "case tiles=%lld tile=%lld inner=%lld repeats=%d wait=%s\n", static_cast<long long>(p.tiles),
    static_cast<long long>(p.tile), static_cast<long long>(p.inner), p.repeats, p.v3_waits ? "v3" : "v2");

  const Result sync = measure(0, p, ref);
  const Result serial = measure(1, p, ref);
  print_row("sync", p, sync, 0.0);
  print_row("serial", p, serial, 0.0);

  Params q = p;
  if (p.single) {
    const Result overlap = measure(2, q, ref);
    print_row("overlap", q, overlap, serial.ms);
  } else {
    for (const int64_t n : {int64_t{2}, int64_t{3}, int64_t{4}}) {
      q.stages_in  = n;
      q.stages_out = std::min<int64_t>(n, 2);
      const Result overlap = measure(2, q, ref);
      print_row("overlap", q, overlap, serial.ms);
    }
  }
  std::printf("\n");
}

void run_edge_cases()
{
  std::printf("edge cases (K < N, K == 1)\n");
  for (const auto& [k, n] : {std::pair<int64_t, int64_t>{1, 4}, {2, 4}, {3, 4}, {5, 2}}) {
    Params p;
    p.tiles      = k;
    p.tile       = 128;
    p.inner      = 4;
    p.stages_in  = n;
    p.stages_out = 2;
    p.repeats    = 2;
    std::vector<float> in(static_cast<size_t>(k * p.tile));
    std::vector<float> ref(static_cast<size_t>(k * p.tile));
    fill_input(in);
    reference(p, in, ref);
    const Result r = measure(2, p, ref);
    std::printf(
      "  K=%lld N=%lld  %7.3f ms  err=%.1g\n", static_cast<long long>(k), static_cast<long long>(n), r.ms,
      r.max_err);
  }
  std::printf("\n");
}

bool parse(int argc, char** argv, Params& p, bool& sweep, bool& edge)
{
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> int64_t {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", arg.c_str());
        std::exit(2);
      }
      return std::strtoll(argv[++i], nullptr, 10);
    };
    if (arg == "--tiles") p.tiles = next();
    else if (arg == "--tile") p.tile = next();
    else if (arg == "--inner") p.inner = next();
    else if (arg == "--stages") p.stages_in = next();
    else if (arg == "--out-stages") p.stages_out = next();
    else if (arg == "--repeats") p.repeats = static_cast<int>(next());
    else if (arg == "--sweep") sweep = true;
    else if (arg == "--edge") edge = true;
    else if (arg == "--single") p.single = true;
    else if (arg == "--v3") p.v3_waits = true;
    else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return false;
    }
  }
  return true;
}

} // namespace

int main(int argc, char** argv)
{
  Params p;
  bool sweep = false;
  bool edge  = false;
  if (!parse(argc, argv, p, sweep, edge)) return 2;

  std::printf("hw threads=%u  pool workers=%u\n", std::thread::hardware_concurrency(),
              std::max(2u, std::min(std::thread::hardware_concurrency(), 16u)));

  if (edge) run_edge_cases();

  if (sweep) {
    for (const int64_t inner : {int64_t{0}, int64_t{16}, int64_t{64}, int64_t{256}}) {
      Params q = p;
      q.inner  = inner;
      run_case(q);
    }
  } else {
    run_case(p);
  }
  return 0;
}