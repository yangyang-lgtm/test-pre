#ifndef PUC_RUNTIME_H
#define PUC_RUNTIME_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#define DRAM
#define SRAM
#define GLOBAL
#define DEVICE
#define DEVICE_INLINE
#define OUTLINED
#define MEM0
#define MEM1
#define MEM2
#define MEM3
#define MEM4
#define ALIGNED(align)
#define outlined

void set_physical_threads_id(int64_t id);
int64_t get_physical_id();

void* alloc(int64_t bytes, int64_t alignment);
void dealloc(void* ptr);
void reset_alloc();

inline void pumem_sync() {}
inline void pumem_wait() {}

namespace detail {
void box_copy(
  void* dst, const void* src, size_t rank, const size_t* sizes, const size_t* dst_strides, const size_t* src_strides,
  size_t elem_size
);

template <size_t Offset, typename Tuple, size_t... Is>
void fill_ints(int64_t* out, const Tuple& tuple, std::index_sequence<Is...>)
{
  ((out[Is] = static_cast<int64_t>(std::get<Offset + Is>(tuple))), ...);
}

template <int Rank, typename... Indices> void copy_box(void* dst, void* src, int64_t elem_size, Indices... indices)
{
  static_assert(sizeof...(Indices) == 3 * size_t(Rank), "copy: expected sizes, dst_strides and src_strides");
  size_t all[] = {static_cast<size_t>(indices)...};
  box_copy(dst, src, size_t(Rank), all, all + Rank, all + 2 * Rank, size_t(elem_size));
}
} // namespace detail

template <int Rank, typename... Indices>
void copy_dram_to_dram(void* dst, void* src, int64_t elem_size, Indices... indices)
{
  detail::copy_box<Rank>(dst, src, elem_size, indices...);
}

template <int Rank, typename... Indices>
void copy_dram_to_sram(void* dst, void* src, int64_t elem_size, Indices... indices)
{
  detail::copy_box<Rank>(dst, src, elem_size, indices...);
}

template <int Rank, typename... Indices>
void copy_sram_to_dram(void* dst, void* src, int64_t elem_size, Indices... indices)
{
  detail::copy_box<Rank>(dst, src, elem_size, indices...);
}

template <int Rank, typename... Indices>
void copy_sram_to_sram(void* dst, void* src, int64_t elem_size, Indices... indices)
{
  detail::copy_box<Rank>(dst, src, elem_size, indices...);
}

struct float16
{
  uint16_t value;

  constexpr float16() : value(0) {}
  constexpr float16(float v) : value(toBits(v)) {}
  constexpr operator float() const { return toFloat(value); }

private:
  static constexpr uint16_t toBits(float v)
  {
    uint32_t bits = __builtin_bit_cast(uint32_t, v);
    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t exp  = (bits >> 23) & 0xFFu;
    uint32_t mant = bits & 0x7FFFFFu;

    if (exp == 0xFFu) return uint16_t(sign | 0x7C00u | (mant != 0 ? 0x200u : 0u));

    int32_t halfExp = int32_t(exp) - 127 + 15;
    if (halfExp >= 0x1F) return uint16_t(sign | 0x7C00u);

    if (halfExp <= 0) {
      if (halfExp < -10) return uint16_t(sign);
      mant             |= 0x800000u;
      uint32_t shift    = uint32_t(14 - halfExp);
      uint32_t rounded  = mant + (1u << (shift - 1)) - 1 + ((mant >> shift) & 1u);
      return uint16_t(sign | (rounded >> shift));
    }

    uint32_t rounded = mant + 0x0FFFu + ((mant >> 13) & 1u);
    if (rounded & 0x800000u) {
      rounded = 0;
      ++halfExp;
      if (halfExp >= 0x1F) return uint16_t(sign | 0x7C00u);
    }
    return uint16_t(sign | (uint32_t(halfExp) << 10) | (rounded >> 13));
  }

  static constexpr float toFloat(uint16_t half)
  {
    uint32_t sign = (uint32_t(half) & 0x8000u) << 16;
    uint32_t exp  = (half >> 10) & 0x1Fu;
    uint32_t mant = half & 0x3FFu;
    uint32_t bits = 0;

    if (exp == 0) {
      if (mant == 0) {
        bits = sign;
      } else {
        int32_t shift = 0;
        while ((mant & 0x400u) == 0) {
          mant <<= 1;
          ++shift;
        }
        mant &= 0x3FFu;
        bits  = sign | (uint32_t(113 - shift) << 23) | (mant << 13);
      }
    } else if (exp == 0x1F) {
      bits = sign | 0x7F800000u | (mant << 13);
    } else {
      bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    return __builtin_bit_cast(float, bits);
  }
};

struct bfloat16
{
  uint16_t value;

  constexpr bfloat16() : value(0) {}
  constexpr bfloat16(float v) : value(toBits(v)) {}
  constexpr operator float() const { return toFloat(value); }

private:
  static constexpr uint16_t toBits(float v)
  {
    uint32_t bits = __builtin_bit_cast(uint32_t, v);
    if ((bits & 0x7F800000u) != 0x7F800000u) bits += 0x7FFFu + ((bits >> 16) & 1u);
    return uint16_t(bits >> 16);
  }

  static constexpr float toFloat(uint16_t bits) { return __builtin_bit_cast(float, uint32_t(bits) << 16); }
};

inline float f32_pos_inf()
{
  return std::numeric_limits<float>::infinity();
}
inline float f32_neg_inf()
{
  return -std::numeric_limits<float>::infinity();
}
inline double f64_pos_inf()
{
  return std::numeric_limits<double>::infinity();
}
inline double f64_neg_inf()
{
  return -std::numeric_limits<double>::infinity();
}
inline float16 f16_pos_inf()
{
  return float16(f32_pos_inf());
}
inline float16 f16_neg_inf()
{
  return float16(f32_neg_inf());
}
inline bfloat16 bf16_pos_inf()
{
  return bfloat16(f32_pos_inf());
}
inline bfloat16 bf16_neg_inf()
{
  return bfloat16(f32_neg_inf());
}

template <typename T, int64_t M, int64_t N, int64_t K> void runtime_matmul(T* A, T* B, T* C)
{
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) acc += static_cast<float>(A[i * K + k]) * static_cast<float>(B[k * N + j]);
      C[i * N + j] = static_cast<T>(acc);
    }
  }
}

template <typename T, int64_t Rows, int64_t Cols> void runtime_transpose(T* src, T* dst)
{
  for (int64_t i = 0; i < Rows; ++i)
    for (int64_t j = 0; j < Cols; ++j) dst[j * Rows + i] = src[i * Cols + j];
}

namespace detail {

template <int Rank, typename T, typename I>
void gather_impl(
  const T* src, int64_t src_stride, const I* indices, T* dst, const int64_t (&sizes)[Rank],
  const int64_t (&index_strides)[Rank], const int64_t (&dst_strides)[Rank], const int8_t* mask,
  const int64_t (&mask_strides)[Rank], const T* other
)
{
  int64_t total = 1;
  for (int d = 0; d < Rank; ++d) total *= sizes[d];

  int64_t index[Rank] = {};
  for (int64_t flat = 0; flat < total; ++flat) {
    int64_t remaining = flat;
    for (int d = Rank - 1; d >= 0; --d) {
      index[d]   = remaining % sizes[d];
      remaining /= sizes[d];
    }
    int64_t offset = 0;
    for (int d = 0; d < Rank; ++d) offset += index[d] * dst_strides[d];

    bool on = true;
    if (mask != nullptr) {
      int64_t mask_offset = 0;
      for (int d = 0; d < Rank; ++d) mask_offset += index[d] * mask_strides[d];
      on = mask[mask_offset] != 0;
    }
    if (on) {
      int64_t index_offset = 0;
      for (int d = 0; d < Rank; ++d) index_offset += index[d] * index_strides[d];
      dst[offset] = src[int64_t(indices[index_offset]) * src_stride];
    } else if (other != nullptr) {
      dst[offset] = *other;
    }
  }
}

template <int Rank, typename T, typename I>
void scatter_impl(
  const T* src, const I* indices, T* dst, int64_t dst_stride, const int64_t (&sizes)[Rank],
  const int64_t (&src_strides)[Rank], const int64_t (&index_strides)[Rank], const int8_t* mask,
  const int64_t (&mask_strides)[Rank]
)
{
  int64_t total = 1;
  for (int d = 0; d < Rank; ++d) total *= sizes[d];

  int64_t index[Rank] = {};
  for (int64_t flat = 0; flat < total; ++flat) {
    int64_t remaining = flat;
    for (int d = Rank - 1; d >= 0; --d) {
      index[d]   = remaining % sizes[d];
      remaining /= sizes[d];
    }
    bool on = true;
    if (mask != nullptr) {
      int64_t mask_offset = 0;
      for (int d = 0; d < Rank; ++d) mask_offset += index[d] * mask_strides[d];
      on = mask[mask_offset] != 0;
    }
    if (!on) continue;

    int64_t src_offset = 0;
    for (int d = 0; d < Rank; ++d) src_offset += index[d] * src_strides[d];
    int64_t index_offset = 0;
    for (int d = 0; d < Rank; ++d) index_offset += index[d] * index_strides[d];
    dst[int64_t(indices[index_offset]) * dst_stride] = src[src_offset];
  }
}

} // namespace detail

template <int Rank, bool HasMask, bool HasOther, typename... Args> void gather(Args... args)
{
  static_assert(
    sizeof...(Args) == 4 + 3 * size_t(Rank) + (HasMask ? 1 + size_t(Rank) : 0) + (HasOther ? 1 : 0),
    "gather: unexpected operand count"
  );
  auto tuple = std::forward_as_tuple(args...);
  using T = std::remove_cv_t<std::remove_pointer_t<std::remove_reference_t<std::tuple_element_t<0, decltype(tuple)>>>>;
  using I = std::remove_cv_t<std::remove_pointer_t<std::remove_reference_t<std::tuple_element_t<2, decltype(tuple)>>>>;

  int64_t sizes[Rank];
  int64_t index_strides[Rank];
  int64_t dst_strides[Rank];
  int64_t mask_strides[Rank] = {};
  detail::fill_ints<4>(sizes, tuple, std::make_index_sequence<Rank>{});
  detail::fill_ints<4 + Rank>(index_strides, tuple, std::make_index_sequence<Rank>{});
  detail::fill_ints<4 + 2 * Rank>(dst_strides, tuple, std::make_index_sequence<Rank>{});

  const int8_t* mask = nullptr;
  if constexpr (HasMask) {
    mask = std::get<4 + 3 * Rank>(tuple);
    detail::fill_ints<4 + 3 * Rank + 1>(mask_strides, tuple, std::make_index_sequence<Rank>{});
  }

  T other_value{};
  const T* other = nullptr;
  if constexpr (HasOther) {
    other_value = static_cast<T>(std::get<4 + 3 * Rank + (HasMask ? 1 + Rank : 0)>(tuple));
    other       = &other_value;
  }

  detail::gather_impl<Rank>(
    std::get<0>(tuple), static_cast<int64_t>(std::get<1>(tuple)), std::get<2>(tuple), std::get<3>(tuple), sizes,
    index_strides, dst_strides, mask, mask_strides, other
  );
}

template <int Rank, bool HasMask, typename... Args> void scatter(Args... args)
{
  static_assert(
    sizeof...(Args) == 4 + 3 * size_t(Rank) + (HasMask ? 1 + size_t(Rank) : 0), "scatter: unexpected operand count"
  );
  auto tuple = std::forward_as_tuple(args...);
  using T = std::remove_cv_t<std::remove_pointer_t<std::remove_reference_t<std::tuple_element_t<0, decltype(tuple)>>>>;
  using I = std::remove_cv_t<std::remove_pointer_t<std::remove_reference_t<std::tuple_element_t<1, decltype(tuple)>>>>;

  int64_t sizes[Rank];
  int64_t src_strides[Rank];
  int64_t index_strides[Rank];
  int64_t mask_strides[Rank] = {};
  detail::fill_ints<4>(sizes, tuple, std::make_index_sequence<Rank>{});
  detail::fill_ints<4 + Rank>(src_strides, tuple, std::make_index_sequence<Rank>{});
  detail::fill_ints<4 + 2 * Rank>(index_strides, tuple, std::make_index_sequence<Rank>{});

  const int8_t* mask = nullptr;
  if constexpr (HasMask) {
    mask = std::get<4 + 3 * Rank>(tuple);
    detail::fill_ints<4 + 3 * Rank + 1>(mask_strides, tuple, std::make_index_sequence<Rank>{});
  }

  detail::scatter_impl<Rank>(
    std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple), static_cast<int64_t>(std::get<3>(tuple)), sizes,
    src_strides, index_strides, mask, mask_strides
  );
}

//===----------------------------------------------------------------------===//
// Scalar value operations
//===----------------------------------------------------------------------===//

template <class T> T scalar_addi(T a, T b)
{
  return a + b;
}
template <class T> T scalar_subi(T a, T b)
{
  return a - b;
}
template <class T> T scalar_muli(T a, T b)
{
  return a * b;
}

template <class T> T scalar_divsi(T a, T b)
{
  return a / b;
}
template <class T> T scalar_divui(T a, T b)
{
  return T(static_cast<std::make_unsigned_t<T>>(a) / static_cast<std::make_unsigned_t<T>>(b));
}
template <class T> T scalar_remsi(T a, T b)
{
  return a % b;
}
template <class T> T scalar_remui(T a, T b)
{
  return T(static_cast<std::make_unsigned_t<T>>(a) % static_cast<std::make_unsigned_t<T>>(b));
}
template <class T> T scalar_floordivsi(T a, T b)
{
  T q = a / b;
  T r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) --q;
  return q;
}
template <class T> T scalar_ceildivsi(T a, T b)
{
  T q = a / b;
  T r = a % b;
  if (r != 0 && ((r < 0) == (b < 0))) ++q;
  return q;
}
template <class T> T scalar_ceildivui(T a, T b)
{
  using U = std::make_unsigned_t<T>;
  return T(U(U(a) + U(b) - 1) / U(b));
}

template <class T> T scalar_andi(T a, T b)
{
  return a & b;
}
template <class T> T scalar_ori(T a, T b)
{
  return a | b;
}
template <class T> T scalar_xori(T a, T b)
{
  return a ^ b;
}
template <class T> T scalar_shli(T a, T b)
{
  return T(static_cast<std::make_unsigned_t<T>>(a) << b);
}
template <class T> T scalar_shrui(T a, T b)
{
  return T(static_cast<std::make_unsigned_t<T>>(a) >> b);
}
template <class T> T scalar_shrsi(T a, T b)
{
  return T(a >> b);
}

template <class T> T scalar_maxsi(T a, T b)
{
  return a < b ? b : a;
}
template <class T> T scalar_minsi(T a, T b)
{
  return a < b ? a : b;
}
template <class T> T scalar_maxui(T a, T b)
{
  using U = std::make_unsigned_t<T>;
  return U(a) < U(b) ? b : a;
}
template <class T> T scalar_minui(T a, T b)
{
  using U = std::make_unsigned_t<T>;
  return U(a) < U(b) ? a : b;
}

template <class T> T scalar_addf(T a, T b)
{
  return a + b;
}
template <class T> T scalar_subf(T a, T b)
{
  return a - b;
}
template <class T> T scalar_mulf(T a, T b)
{
  return a * b;
}
template <class T> T scalar_divf(T a, T b)
{
  return a / b;
}
template <class T> T scalar_remf(T a, T b)
{
  return std::fmod(a, b);
}
template <class T> T scalar_negf(T a)
{
  return -a;
}
template <class T> T scalar_maxnumf(T a, T b)
{
  return std::isnan(a) ? b : (std::isnan(b) ? a : (a < b ? b : a));
}
template <class T> T scalar_minnumf(T a, T b)
{
  return std::isnan(a) ? b : (std::isnan(b) ? a : (b < a ? b : a));
}
template <class T> T scalar_maximumf(T a, T b)
{
  if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<T>::quiet_NaN();
  return a < b ? b : a;
}
template <class T> T scalar_minimumf(T a, T b)
{
  if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<T>::quiet_NaN();
  return b < a ? b : a;
}

template <class T> int8_t scalar_cmpi_eq(T a, T b)
{
  return a == b;
}
template <class T> int8_t scalar_cmpi_ne(T a, T b)
{
  return a != b;
}
template <class T> int8_t scalar_cmpi_slt(T a, T b)
{
  return a < b;
}
template <class T> int8_t scalar_cmpi_sle(T a, T b)
{
  return a <= b;
}
template <class T> int8_t scalar_cmpi_sgt(T a, T b)
{
  return a > b;
}
template <class T> int8_t scalar_cmpi_sge(T a, T b)
{
  return a >= b;
}
template <class T> int8_t scalar_cmpi_ult(T a, T b)
{
  using U = std::make_unsigned_t<T>;
  return U(a) < U(b);
}
template <class T> int8_t scalar_cmpi_ule(T a, T b)
{
  using U = std::make_unsigned_t<T>;
  return U(a) <= U(b);
}
template <class T> int8_t scalar_cmpi_ugt(T a, T b)
{
  using U = std::make_unsigned_t<T>;
  return U(a) > U(b);
}
template <class T> int8_t scalar_cmpi_uge(T a, T b)
{
  using U = std::make_unsigned_t<T>;
  return U(a) >= U(b);
}

template <class T> int8_t scalar_cmpf_false(T, T)
{
  return 0;
}
template <class T> int8_t scalar_cmpf_true(T, T)
{
  return 1;
}
template <class T> int8_t scalar_cmpf_oeq(T a, T b)
{
  return a == b;
}
template <class T> int8_t scalar_cmpf_ogt(T a, T b)
{
  return a > b;
}
template <class T> int8_t scalar_cmpf_oge(T a, T b)
{
  return a >= b;
}
template <class T> int8_t scalar_cmpf_olt(T a, T b)
{
  return a < b;
}
template <class T> int8_t scalar_cmpf_ole(T a, T b)
{
  return a <= b;
}
template <class T> int8_t scalar_cmpf_one(T a, T b)
{
  return a != b;
}
template <class T> int8_t scalar_cmpf_ord(T a, T b)
{
  return !std::isnan(a) && !std::isnan(b);
}
template <class T> int8_t scalar_cmpf_ueq(T a, T b)
{
  return std::isnan(a) || std::isnan(b) || a == b;
}
template <class T> int8_t scalar_cmpf_ugt(T a, T b)
{
  return std::isnan(a) || std::isnan(b) || a > b;
}
template <class T> int8_t scalar_cmpf_uge(T a, T b)
{
  return std::isnan(a) || std::isnan(b) || a >= b;
}
template <class T> int8_t scalar_cmpf_ult(T a, T b)
{
  return std::isnan(a) || std::isnan(b) || a < b;
}
template <class T> int8_t scalar_cmpf_ule(T a, T b)
{
  return std::isnan(a) || std::isnan(b) || a <= b;
}
template <class T> int8_t scalar_cmpf_une(T a, T b)
{
  return std::isnan(a) || std::isnan(b) || a != b;
}
template <class T> int8_t scalar_cmpf_uno(T a, T b)
{
  return std::isnan(a) || std::isnan(b);
}

inline int8_t scalar_extsi_i1_i8(int8_t v)
{
  return v != 0;
}
inline int16_t scalar_extsi_i1_i16(int8_t v)
{
  return v != 0;
}
inline int32_t scalar_extsi_i1_i32(int8_t v)
{
  return v != 0;
}
inline int64_t scalar_extsi_i1_i64(int8_t v)
{
  return v != 0;
}
inline int16_t scalar_extsi_i8_i16(int8_t v)
{
  return int16_t(v);
}
inline int32_t scalar_extsi_i8_i32(int8_t v)
{
  return int32_t(v);
}
inline int64_t scalar_extsi_i8_i64(int8_t v)
{
  return int64_t(v);
}
inline int32_t scalar_extsi_i16_i32(int16_t v)
{
  return int32_t(v);
}
inline int64_t scalar_extsi_i16_i64(int16_t v)
{
  return int64_t(v);
}
inline int64_t scalar_extsi_i32_i64(int32_t v)
{
  return int64_t(v);
}

inline int8_t scalar_extui_u1_u8(int8_t v)
{
  return v != 0;
}
inline int16_t scalar_extui_u1_i16(int8_t v)
{
  return int16_t(uint8_t(v != 0));
}
inline int32_t scalar_extui_u1_i32(int8_t v)
{
  return int32_t(uint8_t(v != 0));
}
inline int64_t scalar_extui_u1_i64(int8_t v)
{
  return int64_t(uint8_t(v != 0));
}
inline int16_t scalar_extui_u8_i16(int8_t v)
{
  return int16_t(uint8_t(v));
}
inline int32_t scalar_extui_u8_i32(int8_t v)
{
  return int32_t(uint8_t(v));
}
inline int64_t scalar_extui_u8_i64(int8_t v)
{
  return int64_t(uint8_t(v));
}
inline int32_t scalar_extui_u16_i32(int16_t v)
{
  return int32_t(uint16_t(v));
}
inline int64_t scalar_extui_u16_i64(int16_t v)
{
  return int64_t(uint16_t(v));
}
inline int64_t scalar_extui_u32_i64(int32_t v)
{
  return int64_t(uint32_t(v));
}

inline int8_t scalar_trunci_i8_i1(int8_t v)
{
  return v & 1;
}
inline int8_t scalar_trunci_i16_i8(int16_t v)
{
  return int8_t(v);
}
inline int8_t scalar_trunci_i32_i8(int32_t v)
{
  return int8_t(v);
}
inline int8_t scalar_trunci_i64_i8(int64_t v)
{
  return int8_t(v);
}
inline int16_t scalar_trunci_i32_i16(int32_t v)
{
  return int16_t(v);
}
inline int16_t scalar_trunci_i64_i16(int64_t v)
{
  return int16_t(v);
}
inline int32_t scalar_trunci_i64_i32(int64_t v)
{
  return int32_t(v);
}

inline double scalar_extf_f32_f64(float v)
{
  return double(v);
}
inline float scalar_extf_f16_f32(float16 v)
{
  return float(v);
}
inline double scalar_extf_f16_f64(float16 v)
{
  return double(float(v));
}
inline float scalar_extf_bf16_f32(bfloat16 v)
{
  return float(v);
}
inline double scalar_extf_bf16_f64(bfloat16 v)
{
  return double(float(v));
}
inline float scalar_truncf_f64_f32(double v)
{
  return float(v);
}
inline float16 scalar_truncf_f32_f16(float v)
{
  return float16(v);
}
inline bfloat16 scalar_truncf_f32_bf16(float v)
{
  return bfloat16(v);
}
inline double scalar_convertf_f32_f64(float v)
{
  return double(v);
}
inline float scalar_convertf_f64_f32(double v)
{
  return float(v);
}

inline float scalar_sitofp_i8_f32(int8_t v)
{
  return float(v);
}
inline float scalar_sitofp_i16_f32(int16_t v)
{
  return float(v);
}
inline float scalar_sitofp_i32_f32(int32_t v)
{
  return float(v);
}
inline float scalar_sitofp_i64_f32(int64_t v)
{
  return float(v);
}
inline double scalar_sitofp_i8_f64(int8_t v)
{
  return double(v);
}
inline double scalar_sitofp_i16_f64(int16_t v)
{
  return double(v);
}
inline double scalar_sitofp_i32_f64(int32_t v)
{
  return double(v);
}
inline double scalar_sitofp_i64_f64(int64_t v)
{
  return double(v);
}
inline float16 scalar_sitofp_i8_f16(int8_t v)
{
  return float16(float(v));
}
inline float16 scalar_sitofp_i16_f16(int16_t v)
{
  return float16(float(v));
}
inline float16 scalar_sitofp_i32_f16(int32_t v)
{
  return float16(float(v));
}
inline float16 scalar_sitofp_i64_f16(int64_t v)
{
  return float16(float(v));
}
inline bfloat16 scalar_sitofp_i8_bf16(int8_t v)
{
  return bfloat16(float(v));
}
inline bfloat16 scalar_sitofp_i16_bf16(int16_t v)
{
  return bfloat16(float(v));
}
inline bfloat16 scalar_sitofp_i32_bf16(int32_t v)
{
  return bfloat16(float(v));
}
inline bfloat16 scalar_sitofp_i64_bf16(int64_t v)
{
  return bfloat16(float(v));
}

inline float scalar_uitofp_u8_f32(int8_t v)
{
  return float(uint8_t(v));
}
inline float scalar_uitofp_u16_f32(int16_t v)
{
  return float(uint16_t(v));
}
inline float scalar_uitofp_u32_f32(int32_t v)
{
  return float(uint32_t(v));
}
inline float scalar_uitofp_u64_f32(int64_t v)
{
  return float(uint64_t(v));
}
inline double scalar_uitofp_u8_f64(int8_t v)
{
  return double(uint8_t(v));
}
inline double scalar_uitofp_u16_f64(int16_t v)
{
  return double(uint16_t(v));
}
inline double scalar_uitofp_u32_f64(int32_t v)
{
  return double(uint32_t(v));
}
inline double scalar_uitofp_u64_f64(int64_t v)
{
  return double(uint64_t(v));
}
inline float16 scalar_uitofp_u8_f16(int8_t v)
{
  return float16(float(uint8_t(v)));
}
inline float16 scalar_uitofp_u16_f16(int16_t v)
{
  return float16(float(uint16_t(v)));
}
inline float16 scalar_uitofp_u32_f16(int32_t v)
{
  return float16(float(uint32_t(v)));
}
inline float16 scalar_uitofp_u64_f16(int64_t v)
{
  return float16(float(uint64_t(v)));
}
inline bfloat16 scalar_uitofp_u8_bf16(int8_t v)
{
  return bfloat16(float(uint8_t(v)));
}
inline bfloat16 scalar_uitofp_u16_bf16(int16_t v)
{
  return bfloat16(float(uint16_t(v)));
}
inline bfloat16 scalar_uitofp_u32_bf16(int32_t v)
{
  return bfloat16(float(uint32_t(v)));
}
inline bfloat16 scalar_uitofp_u64_bf16(int64_t v)
{
  return bfloat16(float(uint64_t(v)));
}

inline int8_t scalar_fptosi_f32_i8(float v)
{
  return int8_t(v);
}
inline int16_t scalar_fptosi_f32_i16(float v)
{
  return int16_t(v);
}
inline int32_t scalar_fptosi_f32_i32(float v)
{
  return int32_t(v);
}
inline int64_t scalar_fptosi_f32_i64(float v)
{
  return int64_t(v);
}
inline int8_t scalar_fptosi_f64_i8(double v)
{
  return int8_t(v);
}
inline int16_t scalar_fptosi_f64_i16(double v)
{
  return int16_t(v);
}
inline int32_t scalar_fptosi_f64_i32(double v)
{
  return int32_t(v);
}
inline int64_t scalar_fptosi_f64_i64(double v)
{
  return int64_t(v);
}
inline int8_t scalar_fptosi_f16_i8(float16 v)
{
  return int8_t(float(v));
}
inline int16_t scalar_fptosi_f16_i16(float16 v)
{
  return int16_t(float(v));
}
inline int32_t scalar_fptosi_f16_i32(float16 v)
{
  return int32_t(float(v));
}
inline int64_t scalar_fptosi_f16_i64(float16 v)
{
  return int64_t(float(v));
}
inline int8_t scalar_fptosi_bf16_i8(bfloat16 v)
{
  return int8_t(float(v));
}
inline int16_t scalar_fptosi_bf16_i16(bfloat16 v)
{
  return int16_t(float(v));
}
inline int32_t scalar_fptosi_bf16_i32(bfloat16 v)
{
  return int32_t(float(v));
}
inline int64_t scalar_fptosi_bf16_i64(bfloat16 v)
{
  return int64_t(float(v));
}

inline int8_t scalar_fptoui_f32_u8(float v)
{
  return int8_t(uint8_t(v));
}
inline int16_t scalar_fptoui_f32_u16(float v)
{
  return int16_t(uint16_t(v));
}
inline int32_t scalar_fptoui_f32_u32(float v)
{
  return int32_t(uint32_t(v));
}
inline int64_t scalar_fptoui_f32_u64(float v)
{
  return int64_t(uint64_t(v));
}
inline int8_t scalar_fptoui_f64_u8(double v)
{
  return int8_t(uint8_t(v));
}
inline int16_t scalar_fptoui_f64_u16(double v)
{
  return int16_t(uint16_t(v));
}
inline int32_t scalar_fptoui_f64_u32(double v)
{
  return int32_t(uint32_t(v));
}
inline int64_t scalar_fptoui_f64_u64(double v)
{
  return int64_t(uint64_t(v));
}
inline int8_t scalar_fptoui_f16_u8(float16 v)
{
  return int8_t(uint8_t(float(v)));
}
inline int16_t scalar_fptoui_f16_u16(float16 v)
{
  return int16_t(uint16_t(float(v)));
}
inline int32_t scalar_fptoui_f16_u32(float16 v)
{
  return int32_t(uint32_t(float(v)));
}
inline int64_t scalar_fptoui_f16_u64(float16 v)
{
  return int64_t(uint64_t(float(v)));
}
inline int8_t scalar_fptoui_bf16_u8(bfloat16 v)
{
  return int8_t(uint8_t(float(v)));
}
inline int16_t scalar_fptoui_bf16_u16(bfloat16 v)
{
  return int16_t(uint16_t(float(v)));
}
inline int32_t scalar_fptoui_bf16_u32(bfloat16 v)
{
  return int32_t(uint32_t(float(v)));
}
inline int64_t scalar_fptoui_bf16_u64(bfloat16 v)
{
  return int64_t(uint64_t(float(v)));
}

inline int32_t scalar_bitcast_f32_i32(float v)
{
  int32_t out;
  std::memcpy(&out, &v, sizeof(out));
  return out;
}
inline float scalar_bitcast_i32_f32(int32_t v)
{
  float out;
  std::memcpy(&out, &v, sizeof(out));
  return out;
}
inline int64_t scalar_bitcast_f64_i64(double v)
{
  int64_t out;
  std::memcpy(&out, &v, sizeof(out));
  return out;
}
inline double scalar_bitcast_i64_f64(int64_t v)
{
  double out;
  std::memcpy(&out, &v, sizeof(out));
  return out;
}
inline int16_t scalar_bitcast_f16_i16(float16 v)
{
  return int16_t(v.value);
}
inline float16 scalar_bitcast_i16_f16(int16_t v)
{
  float16 out;
  out.value = uint16_t(v);
  return out;
}
inline int16_t scalar_bitcast_bf16_i16(bfloat16 v)
{
  return int16_t(v.value);
}
inline bfloat16 scalar_bitcast_i16_bf16(int16_t v)
{
  bfloat16 out;
  out.value = uint16_t(v);
  return out;
}

template <class T> T scalar_absf(T v)
{
  return T(std::fabs(float(v)));
}
template <class T> T scalar_absi(T v)
{
  return v < T(0) ? T(-v) : v;
}

template <class T> T scalar_sqrt(T v)
{
  return T(std::sqrt(float(v)));
}
template <class T> T scalar_rsqrt(T v)
{
  return T(1.0f / std::sqrt(float(v)));
}
template <class T> T scalar_cbrt(T v)
{
  return T(std::cbrt(float(v)));
}
template <class T> T scalar_exp(T v)
{
  return T(std::exp(float(v)));
}
template <class T> T scalar_exp2(T v)
{
  return T(std::exp2(float(v)));
}
template <class T> T scalar_expm1(T v)
{
  return T(std::expm1(float(v)));
}
template <class T> T scalar_log(T v)
{
  return T(std::log(float(v)));
}
template <class T> T scalar_log2(T v)
{
  return T(std::log2(float(v)));
}
template <class T> T scalar_log10(T v)
{
  return T(std::log10(float(v)));
}
template <class T> T scalar_log1p(T v)
{
  return T(std::log1p(float(v)));
}
template <class T> T scalar_sin(T v)
{
  return T(std::sin(float(v)));
}
template <class T> T scalar_cos(T v)
{
  return T(std::cos(float(v)));
}
template <class T> T scalar_tan(T v)
{
  return T(std::tan(float(v)));
}
template <class T> T scalar_asin(T v)
{
  return T(std::asin(float(v)));
}
template <class T> T scalar_acos(T v)
{
  return T(std::acos(float(v)));
}
template <class T> T scalar_atan(T v)
{
  return T(std::atan(float(v)));
}
template <class T> T scalar_sinh(T v)
{
  return T(std::sinh(float(v)));
}
template <class T> T scalar_cosh(T v)
{
  return T(std::cosh(float(v)));
}
template <class T> T scalar_tanh(T v)
{
  return T(std::tanh(float(v)));
}
template <class T> T scalar_asinh(T v)
{
  return T(std::asinh(float(v)));
}
template <class T> T scalar_acosh(T v)
{
  return T(std::acosh(float(v)));
}
template <class T> T scalar_atanh(T v)
{
  return T(std::atanh(float(v)));
}
template <class T> T scalar_erf(T v)
{
  return T(std::erf(float(v)));
}
template <class T> T scalar_erfc(T v)
{
  return T(std::erfc(float(v)));
}
template <class T> T scalar_powf(T a, T b)
{
  return T(std::pow(float(a), float(b)));
}
template <class T> T scalar_atan2(T a, T b)
{
  return T(std::atan2(float(a), float(b)));
}
template <class T> T scalar_copysign(T a, T b)
{
  return T(std::copysign(float(a), float(b)));
}
template <class T> T scalar_fma(T a, T b, T c)
{
  return T(std::fma(float(a), float(b), float(c)));
}
template <class T> T scalar_ceil(T v)
{
  return T(std::ceil(float(v)));
}
template <class T> T scalar_floor(T v)
{
  return T(std::floor(float(v)));
}
template <class T> T scalar_trunc(T v)
{
  return T(std::trunc(float(v)));
}
template <class T> T scalar_round(T v)
{
  return T(std::round(float(v)));
}
template <class T> T scalar_roundeven(T v)
{
  return T(std::nearbyint(float(v)));
}
template <class T> T scalar_clampf(T v, T lo, T hi)
{
  if (std::isnan(v)) return v;
  return v < lo ? lo : (hi < v ? hi : v);
}
template <class T> T scalar_flush_denormals(T v)
{
  return v;
}
template <class T> int8_t scalar_isfinite(T v)
{
  return std::isfinite(float(v));
}
template <class T> int8_t scalar_isinf(T v)
{
  return std::isinf(float(v));
}
template <class T> int8_t scalar_isnan(T v)
{
  return std::isnan(float(v));
}
template <class T> int8_t scalar_isnormal(T v)
{
  return std::isnormal(float(v));
}

template <class T> T scalar_ctlz(T v)
{
  if (v == 0) return T(sizeof(T) * 8);
  return T(
    __builtin_clzll(static_cast<unsigned long long>(static_cast<std::make_unsigned_t<T>>(v))) -
    (64 - int(sizeof(T) * 8))
  );
}
template <class T> T scalar_ctlz(T v, int8_t)
{
  return scalar_ctlz(v);
}
template <class T> T scalar_cttz(T v)
{
  if (v == 0) return T(sizeof(T) * 8);
  return T(__builtin_ctzll(static_cast<unsigned long long>(static_cast<std::make_unsigned_t<T>>(v))));
}
template <class T> T scalar_cttz(T v, int8_t)
{
  return scalar_cttz(v);
}
template <class T> T scalar_ctpop(T v)
{
  return T(__builtin_popcountll(static_cast<unsigned long long>(static_cast<std::make_unsigned_t<T>>(v))));
}
template <class T> T scalar_ipowi(T base, T exponent)
{
  T result = T(1);
  T e      = exponent < 0 ? T(-exponent) : exponent;
  while (e > 0) {
    if (e & 1) result *= base;
    base  *= base;
    e    >>= 1;
  }
  return exponent < 0 ? T(1) / result : result;
}
template <class T> T scalar_fpowi(T base, int32_t exponent)
{
  return T(std::pow(float(base), float(exponent)));
}
template <class T> T scalar_select(int8_t cond, T a, T b)
{
  return cond ? a : b;
}

#endif
