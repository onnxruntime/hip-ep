/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstring>

namespace {

using Fragment = float __attribute__((ext_vector_type(4)));

__device__ float Decode(int high, int low, float scale, int residual) {
  return residual ? static_cast<float>(256 * high + low + 128) * (scale / 256.0f)
                  : static_cast<float>(high) * scale;
}

__global__ __launch_bounds__(256) void AttendInt8Gqa6Mfma(
    const float *query, const int8_t *key, const int8_t *value,
    const int32_t *cumulative, const int32_t *past, const int32_t *table,
    const float *ks, const float *vs, float *partials, int columns,
    int page_size, int heads, int kv_heads, int splits, float scale,
    const int *error, int residual) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx942__)
  if (*error)
    return;
  const int request = blockIdx.z;
  const int token = cumulative[request];
  const int kv = blockIdx.y;
  const int total = past[request] + 1;
  const int tiles = (total + splits * 16 - 1) / (splits * 16);
  const int begin = blockIdx.x * tiles * 16;
  const int end = min(total, begin + tiles * 16);
  // MFMA columns are lane%16 and row groups are lane/16; each wave owns H/4.
  const int lane = threadIdx.x % 64;
  const int wave = threadIdx.x / 64;
  const int column = lane % 16;
  const int group = lane / 16;
  const int width = residual ? 512 : 256;
  float q[16];
#pragma unroll
  for (int chunk = 0; chunk < 16; ++chunk)
    q[chunk] = column < 6
        ? query[(static_cast<int64_t>(token) * heads + kv * 6 + column) * 256 +
                wave * 64 + chunk * 4 + group] : 0;
  Fragment output[4] = {};
  float maximum[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
  float denominator[4] = {};
  // The next producer barrier also fences the preceding buffer's consumers.
  __shared__ float dots_shared[2][4][6][16];
  for (int base = begin; base < end; base += 16) {
    const int parity = (base / 16) & 1;
    const int page = table[static_cast<int64_t>(request) * columns + base / page_size];
    const int64_t page_offset =
        ((static_cast<int64_t>(page) * page_size + base % page_size) *
             kv_heads + kv) * width;
    const int load_row = lane / 4;
    const int load_channel = lane % 4;
    Fragment dots = {};
#pragma unroll
    for (int chunk = 0; chunk < 16; ++chunk) {
      const int channel = wave * 64 + chunk * 4;
      const int64_t offset = page_offset +
          static_cast<int64_t>(load_row) * kv_heads * width + channel;
      float decoded = 0;
      if (base + load_row < end) {
        const uint32_t high = *reinterpret_cast<const uint32_t *>(key + offset);
        const uint32_t low = residual
            ? *reinterpret_cast<const uint32_t *>(key + offset + 256) : 0;
        decoded = Decode(static_cast<int8_t>(high >> (load_channel * 8)),
                         static_cast<int8_t>(low >> (load_channel * 8)),
                         ks[kv * 256 + channel + load_channel], residual);
      }
      const float k = __shfl(decoded, column * 4 + group, 64);
      dots = __builtin_amdgcn_mfma_f32_16x16x4f32(q[chunk], k, dots, 0, 0, 0);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (group * 4 + i < 6)
        dots_shared[parity][wave][group * 4 + i][column] = dots[i];
    __syncthreads();
    float probability[4], keep[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = group * 4 + i;
      float score = -INFINITY;
      if (row < 6 && base + column < end)
        score = ((dots_shared[parity][0][row][column] +
                  dots_shared[parity][1][row][column]) +
                 (dots_shared[parity][2][row][column] +
                  dots_shared[parity][3][row][column])) * scale;
      float tile_max = score;
      for (int offset = 8; offset; offset /= 2)
        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, offset, 16));
      const float next_max = fmaxf(maximum[i], tile_max);
      keep[i] = maximum[i] == -INFINITY ? 0 : expf(maximum[i] - next_max);
      probability[i] = score == -INFINITY ? 0 : expf(score - next_max);
      float tile_sum = probability[i];
      for (int offset = 8; offset; offset /= 2)
        tile_sum += __shfl_xor(tile_sum, offset, 16);
      denominator[i] = fmaf(denominator[i], keep[i], tile_sum);
      maximum[i] = next_max;
    }
    float p[4];
    // Transpose query-row probabilities into the MFMA A-fragment layout.
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
      const int source_lane = (column / 4) * 16 + chunk * 4 + group;
      const float p0 = __shfl(probability[0], source_lane, 64);
      const float p1 = __shfl(probability[1], source_lane, 64);
      const float p2 = __shfl(probability[2], source_lane, 64);
      const float p3 = __shfl(probability[3], source_lane, 64);
      p[chunk] = column >= 6 ? 0
          : column % 4 == 0 ? p0 : column % 4 == 1 ? p1
          : column % 4 == 2 ? p2 : p3;
    }
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
#pragma unroll
      for (int i = 0; i < 4; ++i)
        output[tile][i] *= keep[i];
#pragma unroll
      for (int chunk = 0; chunk < 4; ++chunk) {
        const int position = base + chunk * 4 + group;
        const int channel = wave * 64 + tile * 16 + column;
        const int64_t offset = page_offset +
            static_cast<int64_t>(chunk * 4 + group) * kv_heads * width + channel;
        const float v = position < end
            ? Decode(value[offset], residual ? value[offset + 256] : 0,
                     vs[kv * 256 + channel], residual) : 0;
        output[tile] = __builtin_amdgcn_mfma_f32_16x16x4f32(
            p[chunk], v, output[tile], 0, 0, 0);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if (group * 4 + i >= 6)
      continue;
    const int head = kv * 6 + group * 4 + i;
    float *result = partials +
        ((static_cast<int64_t>(token) * heads + head) * splits + blockIdx.x) * 258;
#pragma unroll
    for (int tile = 0; tile < 4; ++tile)
      result[wave * 64 + tile * 16 + column] = output[tile][i];
    if (wave == 0 && column == 0) {
      result[256] = maximum[i];
      result[257] = denominator[i];
    }
  }
#endif
}

__global__ void MergeInt8Gqa6Mfma(const float *partials, float *output, int heads, int splits,
                      const int *error) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx942__)
  if (*error)
    return;
  const int row = blockIdx.x * heads + blockIdx.y;
  const float *source = partials + static_cast<int64_t>(row) * splits * 258;
  __shared__ float wave_maximum[4];
  __shared__ float factors[512], denominators[512], total_denominator;
  float maximum = -INFINITY;
  for (int split = threadIdx.x; split < splits; split += blockDim.x)
    maximum = fmaxf(maximum, source[split * 258 + 256]);
  for (int offset = 32; offset; offset /= 2)
    maximum = fmaxf(maximum, __shfl_xor(maximum, offset, 64));
  if (threadIdx.x % 64 == 0)
    wave_maximum[threadIdx.x / 64] = maximum;
  __syncthreads();
  maximum = fmaxf(fmaxf(wave_maximum[0], wave_maximum[1]),
                  fmaxf(wave_maximum[2], wave_maximum[3]));
  // Share split factors while retaining sequential denominator/channel FMAs.
  for (int split = threadIdx.x; split < splits; split += blockDim.x) {
    const float *part = source + split * 258;
    factors[split] = part[256] == -INFINITY ? 0 : expf(part[256] - maximum);
    denominators[split] = part[257];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float denominator = 0;
    for (int split = 0; split < splits; ++split)
      denominator = fmaf(denominators[split], factors[split], denominator);
    total_denominator = denominator;
  }
  float sum = 0;
  int split = 0;
  for (; split + 3 < splits; split += 4) {
    const float v0 = source[split * 258 + threadIdx.x];
    const float v1 = source[(split + 1) * 258 + threadIdx.x];
    const float v2 = source[(split + 2) * 258 + threadIdx.x];
    const float v3 = source[(split + 3) * 258 + threadIdx.x];
    sum = fmaf(v0, factors[split], sum);
    sum = fmaf(v1, factors[split + 1], sum);
    sum = fmaf(v2, factors[split + 2], sum);
    sum = fmaf(v3, factors[split + 3], sum);
  }
  for (; split < splits; ++split)
    sum = fmaf(source[split * 258 + threadIdx.x], factors[split], sum);
  __syncthreads();
  output[static_cast<int64_t>(row) * 256 + threadIdx.x] = sum / total_denominator;
#endif
}

} // namespace

extern "C" HIP_KERNEL_API int hip_paged_int8_decode(
    void *stream, const float *query, const int8_t *key, const int8_t *value,
    const int32_t *cumulative, const int32_t *past, const int32_t *table,
    const float *key_scale, const float *value_scale, float *partials,
    float *output, int columns, int page_size, int splits, float scale,
    const int *error, int residual, int *used) {
  if (!used)
    return -1;
  *used = 0;
  if (!query || !key || !value || !cumulative || !past || !table ||
      !key_scale || !value_scale || !partials || !output || !error ||
      columns <= 0 || page_size < 16 || page_size > 4096 ||
      (page_size & (page_size - 1)) || splits < 1 || splits > 512 ||
      !std::isfinite(scale) || scale <= 0 || (residual != 0 && residual != 8))
    return -1;
  if ((reinterpret_cast<uintptr_t>(key) |
       reinterpret_cast<uintptr_t>(value)) & 3)
    return 0;
#if defined(HIP_EP_PAGED_DECODE_GFX942) && HIP_EP_PAGED_DECODE_GFX942
  int device = 0;
  hipDeviceProp_t properties{};
  auto status = hipGetDevice(&device);
  if (status == hipSuccess)
    status = hipGetDeviceProperties(&properties, device);
  if (status != hipSuccess)
    return hipdnn_device_execution_status(status);
  if (std::strncmp(properties.gcnArchName, "gfx942", 6) != 0 ||
      (properties.gcnArchName[6] != '\0' && properties.gcnArchName[6] != ':'))
    return 0;
  *used = 1;
  AttendInt8Gqa6Mfma<<<dim3(splits, 4, 1), 256, 0, static_cast<hipStream_t>(stream)>>>(
      query, key, value, cumulative, past, table, key_scale, value_scale,
      partials, columns, page_size, 24, 4, splits, scale, error, residual);
  status = hipGetLastError();
  if (status != hipSuccess)
    return hipdnn_device_execution_status(status);
  MergeInt8Gqa6Mfma<<<dim3(1, 24), 256, 0, static_cast<hipStream_t>(stream)>>>(
      partials, output, 24, splits, error);
  return hipdnn_device_execution_status(hipGetLastError());
#else
  return 0;
#endif
}
