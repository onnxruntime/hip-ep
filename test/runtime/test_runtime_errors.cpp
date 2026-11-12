/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hipdnn_ep_runtime.h"
#include "runtime_state_internal.h"

#include <dlfcn.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" int8_t hipdnn_ep_op_state_construct_matmul_nbits(RuntimeState *,
                                                            int32_t);

namespace {
bool fail_allocation = false;
int fail_sync_after = -1;
int matmul_calls = 0;
int failures = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);             \
      ++failures;                                                              \
    }                                                                          \
  } while (0)
} // namespace

// Inject only the requested failure; all other operations use the HIP runtime.
extern "C" hipError_t hipMalloc(void **ptr, size_t size) {
  if (fail_allocation) {
    *ptr = nullptr;
    return hipErrorOutOfMemory;
  }
  using Allocate = hipError_t (*)(void **, size_t);
  static auto allocate =
      reinterpret_cast<Allocate>(dlsym(RTLD_NEXT, "hipMalloc"));
  if (!allocate)
    std::abort();
  return allocate(ptr, size);
}

extern "C" hipError_t hipStreamSynchronize(hipStream_t stream) {
  if (fail_sync_after == 0) {
    fail_sync_after = -1;
    return hipErrorUnknown;
  }
  if (fail_sync_after > 0)
    --fail_sync_after;
  static auto synchronize = reinterpret_cast<decltype(&hipStreamSynchronize)>(
      dlsym(RTLD_NEXT, "hipStreamSynchronize"));
  if (!synchronize)
    std::abort();
  return synchronize(stream);
}

extern "C" hipblasStatus_t
hipblasLtMatmul(hipblasLtHandle_t, hipblasLtMatmulDesc_t, const void *,
                const void *, hipblasLtMatrixLayout_t, const void *,
                hipblasLtMatrixLayout_t, const void *, const void *,
                hipblasLtMatrixLayout_t, void *, hipblasLtMatrixLayout_t,
                const hipblasLtMatmulAlgo_t *, void *, size_t, hipStream_t) {
  ++matmul_calls;
  return HIPBLAS_STATUS_EXECUTION_FAILED;
}

int main() {
  RuntimeState state{};
  if (hipStreamCreate(&state.stream) != hipSuccess)
    return 1;
  CHECK(hipMalloc(reinterpret_cast<void **>(&state.device_error_flag),
                  sizeof(int)) == hipSuccess);
  CHECK(hipdnn_ep_state_reset_error_flag(&state) == 0);

  for (int host : {0, -7}) {
    for (int device : {0, -9}) {
      CHECK(hipdnn_ep_state_reset_error_flag(&state) == 0);
      state.host_error_flag = host;
      CHECK(hipdnn_ep_state_set_error_flag(&state, device) == 0);
      CHECK(hipdnn_ep_state_read_and_clear_error_flag(&state) ==
            (host ? host : device));
      CHECK(state.host_error_flag == 0);
      CHECK(hipdnn_ep_state_read_and_clear_error_flag(&state) == 0);
    }
  }
  state.host_error_flag = -7;
  fail_sync_after = 0;
  CHECK(hipdnn_ep_state_read_and_clear_error_flag(&state) != 0);
  CHECK(state.host_error_flag == -7);
  CHECK(hipdnn_ep_state_read_and_clear_error_flag(&state) == -7);
  CHECK(hipdnn_ep_state_read_and_clear_error_flag(&state) == 0);
  fail_sync_after = -1;
  CHECK(hipdnn_ep_state_reset_error_flag(&state) == 0);

  constexpr int64_t count = 512;
  constexpr int64_t capacity = 1024;
  void *data = nullptr, *values = nullptr, *indices = nullptr, *k = nullptr;
  CHECK(hipMalloc(&data, capacity * sizeof(float)) == hipSuccess);
  CHECK(hipMalloc(&values, capacity * sizeof(float)) == hipSuccess);
  CHECK(hipMalloc(&indices, capacity * sizeof(int64_t)) == hipSuccess);
  CHECK(hipMalloc(&k, sizeof(int64_t)) == hipSuccess);
  std::vector<float> input(capacity, 1.0f);
  CHECK(hipMemcpy(data, input.data(), capacity * sizeof(float),
                  hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(k, &count, sizeof(count), hipMemcpyHostToDevice) ==
        hipSuccess);
  const auto topk = [&] {
    return wrap_top_k(&state, data, k, values, indices, 0, 1, 1, 1, &count,
                      count, HIPDNN_EP_DATATYPE_FLOAT);
  };
  CHECK(hipMemset(values, 0x5a, count * sizeof(float)) == hipSuccess);
  fail_allocation = true;
  CHECK(topk() != 0);
  fail_allocation = false;
  CHECK(hipdnn_ep_state_read_and_clear_error_flag(&state) != 0);
  CHECK(hipdnn_ep_state_reset_error_flag(&state) == 0);
  std::vector<unsigned char> untouched(count * sizeof(float));
  CHECK(hipMemcpy(untouched.data(), values, untouched.size(),
                  hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(std::all_of(untouched.begin(), untouched.end(),
                    [](unsigned char byte) { return byte == 0x5a; }));
  CHECK(topk() == 0);
  CHECK(hipdnn_ep_state_read_and_clear_error_flag(&state) == 0);
  CHECK(hipMemcpy(input.data(), values, count * sizeof(float),
                  hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(std::all_of(input.begin(), input.begin() + count,
                    [](float value) { return value == 1.0f; }));

  // Growing an existing workspace must not hide a failed drain with host TopK.
  const int64_t larger_count = 1024;
  CHECK(hipMemcpy(k, &larger_count, sizeof(larger_count),
                  hipMemcpyHostToDevice) == hipSuccess);
  fail_sync_after = 1; // TopK's initial sync succeeds; workspace drain fails.
  CHECK(wrap_top_k(&state, data, k, values, indices, 0, 1, 1, 1, &larger_count,
                   larger_count, HIPDNN_EP_DATATYPE_FLOAT) != 0);
  CHECK(hipdnn_ep_state_read_and_clear_error_flag(&state) != 0);
  CHECK(hipdnn_ep_state_reset_error_flag(&state) == 0);

  CHECK(hipdnn_ep_op_states_alloc(&state, 1));
  CHECK(hipdnn_ep_op_state_construct_matmul_nbits(&state, 0) == 0);
  const auto matmul = [&](int slot, int64_t scale_bytes, const void *zeros) {
    return wrap_matmul_nbits(&state, slot, data, data, data, zeros, nullptr,
                             nullptr, values, 16, 64, 64, 1, 4, 32, 2, 1,
                             scale_bytes);
  };
  CHECK(matmul(1, 2, nullptr) != 0);
  CHECK(state.host_error_flag != 0);
  CHECK(hipdnn_ep_state_read_and_clear_error_flag(&state) != 0);
  CHECK(hipdnn_ep_state_reset_error_flag(&state) == 0);
  fail_allocation = true;
  CHECK(matmul(0, 4, nullptr) != 0);
  CHECK(state.host_error_flag != 0);
  CHECK(hipdnn_ep_state_read_and_clear_error_flag(&state) != 0);
  CHECK(hipdnn_ep_state_reset_error_flag(&state) == 0);
  CHECK(matmul(0, 2, data) != 0);
  CHECK(state.host_error_flag != 0);
  CHECK(hipdnn_ep_state_read_and_clear_error_flag(&state) != 0);
  CHECK(hipdnn_ep_state_reset_error_flag(&state) == 0);
  fail_allocation = false;
  // Valid op state but no hipBLASLt handle: prefill returns a terminal error.
  CHECK(matmul(0, 2, nullptr) != 0);
  CHECK(state.host_error_flag != 0);
  CHECK(hipdnn_ep_state_read_and_clear_error_flag(&state) != 0);
  CHECK(hipdnn_ep_state_reset_error_flag(&state) == 0);
  CHECK(hipblasLtCreate(&state.hipblas_handle) == HIPBLAS_STATUS_SUCCESS);
  CHECK(matmul(0, 2, nullptr) != 0);
  CHECK(matmul_calls > 0);
  CHECK(state.host_error_flag != 0);
  CHECK(hipdnn_ep_state_read_and_clear_error_flag(&state) != 0);
  CHECK(hipdnn_ep_state_read_and_clear_error_flag(&state) == 0);

  CHECK(hipblasLtDestroy(state.hipblas_handle) == HIPBLAS_STATUS_SUCCESS);
  state.op_states[0]->deletor(state.op_states[0]);
  std::free(state.op_states);
  CHECK(hipFree(state.workspace) == hipSuccess);
  CHECK(hipFree(state.matmul_nbits_fp16_scratch) == hipSuccess);
  CHECK(hipFree(data) == hipSuccess);
  CHECK(hipFree(values) == hipSuccess);
  CHECK(hipFree(indices) == hipSuccess);
  CHECK(hipFree(k) == hipSuccess);
  CHECK(hipFree(state.device_error_flag) == hipSuccess);
  CHECK(hipStreamDestroy(state.stream) == hipSuccess);
  std::printf("Runtime error propagation: %d failures\n", failures);
  return failures ? 1 : 0;
}
