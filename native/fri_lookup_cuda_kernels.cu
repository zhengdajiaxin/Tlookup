#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

namespace {
constexpr uint8_t LEAF_0 = static_cast<uint8_t>('l');
constexpr uint8_t LEAF_1 = static_cast<uint8_t>('e');
constexpr uint8_t LEAF_2 = static_cast<uint8_t>('a');
constexpr uint8_t LEAF_3 = static_cast<uint8_t>('f');
constexpr uint8_t NODE_0 = static_cast<uint8_t>('n');
constexpr uint8_t NODE_1 = static_cast<uint8_t>('o');
constexpr uint8_t NODE_2 = static_cast<uint8_t>('d');
constexpr uint8_t NODE_3 = static_cast<uint8_t>('e');
constexpr size_t FIELD_LIMBS = 8;
constexpr size_t FIELD_BYTES = 32;
constexpr size_t DIGEST_BYTES = 32;
constexpr size_t LEAF_PREIMAGE_BYTES = 4 + FIELD_BYTES;
constexpr size_t ROW2_LEAF_PREIMAGE_BYTES = 4 + 2 * FIELD_BYTES;
constexpr size_t ROW3_LEAF_PREIMAGE_BYTES = 4 + 3 * FIELD_BYTES;
constexpr size_t ROW5_LEAF_PREIMAGE_BYTES = 4 + 5 * FIELD_BYTES;
constexpr size_t NODE_PREIMAGE_BYTES = 4 + 2 * DIGEST_BYTES;
__constant__ uint32_t BN254_SCALAR_MODULUS[FIELD_LIMBS] = {
    4026531841u, 1138881939u, 2042196113u, 674490440u,
    2172737629u, 3092268470u, 3778125865u, 811880050u};

__device__ bool geq_modulus(const uint32_t limbs[FIELD_LIMBS]) {
  for (int limb = static_cast<int>(FIELD_LIMBS) - 1; limb >= 0; --limb) {
    if (limbs[limb] > BN254_SCALAR_MODULUS[limb]) {
      return true;
    }
    if (limbs[limb] < BN254_SCALAR_MODULUS[limb]) {
      return false;
    }
  }
  return true;
}

__device__ void subtract_modulus(uint32_t limbs[FIELD_LIMBS]) {
  uint64_t borrow = 0;
  #pragma unroll
  for (size_t limb = 0; limb < FIELD_LIMBS; ++limb) {
    const uint64_t value = static_cast<uint64_t>(limbs[limb]);
    const uint64_t subtrahend = static_cast<uint64_t>(BN254_SCALAR_MODULUS[limb]) + borrow;
    if (value >= subtrahend) {
      limbs[limb] = static_cast<uint32_t>(value - subtrahend);
      borrow = 0;
    } else {
      limbs[limb] = static_cast<uint32_t>((uint64_t{1} << 32) + value - subtrahend);
      borrow = 1;
    }
  }
}

__device__ void canonicalize_scalar(uint32_t limbs[FIELD_LIMBS]) {
  #pragma unroll
  for (size_t round = 0; round < 6; ++round) {
    if (geq_modulus(limbs)) {
      subtract_modulus(limbs);
    }
  }
}

__device__ void write_canonical_scalar_bytes(
    const uint32_t* fields,
    size_t index,
    uint8_t* out) {
  const uint32_t* input_limbs = fields + index * FIELD_LIMBS;
  uint32_t limbs[FIELD_LIMBS];
  #pragma unroll
  for (size_t limb = 0; limb < FIELD_LIMBS; ++limb) {
    limbs[limb] = input_limbs[limb];
  }
  canonicalize_scalar(limbs);

  #pragma unroll
  for (size_t limb = 0; limb < FIELD_LIMBS; ++limb) {
    const uint32_t value = limbs[limb];
    const size_t offset = limb * 4;
    out[offset + 0] = static_cast<uint8_t>(value & 0xffu);
    out[offset + 1] = static_cast<uint8_t>((value >> 8) & 0xffu);
    out[offset + 2] = static_cast<uint8_t>((value >> 16) & 0xffu);
    out[offset + 3] = static_cast<uint8_t>((value >> 24) & 0xffu);
  }
}

__global__ void row2_leaf_preimages_kernel(
    const uint32_t* column0,
    const uint32_t* column1,
    uint8_t* output,
    size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }

  uint8_t* out = output + index * ROW2_LEAF_PREIMAGE_BYTES;
  out[0] = LEAF_0;
  out[1] = LEAF_1;
  out[2] = LEAF_2;
  out[3] = LEAF_3;
  write_canonical_scalar_bytes(column0, index, out + 4);
  write_canonical_scalar_bytes(column1, index, out + 4 + FIELD_BYTES);
}

__global__ void row3_leaf_preimages_kernel(
    const uint32_t* column0,
    const uint32_t* column1,
    const uint32_t* column2,
    uint8_t* output,
    size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }

  uint8_t* out = output + index * ROW3_LEAF_PREIMAGE_BYTES;
  out[0] = LEAF_0;
  out[1] = LEAF_1;
  out[2] = LEAF_2;
  out[3] = LEAF_3;
  write_canonical_scalar_bytes(column0, index, out + 4);
  write_canonical_scalar_bytes(column1, index, out + 4 + FIELD_BYTES);
  write_canonical_scalar_bytes(column2, index, out + 4 + 2 * FIELD_BYTES);
}

__global__ void row5_leaf_preimages_kernel(
    const uint32_t* column0,
    const uint32_t* column1,
    const uint32_t* column2,
    const uint32_t* column3,
    const uint32_t* column4,
    uint8_t* output,
    size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }

  uint8_t* out = output + index * ROW5_LEAF_PREIMAGE_BYTES;
  out[0] = LEAF_0;
  out[1] = LEAF_1;
  out[2] = LEAF_2;
  out[3] = LEAF_3;
  write_canonical_scalar_bytes(column0, index, out + 4);
  write_canonical_scalar_bytes(column1, index, out + 4 + FIELD_BYTES);
  write_canonical_scalar_bytes(column2, index, out + 4 + 2 * FIELD_BYTES);
  write_canonical_scalar_bytes(column3, index, out + 4 + 3 * FIELD_BYTES);
  write_canonical_scalar_bytes(column4, index, out + 4 + 4 * FIELD_BYTES);
}

__global__ void field_leaf_preimages_kernel(
    const uint32_t* fields,
    uint8_t* output,
    size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }

  uint8_t* out = output + index * LEAF_PREIMAGE_BYTES;
  out[0] = LEAF_0;
  out[1] = LEAF_1;
  out[2] = LEAF_2;
  out[3] = LEAF_3;

  const uint32_t* input_limbs = fields + index * FIELD_LIMBS;
  uint32_t limbs[FIELD_LIMBS];
  #pragma unroll
  for (size_t limb = 0; limb < FIELD_LIMBS; ++limb) {
    limbs[limb] = input_limbs[limb];
  }
  canonicalize_scalar(limbs);

  #pragma unroll
  for (size_t limb = 0; limb < FIELD_LIMBS; ++limb) {
    const uint32_t value = limbs[limb];
    const size_t offset = 4 + limb * 4;
    out[offset + 0] = static_cast<uint8_t>(value & 0xffu);
    out[offset + 1] = static_cast<uint8_t>((value >> 8) & 0xffu);
    out[offset + 2] = static_cast<uint8_t>((value >> 16) & 0xffu);
    out[offset + 3] = static_cast<uint8_t>((value >> 24) & 0xffu);
  }
}

__global__ void node_preimages_kernel(
    const uint8_t* digests,
    uint8_t* output,
    size_t pair_count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pair_count) {
    return;
  }

  uint8_t* out = output + index * NODE_PREIMAGE_BYTES;
  out[0] = NODE_0;
  out[1] = NODE_1;
  out[2] = NODE_2;
  out[3] = NODE_3;

  const uint8_t* left = digests + (2 * index) * DIGEST_BYTES;
  const uint8_t* right = left + DIGEST_BYTES;
  #pragma unroll
  for (size_t byte = 0; byte < DIGEST_BYTES; ++byte) {
    out[4 + byte] = left[byte];
    out[4 + DIGEST_BYTES + byte] = right[byte];
  }
}

__device__ void copy_field_limbs(const uint32_t* input_limbs, uint32_t* out_limbs) {
  #pragma unroll
  for (size_t limb = 0; limb < FIELD_LIMBS; ++limb) {
    out_limbs[limb] = input_limbs[limb];
  }
}

__global__ void gather_fields_kernel(
    const uint32_t* fields,
    const size_t* indices,
    uint32_t* output,
    size_t count) {
  const size_t out_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (out_index >= count) {
    return;
  }

  const size_t source_index = indices[out_index];
  const uint32_t* input_limbs = fields + source_index * FIELD_LIMBS;
  uint32_t* out_limbs = output + out_index * FIELD_LIMBS;
  copy_field_limbs(input_limbs, out_limbs);
}

__global__ void gather_row_fields_kernel(
    const uint32_t* column0,
    const uint32_t* column1,
    const uint32_t* column2,
    const uint32_t* column3,
    const uint32_t* column4,
    const size_t* indices,
    uint32_t* output,
    size_t count,
    size_t width) {
  const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= count) {
    return;
  }

  const size_t source_index = indices[row];
  const uint32_t* columns[5] = {column0, column1, column2, column3, column4};
  for (size_t column = 0; column < width; ++column) {
    const uint32_t* input_limbs = columns[column] + source_index * FIELD_LIMBS;
    uint32_t* out_limbs = output + (column * count + row) * FIELD_LIMBS;
    copy_field_limbs(input_limbs, out_limbs);
  }
}

__global__ void gather_digests_kernel(
    const uint8_t* digests,
    const size_t* indices,
    uint8_t* output,
    size_t count) {
  const size_t out_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (out_index >= count) {
    return;
  }

  const size_t source_index = indices[out_index];
  const uint8_t* input = digests + source_index * DIGEST_BYTES;
  uint8_t* out = output + out_index * DIGEST_BYTES;
  #pragma unroll
  for (size_t byte = 0; byte < DIGEST_BYTES; ++byte) {
    out[byte] = input[byte];
  }
}

int launch_config(size_t count, dim3* grid, dim3* block) {
  if (count == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr unsigned threads = 256;
  *block = dim3(threads);
  *grid = dim3(static_cast<unsigned>((count + threads - 1) / threads));
  return static_cast<int>(cudaSuccess);
}
} // namespace

extern "C" int fri_lookup_bn254_leaf_preimages_cuda(
    const void* fields,
    uint8_t* output,
    size_t count,
    void* stream_handle) {
  dim3 grid;
  dim3 block;
  const int config_error = launch_config(count, &grid, &block);
  if (config_error != static_cast<int>(cudaSuccess)) {
    return config_error;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  field_leaf_preimages_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(fields), output, count);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int fri_lookup_cuda_memset_zero(
    void* output,
    size_t byte_count,
    void* stream_handle) {
  if (byte_count == 0) {
    return static_cast<int>(cudaSuccess);
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  return static_cast<int>(cudaMemsetAsync(output, 0, byte_count, stream));
}

extern "C" int fri_lookup_bn254_row2_leaf_preimages_cuda(
    const void* column0,
    const void* column1,
    uint8_t* output,
    size_t count,
    void* stream_handle) {
  dim3 grid;
  dim3 block;
  const int config_error = launch_config(count, &grid, &block);
  if (config_error != static_cast<int>(cudaSuccess)) {
    return config_error;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  row2_leaf_preimages_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(column0),
      reinterpret_cast<const uint32_t*>(column1),
      output,
      count);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int fri_lookup_bn254_row3_leaf_preimages_cuda(
    const void* column0,
    const void* column1,
    const void* column2,
    uint8_t* output,
    size_t count,
    void* stream_handle) {
  dim3 grid;
  dim3 block;
  const int config_error = launch_config(count, &grid, &block);
  if (config_error != static_cast<int>(cudaSuccess)) {
    return config_error;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  row3_leaf_preimages_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(column0),
      reinterpret_cast<const uint32_t*>(column1),
      reinterpret_cast<const uint32_t*>(column2),
      output,
      count);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int fri_lookup_bn254_row5_leaf_preimages_cuda(
    const void* column0,
    const void* column1,
    const void* column2,
    const void* column3,
    const void* column4,
    uint8_t* output,
    size_t count,
    void* stream_handle) {
  dim3 grid;
  dim3 block;
  const int config_error = launch_config(count, &grid, &block);
  if (config_error != static_cast<int>(cudaSuccess)) {
    return config_error;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  row5_leaf_preimages_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(column0),
      reinterpret_cast<const uint32_t*>(column1),
      reinterpret_cast<const uint32_t*>(column2),
      reinterpret_cast<const uint32_t*>(column3),
      reinterpret_cast<const uint32_t*>(column4),
      output,
      count);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int fri_lookup_merkle_node_preimages_cuda(
    const uint8_t* digests,
    uint8_t* output,
    size_t pair_count,
    void* stream_handle) {
  dim3 grid;
  dim3 block;
  const int config_error = launch_config(pair_count, &grid, &block);
  if (config_error != static_cast<int>(cudaSuccess)) {
    return config_error;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  node_preimages_kernel<<<grid, block, 0, stream>>>(digests, output, pair_count);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int fri_lookup_bn254_gather_fields_cuda(
    const void* fields,
    const size_t* indices,
    void* output,
    size_t count,
    void* stream_handle) {
  dim3 grid;
  dim3 block;
  const int config_error = launch_config(count, &grid, &block);
  if (config_error != static_cast<int>(cudaSuccess)) {
    return config_error;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  gather_fields_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(fields),
      indices,
      reinterpret_cast<uint32_t*>(output),
      count);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int fri_lookup_bn254_gather_row_fields_cuda(
    const void* column0,
    const void* column1,
    const void* column2,
    const void* column3,
    const void* column4,
    const size_t* indices,
    void* output,
    size_t count,
    size_t width,
    void* stream_handle) {
  if (width == 0 || width > 5) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  dim3 grid;
  dim3 block;
  const int config_error = launch_config(count, &grid, &block);
  if (config_error != static_cast<int>(cudaSuccess)) {
    return config_error;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  gather_row_fields_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(column0),
      reinterpret_cast<const uint32_t*>(column1),
      reinterpret_cast<const uint32_t*>(column2),
      reinterpret_cast<const uint32_t*>(column3),
      reinterpret_cast<const uint32_t*>(column4),
      indices,
      reinterpret_cast<uint32_t*>(output),
      count,
      width);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int fri_lookup_gather_digests_cuda(
    const uint8_t* digests,
    const size_t* indices,
    uint8_t* output,
    size_t count,
    void* stream_handle) {
  dim3 grid;
  dim3 block;
  const int config_error = launch_config(count, &grid, &block);
  if (config_error != static_cast<int>(cudaSuccess)) {
    return config_error;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  gather_digests_kernel<<<grid, block, 0, stream>>>(digests, indices, output, count);
  return static_cast<int>(cudaGetLastError());
}

