#include <stdint.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <assert.h>

extern "C" void compute2d_acc(uint8_t **, int, int, uint8_t *, uint32_t *);

#define BLOCK_SIZE 256

__global__ void run(uint8_t **input, uint8_t *metric, uint32_t num_rows, uint32_t num_cols, uint32_t *block_idxs, uint32_t *output) {
  uint64_t index = blockIdx.x;
  uint32_t first_col = block_idxs[index * 2];
  uint32_t second_col = block_idxs[index * 2 + 1];
  uint8_t *col1 = input[first_col];
  uint8_t *col2 = input[second_col];
  uint32_t *our_output = output + index * 512;

  __shared__ uint32_t counts[512];
  for (int i = threadIdx.x; i < 512; i += blockDim.x) {
    counts[i] = 0;
  }
  __syncthreads();

  uint32_t *met_32 = (uint32_t *)metric;
  uint32_t *col1_32 = (uint32_t *)col1;
  uint32_t *col2_32 = (uint32_t *)col2;
  for (uint32_t i = threadIdx.x; i < num_rows / 4; i += blockDim.x) {
    uint32_t counts_idx = (col1_32[i] << 4) | col2_32[i];
    uint32_t cur_met = met_32[i];
    for (int j = 0; j < 4; j++) {
      uint8_t single_idx = counts_idx & 255;
      atomicAdd(&counts[2 * single_idx], cur_met & 255);
      atomicAdd(&counts[2 * single_idx + 1], 1);
      counts_idx >>= 8;
      cur_met >>= 8;
    }
  }
  for (uint32_t i = num_rows / 4 * 4 + threadIdx.x; i < num_rows; i += blockDim.x) {
    uint8_t counts_idx = (col1[i] << 4) | col2[i];
    atomicAdd(&counts[2 * counts_idx], metric[i]);
    atomicAdd(&counts[2 * counts_idx + 1], 1);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < 512; i += blockDim.x) {
    our_output[i] = counts[i];
  }
}

void compute2d_acc(uint8_t **cols, int num_rows, int num_cols, uint8_t *metric, uint32_t *stats) {
  assert(cudaSetDevice(0) == cudaSuccess);

  uint8_t **input_dev;
  assert(cudaMalloc((void **) &input_dev, sizeof(uint8_t *) * num_cols) == cudaSuccess);
  uint8_t **col_dev_ptrs = (uint8_t **)malloc(sizeof(uint8_t *) * num_cols);
  for (int i = 0; i < num_cols; i++) {
    assert(cudaMalloc((void **) &col_dev_ptrs[i], num_rows) == cudaSuccess);
    cudaMemcpy(col_dev_ptrs[i], cols[i], num_rows, cudaMemcpyHostToDevice);
  }
  cudaMemcpy(input_dev, col_dev_ptrs, sizeof(uint8_t *) * num_cols, cudaMemcpyHostToDevice);
  uint8_t *metric_dev;
  assert(cudaMalloc((void **) &metric_dev, num_rows) == cudaSuccess);
  cudaMemcpy(metric_dev, metric, num_rows, cudaMemcpyHostToDevice);

  int num_pairs = num_cols * (num_cols + 1) / 2;
  int block_idxs_size = sizeof(uint32_t) * 2 * num_pairs;
  uint32_t *block_idxs = (uint32_t *)malloc(block_idxs_size);
  int pair_cnt = 0;
  for (int i = 0; i < num_cols; i++) {
    for (int j = i; j < num_cols; j++) {
      block_idxs[2 * pair_cnt] = i;
      block_idxs[2 * pair_cnt + 1] = j;
      pair_cnt++;
    }
  }
  uint32_t *block_idxs_dev;
  assert(cudaMalloc((void **) &block_idxs_dev, block_idxs_size) == cudaSuccess);
  cudaMemcpy(block_idxs_dev, block_idxs, block_idxs_size, cudaMemcpyHostToDevice);

  uint32_t *output_dev;
  int output_size = sizeof(uint32_t) * 512 * num_pairs;
  assert(cudaMalloc((void **) &output_dev, output_size) == cudaSuccess);

  run<<<num_pairs, BLOCK_SIZE>>>(input_dev, metric_dev, num_rows, num_cols, block_idxs_dev, output_dev);
  assert(cudaDeviceSynchronize() == cudaSuccess);

  cudaMemcpy(stats, output_dev, output_size, cudaMemcpyDeviceToHost);

  cudaFree(output_dev);
  cudaFree(block_idxs_dev);
  cudaFree(input_dev);
  for (int i = 0; i < num_cols; i++) {
    cudaFree(col_dev_ptrs[i]);
  }
  cudaFree(metric_dev);
  free(col_dev_ptrs);
  free(block_idxs);
}
