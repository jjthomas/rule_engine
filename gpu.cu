#include <stdint.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <assert.h>

extern "C" void compute2d_acc(uint8_t **, int, int, uint8_t *, uint32_t *);

#define BLOCK_SIZE 256

__global__ void run(uint8_t *input, uint32_t num_rows, uint32_t num_cols, uint32_t *block_idxs, uint32_t *output) {
  uint64_t index = blockIdx.x;
  uint32_t first_col = block_idxs[index * 2];
  uint32_t second_col = block_idxs[index * 2 + 1];
  uint32_t *our_output = output + index * 512;
  uint8_t *metric = input;
  uint8_t *col1 = input + (first_col + 1) * num_rows;
  uint8_t *col2 = input + (second_col + 1) * num_rows;

  __shared__ uint32_t counts[512];
  for (int i = threadIdx.x; i < 512; i += blockDim.x) {
    counts[i] = 0;
  }
  __syncthreads();

  for (uint32_t i = threadIdx.x; i < num_rows; i += blockDim.x) {
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

  uint64_t input_size = ((uint64_t)num_rows) * (num_cols + 1);
  uint8_t *input = (uint8_t *)malloc(input_size);
  memcpy(input, metric, num_rows);
  #pragma omp parallel for
  for (int i = 0; i < num_cols; i++) {
    memcpy(input + (i + 1) * num_rows, cols[i], num_rows);
  }

  int num_pairs = num_cols * (num_cols + 1) / 2;
  int block_idxs_size = sizeof(uint32_t) * 2 * num_pairs;
  uint32_t *block_idxs = (uint32_t *)calloc(block_idxs_size, 1);
  int pair_cnt = 0;
  for (int i = 0; i < num_cols; i++) {
    for (int j = i; j < num_cols; j++) {
      block_idxs[2 * pair_cnt] = i;
      block_idxs[2 * pair_cnt + 1] = j;
      pair_cnt++;
    }
  }

  uint8_t *input_dev;
  uint32_t *block_idxs_dev;
  uint32_t *output_dev;
  int output_size = sizeof(uint32_t) * 512 * num_pairs;
  assert(cudaMalloc((void **) &output_dev, output_size) == cudaSuccess);
  assert(cudaMalloc((void **) &input_dev, input_size) == cudaSuccess);
  assert(cudaMalloc((void **) &block_idxs_dev, block_idxs_size) == cudaSuccess);
  cudaMemcpy(input_dev, input, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(block_idxs_dev, block_idxs, block_idxs_size, cudaMemcpyHostToDevice);

  run<<<num_pairs, BLOCK_SIZE>>>(input_dev, num_rows, num_cols, block_idxs_dev, output_dev);
  assert(cudaDeviceSynchronize() == cudaSuccess);

  cudaMemcpy(stats, output_dev, output_size, cudaMemcpyDeviceToHost);

  cudaFree(output_dev);
  cudaFree(input_dev);
  cudaFree(block_idxs_dev);
  free(input);
  free(block_idxs);
}
