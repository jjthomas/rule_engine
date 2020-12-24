#include <stdint.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <assert.h>

extern "C" void compute2d_acc(uint8_t **, int, int, uint8_t *, uint32_t *);

#define BLOCK_SIZE 256

__global__ void run(uint8_t *input, uint32_t num_rows, uint32_t num_cols, uint32_t *thread_idxs, uint32_t *output) {
  uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t first_col = thread_idxs[index * 2];
  uint32_t second_col = thread_idxs[index * 2 + 1];
  uint32_t *our_output = output + index * 512;
  uint8_t *input_ptr = input;

  uint32_t counts[512] = {0};

  for (uint32_t i = 0; i < num_rows; i++) {
    uint8_t metric = *input_ptr;
    input_ptr += 1;
    uint8_t counts_idx = (input_ptr[first_col] << 4) | input_ptr[second_col];
    counts[2 * counts_idx] += metric;
    counts[2 * counts_idx + 1]++;
    input_ptr += num_cols;
  }
  for (uint32_t i = 0; i < 512; i++) {
    our_output[i] = counts[i];
  }
}

void compute2d_acc(uint8_t **cols, int num_rows, int num_cols, uint8_t *metric, uint32_t *stats) {
  assert(cudaSetDevice(0) == cudaSuccess);

  uint64_t input_size = (sizeof(uint8_t) + sizeof(uint8_t) * num_cols) * ((uint64_t)num_rows);
  uint8_t *input = (uint8_t *)malloc(input_size);
  // column to row
  #pragma omp parallel for
  for (int i = 0; i < num_rows; i++) {
    uint8_t *cur = input + (num_cols + 1) * i;
    *cur++ = metric[i];
    for (int j = 0; j < num_cols; j++) {
      *cur++ = cols[j][i];
    }
  }

  int num_pairs = num_cols * (num_cols + 1) / 2;
  int num_blocks = (num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int num_threads = num_blocks * BLOCK_SIZE;
  int thread_idxs_size = sizeof(uint32_t) * 2 * num_threads;
  uint32_t *thread_idxs = (uint32_t *)calloc(thread_idxs_size, 1);
  int pair_cnt = 0;
  for (int i = 0; i < num_cols; i++) {
    for (int j = i; j < num_cols; j++) {
      thread_idxs[2 * pair_cnt] = i;
      thread_idxs[2 * pair_cnt + 1] = j;
      pair_cnt++;
    }
  }

  uint8_t *input_dev;
  uint32_t *thread_idxs_dev;
  uint32_t *output_dev;
  int output_size = sizeof(uint32_t) * 512 * num_threads;
  assert(cudaMalloc((void **) &output_dev, output_size) == cudaSuccess);
  assert(cudaMalloc((void **) &input_dev, input_size) == cudaSuccess);
  assert(cudaMalloc((void **) &thread_idxs_dev, thread_idxs_size) == cudaSuccess);
  cudaMemcpy(input_dev, input, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(thread_idxs_dev, thread_idxs, thread_idxs_size, cudaMemcpyHostToDevice);

  run<<<num_blocks, BLOCK_SIZE>>>(input_dev, num_rows, num_cols, thread_idxs_dev, output_dev);
  assert(cudaDeviceSynchronize() == cudaSuccess);

  cudaMemcpy(stats, output_dev, 512 * sizeof(uint32_t) * num_pairs, cudaMemcpyDeviceToHost);

  cudaFree(output_dev);
  cudaFree(input_dev);
  cudaFree(thread_idxs_dev);
  free(input);
  free(thread_idxs);
}
