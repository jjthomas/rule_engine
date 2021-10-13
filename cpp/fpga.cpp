#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <thread>
#include <sys/mman.h>
#include <sys/time.h>

#include "fpga_pci.h"
#include "fpga_mgmt.h"
#include "fpga_dma.h"

extern "C" void compute2d_acc(uint8_t **, int, int, uint8_t *, uint32_t *);

// The block encoding code is mostly specific to this size
#define BLOCK_SIZE 48
// Must be from 1-4
#define NUM_WRITE_THREADS 4

#define FINE_TIME

void write_data(int write_fd, uint8_t *buf0, uint8_t *buf1, int length, int addr) {
  fpga_dma_burst_write(write_fd, buf0, length, addr);
  fpga_dma_burst_write(write_fd, buf1, length, 1000000000 + addr);
}

void run(int read_fd, int write_fds[NUM_WRITE_THREADS], pci_bar_handle_t pci_bar_handle, uint8_t *input_buf0,
  uint8_t *input_buf1, int input_buf_size, uint8_t *output_buf, int output_buf_size, bool buf_c) {
  if (input_buf0 != NULL) {
    if (NUM_WRITE_THREADS == 1) {
      write_data(write_fds[0], input_buf0, input_buf1, input_buf_size, 0);
    } else {
      int chunk_size = input_buf_size / NUM_WRITE_THREADS;
      std::vector<std::thread> threads;
      for (int i = 0; i < NUM_WRITE_THREADS; i++) {
        int offset = i * chunk_size;
        threads.push_back(std::thread(write_data, write_fds[i], input_buf0 + offset, input_buf1 + offset,
          i == NUM_WRITE_THREADS - 1 ? input_buf_size - offset : chunk_size, offset));
      }
      for (auto& t : threads) {
        t.join();
      }
    }
  }
#ifdef FINE_TIME
  struct timeval start, end, diff;
  gettimeofday(&start, 0);
#endif
  uint32_t reg_peek;
  do {
    fpga_pci_peek(pci_bar_handle, 0x600, &reg_peek);
    usleep(1000);
  } while (reg_peek != 0);
#ifdef FINE_TIME
  gettimeofday(&end, 0);
  timersub(&end, &start, &diff);
  printf("FPGA wait time: %ld.%06ld\n", (long)diff.tv_sec, (long)diff.tv_usec);
#endif
  fpga_pci_poke(pci_bar_handle, 0x800, buf_c ? 1 : 0);
  if (input_buf0 != NULL) {
    fpga_pci_poke(pci_bar_handle, 0x600, 1);
  }
  if (output_buf != NULL) {
    fpga_dma_burst_read(read_fd, output_buf, output_buf_size, 0);
  }
}

void compute2d_acc(uint8_t **cols, int num_rows, int num_cols, uint8_t *metric, uint32_t *stats) {
  if (fpga_mgmt_init() != 0) {
    printf("ERROR: fpga_mgmt_init()\n");
  }
  int read_fd = fpga_dma_open_queue(FPGA_DMA_XDMA, 0, 0, true);
  if (read_fd < 0) {
    printf("ERROR: unable to get XDMA read handle\n");
  }
  int write_fds[NUM_WRITE_THREADS];
  for (int i = 0; i < NUM_WRITE_THREADS; i++) {
    write_fds[i] = fpga_dma_open_queue(FPGA_DMA_XDMA, 0, i, false);
  }
  pci_bar_handle_t pci_bar_handle = PCI_BAR_HANDLE_INIT;
  if (fpga_pci_attach(0, 0, 0, 0, &pci_bar_handle) != 0) {
    printf("ERROR: PCI attach\n");
  }

  int block_bytes = 64 * ((num_rows + 1) / 2 + 1);
  int num_blocks = (num_cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
  uint64_t input_buf_size = (uint64_t)block_bytes * num_blocks;
  uint8_t *input_buf = (uint8_t *)mmap(NULL, input_buf_size, PROT_READ | PROT_WRITE,
    MAP_ANON | MAP_PRIVATE, -1, 0);
  int output_buf_size = sizeof(uint32_t) * 512 * BLOCK_SIZE * BLOCK_SIZE;
  uint32_t *output_buf = (uint32_t *)mmap(NULL, output_buf_size,
    PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);

  for (int i = 0; i < num_blocks; i++) {
    uint8_t *block = input_buf + block_bytes * i;
    *((uint32_t *)block) = num_rows;
    block += 64;
    int start_col = i * BLOCK_SIZE;
    int end_col = std::min(num_cols, start_col + BLOCK_SIZE);
    #pragma omp parallel for
    for (int j = 0; j < (num_rows + 1) / 2; j++) {
      int start_row = j * 2;
      uint64_t *block_row = (uint64_t *)(block + j * 64);
      uint64_t metric_word = 0;
      for (int k = 0; k < std::min(num_rows - start_row, 2); k++) {
        for (int l = 0; l < 3; l++) {
          uint64_t word = 0;
          for (int m = std::min(end_col, start_col + 16 * (l + 1)) - 1; m >= start_col + 16 * l; m--) {
            word = (word << 4) | cols[m][start_row + k];
          }
          block_row[3 * k + l] = word;
        }
        metric_word |= metric[start_row + k] << (8 * k);
      }
      block_row[6] = metric_word;
    }
  }

  fpga_pci_poke(pci_bar_handle, 0x800, 0); // set buf to DDR C
  bool buf_c = true;
  int last_i = 0, last_j = 0;
  struct timeval start, end, diff;
  gettimeofday(&start, 0);
  for (int i = 0; i < num_blocks; i++) {
    // final iteration to collect last output
    int bound = i == num_blocks - 1 ? num_blocks + 1 : num_blocks;
    for (int j = i; j < bound; j++) {
#ifdef FINE_TIME
      struct timeval it_start, it_end, it_diff;
      gettimeofday(&it_start, 0);
#endif
      run(read_fd, write_fds, pci_bar_handle,
        j == num_blocks ? NULL : input_buf + block_bytes * i,
        j == num_blocks ? NULL : input_buf + block_bytes * j, block_bytes,
        i == 0 && j == 0 ? NULL : (uint8_t *)output_buf, output_buf_size, buf_c);
      if (!(i == 0 && j == 0)) {
        int i_start_col = last_i * BLOCK_SIZE;
        int j_start_col = last_j * BLOCK_SIZE;
        int i_end_col = std::min(num_cols, i_start_col + BLOCK_SIZE);
        int j_end_col = std::min(num_cols, j_start_col + BLOCK_SIZE);
        for (int k = i_start_col; k < i_end_col; k++) {
          for (int l = last_i == last_j ? k + 1 : j_start_col; l < j_end_col; l++) {
            int remaining_triangle_base = num_cols - k;
            // idx in output triangle
            int target_idx = num_cols * (num_cols - 1) / 2 -
              remaining_triangle_base * (remaining_triangle_base - 1) / 2 +
              (l - k - 1);
            int source_idx = (k - i_start_col) * BLOCK_SIZE + (l - j_start_col);
            memcpy(stats + 512 * target_idx, output_buf + 512 * source_idx,
              512 * sizeof(uint32_t));
          }
        }
      }
#ifdef FINE_TIME
      gettimeofday(&it_end, 0);
      timersub(&it_end, &it_start, &it_diff);
      printf("FPGA iteration time: %ld.%06ld\n", (long)it_diff.tv_sec, (long)it_diff.tv_usec);
#endif
      last_i = i;
      last_j = j;
      buf_c = !buf_c;
    }
  }
  gettimeofday(&end, 0);
  timersub(&end, &start, &diff);
  printf("FPGA accelerator time: %ld.%06ld\n", (long)diff.tv_sec, (long)diff.tv_usec);
  munmap(input_buf, input_buf_size);
  munmap(output_buf, output_buf_size);
  close(read_fd);
  for (int i = 0; i < NUM_WRITE_THREADS; i++) {
    close(write_fds[i]);
  }
}
