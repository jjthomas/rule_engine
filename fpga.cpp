#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <thread>
#include <sys/mman.h>

#include "fpga_pci.h"
#include "fpga_mgmt.h"
#include "fpga_dma.h"

extern "C" void compute2d_acc(uint8_t **, int, int, uint8_t *, uint32_t *);

// The block encoding code is mostly specific to this size
#define BLOCK_SIZE 48
// Must be from 1-4
#define NUM_WRITE_THREADS 4

void write_data(int write_fd, uint8_t *buf, int length, int addr) {
  fpga_dma_burst_write(write_fd, buf, length, addr);
}

void run(int read_fd, int write_fds[NUM_WRITE_THREADS], pci_bar_handle_t pci_bar_handle, uint8_t *input_buf,
  int input_buf_size, uint8_t *output_buf, int output_buf_size, bool buf_c) {
  if (input_buf != NULL) {
    if (NUM_WRITE_THREADS == 1) {
      write_data(write_fds[0], input_buf, input_buf_size, 0);
    } else {
      int chunk_size = input_buf_size / NUM_WRITE_THREADS;
      std::vector<std::thread> threads;
      for (int i = 0; i < NUM_WRITE_THREADS; i++) {
        int offset = i * chunk_size;
        threads.push_back(std::thread(write_data, write_fds[i], input_buf + offset,
          i == NUM_WRITE_THREADS - 1 ? input_buf_size - offset : chunk_size, offset));
      }
      for (auto& t : threads) {
        t.join();
      }
    }
  }
  uint32_t reg_peek;
  do {
    fpga_pci_peek(pci_bar_handle, 0x600, &reg_peek);
    usleep(1000);
  } while (reg_peek != 0);
  fpga_pci_poke(pci_bar_handle, 0x800, buf_c ? 1 : 0);
  if (input_buf != NULL) {
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

  int input_buf_size = 64 * (num_rows + 1);
  uint8_t *input_buf = (uint8_t *)mmap(NULL, input_buf_size, PROT_READ | PROT_WRITE,
    MAP_ANON | MAP_PRIVATE, -1, 0);
  *((uint32_t *)input_buf) = num_rows; // store length in first line
  uint8_t *input_data = input_buf + 64;
  int output_buf_size = sizeof(uint32_t) * 512 * BLOCK_SIZE * BLOCK_SIZE;
  uint32_t *output_buf = (uint32_t *)mmap(NULL, output_buf_size,
    PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);

  int num_blocks = (num_cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
  std::vector<std::vector<uint64_t>> blocks(num_blocks);
  for (int i = 0; i < num_blocks; i++) {
    std::vector<uint64_t> block(num_rows * 3); // 24 bytes per block row
    int start_col = i * BLOCK_SIZE;
    int end_col = std::min(num_cols, start_col + BLOCK_SIZE);
    #pragma omp parallel for
    for (int j = 0; j < num_rows; j++) {
      for (int k = 0; k < 3; k++) {
        uint64_t word = 0;
        for (int l = std::min(end_col, start_col + 16 * (k + 1)) - 1; l >= start_col + 16 * k; l--) {
          word = (word << 4) | cols[l][j];
        }
        block[3 * j + k] = word;
      }
    }
    blocks[i] = std::move(block);
  }
  #pragma omp parallel for
  for (int i = 0; i < num_rows; i++) {
    input_data[64 * i + 48] = metric[i];
  }

  fpga_pci_poke(pci_bar_handle, 0x800, 0); // set buf to DDR C
  bool buf_c = true;
  int last_i = 0, last_j = 0;
  for (int i = 0; i < num_blocks; i++) {
    #pragma omp parallel for
    for (int j = 0; j < num_rows; j++) {
      uint64_t *cur_row = (uint64_t *)(input_data + 64 * j);
      for (int k = 0; k < 3; k++) {
        cur_row[k] = blocks[i][3 * j + k];
      }
    }
    // final iteration to collect last output
    int bound = i == num_blocks - 1 ? num_blocks + 1 : num_blocks;
    for (int j = i; j < bound; j++) {
      if (j != num_blocks) {
        #pragma omp parallel for
        for (int k = 0; k < num_rows; k++) {
          uint64_t *cur_row = (uint64_t *)(input_data + 64 * k + 24);
          for (int l = 0; l < 3; l++) {
            cur_row[l] = blocks[j][3 * k + l];
          }
        }
      }
      run(read_fd, write_fds, pci_bar_handle,
        j == num_blocks ? NULL : input_buf, input_buf_size,
        i == 0 && j == 0 ? NULL : (uint8_t *)output_buf, output_buf_size, buf_c);
      if (!(i == 0 && j == 0)) {
        int i_start_col = last_i * BLOCK_SIZE;
        int j_start_col = last_j * BLOCK_SIZE;
        int i_end_col = std::min(num_cols, i_start_col + BLOCK_SIZE);
        int j_end_col = std::min(num_cols, j_start_col + BLOCK_SIZE);
        for (int k = i_start_col; k < i_end_col; k++) {
          for (int l = last_i == last_j ? k : j_start_col; l < j_end_col; l++) {
            int remaining_triangle_base = num_cols - k;
            // idx in output triangle
            int target_idx = num_cols * (num_cols + 1) / 2 -
              remaining_triangle_base * (remaining_triangle_base + 1) / 2 +
              (l - k);
            int source_idx = (k - i_start_col) * BLOCK_SIZE + (l - j_start_col);
            memcpy(stats + 512 * target_idx, output_buf + 512 * source_idx,
              512 * sizeof(uint32_t));
          }
        }
      }
      last_i = i;
      last_j = j;
      buf_c = !buf_c;
    }
  }
  munmap(input_buf, input_buf_size);
  munmap(output_buf, output_buf_size);
  close(read_fd);
  for (int i = 0; i < NUM_WRITE_THREADS; i++) {
    close(write_fds[i]);
  }
}
