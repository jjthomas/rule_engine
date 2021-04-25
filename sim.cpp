#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>

extern "C" void compute2d_acc(uint8_t **, int, int, uint8_t *, uint32_t *);

void compute2d_acc(uint8_t **cols, int num_rows, int num_cols, uint8_t *metric, uint32_t *stats) {
  int num_pairs = num_cols * (num_cols - 1) / 2;
  memset(stats, 0, 512 * sizeof(uint32_t) * num_pairs);
  std::vector<std::pair<int, int>> idx_mapping;
  for (int i = 0; i < num_cols; i++) {
    for (int j = i + 1; j < num_cols; j++) {
      idx_mapping.emplace_back(i, j);
    }
  }
  #pragma omp parallel for
  for (int pair_idx = 0; pair_idx < num_pairs; pair_idx++) {
    int i = idx_mapping[pair_idx].first;
    int j = idx_mapping[pair_idx].second;
    uint32_t *our_stats = stats + 512 * pair_idx;
    for (int k = 0; k < num_rows; k++) {
      int stats_idx = (cols[i][k] << 4) | cols[j][k];
      our_stats[2 * stats_idx] += metric[k];
      our_stats[2 * stats_idx + 1]++;
    }
  }
}
