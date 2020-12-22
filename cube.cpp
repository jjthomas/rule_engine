#define _GLIBCXX_USE_CXX11_ABI 0
#include <Python.h>
#include <arrow/python/pyarrow.h>
#include <arrow/python/common.h>
#include <arrow/api.h>
#include <algorithm>
#include <cmath>
#include <set>
#include <map>
#include <inttypes.h>
#include <sys/time.h>

extern "C" void compute_stats(PyObject *, int, double, int, int);

#define STRING_CARD_LIMIT 100
#define NUM_CARD_LIMIT 50
#define DISC_COUNT 15

inline bool not_null(const uint8_t *null_map, int64_t pos) {
  return null_map == nullptr || ((null_map[pos >> 3] >> (pos & 7)) & 1);
}

template <typename T>
bool categorical(const T *data, int64_t len, const uint8_t *null_map, int limit) {
  std::set<T> s;
  int has_null = 0;
  for (int64_t i = 0; i < len; i++) {
    if (not_null(null_map, i)) {
      s.insert(data[i]);
    } else {
      has_null = 1;
    }
    if (s.size() + has_null > limit) {
      return false;
    }
  }
  return true;
}

template <typename T>
void enumerate_cat(const T *data, int64_t len, const uint8_t *null_map,
  std::vector<uint8_t>& result, std::vector<T> &rev_mapping) {
  bool has_null = null_map != nullptr;
  // save 0 id for null if it is present
  uint8_t cur_id = has_null ? 1 : 0;
  std::map<T, uint8_t> mapping;
  for (int64_t i = 0; i < len; i++) {
    if (not_null(null_map, i)) {
      if (mapping.find(data[i]) == mapping.end()) {
        mapping[data[i]] = cur_id++;
      }
      result[i] = mapping[data[i]];
    } else {
      result[i] = 0;
    }
  }
  rev_mapping.resize(mapping.size());
  for (std::pair<T, uint8_t> e : mapping) {
    rev_mapping[has_null ? e.second - 1 : e.second] = e.first;
  }
}

template <typename T>
std::pair<double, double> discretize_cont(const T *data, int64_t len,
  const uint8_t *null_map, std::vector<uint8_t>& result) {
  int64_t i = 0;
  while (!not_null(null_map, i)) {
    i++;
  }
  T min = data[i];
  T max = data[i];
  for (i = i + 1; i < len; i++) {
    if (not_null(null_map, i)) {
      if (data[i] > max) {
        max = data[i];
      }
      if (data[i] < min) {
        min = data[i];
      }
    }
  }

  double step = ((double)(max - min)) / DISC_COUNT;
  for (i = 0; i < len; i++) {
    if (not_null(null_map, i)) {
      result[i] = 1 + std::min((uint8_t)(DISC_COUNT - 1), (uint8_t)((data[i] - min) / step));
    } else {
      result[i] = 0;
    }
  }

  return std::make_pair((double)min, (double)max);
}

enum ColStatus {
  CONT, CAT, BAD_TYPE, STRING_HIGH_CARD, METRIC_ERROR
};

void compute_stats(PyObject *obj, int metric_idx, double z_thresh, int count_thresh,
  int show_nulls) {
  arrow::py::PyAcquireGIL lock;
  arrow::py::import_pyarrow();
  auto table = arrow::py::unwrap_table(obj).ValueOrDie();
  int64_t num_rows = table->num_rows();
  int num_fields = table->schema()->num_fields();
  std::vector<std::vector<uint8_t>> cols_init(num_fields);
  std::vector<ColStatus> col_status(num_fields);
  std::vector<std::pair<double, double>> min_max_init(num_fields);
  std::vector<std::vector<double>> double_mappings_init(num_fields);
  std::vector<std::vector<int64_t>> int_mappings_init(num_fields);
  std::vector<std::vector<std::string>> string_mappings_init(num_fields);
  struct timeval start, end, diff;
  printf("***Columns***\n");
  gettimeofday(&start, 0);
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < table->schema()->num_fields(); i++) {
    auto f = table->schema()->field(i);
    assert(table->column(i)->num_chunks() == 1);
    switch (f->type()->id()) {
      case arrow::Type::type::DOUBLE: {
        auto arr = std::static_pointer_cast<arrow::DoubleArray>(table->column(i)->chunk(0));
        bool is_cat = categorical(arr->raw_values(), arr->length(), arr->null_bitmap_data(),
          NUM_CARD_LIMIT);
        cols_init[i] = std::vector<uint8_t>(num_rows);
        std::vector<uint8_t>& col = cols_init[i];
        if (is_cat) {
          col_status[i] = CAT;
          if (i == metric_idx) {
            for (int64_t j = 0; j < arr->length(); j++) {
              if (arr->IsNull(j)) {
                printf("ERROR: null metric value at row %" PRId64 "\n", j);
                col_status[i] = METRIC_ERROR;
                break;
              }
              if (!(arr->Value(j) >= 0 && arr->Value(j) < 256)) {
                printf("ERROR: categorical metric has value %.2f "
                  "outside range [0, 256) at row %" PRId64 "\n", arr->Value(j), j);
                col_status[i] = METRIC_ERROR;
                break;
              }
              col[j] = (uint8_t)arr->Value(j);
            }
          } else {
            enumerate_cat(arr->raw_values(), arr->length(), arr->null_bitmap_data(),
              col, double_mappings_init[i]);
          }
        } else {
          col_status[i] = CONT;
          min_max_init[i] = discretize_cont(arr->raw_values(), arr->length(),
            arr->null_bitmap_data(), col);
        }
        break;
      }
      case arrow::Type::type::INT64: {
        auto arr = std::static_pointer_cast<arrow::Int64Array>(table->column(i)->chunk(0));
        bool is_cat = categorical(arr->raw_values(), arr->length(), arr->null_bitmap_data(),
          NUM_CARD_LIMIT);
        cols_init[i] = std::vector<uint8_t>(num_rows);
        std::vector<uint8_t>& col = cols_init[i];
        if (is_cat) {
          col_status[i] = CAT;
          if (i == metric_idx) {
            for (int64_t j = 0; j < arr->length(); j++) {
              if (arr->IsNull(j)) {
                printf("ERROR: null metric value at row %" PRId64 "\n", j);
                col_status[i] = METRIC_ERROR;
                break;
              }
              if (!(arr->Value(j) >= 0 && arr->Value(j) < 256)) {
                printf("ERROR: categorical metric has value %" PRId64
                  " outside range [0, 256) at row %" PRId64 "\n", arr->Value(j), j);
                col_status[i] = METRIC_ERROR;
                break;
              }
              col[j] = (uint8_t)arr->Value(j);
            }
          } else {
            enumerate_cat(arr->raw_values(), arr->length(), arr->null_bitmap_data(),
              col, int_mappings_init[i]);
          }
        } else {
          col_status[i] = CONT;
          min_max_init[i] = discretize_cont(arr->raw_values(), arr->length(),
            arr->null_bitmap_data(), col);
        }
        break;
      }
      case arrow::Type::type::STRING: {
        auto arr = std::static_pointer_cast<arrow::StringArray>(table->column(i)->chunk(0));
        std::vector<std::string> str_arr(arr->length());
        for (int64_t j = 0; j < arr->length(); j++) {
          str_arr[j] = arr->GetString(j);
        }
        bool is_cat = categorical(str_arr.data(), arr->length(), arr->null_bitmap_data(),
          STRING_CARD_LIMIT);
        if (is_cat) {
          col_status[i] = CAT;
          cols_init[i] = std::vector<uint8_t>(num_rows);
          enumerate_cat(str_arr.data(), arr->length(), arr->null_bitmap_data(),
            cols_init[i], string_mappings_init[i]);
        } else {
          col_status[i] = STRING_HIGH_CARD;
        }
        break;
      }
      case arrow::Type::type::BOOL: {
        auto arr = std::static_pointer_cast<arrow::BooleanArray>(table->column(i)->chunk(0));
        col_status[i] = CAT;
        cols_init[i] = std::vector<uint8_t>(num_rows);
        std::vector<uint8_t>& col = cols_init[i];
        int null_offset = arr->null_count() > 0 ? 1 : 0;
        for (int64_t j = 0; j < arr->length(); j++) {
          if (arr->IsNull(j)) {
            if (i == metric_idx) {
              printf("ERROR: null metric value at row %" PRId64 "\n", j);
              col_status[i] = METRIC_ERROR;
              break;
            }
            col[j] = 0;
          } else {
            col[j] = null_offset + (arr->Value(j) ? 1 : 0);
          }
        }
      }
      default:
        col_status[i] = BAD_TYPE;
    }
  }

  if (col_status[metric_idx] == METRIC_ERROR) {
    return;
  }
  std::vector<uint8_t> metric_col = std::move(cols_init[metric_idx]);
  std::pair<double, double> metric_min_max = min_max_init[metric_idx];

  std::vector<std::vector<uint8_t>> cols;
  std::vector<int> sizes;
  std::vector<bool> has_nulls;
  std::vector<int> orig_col_idx;
  std::map<int, std::pair<double, double>> min_max;
  std::map<int, std::vector<double>> double_mappings;
  std::map<int, std::vector<int64_t>> int_mappings;
  std::map<int, std::vector<std::string>> string_mappings;
  std::set<int> bool_cols;
  for (int i = 0; i < table->schema()->num_fields(); i++) {
    auto f = table->schema()->field(i);
    printf("%s: %s\n", f->name().c_str(), f->type()->ToString().c_str());
    if (col_status[i] == BAD_TYPE) {
      printf("  WARN: unsupported type\n");
    } else if (col_status[i] == STRING_HIGH_CARD) {
      printf("  WARN: string cardinality too high to use\n");
    }
    if (i == metric_idx) {
      if (col_status[i] == BAD_TYPE || f->type()->id() == arrow::Type::type::STRING) {
        printf("ERROR: unsupported type for metric\n");
        return;
      }
    } else if (col_status[i] == CONT || col_status[i] == CAT) {
      int cur_col = cols.size();
      int size;
      bool has_null;
      if (col_status[i] == CONT) {
        min_max[cur_col] = min_max_init[i];
        size = DISC_COUNT + 1; // always has null
        has_null = true;
      } else {
        has_null = table->column(i)->null_count() > 0;
        size = has_null ? 1 : 0;
        if (f->type()->id() == arrow::Type::type::DOUBLE) {
          double_mappings[cur_col] = std::move(double_mappings_init[i]);
          size += double_mappings[cur_col].size();
        } else if (f->type()->id() == arrow::Type::type::INT64) {
          int_mappings[cur_col] = std::move(int_mappings_init[i]);
          size += int_mappings[cur_col].size();
        } else if (f->type()->id() == arrow::Type::type::STRING) {
          string_mappings[cur_col] = std::move(string_mappings_init[i]);
          size += string_mappings[cur_col].size();
        } else { // bool
          bool_cols.insert(cur_col);
          size += 2;
        }
      }
      orig_col_idx.push_back(i);
      cols.push_back(std::move(cols_init[i]));
      sizes.push_back(size);
      has_nulls.push_back(has_null);
    }
    if (col_status[i] == CAT) {
      printf("  categorical");
      if (i != metric_idx) {
        printf(" (%d", sizes.back());
        if (has_nulls.back()) {
          printf(", incl. null)");
        } else {
          printf(")");
        }
      }
      printf("\n");
    } else if (col_status[i] == CONT) {
      printf("  continuous (%.2f-%.2f)\n", min_max_init[i].first, min_max_init[i].second);
    }
  }
  gettimeofday(&end, 0);
  timersub(&end, &start, &diff);
  printf("Columns time: %ld.%06ld\n", (long)diff.tv_sec, (long)diff.tv_usec);

  uint64_t global_sum = 0;
  for (int64_t i = 0; i < num_rows; i++) {
    global_sum += metric_col[i];
  }
  double global_avg = (double)global_sum / num_rows;
  double global_dev = 0;
  for (int64_t i = 0; i < num_rows; i++) {
    global_dev += pow(metric_col[i] - global_avg, 2);
  }
  global_dev = sqrt(global_dev / num_rows);
  printf("\n%s global mean: %.2f, global stddev: %.2f\n",
    table->schema()->field(metric_idx)->name().c_str(), global_avg, global_dev);
  printf("\n***1D stats***\n");
  std::vector<std::vector<uint64_t>> col_sums(cols.size());
  std::vector<std::vector<uint64_t>> col_counts(cols.size());
  std::vector<std::vector<double>> col_devs(cols.size());
  gettimeofday(&start, 0);
  #pragma omp parallel for
  for (int i = 0; i < cols.size(); i++) {
    int size = sizes[i];
    std::vector<uint64_t> sums(size);
    std::vector<uint64_t> counts(size);
    std::vector<uint8_t>& col = cols[i];
    for (int64_t j = 0; j < num_rows; j++) {
      sums[col[j]] += metric_col[j];
      counts[col[j]]++;
    }
    std::vector<double> devs(size);
    for (int64_t j = 0; j < num_rows; j++) {
      if (counts[col[j]] < count_thresh) {
        continue;
      }
      double group_avg = (double)sums[col[j]] / counts[col[j]];
      devs[col[j]] += pow(metric_col[j] - group_avg, 2);
    }
    for (int j = 0; j < size; j++) {
      devs[j] = sqrt(devs[j] / counts[j]);
    }
    col_sums[i] = std::move(sums);
    col_counts[i] = std::move(counts);
    col_devs[i] = std::move(devs);
  }

  int null_start = show_nulls ? 0 : 1;
  for (int i = 0; i < cols.size(); i++) {
    std::vector<uint64_t>& sums = col_sums[i];
    std::vector<uint64_t>& counts = col_counts[i];
    bool header_printed = false;
    int value_start_idx = has_nulls[i] ? null_start : 0;
    for (int j = value_start_idx; j < sums.size(); j++) {
      if (counts[j] < count_thresh) {
        continue;
      }
      double group_avg = (double)sums[j] / counts[j];
      double effective_dev = global_dev / sqrt(counts[j]);
      double z_score = (group_avg - global_avg) / effective_dev;
      if (std::abs(z_score) > z_thresh) {
        if (!header_printed) {
          printf("%s:\n", table->schema()->field(orig_col_idx[i])->name().c_str());
          header_printed = true;
        }
        if (j == 0 && has_nulls[i]) {
          printf("  NULL: ");
        } else {
          int idx = has_nulls[i] ? j - 1 : j;
          if (double_mappings.find(i) != double_mappings.end()) {
            printf("  %.2f: ", double_mappings[i][idx]);
          } else if (int_mappings.find(i) != int_mappings.end()) {
            printf("  %" PRId64 ": ", int_mappings[i][idx]);
          } else if (string_mappings.find(i) != string_mappings.end()) {
            printf("  %s: ", string_mappings[i][idx].c_str());
          } else if (bool_cols.find(i) != bool_cols.end()) {
            printf("  %s: ", idx == 0 ? "false" : "true");
          } else { // continuous
            printf("  %d: ", idx);
          }
        }
        printf("%.2f (z:%.4f, #:%" PRIu64 ")\n", group_avg, z_score, counts[j]);
      }
    }
  }
  gettimeofday(&end, 0);
  timersub(&end, &start, &diff);
  printf("1D time: %ld.%06ld\n", (long)diff.tv_sec, (long)diff.tv_usec);

  printf("\n***2D stats***\n");
  std::vector<std::vector<uint64_t>> pair_sums(cols.size() * (cols.size() - 1) / 2);
  std::vector<std::vector<uint64_t>> pair_counts(pair_sums.size());
  std::vector<std::pair<int, int>> idx_mapping;
  for (int i = 0; i < cols.size(); i++) {
    for (int j = i + 1; j < cols.size(); j++) {
      idx_mapping.emplace_back(i, j);
    }
  }
  gettimeofday(&start, 0);
  #pragma omp parallel for
  for (int pair_idx = 0; pair_idx < pair_sums.size(); pair_idx++) {
    int i = idx_mapping[pair_idx].first;
    int j = idx_mapping[pair_idx].second;
    int i_card = col_sums[i].size();
    int j_card = col_sums[j].size();
    std::vector<uint64_t> sums(i_card * j_card);
    std::vector<uint64_t> counts(i_card * j_card);
    std::vector<uint8_t>& i_col = cols[i];
    std::vector<uint8_t>& j_col = cols[j];
    for (int64_t k = 0; k < num_rows; k++) {
      int idx = i_col[k] * j_card + j_col[k];
      sums[idx] += metric_col[k];
      counts[idx]++;
    }
    pair_sums[pair_idx] = std::move(sums);
    pair_counts[pair_idx] = std::move(counts);
  }
  for (int pair_idx = 0; pair_idx < pair_sums.size(); pair_idx++) {
    std::vector<uint64_t>& sums = pair_sums[pair_idx];
    std::vector<uint64_t>& counts = pair_counts[pair_idx];
    int i = idx_mapping[pair_idx].first;
    int j = idx_mapping[pair_idx].second;
    int i_card = col_sums[i].size();
    int j_card = col_sums[j].size();
    bool header_printed = false;
    int i_value_start_idx = has_nulls[i] ? null_start : 0;
    int j_value_start_idx = has_nulls[j] ? null_start : 0;
    for (int k = i_value_start_idx; k < i_card; k++) {
      for (int l = j_value_start_idx; l < j_card; l++) {
        int idx = k * j_card + l;
        if (counts[idx] < count_thresh) {
          continue;
        }
        double group_avg = (double)sums[idx] / counts[idx];
        double i_avg = (double)col_sums[i][k] / col_counts[i][k];
        double j_avg = (double)col_sums[j][l] / col_counts[j][l];
        double i_dev = col_devs[i][k] / sqrt(counts[idx]);
        double j_dev = col_devs[j][l] / sqrt(counts[idx]);
        double i_z_score = (group_avg - i_avg) / i_dev;
        double j_z_score = (group_avg - j_avg) / j_dev;
        if (std::abs(i_z_score) > z_thresh && std::abs(j_z_score) > z_thresh) {
          if (!header_printed) {
            printf("%s/%s:\n", table->schema()->field(orig_col_idx[i])->name().c_str(),
              table->schema()->field(orig_col_idx[j])->name().c_str());
            header_printed = true;
          }
          if (k == 0 && has_nulls[i]) {
            printf("  NULL/");
          } else {
            int i_idx = has_nulls[i] ? k - 1 : k;
            if (double_mappings.find(i) != double_mappings.end()) {
              printf("  %.2f/", double_mappings[i][i_idx]);
            } else if (int_mappings.find(i) != int_mappings.end()) {
              printf("  %" PRId64 "/", int_mappings[i][i_idx]);
            } else if (string_mappings.find(i) != string_mappings.end()) {
              printf("  %s/", string_mappings[i][i_idx].c_str());
            } else if (bool_cols.find(i) != bool_cols.end()) {
              printf("  %s/", i_idx == 0 ? "false" : "true");
            } else { // continuous
              printf("  %d/", i_idx);
            }
          }
          if (l == 0 && has_nulls[j]) {
            printf("NULL: ");
          } else {
            int j_idx = has_nulls[j] ? l - 1 : l;
            if (double_mappings.find(j) != double_mappings.end()) {
              printf("%.2f: ", double_mappings[j][j_idx]);
            } else if (int_mappings.find(j) != int_mappings.end()) {
              printf("%" PRId64 ": ", int_mappings[j][j_idx]);
            } else if (string_mappings.find(j) != string_mappings.end()) {
              printf("%s: ", string_mappings[j][j_idx].c_str());
            } else if (bool_cols.find(j) != bool_cols.end()) {
              printf("%s: ", j_idx == 0 ? "false" : "true");
            } else { // continuous
              printf("%d: ", j_idx);
            }
          }
          double smaller_z = i_z_score;
          if (std::abs(j_z_score) < std::abs(i_z_score)) {
            smaller_z = j_z_score;
          }
          printf("%.2f (z:%.4f, #:%" PRIu64 ")\n", group_avg, smaller_z, counts[idx]);
        }
      }
    }
  }
  gettimeofday(&end, 0);
  timersub(&end, &start, &diff);
  printf("2D time: %ld.%06ld\n", (long)diff.tv_sec, (long)diff.tv_usec);
}
