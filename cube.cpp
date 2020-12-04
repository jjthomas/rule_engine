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
  uint8_t cur_id = 1;
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
    rev_mapping[e.second - 1] = e.first;
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

void compute_stats(PyObject *obj, int metric_idx, double z_thresh, int count_thresh,
  int show_nulls) {
  arrow::py::PyAcquireGIL lock;
  arrow::py::import_pyarrow();
  auto table = arrow::py::unwrap_table(obj).ValueOrDie();
  int64_t num_rows = table->num_rows();
  std::vector<std::vector<uint8_t>> cols;
  std::vector<uint8_t> metric_col(num_rows);
  bool metric_cat = false;
  std::pair<double, double> metric_min_max;
  std::vector<int> orig_col_idx;
  std::map<int, std::pair<double, double>> min_max;
  std::map<int, std::vector<double>> double_mappings;
  std::map<int, std::vector<int64_t>> int_mappings;
  std::map<int, std::vector<std::string>> string_mappings;
  printf("***Columns***\n");
  for (int i = 0; i < table->schema()->num_fields(); i++) {
    auto f = table->schema()->field(i);
    printf("%s: %s\n", f->name().c_str(), f->type()->ToString().c_str());
    assert(table->column(i)->num_chunks() == 1);
    int col_idx = cols.size();
    switch (f->type()->id()) {
      case arrow::Type::type::DOUBLE: {
        auto arr = std::static_pointer_cast<arrow::DoubleArray>(table->column(i)->chunk(0));
        bool is_cat = categorical(arr->raw_values(), arr->length(), arr->null_bitmap_data(),
          NUM_CARD_LIMIT);
        std::vector<uint8_t> *col_ptr = &metric_col;
        if (i != metric_idx) {
          cols.push_back(std::vector<uint8_t>(num_rows));
          col_ptr = &cols.back();
          orig_col_idx.push_back(i);
        }
        std::vector<uint8_t>& col = *col_ptr;
        if (is_cat) {
          if (i == metric_idx) {
            for (int64_t i = 0; i < arr->length(); i++) {
              if (arr->IsNull(i)) {
                printf("ERROR: null metric value at row %" PRId64 "\n", i);
                return;
              }
              if (!(arr->Value(i) >= 0 && arr->Value(i) < 256)) {
                printf("ERROR: categorical metric has value %.2f "
                  "outside range [0, 256) at row %" PRId64 "\n", arr->Value(i), i);
                return;
              }
              col[i] = (uint8_t)arr->Value(i);
            }
            metric_cat = true;
          } else {
            std::vector<double> mapping;
            enumerate_cat(arr->raw_values(), arr->length(), arr->null_bitmap_data(),
              col, mapping);
            double_mappings[col_idx] = std::move(mapping);
          }
        } else {
          std::pair<double, double> bounds = discretize_cont(arr->raw_values(), arr->length(),
            arr->null_bitmap_data(), col);
          if (i == metric_idx) {
            metric_min_max = bounds;
          } else {
            min_max[col_idx] = bounds;
          }
        }
        break;
      }
      case arrow::Type::type::INT64: {
        auto arr = std::static_pointer_cast<arrow::Int64Array>(table->column(i)->chunk(0));
        bool is_cat = categorical(arr->raw_values(), arr->length(), arr->null_bitmap_data(),
          NUM_CARD_LIMIT);
        std::vector<uint8_t> *col_ptr = &metric_col;
        if (i != metric_idx) {
          cols.push_back(std::vector<uint8_t>(num_rows));
          col_ptr = &cols.back();
          orig_col_idx.push_back(i);
        }
        std::vector<uint8_t>& col = *col_ptr;
        if (is_cat) {
          if (i == metric_idx) {
            for (int64_t i = 0; i < arr->length(); i++) {
              if (arr->IsNull(i)) {
                printf("ERROR: null metric value at row %" PRId64 "\n", i);
                return;
              }
              if (!(arr->Value(i) >= 0 && arr->Value(i) < 256)) {
                printf("ERROR: categorical metric has value %" PRId64
                  " outside range [0, 256) at row %" PRId64 "\n", arr->Value(i), i);
                return;
              }
              col[i] = (uint8_t)arr->Value(i);
            }
            metric_cat = true;
          } else {
            std::vector<int64_t> mapping;
            enumerate_cat(arr->raw_values(), arr->length(), arr->null_bitmap_data(),
              col, mapping);
            int_mappings[col_idx] = std::move(mapping);
          }
        } else {
          std::pair<double, double> bounds = discretize_cont(arr->raw_values(), arr->length(),
            arr->null_bitmap_data(), col);
          if (i == metric_idx) {
            metric_min_max = bounds;
          } else {
            min_max[col_idx] = bounds;
          }
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
          cols.push_back(std::vector<uint8_t>(num_rows));
          std::vector<uint8_t>& col = cols.back();
          orig_col_idx.push_back(i);
          std::vector<std::string> mapping;
          enumerate_cat(str_arr.data(), arr->length(), arr->null_bitmap_data(),
            col, mapping);
          string_mappings[col_idx] = std::move(mapping);
        } else {
          printf("  string cardinality too high\n");
        }
        break;
      }
      default:
        printf("  unsupported type\n");
    }
  }

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
  printf("\n%s ", table->schema()->field(metric_idx)->name().c_str());
  if (metric_cat) {
    printf("(cat) ");
  } else {
    printf("(%.2f-%.2f) ", metric_min_max.first, metric_min_max.second);
  }
  printf("global mean: %.2f, global stddev: %.2f\n", global_avg, global_dev);
  printf("\n***1D stats***\n");
  std::vector<std::vector<uint64_t>> col_sums(cols.size());
  std::vector<std::vector<uint64_t>> col_counts(cols.size());
  std::vector<std::vector<double>> col_devs(cols.size());
  int value_start_idx = show_nulls ? 0 : 1;
  for (int i = 0; i < cols.size(); i++) {
    int size;
    bool is_cat = true;
    if (double_mappings.find(i) != double_mappings.end()) {
      size = double_mappings[i].size();
    } else if (int_mappings.find(i) != int_mappings.end()) {
      size = int_mappings[i].size();
    } else if (string_mappings.find(i) != string_mappings.end()) {
      size = string_mappings[i].size();
    } else { // continuous
      size = DISC_COUNT;
      is_cat = false;
    }
    size += 1; // account for null
    printf("%s", table->schema()->field(orig_col_idx[i])->name().c_str());
    if (is_cat) {
      printf(" (cat):\n");
    } else {
      printf(" (%.2f-%.2f):\n", min_max[i].first, min_max[i].second);
    }
    std::vector<uint64_t> sums(size);
    std::vector<uint64_t> counts(size);
    std::vector<uint8_t>& col = cols[i];
    for (int64_t j = 0; j < num_rows; j++) {
      sums[col[j]] += metric_col[j];
      counts[col[j]]++;
    }
    for (int j = value_start_idx; j < size; j++) {
      if (counts[j] < count_thresh) {
        continue;
      }
      double group_avg = (double)sums[j] / counts[j];
      double effective_dev = global_dev / sqrt(counts[j]);
      double z_score = (group_avg - global_avg) / effective_dev;
      if (std::abs(z_score) > z_thresh) {
        if (j == 0) {
          printf("  NULL: ");
        } else {
          if (is_cat) {
            if (double_mappings.find(i) != double_mappings.end()) {
              printf("  %.2f: ", double_mappings[i][j - 1]);
            } else if (int_mappings.find(i) != int_mappings.end()) {
              printf("  %" PRId64 ": ", int_mappings[i][j - 1]);
            } else { // string
              printf("  %s: ", string_mappings[i][j - 1].c_str());
            }
          } else {
            printf("  %d: ", j - 1);
          }
        }
        printf("%.2f (z:%.4f, #:%" PRIu64 ")\n", group_avg, z_score, counts[j]);
      }
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
  printf("\n***2D stats***\n");
  for (int i = 0; i < cols.size(); i++) {
    for (int j = i + 1; j < cols.size(); j++) {
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
      bool header_printed = false;
      for (int k = value_start_idx; k < i_card; k++) {
        for (int l = value_start_idx; l < j_card; l++) {
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
            if (k == 0) {
              printf("  NULL/");
            } else {
              if (double_mappings.find(i) != double_mappings.end()) {
                printf("  %.2f/", double_mappings[i][k - 1]);
              } else if (int_mappings.find(i) != int_mappings.end()) {
                printf("  %" PRId64 "/", int_mappings[i][k - 1]);
              } else if (string_mappings.find(i) != string_mappings.end()) {
                printf("  %s/", string_mappings[i][k - 1].c_str());
              } else { // continuous
                printf("  %d/", k - 1);
              }
            }
            if (l == 0) {
              printf("NULL: ");
            } else {
              if (double_mappings.find(j) != double_mappings.end()) {
                printf("%.2f: ", double_mappings[j][l - 1]);
              } else if (int_mappings.find(j) != int_mappings.end()) {
                printf("%" PRId64 ": ", int_mappings[j][l - 1]);
              } else if (string_mappings.find(j) != string_mappings.end()) {
                printf("%s: ", string_mappings[j][l - 1].c_str());
              } else { // continuous
                printf("%d: ", l - 1);
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
  }
}
