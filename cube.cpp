#include <Python.h>
#include <arrow/python/pyarrow.h>
#include <arrow/python/common.h>
#include <arrow/api.h>
#include <algorithm>
#include <cmath>
#include <set>
#include <map>

extern "C" void compute_stats(PyObject *, int);

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
  uint8_t *result, std::vector<T> &rev_mapping) {
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
  const uint8_t *null_map, uint8_t *result) {
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

void compute_stats(PyObject *obj, int metric_idx) {
  arrow::py::PyAcquireGIL lock;
  arrow::py::import_pyarrow();
  auto table = arrow::py::unwrap_table(obj).ValueOrDie();
  int64_t num_rows = table->num_rows();
  std::vector<uint8_t *> cols;
  uint8_t *metric_col = new uint8_t[num_rows];
  std::vector<int> orig_col_idx;
  std::map<int, std::pair<double, double>> min_max;
  std::map<int, std::vector<double>> double_mappings;
  std::map<int, std::vector<int64_t>> int_mappings;
  std::map<int, std::vector<std::string>> string_mappings;
  for (int i = 0; i < table->schema()->num_fields(); i++) {
    auto f = table->schema()->field(i);
    printf("%s: %s\n", f->name().c_str(), f->type()->ToString().c_str());
    assert(table->column(i)->num_chunks() == 1);
    int col_idx = cols.size();
    switch (f->type()->id()) {
      case arrow::Type::type::DOUBLE: {
        auto arr = std::static_pointer_cast<arrow::DoubleArray>(table->column(i)->chunk(0));
        auto is_cat = categorical(arr->raw_values(), arr->length(), arr->null_bitmap_data(),
          NUM_CARD_LIMIT);
	uint8_t *col;
        if (i == metric_idx) {
          col = metric_col;
        } else {
          col = new uint8_t[num_rows];
          cols.push_back(col);
          orig_col_idx.push_back(i);
        }
        if (is_cat) {
          if (i == metric_idx) {
            for (int64_t i = 0; i < arr->length(); i++) {
              assert(!arr->IsNull(i));
              assert(arr->Value(i) >= 0 && arr->Value(i) < 256);
              col[i] = (uint8_t)arr->Value(i);
            }
          } else {
            std::vector<double> mapping;
            enumerate_cat(arr->raw_values(), arr->length(), arr->null_bitmap_data(),
              col, mapping);
            double_mappings[col_idx] = std::move(mapping);
          }
        } else {
          min_max[col_idx] = discretize_cont(arr->raw_values(), arr->length(),
            arr->null_bitmap_data(), col);
        }
	break;
      }
      case arrow::Type::type::INT64: {
        auto arr = std::static_pointer_cast<arrow::Int64Array>(table->column(i)->chunk(0));
        auto is_cat = categorical(arr->raw_values(), arr->length(), arr->null_bitmap_data(),
          NUM_CARD_LIMIT);
	uint8_t *col;
        if (i == metric_idx) {
          col = metric_col;
        } else {
          col = new uint8_t[num_rows];
          cols.push_back(col);
          orig_col_idx.push_back(i);
        }
        if (is_cat) {
          if (i == metric_idx) {
            for (int64_t i = 0; i < arr->length(); i++) {
              assert(!arr->IsNull(i));
              assert(arr->Value(i) >= 0 && arr->Value(i) < 256);
              col[i] = (uint8_t)arr->Value(i);
            }
          } else {
            std::vector<int64_t> mapping;
            enumerate_cat(arr->raw_values(), arr->length(), arr->null_bitmap_data(),
              col, mapping);
            int_mappings[col_idx] = std::move(mapping);
          }
        } else {
          min_max[col_idx] = discretize_cont(arr->raw_values(), arr->length(),
            arr->null_bitmap_data(), col);
        }
	break;
      }
      case arrow::Type::type::STRING: {
        auto arr = std::static_pointer_cast<arrow::StringArray>(table->column(i)->chunk(0));
	std::string *str_arr = new std::string[arr->length()];
	for (int64_t j = 0; j < arr->length(); j++) {
          str_arr[j] = arr->GetString(j);
        }
        auto is_cat = categorical(str_arr, arr->length(), arr->null_bitmap_data(),
          STRING_CARD_LIMIT);
        if (is_cat) {
          uint8_t *col = new uint8_t[num_rows];
          cols.push_back(col);
          orig_col_idx.push_back(i);
          std::vector<std::string> mapping;
          enumerate_cat(str_arr, arr->length(), arr->null_bitmap_data(),
            col, mapping);
          string_mappings[col_idx] = std::move(mapping);
        }
	delete [] str_arr;
	break;
      }
      default:
        printf("  unsupported type for card\n");
    }
  }

  uint32_t global_sum;
  for (int64_t i = 0; i < num_rows; i++) {
    global_sum += metric_col[i];
  }
  double global_avg = (double)global_sum / num_rows;
  double global_dev = 0;
  for (int64_t i = 0; i < num_rows; i++) {
    global_dev += pow(metric_col[i] - global_avg, 2);
  }
  global_dev = sqrt(global_dev / num_rows);
  printf("%s global mean: %.2f, global stddev: %.2f\n",
    table->schema()->field(metric_idx)->name().c_str(), global_avg, global_dev);
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
    uint32_t *sums = (uint32_t *)std::calloc(size, sizeof(uint32_t));
    uint32_t *counts = (uint32_t *)std::calloc(size, sizeof(uint32_t));
    uint8_t *col = cols[i];
    for (int64_t j = 0; j < num_rows; j++) {
      sums[col[j]] += metric_col[j];
      counts[col[j]]++;
    }
    for (int j = 0; j < size; j++) {
      double group_avg = (double)sums[j] / counts[j];
      double effective_dev = global_dev / sqrt(counts[j]);
      double z_score = (group_avg - global_avg) / effective_dev;
      if (abs(z_score) > 2) { // two std devs away
        if (j == 0) {
          printf("  NULL: ");
        } else {
          if (is_cat) {
            if (double_mappings.find(i) != double_mappings.end()) {
              printf("  %.2f: ", double_mappings[i][j - 1]);
            } else if (int_mappings.find(i) != int_mappings.end()) {
              printf("  %lld: ", int_mappings[i][j - 1]);
            } else { // string
              printf("  %s: ", string_mappings[i][j - 1].c_str());
            }
          } else {
            printf("  %d: ", j - 1);
          }
	}
	printf("%.2f (z:%.4f, #:%d)\n", group_avg, z_score, counts[j]);
      }
    }
    std::free(sums);
    std::free(counts);
  }
}
