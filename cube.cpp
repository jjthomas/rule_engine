#define _GLIBCXX_USE_CXX11_ABI 0
#include <Python.h>
#include <arrow/python/pyarrow.h>
#include <arrow/python/common.h>
#include <arrow/api.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <map>
#include <inttypes.h>
#include <sys/time.h>

extern "C" void compute2d_acc(uint8_t **, int, int, uint8_t *, uint32_t *);

#define STRING_CARD_LIMIT 100
#define NUM_CARD_LIMIT 50
#define DISC_COUNT 15

struct Sums {
  std::vector<int> orig_col_idx;
  std::map<int, std::pair<double, double>> min_max;
  std::map<int, std::vector<double>> double_mappings;
  std::map<int, std::vector<int64_t>> int_mappings;
  std::map<int, std::vector<std::string>> string_mappings;
  std::vector<std::vector<uint64_t>> col_sums;
  std::vector<std::vector<uint64_t>> col_counts;
  std::vector<std::pair<int, int>> split_mapping;
  std::vector<uint64_t> pair_sums;
  std::vector<uint64_t> pair_counts;
};

inline bool not_null(const uint8_t *null_map, int64_t pos) {
  return null_map == nullptr || ((null_map[pos >> 3] >> (pos & 7)) & 1);
}

template <typename T>
bool categorical(const T *data, int64_t len, const uint8_t *null_map, int limit) {
  std::set<T> s;
  for (int64_t i = 0; i < len; i++) {
    if (not_null(null_map, i)) {
      s.insert(data[i]);
    }
    if (s.size() > limit) {
      return false;
    }
  }
  return true;
}

template <typename T>
void enumerate_cat(const T *data, int64_t len, const uint8_t *null_map,
  std::vector<uint8_t>& result, std::vector<T> &rev_mapping) {
  // save 0 id for null
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

std::pair<std::vector<uint64_t>, std::vector<uint64_t>>
run_acc(std::vector<std::vector<uint8_t>> cols, std::vector<uint8_t> metric) {
  std::vector<uint64_t> sums(256 * cols.size() * (cols.size() - 1) / 2);
  std::vector<uint64_t> counts(sums.size());
  std::vector<uint32_t> acc_stats(2 * sums.size());
  int chunk_size = 1 << 24; // avoid overflow in acc 32-bit metric sums
  std::vector<uint8_t *> col_ptrs(cols.size());
  for (int64_t i = 0; i < cols[0].size(); i += chunk_size) {
    for (int j = 0; j < cols.size(); j++) {
      col_ptrs[j] = cols[j].data() + i;
    }
    compute2d_acc(col_ptrs.data(), std::min<int64_t>(chunk_size, cols[0].size() - i),
      cols.size(), metric.data() + i, acc_stats.data());
    for (int j = 0; j < sums.size(); j++) {
      sums[j] += acc_stats[2 * j];
      counts[j] += acc_stats[2 * j + 1];
    }
  }
  return std::make_pair(std::move(sums), std::move(counts));
}

// split cols into 4-bit cols for accelerator
std::pair<std::vector<std::vector<uint8_t>>, std::vector<std::pair<int, int>>>
encode_acc(std::vector<std::vector<uint8_t>> cols, std::vector<int>& sizes) {
  std::vector<std::pair<int, int>> mapping;
  for (int i = 0; i < sizes.size(); i++) {
    if (sizes[i] > 16) {
      for (int j = 0; j < (sizes[i] + 14) / 15; j++) {
        mapping.emplace_back(i, j);
      }
    } else {
      mapping.emplace_back(i, -1);
    }
  }
  std::vector<std::vector<uint8_t>> new_cols(mapping.size());
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < new_cols.size(); i++) {
    int orig_idx, chunk;
    std::tie(orig_idx, chunk) = mapping[i];
    std::vector<uint8_t>& cur_col = cols[orig_idx];
    if (chunk >= 0) {
      std::vector<uint8_t> new_col(cur_col.size());
      for (int64_t j = 0; j < cur_col.size(); j++) {
        if (cur_col[j] / 15 == chunk) {
          // 0 reserved for not in range
          new_col[j] = cur_col[j] % 15 + 1;
        }
      }
      new_cols[i] = std::move(new_col);
    } else {
      new_cols[i] = std::move(cur_col);
    }
  }
  return std::make_pair(std::move(new_cols), std::move(mapping));
}

enum ColStatus {
  CONT, CAT, BAD_TYPE, STRING_HIGH_CARD, METRIC_ERROR
};

extern "C" void *compute_sums(PyObject *obj, int metric_idx) {
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
          enumerate_cat(arr->raw_values(), arr->length(), arr->null_bitmap_data(),
            col, double_mappings_init[i]);
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
              if (!(arr->Value(j) >= 0 && arr->Value(j) < 2)) {
                printf("ERROR: categorical metric has value %" PRId64
                  " outside range [0, 2) at row %" PRId64 "\n", arr->Value(j), j);
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
      default:
        col_status[i] = BAD_TYPE;
    }
  }

  if (col_status[metric_idx] == METRIC_ERROR) {
    return nullptr;
  }
  std::vector<uint8_t> metric_col = std::move(cols_init[metric_idx]);
  std::pair<double, double> metric_min_max = min_max_init[metric_idx];

  std::vector<std::vector<uint8_t>> cols;
  std::vector<int> sizes;
  std::vector<int> orig_col_idx;
  std::map<int, std::pair<double, double>> min_max;
  std::map<int, std::vector<double>> double_mappings;
  std::map<int, std::vector<int64_t>> int_mappings;
  std::map<int, std::vector<std::string>> string_mappings;
  for (int i = 0; i < table->schema()->num_fields(); i++) {
    auto f = table->schema()->field(i);
    printf("%s: %s\n", f->name().c_str(), f->type()->ToString().c_str());
    if (col_status[i] == BAD_TYPE) {
      printf("  WARN: unsupported type\n");
    } else if (col_status[i] == STRING_HIGH_CARD) {
      printf("  WARN: string cardinality too high to use\n");
    }
    if (i == metric_idx) {
      if (f->type()->id() != arrow::Type::type::INT64) {
        printf("ERROR: metric must be binary INT64\n");
        return nullptr;
      }
    } else if (col_status[i] == CONT || col_status[i] == CAT) {
      int cur_col = cols.size();
      int size = 1; // always has null
      if (col_status[i] == CONT) {
        min_max[cur_col] = min_max_init[i];
        size += DISC_COUNT;
      } else {
        if (f->type()->id() == arrow::Type::type::DOUBLE) {
          double_mappings[cur_col] = std::move(double_mappings_init[i]);
          size += double_mappings[cur_col].size();
        } else if (f->type()->id() == arrow::Type::type::INT64) {
          int_mappings[cur_col] = std::move(int_mappings_init[i]);
          size += int_mappings[cur_col].size();
        } else { // string
          string_mappings[cur_col] = std::move(string_mappings_init[i]);
          size += string_mappings[cur_col].size();
        }
      }
      orig_col_idx.push_back(i);
      cols.push_back(std::move(cols_init[i]));
      sizes.push_back(size);
    }
    if (col_status[i] == CAT) {
      printf("  categorical");
      if (i != metric_idx) {
        printf(" (%d)", sizes.back());
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
  printf("\n%s global mean: %.2f\n\n",
    table->schema()->field(metric_idx)->name().c_str(), global_avg);

  std::vector<std::vector<uint64_t>> col_sums(cols.size());
  std::vector<std::vector<uint64_t>> col_counts(cols.size());
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
    col_sums[i] = std::move(sums);
    col_counts[i] = std::move(counts);
  }
  gettimeofday(&end, 0);
  timersub(&end, &start, &diff);
  printf("1D time: %ld.%06ld\n", (long)diff.tv_sec, (long)diff.tv_usec);

  gettimeofday(&start, 0);
  std::vector<std::vector<uint8_t>> new_cols;
  std::vector<std::pair<int, int>> new_mapping;
  std::tie(new_cols, new_mapping) = encode_acc(std::move(cols), sizes);
  std::vector<uint64_t> acc_sums, acc_counts;
  std::tie(acc_sums, acc_counts) = run_acc(std::move(new_cols), std::move(metric_col));
  gettimeofday(&end, 0);
  timersub(&end, &start, &diff);
  printf("2D time: %ld.%06ld\n", (long)diff.tv_sec, (long)diff.tv_usec);

  Sums *sums = new Sums;
  sums->orig_col_idx = std::move(orig_col_idx);
  sums->min_max = std::move(min_max);
  sums->double_mappings = std::move(double_mappings);
  sums->int_mappings = std::move(int_mappings);
  sums->string_mappings = std::move(string_mappings);
  sums->col_sums = std::move(col_sums);
  sums->col_counts = std::move(col_counts);
  sums->split_mapping = std::move(new_mapping);
  sums->pair_sums = std::move(acc_sums);
  sums->pair_counts = std::move(acc_counts);

  return sums;
}

extern "C" void free_sums(void *sums) {
  Sums *s = static_cast<Sums *>(sums);
  delete s;
}

extern "C" PyObject *get_rules(void *sums, double pos_thresh, int min_count) {
  arrow::py::PyAcquireGIL lock;
  Sums *s = static_cast<Sums *>(sums);
  arrow::MemoryPool *pool = arrow::default_memory_pool();
  arrow::Int64Builder col1_b(pool), col1val_b(pool), col2_b(pool),
    col2val_b(pool), cnt_b(pool);
  arrow::DoubleBuilder pos_frac_b(pool);

  for (int i = 0; i < s->col_sums.size(); i++) {
    std::vector<uint64_t>& sum = s->col_sums[i];
    std::vector<uint64_t>& count = s->col_counts[i];
    for (int j = 1; j < sum.size(); j++) { // skip null value
      if (count[j] >= min_count && (double)sum[j] / count[j] >= pos_thresh) {
        assert(col1_b.Append(i).ok());
        assert(col1val_b.Append(j - 1).ok());
        assert(col2_b.Append(-1).ok());
        assert(col2val_b.Append(-1).ok());
        assert(cnt_b.Append(count[j]).ok());
        assert(pos_frac_b.Append((double)sum[j] / count[j]).ok());
      }
    }
  }

  int64_t idx = 0;
  for (int i = 0; i < s->split_mapping.size(); i++) {
    for (int j = i + 1; j < s->split_mapping.size(); j++) {
      for (int64_t k = idx; k < idx + 256; k++) {
        if (s->pair_counts[k] >= min_count &&
            (double)s->pair_sums[k] / s->pair_counts[k] >= pos_thresh) {
          int f_val = (k - idx) >> 4;
          int s_val = (k - idx) & 15;
          if (f_val == 0 || s_val == 0) { // either a null or an out of range
            continue;
          }
          int f_idx, f_chunk, s_idx, s_chunk;
          std::tie(f_idx, f_chunk) = s->split_mapping[i];
          std::tie(s_idx, s_chunk) = s->split_mapping[j];
          f_val = f_chunk == -1 ? f_val : 15 * f_chunk + f_val - 1;
          s_val = s_chunk == -1 ? s_val : 15 * s_chunk + s_val - 1;
          if (f_val == 0 || s_val == 0) { // split col null
            continue;
          }
          if ((double)s->col_sums[f_idx][f_val] / s->col_counts[f_idx][f_val] < pos_thresh &&
              (double)s->col_sums[s_idx][s_val] / s->col_counts[s_idx][s_val] < pos_thresh) {
            // both parents below thresh
            assert(col1_b.Append(f_idx).ok());
            assert(col1val_b.Append(f_val - 1).ok());
            assert(col2_b.Append(s_idx).ok());
            assert(col2val_b.Append(s_val - 1).ok());
            assert(cnt_b.Append(s->pair_counts[k]).ok());
            assert(pos_frac_b.Append((double)s->pair_sums[k] / s->pair_counts[k]).ok());
          }
        }
      }
      idx += 256;
    }
  }

  std::shared_ptr<arrow::Array> col1, col1val, col2, col2val, cnt, pos_frac;
  assert(col1_b.Finish(&col1).ok());
  assert(col1val_b.Finish(&col1val).ok());
  assert(col2_b.Finish(&col2).ok());
  assert(col2val_b.Finish(&col2val).ok());
  assert(cnt_b.Finish(&cnt).ok());
  assert(pos_frac_b.Finish(&pos_frac).ok());

  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
    arrow::field("col1", arrow::int64()), arrow::field("col1val", arrow::int64()),
    arrow::field("col2", arrow::int64()), arrow::field("col2val", arrow::int64()),
    arrow::field("count", arrow::int64()), arrow::field("pos_frac", arrow::float64())
  };

  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  auto table = arrow::Table::Make(schema, {col1, col1val, col2, col2val, cnt, pos_frac});
  return arrow::py::wrap_table(table);
}
