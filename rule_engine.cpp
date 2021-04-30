#define _GLIBCXX_USE_CXX11_ABI 0
#include <Python.h>
#include <arrow/python/pyarrow.h>
#include <arrow/python/common.h>
#include <arrow/api.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <inttypes.h>
#include <sys/time.h>

extern "C" void compute2d_acc(uint8_t **, int, int, uint8_t *, uint32_t *);

// these must be <= 255
#define STRING_CARD_LIMIT 100
#define NUM_CARD_LIMIT 50
#define DISC_COUNT 15

struct Sums {
  std::vector<std::string> col_names;
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
bool categorical(const T *data, int64_t len, const uint8_t *null_map, int limit,
  std::vector<T>& rev_mapping) {
  std::unordered_set<T> s;
  for (int64_t i = 0; i < len; i++) {
    if (not_null(null_map, i)) {
      s.insert(data[i]);
      if (s.size() > limit) {
        return false;
      }
    }
  }
  for (T e : s) {
    rev_mapping.push_back(e);
  }
  return true;
}

template <typename T>
void enumerate_cat(const T *data, int64_t len, const uint8_t *null_map,
  std::vector<T>& rev_mapping, std::vector<uint8_t>& result) {
  std::unordered_map<T, uint8_t> mapping;
  for (int i = 0; i < rev_mapping.size(); i++) {
    // save 0 for null
    mapping[rev_mapping[i]] = i + 1;
  }
  for (int64_t i = 0; i < len; i++) {
    if (not_null(null_map, i)) {
      auto e = mapping.find(data[i]);
      result[i] = e == mapping.end() ? 0 : e->second;
    } else {
      result[i] = 0;
    }
  }
}

template <typename T>
std::pair<double, double> get_range_cont(const T *data, int64_t len,
  const uint8_t *null_map) {
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
  return std::make_pair((double)min, (double)max);
}

template <typename T>
void discretize_cont(const T *data, int64_t len, const uint8_t *null_map,
  std::pair<double, double> range, std::vector<uint8_t>& result) {
  double min, max;
  std::tie(min, max) = range;
  double step = (max - min) / DISC_COUNT;
  for (int64_t i = 0; i < len; i++) {
    if (not_null(null_map, i)) {
      double clamped = std::max(min, std::min<double>(data[i], max));
      result[i] = 1 + std::min<uint8_t>(DISC_COUNT - 1, (clamped - min) / step);
    } else {
      result[i] = 0;
    }
  }
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
          NUM_CARD_LIMIT, double_mappings_init[i]);
        cols_init[i] = std::vector<uint8_t>(num_rows);
        std::vector<uint8_t>& col = cols_init[i];
        if (is_cat) {
          col_status[i] = CAT;
          enumerate_cat(arr->raw_values(), arr->length(), arr->null_bitmap_data(),
            double_mappings_init[i], col);
        } else {
          col_status[i] = CONT;
          min_max_init[i] = get_range_cont(arr->raw_values(), arr->length(),
            arr->null_bitmap_data());
          discretize_cont(arr->raw_values(), arr->length(), arr->null_bitmap_data(),
            min_max_init[i], col);
        }
        break;
      }
      case arrow::Type::type::INT64: {
        auto arr = std::static_pointer_cast<arrow::Int64Array>(table->column(i)->chunk(0));
        bool is_cat = categorical(arr->raw_values(), arr->length(), arr->null_bitmap_data(),
          NUM_CARD_LIMIT, int_mappings_init[i]);
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
              col[j] = arr->Value(j);
            }
          } else {
            enumerate_cat(arr->raw_values(), arr->length(), arr->null_bitmap_data(),
              int_mappings_init[i], col);
          }
        } else {
          col_status[i] = CONT;
          min_max_init[i] = get_range_cont(arr->raw_values(), arr->length(),
            arr->null_bitmap_data());
          discretize_cont(arr->raw_values(), arr->length(), arr->null_bitmap_data(),
            min_max_init[i], col);
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
          STRING_CARD_LIMIT, string_mappings_init[i]);
        if (is_cat) {
          col_status[i] = CAT;
          cols_init[i] = std::vector<uint8_t>(num_rows);
          enumerate_cat(str_arr.data(), arr->length(), arr->null_bitmap_data(),
            string_mappings_init[i], cols_init[i]);
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
  std::vector<std::string> col_names;
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
      col_names.push_back(f->name());
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
      printf("  continuous (%.2f to %.2f)\n", min_max_init[i].first, min_max_init[i].second);
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
  sums->col_names = std::move(col_names);
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

std::vector<std::vector<uint8_t>> encode_table(Sums *s,
  std::shared_ptr<arrow::Table> table) {
  std::vector<std::vector<uint8_t>> cols(s->col_names.size());
  std::vector<int> input_idx(s->col_names.size());
  std::unordered_map<std::string, int> name_to_idx;
  auto column_names = table->ColumnNames();
  for (int i = 0; i < column_names.size(); i++) {
    name_to_idx[column_names[i]] = i;
  }
  for (int i = 0; i < s->col_names.size(); i++) {
    auto e = name_to_idx.find(s->col_names[i]);
    assert(e != name_to_idx.end());
    input_idx[i] = e->second;
  }
  int64_t num_rows = table->num_rows();
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < input_idx.size(); i++) {
    int idx = input_idx[i];
    assert(table->column(idx)->num_chunks() == 1);
    auto f = table->schema()->field(idx);
    cols[i] = std::vector<uint8_t>(num_rows);
    switch (f->type()->id()) {
      case arrow::Type::type::DOUBLE: {
        auto arr =
          std::static_pointer_cast<arrow::DoubleArray>(table->column(idx)->chunk(0));
        if (s->min_max.find(i) != s->min_max.end()) {
          discretize_cont(arr->raw_values(), arr->length(), arr->null_bitmap_data(),
            s->min_max[i], cols[i]);
        } else {
          assert(s->double_mappings.find(i) != s->double_mappings.end());
          enumerate_cat(arr->raw_values(), arr->length(), arr->null_bitmap_data(),
            s->double_mappings[i], cols[i]);
        }
        break;
      }
      case arrow::Type::type::INT64: {
        auto arr =
          std::static_pointer_cast<arrow::Int64Array>(table->column(idx)->chunk(0));
        if (s->min_max.find(i) != s->min_max.end()) {
          discretize_cont(arr->raw_values(), arr->length(), arr->null_bitmap_data(),
            s->min_max[i], cols[i]);
        } else {
          assert(s->int_mappings.find(i) != s->int_mappings.end());
          enumerate_cat(arr->raw_values(), arr->length(), arr->null_bitmap_data(),
            s->int_mappings[i], cols[i]);
        }
        break;
      }
      case arrow::Type::type::STRING: {
        auto arr =
          std::static_pointer_cast<arrow::StringArray>(table->column(idx)->chunk(0));
        assert(s->string_mappings.find(i) != s->string_mappings.end());
        std::vector<std::string> str_arr(arr->length());
        for (int64_t j = 0; j < arr->length(); j++) {
          str_arr[j] = arr->GetString(j);
        }
        enumerate_cat(str_arr.data(), arr->length(), arr->null_bitmap_data(),
          s->string_mappings[i], cols[i]);
        break;
      }
      default:
        assert(false);
    }
  }
  return cols;
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
        assert(col1val_b.Append(j).ok());
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
            assert(col1val_b.Append(f_val).ok());
            assert(col2_b.Append(s_idx).ok());
            assert(col2val_b.Append(s_val).ok());
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

void validate_rules(Sums *s, const int64_t *col1, const int64_t *col1val,
  const int64_t *col2, const int64_t *col2val, int64_t len) {
  for (int64_t i = 0; i < len; i++) {
    assert(col1[i] >= 0 && col1[i] < s->col_names.size());
    assert(col1val[i] >= 1 && col1val[i] < s->col_sums[col1[i]].size());
    assert(col2[i] == -1 ||
      (col2[i] >= 0 && col2[i] < s->col_names.size()));
    assert(col2[i] == -1 ||
      (col2val[i] >= 1 && col2val[i] < s->col_sums[col2[i]].size()));
  }
}

extern "C" PyObject *evaluate(void *sums, PyObject *table, PyObject *rules) {
  arrow::py::PyAcquireGIL lock;
  Sums *s = static_cast<Sums *>(sums);
  auto t = arrow::py::unwrap_table(table).ValueOrDie();
  auto r = arrow::py::unwrap_table(rules).ValueOrDie();
  std::vector<std::vector<uint8_t>> cols = encode_table(s, t);
  assert(r->schema()->field(0)->type()->id() == arrow::Type::type::INT64);
  auto col1 =
    std::static_pointer_cast<arrow::Int64Array>(r->column(0)->chunk(0))->raw_values();
  assert(r->schema()->field(1)->type()->id() == arrow::Type::type::INT64);
  auto col1val =
    std::static_pointer_cast<arrow::Int64Array>(r->column(1)->chunk(0))->raw_values();
  assert(r->schema()->field(2)->type()->id() == arrow::Type::type::INT64);
  auto col2 =
    std::static_pointer_cast<arrow::Int64Array>(r->column(2)->chunk(0))->raw_values();
  assert(r->schema()->field(3)->type()->id() == arrow::Type::type::INT64);
  auto col2val =
    std::static_pointer_cast<arrow::Int64Array>(r->column(3)->chunk(0))->raw_values();
  validate_rules(s, col1, col1val, col2, col2val, r->num_rows());

  std::vector<int> pred(cols[0].size());
  #pragma omp parallel for
  for (int64_t i = 0; i < r->num_rows(); i++) {
    for (int64_t j = 0; j < cols[0].size(); j++) {
      if (cols[col1[i]][j] == col1val[i] &&
          (col2[i] == -1 || cols[col2[i]][j] == col2val[i])) {
        pred[j] = 1;
      }
    }
  }

  arrow::MemoryPool *pool = arrow::default_memory_pool();
  arrow::Int64Builder pred_b(pool);
  for (int64_t i = 0; i < pred.size(); i++) {
    assert(pred_b.Append(pred[i]).ok());
  }
  std::shared_ptr<arrow::Array> pred_final;
  assert(pred_b.Finish(&pred_final).ok());
  return arrow::py::wrap_array(pred_final);
}

// rules should be sorted descending by count
extern "C" PyObject *prune_rules(void *sums, PyObject *table, PyObject *rules,
  int metric_idx, double pos_thresh, int min_count) {
  arrow::py::PyAcquireGIL lock;
  Sums *s = static_cast<Sums *>(sums);
  auto t = arrow::py::unwrap_table(table).ValueOrDie();
  auto r = arrow::py::unwrap_table(rules).ValueOrDie();
  std::vector<std::vector<uint8_t>> cols = encode_table(s, t);
  assert(r->schema()->field(0)->type()->id() == arrow::Type::type::INT64);
  auto col1 =
    std::static_pointer_cast<arrow::Int64Array>(r->column(0)->chunk(0))->raw_values();
  assert(r->schema()->field(1)->type()->id() == arrow::Type::type::INT64);
  auto col1val =
    std::static_pointer_cast<arrow::Int64Array>(r->column(1)->chunk(0))->raw_values();
  assert(r->schema()->field(2)->type()->id() == arrow::Type::type::INT64);
  auto col2 =
    std::static_pointer_cast<arrow::Int64Array>(r->column(2)->chunk(0))->raw_values();
  assert(r->schema()->field(3)->type()->id() == arrow::Type::type::INT64);
  auto col2val =
    std::static_pointer_cast<arrow::Int64Array>(r->column(3)->chunk(0))->raw_values();
  validate_rules(s, col1, col1val, col2, col2val, r->num_rows());

  assert(t->column(metric_idx)->num_chunks() == 1);
  assert(t->schema()->field(metric_idx)->type()->id() == arrow::Type::type::INT64);
  auto metric = std::static_pointer_cast<arrow::Int64Array>(
    t->column(metric_idx)->chunk(0))->raw_values();
  std::vector<int> classified(cols[0].size());
  std::vector<int64_t> chosen_rules;
  for (int64_t i = 0; i < r->num_rows(); i++) {
    int64_t pos_new = 0, neg_new = 0;
    #pragma omp parallel for
    for (int64_t j = 0; j < cols[0].size(); j++) {
      if (cols[col1[i]][j] == col1val[i] &&
          (col2[i] == -1 || cols[col2[i]][j] == col2val[i]) && classified[j] == 0) {
        if (metric[j] == 1) {
          #pragma omp atomic
          pos_new++;
        } else {
          #pragma omp atomic
          neg_new++;
        }
        classified[j] = 1;
      }
    }
    if ((double)pos_new / (pos_new + neg_new) >= pos_thresh &&
        pos_new + neg_new >= min_count) {
      chosen_rules.push_back(i);
    }
  }

  arrow::MemoryPool *pool = arrow::default_memory_pool();
  arrow::Int64Builder chosen_b(pool);
  for (int64_t i = 0; i < chosen_rules.size(); i++) {
    assert(chosen_b.Append(chosen_rules[i]).ok());
  }
  std::shared_ptr<arrow::Array> chosen_final;
  assert(chosen_b.Finish(&chosen_final).ok());
  return arrow::py::wrap_array(chosen_final);
}

extern "C" PyObject *get_col_map(void *sums) {
  arrow::py::PyAcquireGIL lock;
  Sums *s = static_cast<Sums *>(sums);
  PyObject *map = PyList_New(s->col_names.size());
  for (int i = 0; i < s->col_names.size(); i++) {
    PyObject *t = PyTuple_New(3);
    PyTuple_SetItem(t, 0, Py_BuildValue("s", s->col_names[i].c_str()));
    if (s->min_max.find(i) != s->min_max.end()) {
      PyTuple_SetItem(t, 1, Py_BuildValue("s", "c"));
      PyTuple_SetItem(t, 2, Py_BuildValue("(dd)", s->min_max[i].first,
        s->min_max[i].second));
    } else if (s->int_mappings.find(i) != s->int_mappings.end()) {
      PyTuple_SetItem(t, 1, Py_BuildValue("s", "i"));
      PyObject *vals = PyList_New(s->int_mappings[i].size());
      for (int j = 0; j < s->int_mappings[i].size(); j++) {
        PyList_SetItem(vals, j, Py_BuildValue("L", s->int_mappings[i][j]));
      }
      PyTuple_SetItem(t, 2, vals);
    } else if (s->double_mappings.find(i) != s->double_mappings.end()) {
      PyTuple_SetItem(t, 1, Py_BuildValue("s", "d"));
      PyObject *vals = PyList_New(s->double_mappings[i].size());
      for (int j = 0; j < s->double_mappings[i].size(); j++) {
        PyList_SetItem(vals, j, Py_BuildValue("d", s->double_mappings[i][j]));
      }
      PyTuple_SetItem(t, 2, vals);
    } else { // string
      PyTuple_SetItem(t, 1, Py_BuildValue("s", "s"));
      PyObject *vals = PyList_New(s->string_mappings[i].size());
      for (int j = 0; j < s->string_mappings[i].size(); j++) {
        PyList_SetItem(vals, j, Py_BuildValue("s",
          s->string_mappings[i][j].c_str()));
      }
      PyTuple_SetItem(t, 2, vals);
    }
    PyList_SetItem(map, i, t);
  }

  return map;
}
