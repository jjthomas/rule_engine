#include <Python.h>
#include <arrow/python/pyarrow.h>
#include <arrow/python/common.h>
#include <arrow/api.h>

extern "C" void print_is_array(PyObject *);

void print_is_array(PyObject *obj) {
  arrow::py::PyAcquireGIL lock;
  arrow::py::import_pyarrow();
  printf("is_table: %d\n", arrow::py::is_table(obj));
  std::shared_ptr<arrow::Table> table;
  arrow::py::unwrap_table(obj, &table);
}

int main() {
  print_is_array(NULL);
}
