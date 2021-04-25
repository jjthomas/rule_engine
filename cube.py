import ctypes
import pyarrow as pa

c_lib = ctypes.CDLL("./libcube")
c_lib.compute_sums.argtypes = [ctypes.py_object, ctypes.c_int]
c_lib.compute_sums.restype = ctypes.c_void_p
c_lib.free_sums.argtypes = [ctypes.c_void_p]
c_lib.get_rules.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int]
c_lib.get_rules.restype = ctypes.py_object

class Sums:
  def __init__(self, ptr):
    self.ptr = ptr

  def __del__(self):
    c_lib.free_sums(self.ptr)

def compute_sums(df, metric_col):
  table = pa.Table.from_pandas(df)
  return Sums(c_lib.compute_sums(table, df.columns.get_loc(metric_col)))

def get_rules(sums, pos_thresh, min_count):
  return c_lib.get_rules(sums.ptr, pos_thresh, min_count).to_pandas()

