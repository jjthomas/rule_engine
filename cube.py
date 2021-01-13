import ctypes
import pyarrow as pa

c_lib = ctypes.CDLL("./libcube")

def compute_stats(df, metric_col, z_thresh, count_thresh, show_nulls=False):
  table = pa.Table.from_pandas(df)
  c_lib.compute_stats(ctypes.py_object(table), df.columns.get_loc(metric_col),
    ctypes.c_double(z_thresh), count_thresh, ctypes.c_bool(show_nulls))

