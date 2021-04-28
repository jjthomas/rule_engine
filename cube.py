import ctypes
import pyarrow as pa

c_lib = ctypes.CDLL("./libcube")
c_lib.compute_sums.argtypes = [ctypes.py_object, ctypes.c_int]
c_lib.compute_sums.restype = ctypes.c_void_p
c_lib.free_sums.argtypes = [ctypes.c_void_p]
c_lib.get_rules.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int]
c_lib.get_rules.restype = ctypes.py_object
c_lib.prune_rules.argtypes = [ctypes.c_void_p, ctypes.py_object,
  ctypes.py_object, ctypes.c_int, ctypes.c_double, ctypes.c_int]
c_lib.prune_rules.restype = ctypes.py_object
c_lib.evaluate.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.py_object]
c_lib.evaluate.restype = ctypes.py_object

class Sums:
  def __init__(self, ptr, metric_col):
    self.ptr = ptr
    self.metric_col = metric_col

  def __del__(self):
    c_lib.free_sums(self.ptr)

  def get_rules(self, pos_thresh, min_count):
    return c_lib.get_rules(self.ptr, pos_thresh, min_count).to_pandas()

  def prune_rules(self, df, rules, pos_thresh, min_pos_count):
    t = pa.Table.from_pandas(df)
    rules = rules.sort_values(by='count', ascending=False)
    r = pa.Table.from_pandas(rules)
    idxs = c_lib.prune_rules(self.ptr, t, r,
      df.columns.get_loc(self.metric_col), pos_thresh, min_pos_count).to_pandas()
    return rules.iloc[idxs].reset_index(drop=True)

  def evaluate(self, df, rules):
    t = pa.Table.from_pandas(df)
    r = pa.Table.from_pandas(rules)
    return c_lib.evaluate(self.ptr, t, r).to_pandas()

  def evaluate_summary(self, df, rules):
    preds = self.evaluate(df, rules)
    total_pred = preds.sum()
    pred_correct = (df[self.metric_col] & preds).sum()
    total_true = df[self.metric_col].sum()
    # precision, recall
    return (pred_correct / total_pred, pred_correct / total_true)

def compute_sums(df, metric_col):
  table = pa.Table.from_pandas(df)
  return Sums(c_lib.compute_sums(table, df.columns.get_loc(metric_col)),
              metric_col)


