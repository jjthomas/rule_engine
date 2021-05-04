import ctypes
import pyarrow as pa

c_lib = ctypes.CDLL("librule")
c_lib.compute_sums.argtypes = [ctypes.py_object, ctypes.c_int]
c_lib.compute_sums.restype = ctypes.c_void_p
c_lib.get_col_map.argtypes = [ctypes.c_void_p]
c_lib.get_col_map.restype = ctypes.py_object
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
    self.map = c_lib.get_col_map(self.ptr)

  def __del__(self):
    c_lib.free_sums(self.ptr)

  def get_rules(self, pos_thresh, min_count):
    r = c_lib.get_rules(self.ptr, pos_thresh, min_count).to_pandas()
    return r.sort_values(by='count', ascending=False).reset_index(drop=True)

  def prune_rules(self, df, rules, pos_thresh, min_count):
    t = pa.Table.from_pandas(df, preserve_index=False)
    rules = rules.sort_values(by='count', ascending=False)
    r = pa.Table.from_pandas(rules, preserve_index=False)
    idxs = c_lib.prune_rules(self.ptr, t, r,
      df.columns.get_loc(self.metric_col), pos_thresh, min_count).to_pandas()
    return rules.iloc[idxs].reset_index(drop=True)

  def display_rules(self, rules):
    def get_str(col, colval):
      colval -= 1 # correct for null at val 0
      res = "{0} = ".format(self.map[col][0])
      if self.map[col][1] == "c":
        l, h = self.map[col][2]
        step = (h - l) / 15
        res += "{0:.2f} to {1:.2f}".format(l + colval * step,
          l + (colval + 1) * step)
      elif self.map[col][1] == "d":
        res += "{0:.2f}".format(self.map[col][2][colval])
      else: # int or string cat
        res += str(self.map[col][2][colval])
      return res

    for row in rules.itertuples():
      res = get_str(row.col1, row.col1val)
      if row.col2 != -1:
        res += ", " + get_str(row.col2, row.col2val)
      res += " ({0} samples, {1:.2f} precision)".format(row.count, row.pos_frac)
      print(res)

  def evaluate(self, df, rules):
    t = pa.Table.from_pandas(df, preserve_index=False)
    r = pa.Table.from_pandas(rules, preserve_index=False)
    return c_lib.evaluate(self.ptr, t, r).to_pandas()

  def evaluate_summary(self, df, rules):
    preds = self.evaluate(df, rules)
    total_pred = preds.sum()
    true_labels = df[self.metric_col].reset_index(drop=True)
    pred_correct = (true_labels & preds).sum()
    total_true = true_labels.sum()
    # precision, recall
    return (pred_correct / total_pred, pred_correct / total_true)

def compute_sums(df, metric_col):
  table = pa.Table.from_pandas(df, preserve_index=False)
  return Sums(c_lib.compute_sums(table, df.columns.get_loc(metric_col)),
              metric_col)


