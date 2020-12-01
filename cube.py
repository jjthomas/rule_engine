import ctypes
import pandas as pd
import pyarrow as pa

c_lib = ctypes.CDLL("./libcube")
df = pd.read_csv("Rookies.csv")
df = df[df["Year"] >= 1995]
table = pa.Table.from_pandas(df)
c_lib.compute_stats(ctypes.py_object(table), df.columns.get_loc("IFAS"),
  ctypes.c_double(3.0), 20)

