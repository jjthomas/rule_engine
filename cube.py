import ctypes
import pandas as pd
import pyarrow as pa

c_lib = ctypes.CDLL("./libcube")
df = pd.DataFrame({"a": [1, 2, 3]})
table = pa.Table.from_pandas(df)
c_lib.print_is_array(ctypes.py_object(table))

