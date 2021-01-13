# python rand.py <num cols> <upper bound of column values, non-inclusive>
import cube
import pandas as pd
import numpy as np
import sys

df = pd.DataFrame()
np.random.seed(0)
for i in range(int(sys.argv[1])):
  df["c%d" % i] = np.random.randint(0, int(sys.argv[2]), 1 << 22)
cube.compute_stats(df, "c0", 2.0, 20)

