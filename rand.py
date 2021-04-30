# python rand.py <num cols> <upper bound of column values, non-inclusive>
import rule_engine as re
import pandas as pd
import numpy as np
import sys

df = pd.DataFrame()
np.random.seed(0)
df["c0"] = np.random.randint(0, 2, 1 << 22)
for i in range(1, int(sys.argv[1])):
  df["c%d" % i] = np.random.randint(0, int(sys.argv[2]), 1 << 22)
s = re.compute_sums(df, "c0")
print(s.get_rules(0.50, 10))

