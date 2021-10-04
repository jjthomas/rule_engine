# python rand.py <num cols> <num trials> <optional: power distribution parameter>
import rule_engine as re
import pandas as pd
import numpy as np
import sys

df = pd.DataFrame()
np.random.seed(0)
df["c0"] = np.random.randint(0, 2, 1 << 22)
use_power = len(sys.argv) > 3
if use_power:
  a = int(sys.argv[3])
for i in range(1, int(sys.argv[1])):
  if use_power:
    data = np.random.power(a, 1 << 22)
    data[0] = 0 # ensure range is 0.0 to 1.0
  else:
    data = np.random.rand(1 << 22)
  df["c%d" % i] = data
for i in range(int(sys.argv[2])):
  s = re.compute_sums(df, "c0")
print(s.get_rules(0.50, 10))

