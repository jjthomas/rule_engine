import pandas as pd
import rule_engine as re

df = pd.read_parquet("train.parquet")
df["resp_bin"] = (df["resp"] > 0.15).astype(int)
df = df.drop(columns=['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4'])
s = re.compute_sums(df, "resp_bin")
print(len(s.get_rules(0.9, 10)))
