import cube
import pandas as pd

df = pd.read_csv("Rookies.csv")
df = df.drop(columns = ['AS'])
s = cube.compute_sums(df, "IFAS")
r = s.get_rules(0.7, 10)
print(s.evaluate_summary(df, r))

