import cube
import pandas as pd

df = pd.read_csv("Rookies.csv")
s = cube.compute_sums(df, "IFAS")
r = s.get_rules(0.7, 10)
print(s.evaluate_summary(df, r))
r = s.prune_rules(df, r, 0.01, 1)
print(s.evaluate_summary(df, r))
s.display_rules(r)

