import cube
import pandas as pd

df = pd.read_csv("Rookies.csv")
df = df.drop(columns = ['AS'])
s = cube.compute_sums(df, "IFAS")
print(cube.get_rules(s, 0.7, 10))

