import cube
import pandas as pd

df = pd.read_csv("Rookies.csv")
cube.compute_stats(df, "IFAS", 3.0, 20)

