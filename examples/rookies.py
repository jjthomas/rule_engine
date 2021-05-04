import rule_engine as re
import pandas as pd

df = pd.read_csv("Rookies.csv")
# Compute # of anomalies and # of total examples classified by each
# possible one- and two-feature decision rule; IFAS is the class variable.
s = re.compute_sums(df, "IFAS")
# Get rules with precision >= 0.7 and total examples classified >= 10.
r = s.get_rules(0.7, 10)
# Print precision, recall on the training set.
print(s.evaluate_summary(df, r))
# Prune rules that classify fewer than 1 new example over prior rules
# or have lower than 0.01 precision on their new classifications.
r = s.prune_rules(df, r, 0.01, 1)
print(s.evaluate_summary(df, r))
s.display_rules(r)

