import pandas as pd

df = pd.read_csv(
    "/Users/apple/Documents/college 4th sem el /RAKSHAK-ICS/data/swat/dataset1.csv",
    low_memory=False
)

numeric_cols = df.select_dtypes(include=['int64','float64']).columns

print("Numeric Features:", len(numeric_cols))
print(numeric_cols.tolist())