import pandas as pd

files = [
    "/Users/apple/Documents/college 4th sem el /RAKSHAK-ICS/data/swat/dataset1.csv",
    "/Users/apple/Documents/college 4th sem el /RAKSHAK-ICS/data/swat/dataset2.csv",
    "/Users/apple/Documents/college 4th sem el /RAKSHAK-ICS/data/swat/dataset3.csv"
]

for file in files:

    print("\n" + "="*70)
    print(f"FILE: {file}")
    print("="*70)

    try:
        df = pd.read_csv(file, low_memory=False)

        print("Rows:", len(df))
        print("Columns:", len(df.columns))

    except Exception as e:
        print("Error:", e)