import pandas as pd

df = pd.read_csv("D:/Fruit_CBIR_NEW/metadata.csv")

print("Meyve türleri:")
print(df["fruit_type"].value_counts(dropna=False))

print("\nTaze/çürük etiketleri:")
print(df["label"].value_counts(dropna=False))

print("\nMeyve ve etiket dağılımı:")
print(pd.crosstab(df["fruit_type"], df["label"]))