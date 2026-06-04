import pandas as pd

h = pd.read_csv('data/hackaprompt_holdout_seed42.csv')
e = pd.read_csv('data/evasion_benchmark_n100.csv')

print('=== holdout ===')
print(f'Rows: {len(h)}')
print(f'Columns: {h.columns.tolist()}')
print(f'Label dist:\n{h["label"].value_counts().to_string()}')

print()
print('=== evasion benchmark ===')
print(f'Rows: {len(e)}')
print(f'Attack only (label=1): {(e["label"]==1).sum()}')
print(f'Sample:')
print(e[['text','label']].head(3).to_string())