import numpy as np
import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')

data = np.load('D:/IPC/result/v5/results/samples.npz', allow_pickle=True)
X = data['X'][:20]
y = data['y'][:20]
print(f'X: {X.shape}, y: {y.shape}')

rows = []
for i in range(20):
    for j in range(X.shape[1]):
        row = [y[i], i, j] + list(X[i, j])
        rows.append(row)

cols = ['label', 'sample_idx', 'frame_idx'] + [f'feat_{f}' for f in range(139)]
df = pd.DataFrame(rows, columns=cols)
print(f'DataFrame: {df.shape}')

out = 'D:/IPC/result/v5/results/samples_first20.xlsx'
df.to_excel(out, index=False)
print(f'Saved: {os.path.getsize(out)} bytes')