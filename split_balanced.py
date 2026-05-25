import numpy as np
from sklearn.model_selection import train_test_split

# 加载数据
data = np.load('D:/IPC/result/v4/samples.npz')
X, y = data['X'], data['y']

print(f"总样本: {len(y)}, NOFALL: {np.sum(y==0)}, FALL: {np.sum(y==1)}")

# 分割比例 9:1
val_size = 0.1
test_size = 0.111111  # 验证集占训练+验证的1/9

# 分层分割，保持类别比例
X_temp, X_val, y_temp, y_val = train_test_split(
    X, y, test_size=val_size, random_state=42, stratify=y
)

print(f"\n分割后:")
print(f"验证集: {len(y_val)}, NOFALL: {np.sum(y_val==0)}, FALL: {np.sum(y_val==1)}")
print(f"训练集: {len(y_temp)}, NOFALL: {np.sum(y_temp==0)}, FALL: {np.sum(y_temp==1)}")

# 训练集均衡采样 - 让FALL和NOFALL数量接近
fall_idx = np.where(y_temp == 1)[0]
nofall_idx = np.where(y_temp == 0)[0]

print(f"\n训练集均衡化前: NOFALL={len(nofall_idx)}, FALL={len(fall_idx)}")

# 均衡策略：将多数类下采样到接近少数类的数量
target_size = min(len(fall_idx), len(nofall_idx))
np.random.seed(42)

if len(nofall_idx) > len(fall_idx):
    nofall_balanced = np.random.choice(nofall_idx, size=target_size, replace=False)
    balanced_idx = np.concatenate([fall_idx, nofall_balanced])
else:
    fall_balanced = np.random.choice(fall_idx, size=target_size, replace=False)
    balanced_idx = np.concatenate([nofall_idx, fall_balanced])

np.random.shuffle(balanced_idx)

X_train = X_temp[balanced_idx]
y_train = y_temp[balanced_idx]

print(f"训练集均衡化后: NOFALL={np.sum(y_train==0)}, FALL={np.sum(y_train==1)}")
print(f"总计: 训练集{len(y_train)}, 验证集{len(y_val)}")

# 保存
np.savez('D:/IPC/result/v4/split_balanced/train.npz', X=X_train, y=y_train)
np.savez('D:/IPC/result/v4/split_balanced/val.npz', X=X_val, y=y_val)
print("\n已保存到 D:/IPC/result/v4/split_balanced/")
