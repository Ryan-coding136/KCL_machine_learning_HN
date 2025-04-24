import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
X_seq = np.load('X_esm2_embeddings.npy')
X_struct = np.load('X_structural_features.npy')
X_host = np.load('X_host_label.npy')
y = np.load('y_binding_score.npy')

# 合并特征
X = np.concatenate([X_seq, X_struct, X_host.reshape(-1, 1)], axis=1)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 转换为 tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# 定义模型
class SimpleDNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleDNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.model(x)

model = SimpleDNN(X.shape[1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加权损失函数
class_counts = np.bincount(y)
weights = 1. / class_counts
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor.to(device))
    loss = criterion(outputs, y_train_tensor.to(device))
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # 验证
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor.to(device))
        val_loss = criterion(val_outputs, y_val_tensor.to(device))
        val_losses.append(val_loss.item())

        _, predicted = torch.max(val_outputs, 1)
        val_acc = (predicted == y_val_tensor.to(device)).float().mean().item()
        val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}, Val Acc: {val_acc:.4f}")

# 可视化 loss
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title("Loss Curve")
plt.legend()
plt.savefig("loss_curve_weighted.png")
plt.close()

# 混淆矩阵
model.eval()
with torch.no_grad():
    val_pred = model(X_val_tensor.to(device))
    y_pred = torch.argmax(val_pred, dim=1).cpu().numpy()
    cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix_weighted.png")
plt.close()

# ROC curve
y_score = torch.softmax(val_pred, dim=1).cpu().numpy()
y_true_bin = label_binarize(y_val, classes=[0,1,2])

plt.figure()
for i in range(3):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (One-vs-Rest)")
plt.legend()
plt.savefig("roc_curve_weighted.png")
plt.close()
