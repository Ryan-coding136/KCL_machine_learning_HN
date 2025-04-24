
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
X_esm = np.load("X_esm2_embeddings.npy")
X_struct = np.load("X_structural_features.npy")
X_host = np.load("X_host_label.npy")
y = np.load("y_binding_score.npy")

# 合并所有输入特征
X = np.concatenate([X_esm, X_struct, X_host.reshape(-1, 1)], axis=1)
num_classes = len(np.unique(y))

# 数据划分
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 转为 tensor
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.long).to(device)

# 类别权重
class_sample_count = np.array([np.sum(y == t) for t in np.unique(y)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in y])
class_weights = torch.tensor(weight, dtype=torch.float32).to(device)

# 模型定义
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SimpleMLP, self).__init__()
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

model = SimpleMLP(X.shape[1], [128, 64, 32], num_classes).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train_losses, val_losses, val_accuracies = [], [], []
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # 验证
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()
        val_losses.append(val_loss)
        preds = torch.argmax(val_outputs, dim=1)
        acc = (preds == y_val).float().mean().item()
        val_accuracies.append(acc)
    print(f"Epoch {epoch+1}, Val Acc: {acc:.4f}")

# 保存模型
torch.save(model.state_dict(), "best_model_weighted.pth")

# 绘图：损失
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title("Loss Curve")
plt.savefig("loss_curve_weighted.png")

# 混淆矩阵
conf_matrix = confusion_matrix(y_val.cpu(), preds.cpu())
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_weighted.png")

# ROC 曲线
y_true_bin = label_binarize(y_val.cpu().numpy(), classes=[0, 1, 2])
y_score = val_outputs.softmax(dim=1).detach().cpu().numpy()
plt.figure()
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    plt.plot(fpr, tpr, label=f"Class {i} (AUC={auc(fpr, tpr):.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("ROC Curve")
plt.savefig("roc_curve_weighted.png")
