import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt

# ======================================
class EfficientESM2SumRegressionDataset(Dataset):
    def __init__(self, embedding_folder, label_path):
        self.embedding_files = sorted(glob.glob(f"{embedding_folder}/split_*_embeddings.npy"))
        self.file_sample_counts = [np.load(f, mmap_mode='r').shape[0] for f in self.embedding_files]
        self.file_offsets = np.cumsum([0] + self.file_sample_counts[:-1])
        self.total_samples = sum(self.file_sample_counts)

        self.labels = np.load(label_path)
        assert self.total_samples == len(self.labels), \
            f"The total number of embeddings {self.total_samples} does not match the number of labels {len(self.labels)}！"

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        for file_idx, offset in enumerate(self.file_offsets):
            if idx < offset + self.file_sample_counts[file_idx]:
                local_idx = idx - offset
                x = np.load(self.embedding_files[file_idx], mmap_mode='r')[local_idx]
                y = self.labels[idx]
                return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        raise IndexError("The index is out of range")

# =================== DNN Regression Model Definition ===================
class DNNRegressionModel(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512):
        super(DNNRegressionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1) 
        )

    def forward(self, x):
        x = x.mean(dim=1)
        return self.net(x).squeeze(-1)

# =================== Validation set evaluation function ===================
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            preds.extend(outputs.cpu().numpy())
            trues.extend(y.cpu().numpy())
    return total_loss / len(dataloader), preds, trues

# =================== Train the main function ===================
def train():
    dataset = EfficientESM2SumRegressionDataset(
        embedding_folder="split1",
        label_path="ordered_score_sum_labels.npy"
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNNRegressionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    train_losses, val_losses = [], []
    log_data = []
    best_val_loss = float('inf')
    patience_counter = 0
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 5

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        log_data.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        print(f"📘 Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_dnn_model_sum.pth")
            patience_counter = 0
            print(f"✅ The new best model has been saved（Val Loss: {val_loss:.4f}）")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("🛑 Early stopping")
                break


    pd.DataFrame(log_data).to_csv("training_log_sum.csv", index=False)
    print("📄 Log saved as training_log_sum.csv")

   
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Regression Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curves_sum.png")
    print("📈 Curve saved as training_curves_sum.png")

if __name__ == "__main__":
    train()
