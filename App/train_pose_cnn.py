import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class PoseCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(99, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Load dataset
data = np.load('dataset/pose_sequences.npy', allow_pickle=True).item()
keypoints = np.array(data['keypoints'])
labels = np.array(data['labels'])

# Convert labels to integers
label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
y = np.array([label_to_idx[label] for label in labels])

# Save labels.txt in correct order
with open("labels.txt", "w") as f:
    for label in sorted(label_to_idx, key=label_to_idx.get):
        f.write(label + "\n")

X_tensor = torch.tensor(keypoints, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = PoseCNN(num_classes=len(label_to_idx))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

# Save model
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/pose_cnn.pt")
print("✅ Training complete. Saved model to model/pose_cnn.pt")
