import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from preprocessing import load_data
from model import XAI_CTI_Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#data loading
X_train, X_test, y_train, y_test, feature_names = load_data("data/full_week.csv")

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=8,
    shuffle=True
)

model = XAI_CTI_Model(input_dim=X_train.shape[1]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5

print("Starting training...")

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

print("Training complete.")

# Evaluation of the data
model.eval()
test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=16,
    shuffle=False
)

all_preds = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        outputs = model(xb)
        preds = outputs.argmax(dim=1).cpu()
        all_preds.extend(preds.numpy())

preds = np.array(all_preds)

accuracy = accuracy_score(y_test.numpy(), preds)
precision = precision_score(y_test.numpy(), preds)
recall = recall_score(y_test.numpy(), preds)
f1 = f1_score(y_test.numpy(), preds)

print("\nFinal Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test.numpy(), preds)
print("\nConfusion Matrix:")
print(cm)

torch.save(model.state_dict(), "xai_cti_model.pth")
print("\nModel saved successfully.")