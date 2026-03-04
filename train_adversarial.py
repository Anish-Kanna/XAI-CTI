import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from preprocessing import load_data
from model import XAI_CTI_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading data...")
X_train, X_test, y_train, y_test, feature_names = load_data("data/full_week.csv")

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

model = XAI_CTI_Model(input_dim=X_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epsilon = 0.01
epochs = 5

def fgsm_attack(model, data, target, epsilon):
    data.requires_grad = True
    output = model(data)
    loss = criterion(output, target)

    model.zero_grad()
    loss.backward()

    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()

    return perturbed_data.detach()

print("Starting adversarial training...")

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        # Normal loss
        output = model(xb)
        loss = criterion(output, yb)

        # Adversarial loss
        adv_x = fgsm_attack(model, xb.clone(), yb, epsilon)
        adv_output = model(adv_x)
        adv_loss = criterion(adv_output, yb)

        total_combined_loss = (loss + adv_loss) / 2

        optimizer.zero_grad()
        total_combined_loss.backward()
        optimizer.step()

        total_loss += total_combined_loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "xai_cti_model_adv.pth")
print("Adversarially trained model saved.")