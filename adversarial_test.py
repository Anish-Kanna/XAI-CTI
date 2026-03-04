import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from preprocessing import load_data
from model import XAI_CTI_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading data...")
X_train, X_test, y_train, y_test, feature_names = load_data("data/full_week.csv")

model = XAI_CTI_Model(input_dim=X_train.shape[1]).to(device)
model.load_state_dict(torch.load("xai_cti_model_adv.pth", map_location=device))
model.eval()

criterion = nn.CrossEntropyLoss()

# FGSM attack function
def fgsm_attack(model, data, target, epsilon):
    data.requires_grad = True

    output = model(data)
    loss = criterion(output, target)

    model.zero_grad()
    loss.backward()

    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()

    return perturbed_data.detach()

print("Evaluating normal accuracy...")

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=32,
    shuffle=False
)

normal_preds = []
adv_preds = []

epsilon = 0.1  # small perturbation strength

for xb, yb in test_loader:
    xb, yb = xb.to(device), yb.to(device)

    # Normal prediction
    with torch.no_grad():
        output = model(xb)
        preds = output.argmax(dim=1)
        normal_preds.extend(preds.cpu().numpy())

    # Generate adversarial example
    adv_x = fgsm_attack(model, xb.clone(), yb, epsilon)

    with torch.no_grad():
        adv_output = model(adv_x)
        adv_pred = adv_output.argmax(dim=1)
        adv_preds.extend(adv_pred.cpu().numpy())

normal_acc = accuracy_score(y_test.numpy(), normal_preds)
adv_acc = accuracy_score(y_test.numpy(), adv_preds)

print("\nNormal Accuracy:", normal_acc)
print("Adversarial Accuracy (FGSM):", adv_acc)
print("Accuracy Drop:", normal_acc - adv_acc)