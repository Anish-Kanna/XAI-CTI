import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from preprocessing import load_data
from model import XAI_CTI_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, X_test, y_train, y_test, feature_names = load_data("data/full_week.csv")

model = XAI_CTI_Model(input_dim=X_train.shape[1]).to(device)
model.load_state_dict(torch.load("xai_cti_model_adv.pth", map_location=device))
model.eval()

criterion = nn.CrossEntropyLoss()

def fgsm_attack(model, data, target, epsilon):
    data.requires_grad = True
    output = model(data)
    loss = criterion(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    return perturbed_data.detach()

epsilons = [0, 0.05, 0.1, 0.15, 0.2]
accuracies = []

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=32,
    shuffle=False
)

for eps in epsilons:
    preds = []

    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)

        if eps != 0:
            xb = fgsm_attack(model, xb.clone(), yb, eps)

        with torch.no_grad():
            output = model(xb)
            pred = output.argmax(dim=1)
            preds.extend(pred.cpu().numpy())

    acc = accuracy_score(y_test.numpy(), preds)
    accuracies.append(acc)
    print(f"Epsilon: {eps}, Accuracy: {acc}")

plt.figure()
plt.plot(epsilons, accuracies, marker='o')
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.title("Robustness Curve (FGSM Attack)")
plt.savefig("robustness_curve.png")
plt.show()