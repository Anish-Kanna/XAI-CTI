import torch
import numpy as np
from preprocessing import load_data
from model import XAI_CTI_Model

device = torch.device("cpu")

print("Loading data...")
X_train, X_test, y_train, y_test, feature_names = load_data("data/cic_ids.csv")

print("Loading model...")
model = XAI_CTI_Model(input_dim=X_train.shape[1])
model.load_state_dict(torch.load("xai_cti_model.pth", map_location=device))
model.eval()

# Pick one test sample
#sample = X_test[0].unsqueeze(0)

# Find first attack sample
for i in range(len(y_test)):
    if y_test[i] == 1:
        sample = X_test[i].unsqueeze(0)
        break

with torch.no_grad():
    output = model(sample)
    probs = torch.softmax(output, dim=1)
    prediction = torch.argmax(probs).item()
    confidence = probs[0][prediction].item()

label = "ATTACK" if prediction == 1 else "BENIGN"

print("\nPrediction:", label)
print("Confidence:", round(confidence, 4))

# Simple symbolic reasoning rules
reasoning = []

sample_np = sample.numpy()[0]

for i, value in enumerate(sample_np):
    if abs(value) > 2.5:  # strong deviation after scaling
        reasoning.append(feature_names[i])

print("\nTop Triggered Features:")
for r in reasoning[:5]:
    print("-", r)

# Threat level logic
if prediction == 1 and confidence > 0.9:
    threat_level = "HIGH"
elif prediction == 1:
    threat_level = "MEDIUM"
else:
    threat_level = "LOW"

print("\nThreat Level:", threat_level)