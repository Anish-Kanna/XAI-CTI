import torch
import shap
import numpy as np
from preprocessing import load_data
from model import XAI_CTI_Model
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cpu")

print("\n========== XAI-CTI DEMO ==========\n")

X_train, X_test, y_train, y_test, feature_names = load_data("data/full_week.csv")

# Loading the adversarially trained model
model = XAI_CTI_Model(input_dim=X_train.shape[1])
model.load_state_dict(torch.load("xai_cti_model_adv.pth", map_location=device))
model.eval()

# Attack sample
for i in range(len(y_test)):
    if y_test[i] == 1:
        sample = X_test[i].unsqueeze(0)
        break

# Prediction
with torch.no_grad():
    output = model(sample)
    probs = torch.softmax(output, dim=1)
    prediction = torch.argmax(probs).item()
    confidence = probs[0][prediction].item()

label = "ATTACK" if prediction == 1 else "BENIGN"

print(f"Prediction: {label}")
print(f"Confidence: {confidence:.4f}")

# Threat level logic
if prediction == 1 and confidence > 0.9:
    threat_level = "HIGH"
elif prediction == 1:
    threat_level = "MEDIUM"
else:
    threat_level = "LOW"

print(f"Threat Level: {threat_level}\n")

# ---------------- SHAP Explanation ----------------
background = X_train[:50]
explainer = shap.DeepExplainer(model, background)

shap_values = explainer.shap_values(sample, check_additivity=False)
shap_vals = np.array(shap_values)

if len(shap_vals.shape) == 3:
    shap_vals = shap_vals[:, :, 1]

# Get top 3 important features
feature_importance = np.abs(shap_vals[0])
top_indices = np.argsort(feature_importance)[-3:][::-1]

print("Top Contributing Features (SHAP):")
for idx in top_indices:
    print(f"- {feature_names[idx]}")

print("\nSymbolic Explanation:")

for idx in top_indices:
    fname = feature_names[idx]

    if "Bytes" in fname:
        print("- Abnormally high traffic volume detected")
    elif "Packets" in fname:
        print("- Elevated packet transmission rate observed")
    elif "Length" in fname:
        print("- Suspicious packet size distribution")
    else:
        print(f"- Significant deviation in {fname}")

print("\n===================================")