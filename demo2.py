import torch
import shap
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model import XAI_CTI_Model
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cpu")

print("\n========== XAI-CTI LIVE DEMO ==========\n")

# ---------------- LOAD TRAIN DATA (for scaler) ----------------
train_df = pd.read_csv("data/full_week.csv")
train_df.columns = train_df.columns.str.strip()

train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna()
train_df["Label"] = train_df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

X_train = train_df.drop("Label", axis=1)
feature_names = X_train.columns.tolist()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)

# ---------------- LOAD NEW DATA (REAL TRAFFIC) ----------------
print("Loading new traffic data...")

new_df = pd.read_csv("data/insane_dos.csv")  # 👈 your Wireshark converted file
new_df.columns = new_df.columns.str.strip()

new_df = new_df.replace([np.inf, -np.inf], np.nan).dropna()

# If no Label column (real traffic), just ignore
if "Label" in new_df.columns:
    new_df = new_df.drop("Label", axis=1)

# Fix column names
new_df.columns = new_df.columns.str.strip()

# Create missing columns if not present
for col in feature_names:
    if col not in new_df.columns:
        new_df[col] = 0  # fill missing features with 0

# Remove extra columns
X_new = new_df[feature_names]

X_new_scaled = scaler.transform(X_new)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

# ---------------- LOAD MODEL ----------------
model = XAI_CTI_Model(input_dim=X_new_tensor.shape[1])
model.load_state_dict(torch.load("xai_cti_model_adv.pth", map_location=device))
model.eval()

# ---------------- PICK SAMPLE ----------------
sample = X_new_tensor[0].unsqueeze(0)  # you can change index

# ---------------- PREDICTION ----------------
with torch.no_grad():
    output = model(sample)
    probs = torch.softmax(output, dim=1)
    prediction = torch.argmax(probs).item()
    confidence = probs[0][prediction].item()

label = "ATTACK" if prediction == 1 else "BENIGN"

print(f"Prediction: {label}")
print(f"Confidence: {confidence:.4f}")

# ---------------- THREAT LEVEL ----------------
if prediction == 1 and confidence > 0.9:
    threat_level = "HIGH"
elif prediction == 1:
    threat_level = "MEDIUM"
else:
    threat_level = "LOW"

print(f"Threat Level: {threat_level}\n")

# ---------------- SHAP EXPLANATION ----------------
background = X_train_tensor[:50]
explainer = shap.DeepExplainer(model, background)

shap_values = explainer.shap_values(sample, check_additivity=False)
shap_vals = np.array(shap_values)

if len(shap_vals.shape) == 3:
    shap_vals = shap_vals[:, :, 1]

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