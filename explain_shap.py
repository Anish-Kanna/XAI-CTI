import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import load_data
from model import XAI_CTI_Model

# Forcing the use of cpu
device = torch.device("cpu")

print("Loading data...")
X_train, X_test, y_train, y_test, feature_names = load_data("data/full_week.csv")

print("Loading trained model...")
model = XAI_CTI_Model(input_dim=X_train.shape[1])
model.load_state_dict(torch.load("xai_cti_model_adv.pth", map_location=device))
model.eval()

# Small background set for SHAP
background = X_train[:100]

# Small sample to explain
sample = X_test[:20]

print("Creating DeepExplainer...")
explainer = shap.DeepExplainer(model, background)

print("Computing SHAP values...")
shap_values = explainer.shap_values(sample, check_additivity=False)

# Convert to numpy safely
shap_vals = np.array(shap_values)

# Handling the shape automatically
# If shape is (samples, features, classes)
if len(shap_vals.shape) == 3:
    shap_vals = shap_vals[:, :, 1]  # Select class 1 (Attack)

print("Generating SHAP summary plot...")

shap.summary_plot(
    shap_vals,
    sample.numpy(),
    feature_names=feature_names,
    show=False
)

plt.savefig("shap_summary.png", bbox_inches="tight")
print("SHAP summary plot saved as shap_summary.png")