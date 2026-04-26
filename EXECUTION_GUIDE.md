# Hybrid Transformer–CNN Neuro-Symbolic Explainable AI for Cyber Threat Intelligence: Advancing Transparency and Adversarial Robustness
### Explainable AI for Cyber Threat Intelligence

Quick start guide for executing the project.

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- (Optional) NVIDIA GPU with CUDA 11.8+ for faster training

### Step 1: Clone the repository
```bash
git clone https://github.com/your-username/XAI_CTI_Project.git
cd XAI_CTI_Project
```

### Step 2: Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Prepare the dataset

Place CIC-IDS 2017 CSV files in the `data/` folder, then run:
```bash
python merge_fullweek.py
```
This creates `data/full_week.csv`.

---

## Quick Start - Execute Commands

### Train standard model
```bash
python train_gpu.py
```

### Train adversarial model (recommended)
```bash
python train_adversarial.py
```

### Run demo
```bash
python demo.py
```

### Run demo with new data
```bash
python demo2.py
```

### Generate SHAP explanations
```bash
python explain_shap.py
```

### Test robustness
```bash
python adversarial_test.py
```

### Plot robustness curve
```bash
python robustness_curve.py
```

---

## Output Files

| File | Description |
|------|-------------|
| `xai_cti_model.pth` | Trained standard model |
| `xai_cti_model_adv.pth` | Adversarially trained model |
| `scaler.pkl` | Feature scaler |
| `shap_summary.png` | Feature importance plot |
| `robustness_curve.png` | Robustness plot |

---

## References & Resources

- **Research Paper**: [Hybrid Transformer-CNN Neuro-Symbolic Explainable AI for Cyber Threat Intelligence](https://www.researchgate.net/publication/398320038_Hybrid_Transformer-CNN_Neuro-Symbolic_Explainable_AI_for_Cyber_Threat_Intelligence_Advancing_Transparency_and_Adversarial_Robustness)
- **Dataset**: [Network Intrusion Dataset on Kaggle](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)

---
