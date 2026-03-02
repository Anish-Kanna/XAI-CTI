import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import joblib

def load_data(path):
    print("Loading dataset...")
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    print("Original shape:", df.shape)

    # Convert labels
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    # Keep numeric only
    df = df.select_dtypes(include=[np.number])

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows containing NaN
    df.dropna(inplace=True) 

    X = df.drop('Label', axis=1)
    y = df['Label']

    feature_names = X.columns.tolist()

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "scaler.pkl")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y.values, test_size=0.2, random_state=42
    )

    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long),
        feature_names
    )