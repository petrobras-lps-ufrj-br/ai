import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from ai.preprocessing import StandardScale
from ai.trainers.time_series import Trainer
from ai.models.model_v1 import Model_v1
from ai.evaluation import Summary
import torch.nn as nn
import torch
import pytorch_lightning as pl

def create_windowed_dataset(data, target, window_size):
    X_windows = []
    y_windows = []
    for i in range(len(data) - window_size):
        X_windows.append(data[i:i+window_size])
        y_windows.append(target[i+window_size])
    return np.array(X_windows), np.array(y_windows)

def main():
    # Load Data
    csv_path = 'notebooks/compressor.csv'
    if not os.path.exists(csv_path):
        # Fallback for notebook location
        csv_path = '../notebooks/compressor.csv'
        if not os.path.exists(csv_path):
             csv_path = 'compressor.csv'
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # 1. Resample and Interpolate
    # Assuming the data is roughly 30s, we enforce it to handle gaps
    df = df.resample('30S').mean()
    df = df.interpolate(method='linear')
    df = df.dropna()
    
    print("Data Head after preprocessing:")
    print(df.head())

    # Define Inputs and Output
    input_cols = [
        'PH (CBM) 1st Stage ActShaft Power', 
        'PH (CBM) 1st Stage ActPress Ratio', 
        'PH (CBM) 1st Stage ExpPress Ratio'
    ]
    output_col = ['PH (CBM) 1st Stg ActCompr Poly Head']

    X_raw = df[input_cols].values
    y_raw = df[output_col].values
    
    # 2. Windowing
    window_size = 20
    X_windows, y_windows = create_windowed_dataset(X_raw, y_raw, window_size)
    
    # Flatten windows for MLP (N, 20, F) -> (N, 20*F)
    N, T, F = X_windows.shape
    X_flattened = X_windows.reshape(N, T * F)
    
    print(f"Original Input Shape: {X_raw.shape}")
    print(f"Windowed Input Shape: {X_windows.shape}")
    print(f"Flattened Input Shape: {X_flattened.shape}")
    print(f"Output Shape: {y_windows.shape}")

    # Setup Components
    # 3. TimeSeriesSplit
    cv = TimeSeriesSplit(n_splits=4)
    
    # Preprocessing
    # Note: For time series, fitting scaler on windowed flattened data is tricky if we want to share statistics across time steps.
    # Ideally, we fit on the original raw data of the training fold.
    # But the Trainer's preprocessing inputs are applied to the data passed to fit().
    # Here X is flattened. StandardScale on (N, 20*F) will calculate mean/std for each of the 20*F features independently.
    # This is acceptable (scaling each lag independently).
    input_proc = [StandardScale()]
    output_proc = [StandardScale()]
    
    # Model configuration
    # Input dim is now window_size * n_features
    model = Model_v1(input_dim=X_flattened.shape[1], n_hidden=32)
    
    evaluators = [Summary(name="metrics")]
    
    params = {
        "batch_size": 32,
        "num_epochs": 5,
        "lr": 1e-3,
        "optimizer": "Adam"
    }

    trainer = Trainer(
        model=model,
        cv_strategy=cv,
        input_preprocessing=input_proc,
        output_preprocessing=output_proc,
        callbacks=[],
        evaluators=evaluators,
        params=params
    )

    # Train
    print("Starting training...")
    # output_dir inside the current directory
    trainer.fit(X_flattened, y_windows, output_dir="output")
    print("Training complete. Artifacts saved in 'output'.")

if __name__ == "__main__":
    main()
