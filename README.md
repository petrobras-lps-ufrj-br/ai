# 🧠 AI Repository 

Welcome to the **AI Repository**! This project is a modular framework designed for developing, training, and testing artificial intelligence models, with a strong focus on **Time Series Forecasting** using **PyTorch** and **PyTorch Lightning**.

## 🎯 Purpose

The goal of this repository is to provide a clean, extensible structure for AI experiments. It abstracts away common boilerplate code for data loading, preprocessing, and training loops, allowing researchers and developers to focus on model architecture and feature engineering.

---

## 📂 Repository Structure

The codebase is organized into a Python package named `ai` with clear separation of concerns:

```text
.
├── 📂 ai/                  # Core Python package
│   ├── 📂 clients/         # External API clients (e.g., Cognite)
│   ├── 📂 evaluation/      # Metrics and model evaluation tools
│   ├── 📂 loaders/         # Custom DataLoaders (e.g., Time Series Windowing)
│   ├── 📂 models/          # PyTorch Model architectures (e.g., MLP)
│   ├── 📂 preprocessing/   # Sktime & Sklearn pipelines (Scaling, Lags, etc.)
│   ├── 📂 runners/         # Scripts to execute training/inference jobs
│   ├── 📂 trainers/        # Training loops (Cross-Validation + Lightning)
│   └── 📂 visualization/   # Training and inference visualization tools
│
├── 📂 notebooks/           # Jupyter Notebooks for exploration and demos
├── 📂 scripts/             # Helper scripts
├── 📜 activate.sh          # Environment setup script
├── 📜 Makefile             # Shortcuts for installation and running
└── 📜 requirements.txt     # Python dependencies
```

---

## ✨ Key Features

*   **⚡ PyTorch Lightning Integration**: Robust training loops with built-in logging, checkpointing, and GPU support.
*   **🔄 Automated Cross-Validation**: The `Time Series Trainer` handles CV splits (e.g., TimeSeriesSplit) automatically.
*   **🛠 Advanced Preprocessing**: Modular pipelines using `sktime` and `sklearn` for easy feature engineering (e.g., Lag features, Standard Scaling).
*   **🧵 Custom DataLoaders**: Flexible loaders that accept raw dataframes and handle windowing and batching on-the-fly.

---

## 🚀 Getting Started

### Prerequisites

*   **OS**: Mac/Linux
*   **Tools**: `make`, `python3`, `virtualenv`

### Installation

To set up the environment and install dependencies, simply run:

```bash
make
```

This command will source `activate.sh`, create a virtual environment in `.ai-env` (if it doesn't exist), and install the required packages.

### 📓 Running Notebooks

To launch a Jupyter Lab instance with the environment pre-configured:

```bash
make jupyter
```

---

## 🛠 Usage Example

Here is a simplified example of how to set up a training pipeline using the `ai` modules:

```python
import os
from sklearn.model_selection import TimeSeriesSplit
from ai.loaders import DataLoader_v1
from ai.models import Model_v1
from ai.trainers.time_series import Trainer
from ai.preprocessing import StandardScale, Lag

# 1. Define Data & Features
col_names = {"Raw_Sensor_1": "input_1", "Raw_Target": "target"}
feature_cols = ["input_1"]
target_col = ["target"]

# 2. Initialize DataLoader with Preprocessing
dataset = DataLoader_v1(
    path="data.csv",
    features=col_names,
    input_features=feature_cols,
    target_feature=target_col,
    lags={"input_1": Lag(10)}, 
    preprocessors={"input_1": StandardScale()}
)

# 3. Setup Model & Trainer
model = Model_v1(dataset=dataset, n_hidden=32)
trainer = Trainer(
    model=model,
    cv_strategy=TimeSeriesSplit(n_splits=4),
    accelerator='auto'
)

# 4. Train
trainer.fit(dataset, num_epochs=10)
```

---

## 🤝 Contributing

Feel free to open issues or submit pull requests. Ensure all new modules have appropriate unit tests and follow the existing directory structure.
