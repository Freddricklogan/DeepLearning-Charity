# Deep Learning Charity Success Predictor

![Neural Network Model](images/model_performance.jpg)

## Overview
A sophisticated neural network model designed to predict the success of charity funding applications based on various organizational features. This project leverages deep learning techniques to identify patterns in historical funding data and create a predictive model that can help identify promising charity ventures.

## Features

- **Advanced Data Preprocessing**:
  - Automated binning of rare categorical values
  - Feature scaling using standardization
  - One-hot encoding of categorical variables
  - Feature selection for optimal model performance

- **Neural Network Architecture**:
  - Configurable hidden layers and neurons
  - Dropout layers for regularization
  - Batch normalization for faster training
  - Hyperparameter optimization for maximum performance

- **Model Training & Evaluation**:
  - Early stopping to prevent overfitting
  - Model checkpointing to save best weights
  - Comprehensive evaluation metrics (accuracy, precision, recall, F1-score)
  - ROC curve and AUC analysis

- **Production-Ready Deployment**:
  - Exportable model for future predictions
  - Preprocessing pipeline for new data
  - Inference examples for implementation

## Project Structure

```
DeepLearning-Charity/
├── model.py                          # Monolithic pipeline (load, preprocess, train, evaluate, export)
├── data/
│   ├── charity_data.csv              # Raw charity application dataset
│   └── preprocess.py                 # Modular preprocessing functions (load, clean, encode, split)
├── models/
│   └── model.py                      # Modular model functions (build, train, evaluate, export, predict)
├── notebooks/
│   └── model_training.ipynb          # Interactive Jupyter notebook walking through the full pipeline
├── images/
│   └── model_performance.jpg         # Model performance visualisation
├── output/                           # Generated evaluation plots (created at runtime)
├── .gitignore
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Freddricklogan/DeepLearning-Charity.git
   cd DeepLearning-Charity
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   # venv\Scripts\activate          # Windows
   pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
   ```

3. Place the charity application CSV at `data/charity_data.csv`.

## Usage

### Run the full pipeline (script)

```bash
python model.py
```

This executes data loading, preprocessing, hyperparameter optimisation, training, evaluation, and model export in one go.

### Use the modular API

```python
from data.preprocess import load_data, preprocess_data
from models.model import build_model, train_model, evaluate_model, export_model

df = load_data('data/charity_data.csv')
X_train, X_test, y_train, y_test, feature_names, preprocessors = preprocess_data(df)

model = build_model(input_dim=X_train.shape[1], hidden_layers=[128, 64])
model, history = train_model(model, X_train, y_train, X_test, y_test)
metrics = evaluate_model(model, history, X_test, y_test, output_dir='output')
export_model(model, preprocessors, output_dir='models')
```

### Interactive notebook

Open `notebooks/model_training.ipynb` in Jupyter and step through each cell for an interactive walkthrough of the pipeline.

## Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **Scikit-learn**: Data preprocessing and evaluation metrics
- **Pandas/NumPy**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **Python**: Core programming language

## License

This project is provided for educational purposes.
