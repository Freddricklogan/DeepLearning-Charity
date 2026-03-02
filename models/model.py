"""
Neural Network Model Module for Charity Success Prediction

This module provides functions for building, compiling, training, evaluating,
and exporting a TensorFlow/Keras neural network model. All architecture and
training logic is consistent with the root model.py pipeline.

Author: Freddrick Logan
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ---------------------------------------------------------------------------
# Model Construction
# ---------------------------------------------------------------------------

def build_model(input_dim, hidden_layers=None, activation='relu', dropout_rate=0.2):
    """
    Build a Sequential neural network for binary classification.

    Architecture per hidden layer:
        Dense -> BatchNormalization -> Dropout

    The output layer uses a single neuron with sigmoid activation.

    Parameters:
        input_dim (int): Number of input features.
        hidden_layers (list[int]): Number of neurons in each hidden layer.
                                   Defaults to ``[128, 64]``.
        activation (str): Activation function for hidden layers.
        dropout_rate (float): Dropout probability for regularization.

    Returns:
        tensorflow.keras.Model: A compiled Keras Sequential model.
    """
    if hidden_layers is None:
        hidden_layers = [128, 64]

    print(f"Building neural network with {len(hidden_layers)} hidden layers: "
          f"{hidden_layers}")

    model = Sequential()

    # First hidden layer (requires input_dim)
    model.add(Dense(units=hidden_layers[0], activation=activation,
                    input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Additional hidden layers
    for units in hidden_layers[1:]:
        model.add(Dense(units=units, activation=activation))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Output layer -- binary classification
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    model.summary()
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, X_train, y_train, X_test, y_test,
                batch_size=32, epochs=100, patience=10, output_dir=None):
    """
    Train the model with early stopping and optional model checkpointing.

    Parameters:
        model (tensorflow.keras.Model): Compiled Keras model.
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target labels.
        X_test (numpy.ndarray): Validation / testing features.
        y_test (numpy.ndarray): Validation / testing target labels.
        batch_size (int): Mini-batch size.
        epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs with no improvement before stopping.
        output_dir (str | None): Directory to save the best checkpoint.

    Returns:
        tuple: (trained model, keras History object)
    """
    print(f"Training model  --  batch_size={batch_size}, "
          f"max_epochs={epochs}, patience={patience}")

    # Prepare checkpoint path
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, 'best_model.h5')
    else:
        checkpoint_path = 'best_model.h5'

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal evaluation  --  Loss: {loss:.4f}  |  Accuracy: {accuracy:.4f}")

    return model, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, history, X_test, y_test, output_dir=None):
    """
    Evaluate a trained model and produce visualizations.

    Generates:
      - Training / validation loss and accuracy curves.
      - Confusion matrix heatmap.
      - ROC curve with AUC.

    Parameters:
        model (tensorflow.keras.Model): Trained model.
        history (tensorflow.keras.callbacks.History): Training history.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test target labels.
        output_dir (str | None): Directory to save plots.

    Returns:
        dict: Evaluation metrics (accuracy, precision, recall, f1, roc_auc).
    """
    print("Evaluating model performance...")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # ---- Training history plots ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, 'training_history.png'))

    # ---- Predictions ----
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # ---- Classification report ----
    report = classification_report(y_test, y_pred, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_title('Confusion Matrix')
    ax_cm.set_ylabel('True Label')
    ax_cm.set_xlabel('Predicted Label')
    if output_dir:
        fig_cm.savefig(os.path.join(output_dir, 'confusion_matrix.png'))

    # ---- ROC curve ----
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic')
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True, alpha=0.3)
    if output_dir:
        fig_roc.savefig(os.path.join(output_dir, 'roc_curve.png'))

    plt.show()

    # ---- Collect metrics ----
    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1_score': report['1']['f1-score'],
        'roc_auc': roc_auc,
    }

    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# Hyperparameter Optimisation
# ---------------------------------------------------------------------------

def optimize_hyperparameters(X_train, y_train, X_test, y_test, param_grid=None):
    """
    Grid-search over neural network hyperparameters.

    Parameters:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target.
        X_test (numpy.ndarray): Testing features.
        y_test (numpy.ndarray): Testing target.
        param_grid (dict | None): Hyperparameter grid.  Defaults to a
            sensible set of architectures, activations, and dropout rates.

    Returns:
        dict: Best hyperparameters found during the search.
    """
    if param_grid is None:
        param_grid = {
            'hidden_layers': [[128, 64], [64, 32], [128, 64, 32]],
            'activation': ['relu', 'tanh'],
            'dropout_rate': [0.2, 0.3],
        }

    print("Performing hyperparameter optimisation...")

    input_dim = X_train.shape[1]
    best_accuracy = 0
    best_params = None
    results = []

    total = (len(param_grid['hidden_layers'])
             * len(param_grid['activation'])
             * len(param_grid['dropout_rate']))
    current = 0

    for hidden_layers in param_grid['hidden_layers']:
        for activation in param_grid['activation']:
            for dropout_rate in param_grid['dropout_rate']:
                current += 1
                print(f"\n[{current}/{total}]  layers={hidden_layers}  "
                      f"activation={activation}  dropout={dropout_rate}")

                model = build_model(
                    input_dim=input_dim,
                    hidden_layers=hidden_layers,
                    activation=activation,
                    dropout_rate=dropout_rate,
                )

                model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    batch_size=32,
                    epochs=50,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=5,
                                            restore_best_weights=True)],
                    verbose=0,
                )

                loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                results.append({
                    'hidden_layers': hidden_layers,
                    'activation': activation,
                    'dropout_rate': dropout_rate,
                    'accuracy': accuracy,
                    'loss': loss,
                })
                print(f"  -> Accuracy: {accuracy:.4f}  |  Loss: {loss:.4f}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {
                        'hidden_layers': hidden_layers,
                        'activation': activation,
                        'dropout_rate': dropout_rate,
                    }

    print(f"\nBest accuracy: {best_accuracy:.4f}")
    print(f"Best params:   {best_params}")

    return best_params


# ---------------------------------------------------------------------------
# Export / Persistence
# ---------------------------------------------------------------------------

def export_model(model, preprocessors=None, output_dir='models'):
    """
    Save the trained Keras model (H5) and optional preprocessors (pickle).

    Parameters:
        model (tensorflow.keras.Model): Trained model to export.
        preprocessors (dict | None): Preprocessing transformers to save.
        output_dir (str): Destination directory.

    Returns:
        str: Path to the saved model file.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, 'charity_predictor_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")

    if preprocessors is not None:
        preprocessors_path = os.path.join(output_dir, 'preprocessors.pkl')
        with open(preprocessors_path, 'wb') as f:
            pickle.dump(preprocessors, f)
        print(f"Preprocessors saved to {preprocessors_path}")

    return model_path


def load_trained_model(model_path):
    """
    Load a previously saved Keras model from disk.

    Parameters:
        model_path (str): Path to the ``.h5`` model file.

    Returns:
        tensorflow.keras.Model: Loaded model ready for inference.
    """
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model, X):
    """
    Run inference on preprocessed feature data.

    Parameters:
        model (tensorflow.keras.Model): Trained model.
        X (numpy.ndarray): Preprocessed feature matrix.

    Returns:
        tuple: (predicted_classes, predicted_probabilities)
            - predicted_classes: 0/1 array of shape ``(n_samples,)``
            - predicted_probabilities: float array of shape ``(n_samples,)``
    """
    probabilities = model.predict(X).flatten()
    classes = (probabilities > 0.5).astype(int)
    return classes, probabilities


def inference_example(model_path, preprocessors_path, sample_data):
    """
    Demonstrate end-to-end inference on new data.

    Parameters:
        model_path (str): Path to the saved model.
        preprocessors_path (str): Path to saved preprocessors pickle.
        sample_data (pandas.DataFrame): Raw sample data (before encoding).

    Returns:
        numpy.ndarray: Predicted class labels.
    """
    print("Running inference example...")

    # Load model
    model = load_trained_model(model_path)

    # Load preprocessors
    with open(preprocessors_path, 'rb') as f:
        preprocessors = pickle.load(f)

    num_transformer = preprocessors['num_transformer']
    cat_transformer = preprocessors['cat_transformer']
    num_features = preprocessors['num_features']
    cat_features = preprocessors['cat_features']

    # Transform
    num_data = (num_transformer.transform(sample_data[num_features])
                if num_features else np.empty((sample_data.shape[0], 0)))
    cat_data = (cat_transformer.transform(sample_data[cat_features])
                if cat_features else np.empty((sample_data.shape[0], 0)))

    X_processed = np.hstack((num_data, cat_data))

    # Predict
    classes, probabilities = predict(model, X_processed)

    print("\nInference Results:")
    for i, (prob, cls) in enumerate(zip(probabilities, classes)):
        label = 'Successful' if cls == 1 else 'Unsuccessful'
        print(f"  Sample {i + 1}: Probability = {prob:.4f}, "
              f"Prediction = {label}")

    return classes
