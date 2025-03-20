"""
Deep Learning Charity Success Predictor

This script implements a neural network model to predict the success of charity 
funding applications based on various organizational features. It includes data 
preprocessing, model building, training, evaluation, and export functionality.

Key capabilities:
- Data preprocessing (scaling, encoding, feature selection)
- Neural network architecture with configurable layers
- Hyperparameter optimization
- Model evaluation and visualization
- Model export for deployment

Author: Freddrick Logan
"""

import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def load_data(filepath):
    """
    Load charity application data from CSV file.
    
    Parameters:
        filepath (str): Path to the charity data CSV file
    
    Returns:
        pandas.DataFrame: DataFrame containing the charity application data
    """
    print(f"Loading charity application data from: {filepath}")
    
    try:
        # Read CSV file into a pandas DataFrame
        df = pd.read_csv(filepath)
        print(f"Successfully loaded {len(df)} charity application records.")
        
        # Display basic information
        print("\nData Overview:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {', '.join(df.columns)}")
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            print(f"Warning: Found {missing_values} missing values in the dataset.")
        else:
            print("No missing values found in the dataset.")
            
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def preprocess_data(df, target_column='IS_SUCCESSFUL'):
    """
    Preprocess charity application data for neural network training.
    
    Parameters:
        df (pandas.DataFrame): Raw charity application data
        target_column (str): Name of the target column to predict
    
    Returns:
        tuple: X_train, X_test, y_train, y_test, feature_names, preprocessors
    """
    print("Preprocessing charity application data...")
    
    # Make a copy to avoid modifying the original data
    df_processed = df.copy()
    
    # Extract target variable
    y = df_processed[target_column].values
    
    # Drop non-beneficial ID columns (if they exist)
    columns_to_drop = ['EIN', 'NAME', target_column] 
    df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns])
    
    print(f"\nFeatures used for prediction: {', '.join(df_processed.columns)}")
    
    # Identify categorical features (object and categorical dtypes)
    cat_features = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"\nCategorical features: {', '.join(cat_features)}")
    print(f"Numerical features: {', '.join(num_features)}")
    
    # Bin rare categorical values for APPLICATION_TYPE
    application_counts = df_processed['APPLICATION_TYPE'].value_counts()
    application_to_replace = application_counts[application_counts < 500].index
    df_processed['APPLICATION_TYPE'] = df_processed['APPLICATION_TYPE'].replace(application_to_replace, 'Other')
    
    # Bin rare categorical values for CLASSIFICATION
    classification_counts = df_processed['CLASSIFICATION'].value_counts()
    classifications_to_replace = classification_counts[classification_counts < 1000].index
    df_processed['CLASSIFICATION'] = df_processed['CLASSIFICATION'].replace(classifications_to_replace, 'Other')
    
    print("\nAfter binning rare categories:")
    print(f"APPLICATION_TYPE unique values: {df_processed['APPLICATION_TYPE'].nunique()}")
    print(f"CLASSIFICATION unique values: {df_processed['CLASSIFICATION'].nunique()}")
    
    # Split data into features and target
    X = df_processed
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Preprocessing for numerical features
    num_transformer = StandardScaler()
    
    # Apply numerical preprocessing
    if num_features:
        X_train_num = num_transformer.fit_transform(X_train[num_features])
        X_test_num = num_transformer.transform(X_test[num_features])
    else:
        X_train_num = np.empty((X_train.shape[0], 0))
        X_test_num = np.empty((X_test.shape[0], 0))
    
    # Preprocessing for categorical features
    cat_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')
    
    # Apply categorical preprocessing
    if cat_features:
        X_train_cat = cat_transformer.fit_transform(X_train[cat_features])
        X_test_cat = cat_transformer.transform(X_test[cat_features])
        
        # Get feature names for one-hot encoded features
        encoded_feature_names = cat_transformer.get_feature_names_out(cat_features)
    else:
        X_train_cat = np.empty((X_train.shape[0], 0))
        X_test_cat = np.empty((X_test.shape[0], 0))
        encoded_feature_names = []
    
    # Combine preprocessed data
    X_train_processed = np.hstack((X_train_num, X_train_cat))
    X_test_processed = np.hstack((X_test_num, X_test_cat))
    
    # Combine feature names
    feature_names = num_features + list(encoded_feature_names)
    
    print(f"\nPreprocessed data shape: {X_train_processed.shape} with {len(feature_names)} features")
    
    # Save preprocessing objects for later use
    preprocessors = {
        'num_transformer': num_transformer,
        'cat_transformer': cat_transformer,
        'num_features': num_features,
        'cat_features': cat_features
    }
    
    return X_train_processed, X_test_processed, y_train, y_test, feature_names, preprocessors


def build_model(input_dim, hidden_layers=[128, 64], activation='relu', dropout_rate=0.2):
    """
    Build a neural network model for charity application success prediction.
    
    Parameters:
        input_dim (int): Number of input features
        hidden_layers (list): List of neurons for each hidden layer
        activation (str): Activation function for hidden layers
        dropout_rate (float): Dropout rate for regularization
    
    Returns:
        tensorflow.keras.Model: Compiled neural network model
    """
    print(f"Building neural network with {len(hidden_layers)} hidden layers: {hidden_layers}")
    
    # Initialize a sequential model
    model = Sequential()
    
    # Add the first hidden layer with input dimension
    model.add(Dense(units=hidden_layers[0], activation=activation, input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Add additional hidden layers
    for units in hidden_layers[1:]:
        model.add(Dense(units=units, activation=activation))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    
    # Add output layer with sigmoid activation for binary classification
    model.add(Dense(units=1, activation='sigmoid'))
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    return model


def train_model(model, X_train, y_train, X_test, y_test, batch_size=32, epochs=100, patience=10, output_dir=None):
    """
    Train the neural network model with early stopping.
    
    Parameters:
        model (tensorflow.keras.Model): Neural network model to train
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training target
        X_test (numpy.ndarray): Testing features
        y_test (numpy.ndarray): Testing target
        batch_size (int): Batch size for training
        epochs (int): Maximum number of epochs
        patience (int): Early stopping patience
        output_dir (str): Directory to save model checkpoints
    
    Returns:
        tuple: Trained model and training history
    """
    print(f"Training neural network model with batch size={batch_size}, max epochs={epochs}")
    
    # Create output directory if it doesn't exist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, "best_model.h5")
    else:
        checkpoint_path = "best_model.h5"
    
    # Define callbacks
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        # Model checkpoint to save the best model
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Calculate final metrics
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nFinal model evaluation:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return model, history


def evaluate_model(model, history, X_test, y_test, output_dir=None):
    """
    Evaluate the trained model and visualize results.
    
    Parameters:
        model (tensorflow.keras.Model): Trained neural network model
        history (tensorflow.keras.callbacks.History): Training history
        X_test (numpy.ndarray): Testing features
        y_test (numpy.ndarray): Testing target
        output_dir (str): Directory to save evaluation visualizations
    
    Returns:
        dict: Evaluation metrics
    """
    print("Evaluating model performance...")
    
    # Create output directory if it doesn't exist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    # Make predictions on test data
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    
    # Generate and print classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    
    # Collect evaluation metrics
    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1_score': report['1']['f1-score'],
        'roc_auc': roc_auc
    }
    
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Show all plots
    plt.show()
    
    return metrics


def export_model(model, preprocessors, output_dir='models'):
    """
    Export the trained model and preprocessors for future use.
    
    Parameters:
        model (tensorflow.keras.Model): Trained neural network model
        preprocessors (dict): Dictionary containing preprocessing objects
        output_dir (str): Directory to save exported model
    
    Returns:
        str: Path to exported model
    """
    print(f"Exporting model to {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model in h5 format
    model_path = os.path.join(output_dir, 'charity_predictor_model.h5')
    model.save(model_path)
    
    # Save preprocessors for inference
    preprocessors_path = os.path.join(output_dir, 'preprocessors.pkl')
    with open(preprocessors_path, 'wb') as f:
        pickle.dump(preprocessors, f)
    
    print(f"Model saved to {model_path}")
    print(f"Preprocessors saved to {preprocessors_path}")
    
    return model_path


def inference_example(model_path, preprocessors_path, sample_data):
    """
    Demonstrate model inference with sample data.
    
    Parameters:
        model_path (str): Path to saved model
        preprocessors_path (str): Path to saved preprocessors
        sample_data (pandas.DataFrame): Sample data for inference
    
    Returns:
        numpy.ndarray: Prediction results
    """
    print("Demonstrating model inference with sample data...")
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load preprocessors
    with open(preprocessors_path, 'rb') as f:
        preprocessors = pickle.load(f)
    
    # Extract preprocessors
    num_transformer = preprocessors['num_transformer']
    cat_transformer = preprocessors['cat_transformer']
    num_features = preprocessors['num_features']
    cat_features = preprocessors['cat_features']
    
    # Preprocess sample data
    sample_num = num_transformer.transform(sample_data[num_features]) if num_features else np.empty((sample_data.shape[0], 0))
    sample_cat = cat_transformer.transform(sample_data[cat_features]) if cat_features else np.empty((sample_data.shape[0], 0))
    
    # Combine preprocessed data
    sample_processed = np.hstack((sample_num, sample_cat))
    
    # Make predictions
    predictions_proba = model.predict(sample_processed)
    predictions = (predictions_proba > 0.5).astype(int)
    
    # Print results
    print("\nInference Results:")
    for i, (prob, pred) in enumerate(zip(predictions_proba, predictions)):
        print(f"Sample {i+1}: Probability = {prob[0]:.4f}, Prediction = {'Successful' if pred[0] == 1 else 'Unsuccessful'}")
    
    return predictions


def optimize_hyperparameters(X_train, y_train, X_test, y_test, param_grid):
    """
    Perform hyperparameter optimization for the neural network model.
    
    Parameters:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training target
        X_test (numpy.ndarray): Testing features
        y_test (numpy.ndarray): Testing target
        param_grid (dict): Hyperparameter grid for optimization
    
    Returns:
        dict: Best hyperparameters
    """
    print("Performing hyperparameter optimization...")
    
    # Initialize variables to track best parameters
    best_accuracy = 0
    best_params = None
    results = []
    
    # Get input dimension
    input_dim = X_train.shape[1]
    
    # Iterate through hyperparameter combinations
    total_combinations = len(param_grid['hidden_layers']) * len(param_grid['activation']) * len(param_grid['dropout_rate'])
    current = 0
    
    for hidden_layers in param_grid['hidden_layers']:
        for activation in param_grid['activation']:
            for dropout_rate in param_grid['dropout_rate']:
                current += 1
                print(f"\nTrying combination {current}/{total_combinations}:")
                print(f"Hidden layers: {hidden_layers}, Activation: {activation}, Dropout: {dropout_rate}")
                
                # Build model with current hyperparameters
                model = build_model(
                    input_dim=input_dim,
                    hidden_layers=hidden_layers,
                    activation=activation,
                    dropout_rate=dropout_rate
                )
                
                # Train model with early stopping
                model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    batch_size=32,
                    epochs=50,  # Use fewer epochs for optimization
                    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
                    verbose=0
                )
                
                # Evaluate model
                loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                
                # Store results
                result = {
                    'hidden_layers': hidden_layers,
                    'activation': activation,
                    'dropout_rate': dropout_rate,
                    'accuracy': accuracy,
                    'loss': loss
                }
                results.append(result)
                
                print(f"Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
                
                # Update best parameters if accuracy improves
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {
                        'hidden_layers': hidden_layers,
                        'activation': activation,
                        'dropout_rate': dropout_rate
                    }
    
    # Print optimization results
    print("\nHyperparameter Optimization Results:")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Best parameters: {best_params}")
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    # Plot top 10 configurations
    plt.figure(figsize=(12, 6))
    sns.barplot(x=results_df.head(10).index, y='accuracy', data=results_df.head(10))
    plt.title('Top 10 Model Configurations')
    plt.xlabel('Configuration Index')
    plt.ylabel('Validation Accuracy')
    plt.ylim(results_df['accuracy'].min() - 0.01, results_df['accuracy'].max() + 0.01)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return best_params


def main():
    """
    Main function to orchestrate the charity application success prediction workflow.
    """
    print("Charity Application Success Prediction")
    print("======================================")
    
    # Define file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data', 'charity_data.csv')
    output_dir = os.path.join(current_dir, 'output')
    models_dir = os.path.join(current_dir, 'models')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_data(data_path)
    
    if df is None:
        print("Failed to load data. Exiting program.")
        return
    
    # Preprocess data
    X_train, X_test, y_train, y_test, feature_names, preprocessors = preprocess_data(df)
    
    # Define hyperparameter grid for optimization
    param_grid = {
        'hidden_layers': [[128, 64], [64, 32], [128, 64, 32]],
        'activation': ['relu', 'tanh'],
        'dropout_rate': [0.2, 0.3]
    }
    
    # Optimize hyperparameters
    best_params = optimize_hyperparameters(X_train, y_train, X_test, y_test, param_grid)
    
    # Build model with optimized hyperparameters
    input_dim = X_train.shape[1]
    model = build_model(
        input_dim=input_dim,
        hidden_layers=best_params['hidden_layers'],
        activation=best_params['activation'],
        dropout_rate=best_params['dropout_rate']
    )
    
    # Train model
    model, history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        output_dir=models_dir
    )
    
    # Evaluate model
    metrics = evaluate_model(
        model=model,
        history=history,
        X_test=X_test,
        y_test=y_test,
        output_dir=output_dir
    )
    
    # Export model
    model_path = export_model(
        model=model,
        preprocessors=preprocessors,
        output_dir=models_dir
    )
    
    # Demonstrate inference with sample data (first 5 records for example)
    sample_data = df.drop(columns=['IS_SUCCESSFUL']).head(5)
    preprocessors_path = os.path.join(models_dir, 'preprocessors.pkl')
    inference_example(model_path, preprocessors_path, sample_data)
    
    print("\nCharity application success prediction workflow completed successfully!")


if __name__ == "__main__":
    main()