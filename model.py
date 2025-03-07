# Deep Learning Charity Success Predictor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

print("Charity Funding Predictor: Deep Learning Model")

# Sample data generation (in actual use, load from CSV)
np.random.seed(42)
n_samples = 1000

# Generate synthetic data
data = {
    'APPLICATION_TYPE': np.random.choice(['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10'], n_samples),
    'AFFILIATION': np.random.choice(['Independent', 'CompanySponsored', 'Family/Parent'], n_samples),
    'CLASSIFICATION': np.random.choice(['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10'], n_samples),
    'USE_CASE': np.random.choice(['Healthcare', 'Preservation', 'Education', 'Community', 'Other'], n_samples),
    'ORGANIZATION': np.random.choice(['Trust', 'Association', 'Co-operative', 'Corporation'], n_samples),
    'INCOME_AMT': np.random.choice(['0', '1-9999', '10000-24999', '25000-99999', '100000-499999', '500000-999999', '1M-5M'], n_samples),
    'ASK_AMT': np.random.randint(5000, 200000, n_samples),
    'IS_SUCCESSFUL': np.random.choice([0, 1], n_samples, p=[0.4, 0.6])  # Target variable
}

df = pd.DataFrame(data)
print("Sample data created successfully!")

# Preprocessing
print("\nPreprocessing data...")

# Define features and target
target = df['IS_SUCCESSFUL']
features = df.drop('IS_SUCCESSFUL', axis=1)

# Create a list of categorical variables
categorical_variables = ['APPLICATION_TYPE', 'AFFILIATION', 'CLASSIFICATION', 'USE_CASE', 'ORGANIZATION', 'INCOME_AMT']

# Create a OneHotEncoder instance
enc = OneHotEncoder(sparse=False)

# Encode categorical variables
encoded_categorical_vars = pd.DataFrame(
    enc.fit_transform(features[categorical_variables])
)

# Create column names for encoded variables
encoded_categorical_names = enc.get_feature_names_out(categorical_variables)
encoded_categorical_vars.columns = encoded_categorical_names

# Create numerical feature dataframe
numerical_vars = features.drop(categorical_variables, axis=1)

# Combine numerical and encoded categorical variables
X = pd.concat([numerical_vars.reset_index(drop=True), encoded_categorical_vars], axis=1)
y = target

# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training with {X_train.shape[1]} features after encoding")

# Create a Neural Network Model
def build_model(input_dim, neurons_layer1=80, neurons_layer2=30, dropout_rate=0.2):
    # Create model
    nn = Sequential()
    
    # Add first hidden layer
    nn.add(Dense(units=neurons_layer1, activation='relu', input_dim=input_dim))
    nn.add(BatchNormalization())
    nn.add(Dropout(dropout_rate))
    
    # Add second hidden layer
    nn.add(Dense(units=neurons_layer2, activation='relu'))
    nn.add(BatchNormalization())
    nn.add(Dropout(dropout_rate))
    
    # Add output layer
    nn.add(Dense(units=1, activation='sigmoid'))
    
    # Compile model
    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return nn

# Build the model
model = build_model(X_train.shape[1])

# Define early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Display model summary
model.summary()

# Train the model
print("\nTraining the model...")
n_epochs = 5  # Use a small number for quick run, increase for better results
fit_model = model.fit(
    X_train_scaled,
    y_train,
    epochs=n_epochs,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
print("\nEvaluating the model...")
model_loss, model_accuracy = model.evaluate(X_test_scaled, y_test, verbose=2)
print(f"Loss: {model_loss:.4f}")
print(f"Accuracy: {model_accuracy:.4f}")

# Save model
model.save('charity_model.h5')
print("\nModel saved as 'charity_model.h5'")
