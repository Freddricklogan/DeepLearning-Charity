"""
Data Preprocessing Module for Charity Success Prediction

This module provides functions for loading, cleaning, encoding, and splitting
charity application data for neural network training. All preprocessing logic
is consistent with the root model.py pipeline.

Author: Freddrick Logan
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
import os


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data(filepath):
    """
    Load charity application data from a CSV file.

    Parameters:
        filepath (str): Path to the charity data CSV file.

    Returns:
        pandas.DataFrame: DataFrame containing the charity application data,
                          or None if loading fails.
    """
    print(f"Loading charity application data from: {filepath}")

    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded {len(df)} charity application records.")

        # Display basic information
        print(f"\nData Overview:")
        print(f"  Shape:   {df.shape}")
        print(f"  Columns: {', '.join(df.columns)}")

        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            print(f"  Warning: Found {missing_values} missing values in the dataset.")
        else:
            print("  No missing values found in the dataset.")

        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        return None


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def drop_id_columns(df, columns_to_drop=None):
    """
    Drop non-beneficial identifier columns from the DataFrame.

    Parameters:
        df (pandas.DataFrame): Input DataFrame.
        columns_to_drop (list): Column names to drop.  Defaults to
                                ``['EIN', 'NAME']``.

    Returns:
        pandas.DataFrame: DataFrame without the specified columns.
    """
    if columns_to_drop is None:
        columns_to_drop = ['EIN', 'NAME']

    existing = [col for col in columns_to_drop if col in df.columns]
    if existing:
        print(f"Dropping non-beneficial ID columns: {', '.join(existing)}")
        df = df.drop(columns=existing)
    return df


def bin_rare_categories(df, column, threshold):
    """
    Replace rare categorical values in *column* with ``'Other'``.

    Any category whose count is below *threshold* is replaced.

    Parameters:
        df (pandas.DataFrame): Input DataFrame (modified in place).
        column (str): Name of the column to bin.
        threshold (int): Minimum count required to keep a category.

    Returns:
        pandas.DataFrame: DataFrame with rare values replaced.
    """
    value_counts = df[column].value_counts()
    rare_values = value_counts[value_counts < threshold].index
    df[column] = df[column].replace(rare_values, 'Other')
    print(f"  {column}: binned {len(rare_values)} rare categories "
          f"(threshold={threshold}), {df[column].nunique()} unique values remain")
    return df


# ---------------------------------------------------------------------------
# Core preprocessing pipeline
# ---------------------------------------------------------------------------

def clean_data(df, target_column='IS_SUCCESSFUL',
               application_type_threshold=500,
               classification_threshold=1000):
    """
    Clean the raw charity data:
      1. Drop non-beneficial ID columns (EIN, NAME).
      2. Bin rare categorical values for APPLICATION_TYPE and CLASSIFICATION.

    Parameters:
        df (pandas.DataFrame): Raw charity data.
        target_column (str): Name of the target column.
        application_type_threshold (int): Minimum count for APPLICATION_TYPE.
        classification_threshold (int): Minimum count for CLASSIFICATION.

    Returns:
        tuple: (X DataFrame without target, y Series/array of targets)
    """
    df_processed = df.copy()

    # Extract target
    y = df_processed[target_column].values

    # Drop ID columns and target
    columns_to_drop = ['EIN', 'NAME', target_column]
    df_processed = df_processed.drop(
        columns=[col for col in columns_to_drop if col in df_processed.columns]
    )

    print(f"\nFeatures used for prediction: {', '.join(df_processed.columns)}")

    # Bin rare categorical values
    print("\nBinning rare categorical values:")
    if 'APPLICATION_TYPE' in df_processed.columns:
        df_processed = bin_rare_categories(
            df_processed, 'APPLICATION_TYPE', application_type_threshold
        )
    if 'CLASSIFICATION' in df_processed.columns:
        df_processed = bin_rare_categories(
            df_processed, 'CLASSIFICATION', classification_threshold
        )

    return df_processed, y


def encode_features(X_train, X_test):
    """
    Encode features for neural network consumption.

    - Numerical features are scaled using ``StandardScaler``.
    - Categorical features are one-hot encoded with ``OneHotEncoder``.

    Parameters:
        X_train (pandas.DataFrame): Training feature DataFrame.
        X_test (pandas.DataFrame): Testing feature DataFrame.

    Returns:
        tuple: (X_train_processed, X_test_processed, feature_names, preprocessors)
            - X_train_processed (numpy.ndarray): Scaled/encoded training features.
            - X_test_processed (numpy.ndarray): Scaled/encoded testing features.
            - feature_names (list): Names of all processed features.
            - preprocessors (dict): Fitted transformer objects for later reuse.
    """
    # Identify feature types
    cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"\nCategorical features ({len(cat_features)}): {', '.join(cat_features)}")
    print(f"Numerical features  ({len(num_features)}): {', '.join(num_features)}")

    # --- Numerical scaling ---
    num_transformer = StandardScaler()
    if num_features:
        X_train_num = num_transformer.fit_transform(X_train[num_features])
        X_test_num = num_transformer.transform(X_test[num_features])
    else:
        X_train_num = np.empty((X_train.shape[0], 0))
        X_test_num = np.empty((X_test.shape[0], 0))

    # --- Categorical encoding ---
    cat_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if cat_features:
        X_train_cat = cat_transformer.fit_transform(X_train[cat_features])
        X_test_cat = cat_transformer.transform(X_test[cat_features])
        encoded_feature_names = cat_transformer.get_feature_names_out(cat_features).tolist()
    else:
        X_train_cat = np.empty((X_train.shape[0], 0))
        X_test_cat = np.empty((X_test.shape[0], 0))
        encoded_feature_names = []

    # Combine
    X_train_processed = np.hstack((X_train_num, X_train_cat))
    X_test_processed = np.hstack((X_test_num, X_test_cat))

    feature_names = num_features + encoded_feature_names

    print(f"\nProcessed feature matrix shape: {X_train_processed.shape} "
          f"({len(feature_names)} features)")

    preprocessors = {
        'num_transformer': num_transformer,
        'cat_transformer': cat_transformer,
        'num_features': num_features,
        'cat_features': cat_features,
    }

    return X_train_processed, X_test_processed, feature_names, preprocessors


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split features and target into training and testing sets.

    Uses stratified splitting to maintain class balance.

    Parameters:
        X (pandas.DataFrame or numpy.ndarray): Feature data.
        y (numpy.ndarray): Target labels.
        test_size (float): Fraction of data reserved for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Testing  set shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Full preprocessing pipeline (mirrors root model.py preprocess_data())
# ---------------------------------------------------------------------------

def preprocess_data(df, target_column='IS_SUCCESSFUL', test_size=0.2,
                    random_state=42, application_type_threshold=500,
                    classification_threshold=1000):
    """
    Run the complete preprocessing pipeline on raw charity data.

    Steps:
      1. Clean data (drop IDs, bin rare categories).
      2. Split into train / test.
      3. Encode features (scale numericals, one-hot encode categoricals).

    Parameters:
        df (pandas.DataFrame): Raw charity application data.
        target_column (str): Target column name.
        test_size (float): Fraction of data for testing.
        random_state (int): Random seed.
        application_type_threshold (int): Bin threshold for APPLICATION_TYPE.
        classification_threshold (int): Bin threshold for CLASSIFICATION.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, preprocessors)
    """
    print("=" * 60)
    print("Running full preprocessing pipeline")
    print("=" * 60)

    # Step 1 -- clean
    X, y = clean_data(
        df,
        target_column=target_column,
        application_type_threshold=application_type_threshold,
        classification_threshold=classification_threshold,
    )

    # Step 2 -- split
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    # Step 3 -- encode
    X_train_processed, X_test_processed, feature_names, preprocessors = encode_features(
        X_train, X_test
    )

    return X_train_processed, X_test_processed, y_train, y_test, feature_names, preprocessors


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_preprocessors(preprocessors, filepath):
    """
    Serialize preprocessing objects to disk with pickle.

    Parameters:
        preprocessors (dict): Dictionary of fitted transformers.
        filepath (str): Output file path (e.g. ``'models/preprocessors.pkl'``).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(preprocessors, f)
    print(f"Preprocessors saved to {filepath}")


def load_preprocessors(filepath):
    """
    Load previously saved preprocessing objects.

    Parameters:
        filepath (str): Path to the pickle file.

    Returns:
        dict: Dictionary of fitted transformers.
    """
    with open(filepath, 'rb') as f:
        preprocessors = pickle.load(f)
    print(f"Preprocessors loaded from {filepath}")
    return preprocessors


def transform_new_data(new_data, preprocessors):
    """
    Apply saved preprocessing transformations to new / unseen data.

    Parameters:
        new_data (pandas.DataFrame): New data to transform.
        preprocessors (dict): Fitted transformer objects.

    Returns:
        numpy.ndarray: Preprocessed feature matrix ready for prediction.
    """
    num_transformer = preprocessors['num_transformer']
    cat_transformer = preprocessors['cat_transformer']
    num_features = preprocessors['num_features']
    cat_features = preprocessors['cat_features']

    num_data = (num_transformer.transform(new_data[num_features])
                if num_features else np.empty((new_data.shape[0], 0)))
    cat_data = (cat_transformer.transform(new_data[cat_features])
                if cat_features else np.empty((new_data.shape[0], 0)))

    return np.hstack((num_data, cat_data))


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), 'charity_data.csv'
    )

    df = load_data(data_path)
    if df is not None:
        X_train, X_test, y_train, y_test, feature_names, preprocessors = preprocess_data(df)
        print(f"\nPreprocessing complete.  Training samples: {X_train.shape[0]}, "
              f"Test samples: {X_test.shape[0]}, Features: {len(feature_names)}")
