"""
Data loading and initial cleaning utilities for Financial Sentiment Analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
from config import DATASET_PATH, PREPROCESSING_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(filepath: str = DATASET_PATH) -> pd.DataFrame:
    """
    Load the financial sentiment dataset from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Dataset file not found at {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that the dataset has required columns and proper format.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Validated dataset
    """
    required_columns = ['Sentence', 'Sentiment']
    
    # Check if required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    logger.info("Dataset validation passed")
    return df


def clean_initial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform initial data cleaning operations.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    initial_shape = df.shape
    
    # Remove rows with missing values in critical columns
    df = df.dropna(subset=['Sentence', 'Sentiment'])
    
    # Remove empty sentences
    df = df[df['Sentence'].str.strip() != '']
    
    # Filter by sentence length
    min_length = PREPROCESSING_CONFIG['min_length']
    max_length = PREPROCESSING_CONFIG['max_length']
    
    df = df[(df['Sentence'].str.len() >= min_length) & 
            (df['Sentence'].str.len() <= max_length)]
    
    # Remove duplicate sentences
    df = df.drop_duplicates(subset=['Sentence'])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    logger.info(f"Data cleaning completed. Shape changed from {initial_shape} to {df.shape}")
    
    return df


def standardize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize sentiment labels to consistent format.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with standardized labels
    """
    # Convert to lowercase and strip whitespace
    df['Sentiment'] = df['Sentiment'].str.lower().str.strip()
    
    # Map variations to standard labels
    label_mapping = {
        'pos': 'positive',
        'positive': 'positive',
        'bull': 'positive',
        'bullish': 'positive',
        
        'neg': 'negative',
        'negative': 'negative',
        'bear': 'negative',
        'bearish': 'negative',
        
        'neu': 'neutral',
        'neutral': 'neutral',
        'hold': 'neutral'
    }
    
    df['Sentiment'] = df['Sentiment'].map(label_mapping)
    
    # Remove rows with unmapped labels
    df = df.dropna(subset=['Sentiment'])
    
    logger.info(f"Label standardization completed. Unique labels: {df['Sentiment'].unique()}")
    
    return df


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Get comprehensive information about the dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Dataset information
    """
    info = {
        'total_samples': len(df),
        'features': list(df.columns),
        'sentiment_distribution': df['Sentiment'].value_counts().to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'sentence_length_stats': {
            'mean': df['Sentence'].str.len().mean(),
            'std': df['Sentence'].str.len().std(),
            'min': df['Sentence'].str.len().min(),
            'max': df['Sentence'].str.len().max()
        }
    }
    
    return info


def split_features_labels(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Split dataset into features (sentences) and labels (sentiments).
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        Tuple[pd.Series, pd.Series]: Features and labels
    """
    X = df['Sentence']
    y = df['Sentiment']
    
    logger.info(f"Dataset split into {len(X)} samples with {len(y.unique())} classes")
    
    return X, y


def load_and_prepare_data(filepath: str = DATASET_PATH) -> Tuple[pd.Series, pd.Series, dict]:
    """
    Complete pipeline to load and prepare data for analysis.
    
    Args:
        filepath (str): Path to the dataset file
        
    Returns:
        Tuple[pd.Series, pd.Series, dict]: Features, labels, and dataset info
    """
    logger.info("Starting data loading and preparation pipeline")
    
    # Load dataset
    df = load_dataset(filepath)
    
    # Validate dataset structure
    df = validate_dataset(df)
    
    # Clean initial data
    df = clean_initial_data(df)
    
    # Standardize labels
    df = standardize_labels(df)
    
    # Get dataset information
    dataset_info = get_dataset_info(df)
    
    # Split into features and labels
    X, y = split_features_labels(df)
    
    logger.info("Data loading and preparation completed successfully")
    
    return X, y, dataset_info


if __name__ == "__main__":
    # Test the loader functions
    try:
        X, y, info = load_and_prepare_data()
        print("Dataset Info:")
        for key, value in info.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error: {e}")