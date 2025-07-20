"""
Configuration file for Financial Sentiment Analysis project.
Contains all paths, parameters, and settings used across modules.
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File paths
DATASET_PATH = os.path.join(DATA_DIR, 'financial_sentiment.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'naive_bayes_model.pkl')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'vectorizer.pkl')

# Model parameters
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True,
    'alpha': 1.0  # Naive Bayes smoothing parameter
}

# Text preprocessing parameters
PREPROCESSING_CONFIG = {
    'remove_stopwords': True,
    'remove_punctuation': True,
    'remove_numbers': True,
    'normalize_tickers': True,
    'min_length': 3,
    'max_length': 1000
}

# Feature extraction parameters
VECTORIZER_CONFIG = {
    'vectorizer_type': 'tfidf',  # 'tfidf' or 'count'
    'max_features': 10000,
    'ngram_range': (1, 2),
    'min_df': 1,  # Changed from 2 to 1 for small datasets
    'max_df': 0.95,
    'lowercase': True,
    'stop_words': 'english'
}

# Visualization parameters
PLOT_CONFIG = {
    'figure_size': (12, 8),
    'color_palette': ['#FF6B6B', '#4ECDC4', '#45B7D1'],
    'save_plots': True,
    'plot_format': 'png',
    'dpi': 300
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': os.path.join(OUTPUT_DIR, 'sentiment_analysis.log')
}

# Cross-validation parameters
CV_CONFIG = {
    'cv_folds': 5,
    'scoring': 'accuracy',
    'n_jobs': -1
}

# Django web app configuration
DJANGO_CONFIG = {
    'host': '127.0.0.1',
    'port': 8000,
    'debug': True
}

# Stock ticker patterns (for normalization)
TICKER_PATTERNS = [
    r'\$[A-Z]{2,5}',  # $AAPL, $MSFT
    r'[A-Z]{2,5}\s*\$',  # AAPL$, MSFT $
]