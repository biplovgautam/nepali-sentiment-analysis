#!/usr/bin/env python3
"""
Enhanced Financial Sentiment Analysis ML Pipeline
=================================================

This script provides a complete ML pipeline with:
- Improved data loading and validation
- Advanced preprocessing with before/after visualization
- Intelligent data balancing
- Comprehensive model training and evaluation
- Performance optimization and tuning

Author: AI Assistant
Date: July 20, 2025
"""

import os
import sys
import argparse
import logging
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from utils.loader import load_dataset, validate_dataset, clean_initial_data
from utils.eda import comprehensive_eda, generate_comparison_plots
from utils.preprocess import TextPreprocessor, preprocess_dataset
from utils.vectorizer import create_feature_extractor
from utils.model import SentimentClassifier, train_and_evaluate_model
from utils.evaluator import ModelEvaluator
from config import *

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'ml_pipeline.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class EnhancedMLPipeline:
    """
    Enhanced ML Pipeline for Financial Sentiment Analysis with focus on:
    - Data quality and balancing
    - Preprocessing optimization
    - Model performance
    """
    
    def __init__(self, config=None):
        """Initialize the enhanced ML pipeline."""
        self.config = config or {}
        self.raw_data = None
        self.cleaned_data = None
        self.balanced_data = None
        self.processed_data = None
        self.train_data = None
        self.model = None
        self.vectorizer = None
        self.results = {}
        
        logger.info("Enhanced ML Pipeline initialized")
    
    def load_and_analyze_data(self, dataset_path=DATASET_PATH):
        """Load and perform initial analysis of the dataset."""
        logger.info("Step 1: Loading and analyzing raw data...")
        
        # Load dataset
        self.raw_data = load_dataset(dataset_path)
        self.raw_data = validate_dataset(self.raw_data)
        
        # Generate raw data EDA
        raw_eda_dir = os.path.join(OUTPUT_DIR, 'raw_data_analysis')
        os.makedirs(raw_eda_dir, exist_ok=True)
        
        logger.info("Generating comprehensive EDA for raw data...")
        self.raw_eda_results = comprehensive_eda(
            self.raw_data, 
            output_dir=raw_eda_dir,
            title_prefix="Raw Data"
        )
        
        # Print data overview
        self._print_data_overview(self.raw_data, "Raw Data")
        
        return self.raw_data
    
    def clean_and_balance_data(self):
        """Clean data and apply intelligent balancing."""
        logger.info("Step 2: Cleaning and balancing data...")
        
        # Initial cleaning
        self.cleaned_data = clean_initial_data(self.raw_data)
        
        # Apply intelligent balancing
        self.balanced_data = self._balance_dataset(self.cleaned_data)
        
        # Generate balanced data EDA
        balanced_eda_dir = os.path.join(OUTPUT_DIR, 'balanced_data_analysis')
        os.makedirs(balanced_eda_dir, exist_ok=True)
        
        logger.info("Generating EDA for balanced data...")
        self.balanced_eda_results = comprehensive_eda(
            self.balanced_data,
            output_dir=balanced_eda_dir,
            title_prefix="Balanced Data"
        )
        
        # Print balancing results
        self._print_data_overview(self.balanced_data, "Balanced Data")
        
        return self.balanced_data
    
    def preprocess_data(self):
        """Apply enhanced preprocessing with before/after comparison."""
        logger.info("Step 3: Enhanced text preprocessing...")
        
        # Create preprocessor with optimized config
        preprocessing_config = {
            'lowercase': True,
            'remove_punctuation': False,  # Keep financial punctuation
            'remove_numbers': False,      # Keep numbers for financial context
            'normalize_tickers': True,
            'remove_stopwords': True,
            'min_length': 3,
            'max_length': 1000
        }
        
        preprocessor = TextPreprocessor(preprocessing_config)
        
        # Get original texts
        original_texts = self.balanced_data['Sentence'].tolist()
        
        # Apply preprocessing
        processed_texts = preprocessor.preprocess_texts(original_texts)
        
        # Create processed dataset
        self.processed_data = self.balanced_data.copy()
        self.processed_data['Processed_Sentence'] = processed_texts
        self.processed_data['Original_Length'] = self.processed_data['Sentence'].str.len()
        self.processed_data['Processed_Length'] = self.processed_data['Processed_Sentence'].str.len()
        self.processed_data['Length_Reduction'] = (
            self.processed_data['Original_Length'] - self.processed_data['Processed_Length']
        ) / self.processed_data['Original_Length']
        
        # Generate preprocessing comparison plots
        self._generate_preprocessing_comparison()
        
        # Filter out empty processed texts
        self.processed_data = self.processed_data[
            self.processed_data['Processed_Sentence'].str.len() > 0
        ].reset_index(drop=True)
        
        # Save processed dataset for training
        train_dataset_path = os.path.join(DATA_DIR, 'train_dataset.csv')
        self.processed_data.to_csv(train_dataset_path, index=False)
        logger.info(f"Processed training dataset saved to: {train_dataset_path}")
        
        # Print preprocessing stats
        self._print_preprocessing_stats()
        
        return self.processed_data
    
    def train_and_evaluate_model(self):
        """Train and comprehensively evaluate the model."""
        logger.info("Step 4: Model training and evaluation...")
        
        # Prepare features and labels
        X_text = self.processed_data['Processed_Sentence']
        y = self.processed_data['Sentiment']
        
        # Create vectorizer
        vectorizer_config = {
            'vectorizer_type': self.config.get('vectorizer_type', 'tfidf'),
            'max_features': self.config.get('max_features', 10000),
            'ngram_range': self.config.get('ngram_range', (1, 2)),
            'min_df': self.config.get('min_df', 2),
            'max_df': self.config.get('max_df', 0.95)
        }
        
        self.vectorizer = create_feature_extractor(
            vectorizer_config['vectorizer_type'],
            vectorizer_config
        )
        
        # Transform texts to features
        X = self.vectorizer.fit_transform(X_text)
        
        # Split data
        test_size = self.config.get('test_size', 0.2)
        random_state = self.config.get('random_state', 42)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train model
        model_config = {
            'alpha': self.config.get('alpha', 1.0),
            'fit_prior': self.config.get('fit_prior', True)
        }
        
        self.model, self.results = train_and_evaluate_model(
            X_train, X_test, y_train, y_test, model_config
        )
        
        # Comprehensive evaluation
        evaluator = ModelEvaluator()
        evaluation_dir = os.path.join(OUTPUT_DIR, 'model_evaluation')
        os.makedirs(evaluation_dir, exist_ok=True)
        
        try:
            evaluation_results = evaluator.comprehensive_evaluation(
                self.model, X_test, y_test, output_dir=evaluation_dir
            )
            self.results.update(evaluation_results)
        except Exception as e:
            logger.warning(f"Comprehensive evaluation failed: {e}. Using basic results.")
            # Use basic results if comprehensive evaluation fails
        
        # Save model and vectorizer
        self._save_models()
        
        # Print results
        self._print_results()
        
        return self.results
    
    def _balance_dataset(self, df):
        """Apply intelligent dataset balancing."""
        logger.info("Applying intelligent dataset balancing...")
        
        # Get class distribution
        class_counts = df['Sentiment'].value_counts()
        logger.info(f"Original distribution: {dict(class_counts)}")
        
        # Strategy: Undersample majority class (neutral) to reduce dominance
        # but not too aggressively to maintain information
        neutral_count = class_counts.get('neutral', 0)
        positive_count = class_counts.get('positive', 0)
        negative_count = class_counts.get('negative', 0)
        
        # Target: Make neutral class no more than 1.5x the largest minority class
        max_minority = max(positive_count, negative_count)
        target_neutral = min(neutral_count, int(max_minority * 1.5))
        
        # Undersample neutral class
        neutral_samples = df[df['Sentiment'] == 'neutral'].sample(
            n=target_neutral, random_state=42
        )
        
        # Keep all positive and negative samples
        positive_samples = df[df['Sentiment'] == 'positive']
        negative_samples = df[df['Sentiment'] == 'negative']
        
        # Combine balanced dataset
        balanced_df = pd.concat([
            positive_samples,
            negative_samples,
            neutral_samples
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        new_counts = balanced_df['Sentiment'].value_counts()
        logger.info(f"Balanced distribution: {dict(new_counts)}")
        
        return balanced_df
    
    def _generate_preprocessing_comparison(self):
        """Generate before/after preprocessing comparison plots."""
        comparison_dir = os.path.join(OUTPUT_DIR, 'preprocessing_comparison')
        os.makedirs(comparison_dir, exist_ok=True)
        
        generate_comparison_plots(
            self.processed_data,
            comparison_dir,
            original_col='Sentence',
            processed_col='Processed_Sentence'
        )
    
    def _save_models(self):
        """Save trained models."""
        import joblib
        
        # Save model
        model_path = os.path.join(MODELS_DIR, 'naive_bayes_model.pkl')
        joblib.dump(self.model, model_path)
        
        # Save vectorizer
        vectorizer_path = os.path.join(MODELS_DIR, 'vectorizer.pkl')
        joblib.dump(self.vectorizer, vectorizer_path)
        
        logger.info(f"Models saved to {MODELS_DIR}")
    
    def _print_data_overview(self, df, title):
        """Print data overview."""
        print(f"\n{'='*60}")
        print(f"{title.upper()} OVERVIEW")
        print(f"{'='*60}")
        print(f"Total samples: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print("\nClass distribution:")
        class_counts = df['Sentiment'].value_counts()
        for sentiment, count in class_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {sentiment}: {count} samples ({percentage:.1f}%)")
    
    def _print_preprocessing_stats(self):
        """Print preprocessing statistics."""
        print(f"\n{'='*60}")
        print("PREPROCESSING STATISTICS")
        print(f"{'='*60}")
        
        avg_original = self.processed_data['Original_Length'].mean()
        avg_processed = self.processed_data['Processed_Length'].mean()
        avg_reduction = self.processed_data['Length_Reduction'].mean()
        
        print(f"Average original length: {avg_original:.1f} characters")
        print(f"Average processed length: {avg_processed:.1f} characters")
        print(f"Average length reduction: {avg_reduction:.1%}")
        print(f"Samples ready for training: {len(self.processed_data)}")
    
    def _print_results(self):
        """Print final results."""
        print(f"\n{'='*60}")
        print("FINAL MODEL RESULTS")
        print(f"{'='*60}")
        print(f"Test Accuracy: {self.results.get('accuracy', 0):.4f}")
        print(f"Precision: {self.results.get('precision', 0):.4f}")
        print(f"Recall: {self.results.get('recall', 0):.4f}")
        print(f"F1-Score: {self.results.get('f1_score', 0):.4f}")
        
        if 'cv_scores' in self.results:
            cv_mean = np.mean(self.results['cv_scores'])
            cv_std = np.std(self.results['cv_scores'])
            print(f"Cross-validation: {cv_mean:.4f} ± {cv_std:.4f}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Enhanced Financial Sentiment Analysis ML Pipeline')
    parser.add_argument('--dataset', type=str, default=DATASET_PATH, help='Path to dataset')
    parser.add_argument('--alpha', type=float, default=1.0, help='Naive Bayes smoothing parameter')
    parser.add_argument('--max-features', type=int, default=10000, help='Maximum features for vectorizer')
    parser.add_argument('--vectorizer', choices=['tfidf', 'count'], default='tfidf', help='Vectorizer type')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'alpha': args.alpha,
        'max_features': args.max_features,
        'vectorizer_type': args.vectorizer,
        'test_size': args.test_size,
        'ngram_range': (1, 2),
        'min_df': 2,
        'max_df': 0.95,
        'random_state': 42,
        'fit_prior': True
    }
    
    print("="*80)
    print("ENHANCED FINANCIAL SENTIMENT ANALYSIS ML PIPELINE")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Configuration: {config}")
    print("="*80)
    
    # Run pipeline
    pipeline = EnhancedMLPipeline(config)
    
    try:
        # Execute pipeline steps
        pipeline.load_and_analyze_data(args.dataset)
        pipeline.clean_and_balance_data()
        pipeline.preprocess_data()
        pipeline.train_and_evaluate_model()
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Results saved to: {OUTPUT_DIR}")
        print(f"Models saved to: {MODELS_DIR}")
        print(f"Training dataset: {os.path.join(DATA_DIR, 'train_dataset.csv')}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
