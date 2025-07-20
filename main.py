"""
Main script for Financial Sentiment Analysis using Naïve Bayes.
Controls the entire ML pipeline from data loading to model deployment.
"""

import pandas as pd
import numpy as np
import logging
import os
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Tuple

# Import project modules
from config import *
from utils.loader import load_dataset, validate_dataset
from utils.eda import perform_comprehensive_eda
from utils.preprocess import TextPreprocessor
from utils.tokenizer import create_tokenizer
from utils.vectorizer import create_feature_extractor
from utils.model import create_sentiment_classifier, train_and_evaluate_model
from utils.evaluator import ModelEvaluator

# Set up logging
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG['filename']),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SentimentAnalysisPipeline:
    """
    Complete sentiment analysis pipeline orchestrator.
    """
    
    def __init__(self, config_overrides: Dict[str, Any] = None):
        """
        Initialize the pipeline.
        
        Args:
            config_overrides (Dict[str, Any], optional): Configuration overrides
        """
        self.config = {
            'model': MODEL_CONFIG.copy(),
            'preprocessing': PREPROCESSING_CONFIG.copy(),
            'vectorizer': VECTORIZER_CONFIG.copy(),
            'cv': CV_CONFIG.copy(),
            'plot': PLOT_CONFIG.copy()
        }
        
        # Apply overrides if provided
        if config_overrides:
            for section, overrides in config_overrides.items():
                if section in self.config:
                    self.config[section].update(overrides)
        
        # Initialize components
        self.data = None
        self.preprocessor = None
        self.tokenizer = None
        self.vectorizer = None
        self.model = None
        self.evaluator = None
        
        # Results storage
        self.results = {
            'data_info': {},
            'eda_results': {},
            'preprocessing_results': {},
            'feature_extraction_results': {},
            'model_results': {},
            'evaluation_results': {}
        }
        
        logger.info("SentimentAnalysisPipeline initialized")
    
    def load_data(self, dataset_path: str = DATASET_PATH) -> pd.DataFrame:
        """
        Load and validate the dataset.
        
        Args:
            dataset_path (str): Path to the dataset
            
        Returns:
            pd.DataFrame: Loaded and validated dataset
        """
        logger.info("=" * 60)
        logger.info("STEP 1: DATA LOADING")
        logger.info("=" * 60)
        
        # Load dataset
        self.data = load_dataset(dataset_path)
        
        # Validate dataset
        self.data = validate_dataset(self.data)
        
        # Store data info
        self.results['data_info'] = {
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'class_distribution': self.data['Sentiment'].value_counts().to_dict(),
            'null_values': self.data.isnull().sum().to_dict(),
            'dataset_path': dataset_path
        }
        
        logger.info(f"Dataset loaded successfully. Shape: {self.data.shape}")
        logger.info(f"Class distribution: {self.results['data_info']['class_distribution']}")
        
        return self.data
    
    def perform_eda(self, save_plots: bool = True) -> Dict[str, Any]:
        """
        Perform exploratory data analysis.
        
        Args:
            save_plots (bool): Whether to save EDA plots
            
        Returns:
            Dict[str, Any]: EDA results
        """
        logger.info("=" * 60)
        logger.info("STEP 2: EXPLORATORY DATA ANALYSIS")
        logger.info("=" * 60)
        
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        # Perform comprehensive EDA
        eda_results = perform_comprehensive_eda(
            self.data['Sentence'], 
            self.data['Sentiment'],
            output_dir=OUTPUT_DIR if save_plots else None
        )
        
        self.results['eda_results'] = eda_results
        
        logger.info("EDA completed successfully")
        
        return eda_results
    
    def preprocess_data(self) -> Tuple[pd.Series, pd.Series]:
        """
        Preprocess the text data.
        
        Returns:
            Tuple[pd.Series, pd.Series]: Preprocessed texts and labels
        """
        logger.info("=" * 60)
        logger.info("STEP 3: TEXT PREPROCESSING")
        logger.info("=" * 60)
        
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor(self.config['preprocessing'])
        
        # Preprocess texts
        logger.info("Preprocessing text data...")
        processed_texts = self.preprocessor.preprocess_texts(self.data['Sentence'])
        
        # Store preprocessing results
        self.results['preprocessing_results'] = {
            'original_texts_count': len(self.data['Sentence']),
            'processed_texts_count': len(processed_texts),
            'average_length_before': self.data['Sentence'].str.len().mean(),
            'average_length_after': pd.Series(processed_texts).str.len().mean(),
            'preprocessing_config': self.config['preprocessing']
        }
        
        logger.info("Text preprocessing completed")
        logger.info(f"Average length before: {self.results['preprocessing_results']['average_length_before']:.1f}")
        logger.info(f"Average length after: {self.results['preprocessing_results']['average_length_after']:.1f}")
        
        return pd.Series(processed_texts), self.data['Sentiment']
    
    def extract_features(self, texts: pd.Series, labels: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features using vectorization.
        
        Args:
            texts (pd.Series): Preprocessed texts
            labels (pd.Series): Labels
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Train/test splits
        """
        logger.info("=" * 60)
        logger.info("STEP 4: FEATURE EXTRACTION")
        logger.info("=" * 60)
        
        # Initialize vectorizer
        self.vectorizer = create_feature_extractor(
            vectorizer_type=self.config['vectorizer']['vectorizer_type'],
            config=self.config['vectorizer']
        )
        
        # Split data first
        from sklearn.model_selection import train_test_split
        
        # Check if we have enough samples for stratification
        min_class_count = labels.value_counts().min()
        use_stratify = self.config['model']['stratify'] and min_class_count >= 2
        
        if not use_stratify:
            logger.warning("Disabling stratification due to insufficient samples in some classes")
        
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts, labels,
            test_size=self.config['model']['test_size'],
            random_state=self.config['model']['random_state'],
            stratify=labels if use_stratify else None
        )
        
        # Fit vectorizer on training data and transform both sets
        logger.info("Fitting vectorizer on training data...")
        X_train = self.vectorizer.fit_transform(X_train_text)
        
        logger.info("Transforming test data...")
        X_test = self.vectorizer.transform(X_test_text)
        
        # Store feature extraction results
        self.results['feature_extraction_results'] = {
            'vectorizer_type': self.config['vectorizer']['vectorizer_type'],
            'vocabulary_size': len(self.vectorizer.get_feature_names()),
            'feature_matrix_shape': X_train.shape,
            'train_test_split': {
                'train_size': X_train.shape[0],
                'test_size': X_test.shape[0],
                'test_ratio': self.config['model']['test_size']
            },
            'vectorizer_config': self.config['vectorizer']
        }
        
        logger.info("Feature extraction completed")
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")
        logger.info(f"Vocabulary size: {self.results['feature_extraction_results']['vocabulary_size']}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train: np.ndarray, X_test: np.ndarray, 
                   y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Train and evaluate the sentiment classification model.
        
        Args:
            X_train (np.ndarray): Training features
            X_test (np.ndarray): Test features
            y_train (np.ndarray): Training labels
            y_test (np.ndarray): Test labels
            
        Returns:
            Dict[str, Any]: Model results
        """
        logger.info("=" * 60)
        logger.info("STEP 5: MODEL TRAINING")
        logger.info("=" * 60)
        
        # Train model
        self.model, evaluation_results = train_and_evaluate_model(
            X_train, X_test, y_train, y_test, self.config['model']
        )
        
        # Store model results
        self.results['model_results'] = {
            'model_type': 'MultinomialNB',
            'model_config': self.config['model'],
            'training_accuracy': evaluation_results.get('training_accuracy'),
            'test_accuracy': evaluation_results['accuracy'],
            'cross_validation': evaluation_results.get('cross_validation'),
            'model_info': self.model.get_model_info()
        }
        
        logger.info("Model training completed")
        logger.info(f"Test accuracy: {evaluation_results['accuracy']:.4f}")
        
        return evaluation_results
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, 
                      class_names: list = None) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            class_names (list, optional): Class names
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        logger.info("=" * 60)
        logger.info("STEP 6: MODEL EVALUATION")
        logger.info("=" * 60)
        
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(OUTPUT_DIR)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        # Comprehensive evaluation
        from utils.evaluator import evaluate_model_comprehensive
        
        evaluation_results = evaluate_model_comprehensive(
            y_test, y_pred, y_proba, class_names, "NaiveBayes_FinancialSentiment", OUTPUT_DIR
        )
        
        # Get feature importance
        if hasattr(self.vectorizer, 'get_feature_names'):
            feature_names = self.vectorizer.get_feature_names()
            feature_importance = self.model.get_feature_importance(feature_names, top_n=20)
            evaluation_results['feature_importance'] = feature_importance
        
        self.results['evaluation_results'] = evaluation_results
        
        logger.info("Model evaluation completed")
        logger.info(f"Final accuracy: {evaluation_results['metrics']['accuracy']:.4f}")
        
        return evaluation_results
    
    def save_models(self) -> Dict[str, str]:
        """
        Save trained models and vectorizer.
        
        Returns:
            Dict[str, str]: Paths where models were saved
        """
        logger.info("=" * 60)
        logger.info("STEP 7: SAVING MODELS")
        logger.info("=" * 60)
        
        if self.model is None or self.vectorizer is None:
            raise ValueError("Models must be trained first")
        
        # Save model
        self.model.save_model(MODEL_PATH)
        
        # Save vectorizer
        self.vectorizer.save_vectorizer(VECTORIZER_PATH)
        
        saved_paths = {
            'model': MODEL_PATH,
            'vectorizer': VECTORIZER_PATH
        }
        
        logger.info("Models saved successfully")
        for component, path in saved_paths.items():
            logger.info(f"{component.capitalize()}: {path}")
        
        return saved_paths
    
    def generate_final_report(self) -> str:
        """
        Generate a comprehensive final report.
        
        Returns:
            str: Path to the final report
        """
        logger.info("=" * 60)
        logger.info("STEP 8: GENERATING FINAL REPORT")
        logger.info("=" * 60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(OUTPUT_DIR, f"final_report_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FINANCIAL SENTIMENT ANALYSIS - FINAL REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Data information
            f.write("1. DATASET INFORMATION:\n")
            f.write("-" * 30 + "\n")
            data_info = self.results.get('data_info', {})
            f.write(f"Shape: {data_info.get('shape', 'N/A')}\n")
            f.write(f"Columns: {data_info.get('columns', 'N/A')}\n")
            f.write(f"Class Distribution: {data_info.get('class_distribution', 'N/A')}\n")
            f.write(f"Null Values: {data_info.get('null_values', 'N/A')}\n\n")
            
            # Preprocessing information
            f.write("2. PREPROCESSING RESULTS:\n")
            f.write("-" * 30 + "\n")
            prep_info = self.results.get('preprocessing_results', {})
            f.write(f"Original texts: {prep_info.get('original_texts_count', 'N/A')}\n")
            f.write(f"Processed texts: {prep_info.get('processed_texts_count', 'N/A')}\n")
            f.write(f"Avg length before: {prep_info.get('average_length_before', 'N/A'):.1f}\n")
            f.write(f"Avg length after: {prep_info.get('average_length_after', 'N/A'):.1f}\n\n")
            
            # Feature extraction
            f.write("3. FEATURE EXTRACTION:\n")
            f.write("-" * 30 + "\n")
            feat_info = self.results.get('feature_extraction_results', {})
            f.write(f"Vectorizer: {feat_info.get('vectorizer_type', 'N/A')}\n")
            f.write(f"Vocabulary size: {feat_info.get('vocabulary_size', 'N/A')}\n")
            f.write(f"Feature matrix shape: {feat_info.get('feature_matrix_shape', 'N/A')}\n\n")
            
            # Model results
            f.write("4. MODEL PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            if 'evaluation_results' in self.results and 'metrics' in self.results['evaluation_results']:
                metrics = self.results['evaluation_results']['metrics']
                f.write(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}\n")
                f.write(f"Precision (macro): {metrics.get('precision_macro', 'N/A'):.4f}\n")
                f.write(f"Recall (macro): {metrics.get('recall_macro', 'N/A'):.4f}\n")
                f.write(f"F1-Score (macro): {metrics.get('f1_macro', 'N/A'):.4f}\n")
            
            # Cross-validation
            if 'model_results' in self.results and 'cross_validation' in self.results['model_results']:
                cv = self.results['model_results']['cross_validation']
                f.write(f"CV Mean Score: {cv.get('mean_score', 'N/A'):.4f}\n")
                f.write(f"CV Std Score: {cv.get('std_score', 'N/A'):.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        logger.info(f"Final report saved to: {report_path}")
        return report_path
    
    def run_full_pipeline(self, dataset_path: str = DATASET_PATH, 
                         save_models: bool = True, 
                         generate_report: bool = True) -> Dict[str, Any]:
        """
        Run the complete sentiment analysis pipeline.
        
        Args:
            dataset_path (str): Path to the dataset
            save_models (bool): Whether to save trained models
            generate_report (bool): Whether to generate final report
            
        Returns:
            Dict[str, Any]: Complete pipeline results
        """
        start_time = datetime.now()
        
        logger.info("*" * 80)
        logger.info("STARTING FINANCIAL SENTIMENT ANALYSIS PIPELINE")
        logger.info("*" * 80)
        
        try:
            # Step 1: Load data
            self.load_data(dataset_path)
            
            # Step 2: EDA
            self.perform_eda()
            
            # Step 3: Preprocess
            processed_texts, labels = self.preprocess_data()
            
            # Step 4: Feature extraction
            X_train, X_test, y_train, y_test = self.extract_features(processed_texts, labels)
            
            # Step 5: Train model
            self.train_model(X_train, X_test, y_train, y_test)
            
            # Step 6: Evaluate model
            class_names = sorted(labels.unique())
            self.evaluate_model(X_test, y_test, class_names)
            
            # Step 7: Save models
            if save_models:
                self.save_models()
            
            # Step 8: Generate report
            if generate_report:
                self.generate_final_report()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info("*" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Total execution time: {duration}")
            logger.info("*" * 80)
            
            return {
                'status': 'success',
                'execution_time': duration,
                'results': self.results
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'results': self.results
            }


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Financial Sentiment Analysis Pipeline')
    
    parser.add_argument('--data', type=str, default=DATASET_PATH,
                       help='Path to the dataset CSV file')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save trained models')
    parser.add_argument('--no-report', action='store_true',
                       help='Do not generate final report')
    parser.add_argument('--config', type=str,
                       help='Path to JSON config file for overrides')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Naive Bayes alpha parameter')
    parser.add_argument('--vectorizer', choices=['tfidf', 'count'], default='tfidf',
                       help='Type of vectorizer to use')
    parser.add_argument('--max-features', type=int, default=10000,
                       help='Maximum number of features for vectorizer')
    
    args = parser.parse_args()
    
    # Prepare configuration overrides
    config_overrides = {
        'model': {'alpha': args.alpha},
        'vectorizer': {
            'vectorizer_type': args.vectorizer,
            'max_features': args.max_features
        }
    }
    
    # Load additional config if provided
    if args.config:
        import json
        with open(args.config, 'r') as f:
            additional_config = json.load(f)
            for section, overrides in additional_config.items():
                if section in config_overrides:
                    config_overrides[section].update(overrides)
                else:
                    config_overrides[section] = overrides
    
    # Initialize and run pipeline
    pipeline = SentimentAnalysisPipeline(config_overrides)
    
    results = pipeline.run_full_pipeline(
        dataset_path=args.data,
        save_models=not args.no_save,
        generate_report=not args.no_report
    )
    
    if results['status'] == 'success':
        print(f"\n✅ Pipeline completed successfully in {results['execution_time']}")
        if 'evaluation_results' in results['results']:
            accuracy = results['results']['evaluation_results']['metrics']['accuracy']
            print(f"📊 Final model accuracy: {accuracy:.4f}")
    else:
        print(f"\n❌ Pipeline failed: {results['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
