"""
Enhanced Financial Sentiment Analysis Django App with Advanced Features
- Data balancing to reduce neutral class overfitting
- Model parameter history and optimization
- Comprehensive EDA before/after preprocessing
- Best model tracking and parameter management
"""

import os
import sys
import json
import joblib
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import base64
from io import BytesIO
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from config import MODEL_PATH, VECTORIZER_PATH, DATA_DIR, OUTPUT_DIR, DATASET_PATH

# Import utility modules
from utils.loader import load_dataset, validate_dataset
from utils.eda import perform_comprehensive_eda
from utils.preprocess import TextPreprocessor
from utils.vectorizer import create_feature_extractor
from utils.model import create_sentiment_classifier, train_and_evaluate_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create model history directory
MODEL_HISTORY_DIR = os.path.join(OUTPUT_DIR, 'model_history')
os.makedirs(MODEL_HISTORY_DIR, exist_ok=True)


class EnhancedFinancialSentimentAnalyzer:
    """Enhanced ML pipeline with advanced preprocessing and model management."""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.model_info = {}
        self.eda_results = {}
        self.evaluation_plots = {}
        self.is_trained = False
        self.model_history = []
        self.raw_data = None
        self.processed_data = None
        self.load_model_history()
        self.load_or_train_model()
    
    def load_model_history(self):
        """Load model training history."""
        history_file = os.path.join(MODEL_HISTORY_DIR, 'model_history.json')
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                self.model_history = json.load(f)
        else:
            self.model_history = []
    
    def save_model_history(self, params, results):
        """Save model training results to history."""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'parameters': params,
            'results': results,
            'model_id': len(self.model_history) + 1
        }
        self.model_history.append(history_entry)
        
        # Save to file
        history_file = os.path.join(MODEL_HISTORY_DIR, 'model_history.json')
        with open(history_file, 'w') as f:
            json.dump(self.model_history, f, indent=2)
    
    def get_best_model_params(self):
        """Get parameters of the best performing model."""
        if not self.model_history:
            return None
        
        best_model = max(self.model_history, key=lambda x: x['results'].get('accuracy', 0))
        return best_model
    
    def analyze_raw_data(self):
        """Comprehensive analysis of raw data."""
        logger.info("Analyzing raw financial sentiment data...")
        
        # Load raw data
        self.raw_data = load_dataset(DATASET_PATH)
        self.raw_data = validate_dataset(self.raw_data)
        
        # Basic statistics
        total_samples = len(self.raw_data)
        class_counts = self.raw_data['Sentiment'].value_counts()
        
        logger.info(f"Dataset contains {total_samples} samples")
        for sentiment, count in class_counts.items():
            percentage = (count / total_samples) * 100
            logger.info(f"{sentiment}: {count} samples ({percentage:.1f}%)")
        
        # Generate raw data EDA
        raw_eda_results = perform_comprehensive_eda(
            self.raw_data['Sentence'], 
            self.raw_data['Sentiment'],
            output_dir=os.path.join(OUTPUT_DIR, 'raw_data_eda')
        )
        
        # Text length analysis
        self.raw_data['text_length'] = self.raw_data['Sentence'].str.len()
        self.raw_data['word_count'] = self.raw_data['Sentence'].str.split().str.len()
        
        self.generate_data_quality_plots()
        
        return {
            'total_samples': total_samples,
            'class_distribution': class_counts.to_dict(),
            'class_percentages': (class_counts / total_samples * 100).to_dict(),
            'avg_text_length': self.raw_data['text_length'].mean(),
            'avg_word_count': self.raw_data['word_count'].mean(),
            'eda_results': raw_eda_results
        }
    
    def generate_data_quality_plots(self):
        """Generate data quality and distribution plots."""
        # Text length distribution by sentiment
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        for sentiment in self.raw_data['Sentiment'].unique():
            data = self.raw_data[self.raw_data['Sentiment'] == sentiment]['text_length']
            plt.hist(data, alpha=0.6, label=sentiment, bins=30)
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Frequency')
        plt.title('Text Length Distribution by Sentiment')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        sns.boxplot(data=self.raw_data, x='Sentiment', y='word_count')
        plt.title('Word Count by Sentiment')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        class_counts = self.raw_data['Sentiment'].value_counts()
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', colors=colors)
        plt.title('Class Distribution (Raw Data)')
        
        plt.subplot(2, 2, 4)
        # Sample lengths by sentiment
        lengths_by_sentiment = self.raw_data.groupby('Sentiment')['text_length'].describe()
        lengths_by_sentiment[['mean', '50%', 'max']].plot(kind='bar')
        plt.title('Text Length Statistics by Sentiment')
        plt.ylabel('Characters')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        quality_path = os.path.join(OUTPUT_DIR, 'data_quality_analysis.png')
        plt.savefig(quality_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return self.encode_image_to_base64(quality_path)
    
    def balance_dataset(self, balance_strategy='undersample_neutral'):
        """Balance the dataset to reduce neutral class overfitting."""
        logger.info("Balancing dataset to reduce neutral class dominance...")
        
        # Get class counts
        class_counts = self.raw_data['Sentiment'].value_counts()
        logger.info(f"Original distribution: {class_counts.to_dict()}")
        
        if balance_strategy == 'undersample_neutral':
            # Reduce neutral samples to match the average of positive and negative
            pos_count = class_counts.get('positive', 0)
            neg_count = class_counts.get('negative', 0)
            target_neutral_count = int((pos_count + neg_count) / 2 * 1.2)  # Slightly more than average
            
            # Separate by class
            positive_df = self.raw_data[self.raw_data['Sentiment'] == 'positive']
            negative_df = self.raw_data[self.raw_data['Sentiment'] == 'negative']
            neutral_df = self.raw_data[self.raw_data['Sentiment'] == 'neutral']
            
            # Undersample neutral class
            neutral_balanced = resample(neutral_df, 
                                      replace=False, 
                                      n_samples=min(target_neutral_count, len(neutral_df)),
                                      random_state=42)
            
            # Combine balanced dataset
            balanced_data = pd.concat([positive_df, negative_df, neutral_balanced])
            balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
            
        elif balance_strategy == 'oversample_minority':
            # Oversample minority classes to match majority
            max_count = class_counts.max()
            
            balanced_frames = []
            for sentiment in class_counts.index:
                class_df = self.raw_data[self.raw_data['Sentiment'] == sentiment]
                if len(class_df) < max_count:
                    # Oversample
                    oversampled = resample(class_df, 
                                         replace=True, 
                                         n_samples=max_count,
                                         random_state=42)
                    balanced_frames.append(oversampled)
                else:
                    balanced_frames.append(class_df)
            
            balanced_data = pd.concat(balanced_frames)
            balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        new_counts = balanced_data['Sentiment'].value_counts()
        logger.info(f"Balanced distribution: {new_counts.to_dict()}")
        
        # Generate comparison plot
        self.generate_balance_comparison_plot(class_counts, new_counts)
        
        return balanced_data
    
    def generate_balance_comparison_plot(self, original_counts, balanced_counts):
        """Generate before/after balancing comparison plot."""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        plt.pie(original_counts.values, labels=original_counts.index, autopct='%1.1f%%', colors=colors)
        plt.title('Original Class Distribution')
        
        plt.subplot(1, 2, 2)
        plt.pie(balanced_counts.values, labels=balanced_counts.index, autopct='%1.1f%%', colors=colors)
        plt.title('Balanced Class Distribution')
        
        plt.tight_layout()
        balance_path = os.path.join(OUTPUT_DIR, 'class_balance_comparison.png')
        plt.savefig(balance_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return self.encode_image_to_base64(balance_path)
    
    def enhanced_preprocessing(self, data):
        """Enhanced preprocessing with financial text specific improvements."""
        logger.info("Applying enhanced preprocessing for financial texts...")
        
        # Initialize preprocessor with financial-specific settings
        preprocessor = TextPreprocessor({
            'lowercase': True,
            'remove_punctuation': False,  # Keep some punctuation for financial context
            'remove_stopwords': True,
            'lemmatization': True,
            'remove_short_words': True,
            'normalize_financial': True,
            'handle_contractions': True,
            'remove_urls': True,
            'remove_mentions': True,
            'remove_hashtags': False  # Keep hashtags as they might be relevant
        })
        
        # Process texts
        processed_texts = preprocessor.preprocess_texts(data['Sentence'])
        
        # Create processed dataframe
        processed_df = data.copy()
        processed_df['Processed_Sentence'] = processed_texts
        processed_df['processed_length'] = processed_df['Processed_Sentence'].str.len()
        processed_df['processed_word_count'] = processed_df['Processed_Sentence'].str.split().str.len()
        
        # Generate preprocessing comparison plots
        self.generate_preprocessing_comparison_plots(data, processed_df)
        
        self.processed_data = processed_df
        return processed_texts
    
    def generate_preprocessing_comparison_plots(self, original_df, processed_df):
        """Generate before/after preprocessing comparison."""
        plt.figure(figsize=(15, 10))
        
        # Text length comparison
        plt.subplot(2, 3, 1)
        plt.hist(original_df['Sentence'].str.len(), alpha=0.6, label='Original', bins=30)
        plt.hist(processed_df['processed_length'], alpha=0.6, label='Processed', bins=30)
        plt.xlabel('Text Length')
        plt.ylabel('Frequency')
        plt.title('Text Length: Before vs After Preprocessing')
        plt.legend()
        
        # Word count comparison
        plt.subplot(2, 3, 2)
        original_word_counts = original_df['Sentence'].str.split().str.len()
        plt.hist(original_word_counts, alpha=0.6, label='Original', bins=30)
        plt.hist(processed_df['processed_word_count'], alpha=0.6, label='Processed', bins=30)
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.title('Word Count: Before vs After Preprocessing')
        plt.legend()
        
        # Average length by sentiment - original
        plt.subplot(2, 3, 3)
        orig_by_sentiment = original_df.groupby('Sentiment')['Sentence'].apply(lambda x: x.str.len().mean())
        orig_by_sentiment.plot(kind='bar', color='skyblue')
        plt.title('Average Text Length by Sentiment (Original)')
        plt.ylabel('Characters')
        plt.xticks(rotation=45)
        
        # Average length by sentiment - processed
        plt.subplot(2, 3, 4)
        proc_by_sentiment = processed_df.groupby('Sentiment')['processed_length'].mean()
        proc_by_sentiment.plot(kind='bar', color='lightgreen')
        plt.title('Average Text Length by Sentiment (Processed)')
        plt.ylabel('Characters')
        plt.xticks(rotation=45)
        
        # Sample texts comparison
        plt.subplot(2, 3, 5)
        reduction_ratio = 1 - (processed_df['processed_length'].mean() / original_df['Sentence'].str.len().mean())
        plt.text(0.1, 0.7, f"Average Length Reduction: {reduction_ratio:.1%}", fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        plt.text(0.1, 0.5, f"Original Avg: {original_df['Sentence'].str.len().mean():.0f} chars", fontsize=10)
        plt.text(0.1, 0.3, f"Processed Avg: {processed_df['processed_length'].mean():.0f} chars", fontsize=10)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title('Preprocessing Impact Summary')
        plt.axis('off')
        
        # Word frequency comparison (top 10 words)
        plt.subplot(2, 3, 6)
        # Get most common words from processed texts
        all_processed_text = ' '.join(processed_df['Processed_Sentence'].dropna())
        word_freq = pd.Series(all_processed_text.split()).value_counts().head(10)
        word_freq.plot(kind='bar')
        plt.title('Top 10 Words After Preprocessing')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        preprocess_path = os.path.join(OUTPUT_DIR, 'preprocessing_comparison.png')
        plt.savefig(preprocess_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return self.encode_image_to_base64(preprocess_path)
    
    def train_model_with_params(self, model_params):
        """Train model with specific parameters."""
        logger.info(f"Training model with parameters: {model_params}")
        
        # Analyze raw data
        raw_analysis = self.analyze_raw_data()
        
        # Balance dataset
        balanced_data = self.balance_dataset(model_params.get('balance_strategy', 'undersample_neutral'))
        
        # Enhanced preprocessing
        processed_texts = self.enhanced_preprocessing(balanced_data)
        
        # Split data
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            processed_texts, balanced_data['Sentiment'],
            test_size=model_params.get('test_size', 0.2), 
            random_state=model_params.get('random_state', 42), 
            stratify=balanced_data['Sentiment']
        )
        
        # Create vectorizer with parameters
        vectorizer_config = {
            'vectorizer_type': model_params.get('vectorizer_type', 'tfidf'),
            'max_features': model_params.get('max_features', 10000),
            'ngram_range': tuple(model_params.get('ngram_range', [1, 2])),
            'min_df': model_params.get('min_df', 1),
            'max_df': model_params.get('max_df', 0.95)
        }
        
        self.vectorizer = create_feature_extractor(
            vectorizer_config['vectorizer_type'],
            vectorizer_config
        )
        
        # Transform data
        X_train = self.vectorizer.fit_transform(X_train_text)
        X_test = self.vectorizer.transform(X_test_text)
        
        # Train model
        nb_config = {
            'alpha': model_params.get('alpha', 1.0),
            'fit_prior': model_params.get('fit_prior', True)
        }
        
        self.model, evaluation_results = train_and_evaluate_model(
            X_train, X_test, y_train, y_test, nb_config
        )
        
        # Generate comprehensive evaluation
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        # Calculate detailed metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        detailed_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'class_names': self.model.classes_.tolist(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'raw_data_analysis': raw_analysis
        }
        
        # Generate evaluation plots
        self.generate_comprehensive_evaluation_plots(y_test, y_pred, y_proba, detailed_results)
        
        # Save model and results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(MODEL_HISTORY_DIR, f"model_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model files
        joblib.dump(self.model, os.path.join(model_dir, 'model.pkl'))
        joblib.dump(self.vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))
        
        # Save current best model as main model
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.vectorizer, VECTORIZER_PATH)
        
        # Update model info
        self.model_info = {
            'status': 'trained',
            'model_type': 'Naive Bayes',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classes': self.model.classes_.tolist(),
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'parameters': model_params,
            'model_path': model_dir
        }
        
        # Save to history
        self.save_model_history(model_params, detailed_results)
        
        self.is_trained = True
        logger.info(f"Model trained successfully. Accuracy: {accuracy:.4f}")
        
        return detailed_results
    
    def generate_comprehensive_evaluation_plots(self, y_test, y_pred, y_proba, results):
        """Generate comprehensive evaluation visualizations."""
        plt.style.use('default')
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Confusion Matrix
        plt.subplot(3, 4, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.model.classes_, 
                   yticklabels=self.model.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 2. Classification Report Heatmap
        plt.subplot(3, 4, 2)
        report = results['classification_report']
        metrics_df = pd.DataFrame({
            'Precision': [report[cls]['precision'] for cls in self.model.classes_],
            'Recall': [report[cls]['recall'] for cls in self.model.classes_],
            'F1-Score': [report[cls]['f1-score'] for cls in self.model.classes_]
        }, index=self.model.classes_)
        sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn')
        plt.title('Performance Metrics by Class')
        
        # 3. Class-wise Performance Bar Chart
        plt.subplot(3, 4, 3)
        metrics_df.plot(kind='bar', ax=plt.gca())
        plt.title('Detailed Metrics by Class')
        plt.ylabel('Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        
        # 4. Prediction Confidence Distribution
        plt.subplot(3, 4, 4)
        confidence_scores = np.max(y_proba, axis=1)
        plt.hist(confidence_scores, bins=20, alpha=0.7, color='skyblue')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Prediction Confidence Distribution')
        plt.axvline(confidence_scores.mean(), color='red', linestyle='--', label=f'Mean: {confidence_scores.mean():.3f}')
        plt.legend()
        
        # 5. Class Distribution in Test Set
        plt.subplot(3, 4, 5)
        test_dist = pd.Series(y_test).value_counts()
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        plt.pie(test_dist.values, labels=test_dist.index, autopct='%1.1f%%', colors=colors)
        plt.title('Test Set Class Distribution')
        
        # 6. Model Performance Summary
        plt.subplot(3, 4, 6)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [results['accuracy'], results['precision'], results['recall'], results['f1_score']]
        bars = plt.bar(metrics, values, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
        plt.title('Overall Model Performance')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 7. Feature Importance (Top 20 words)
        plt.subplot(3, 4, 7)
        if hasattr(self.model, 'feature_log_prob_'):
            feature_names = self.vectorizer.get_feature_names_out()
            # Get importance for positive class
            pos_idx = list(self.model.classes_).index('positive') if 'positive' in self.model.classes_ else 0
            importance = self.model.feature_log_prob_[pos_idx]
            top_indices = importance.argsort()[-20:]
            top_features = [feature_names[i] for i in top_indices]
            top_importance = importance[top_indices]
            
            plt.barh(range(len(top_features)), top_importance)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Log Probability')
            plt.title('Top 20 Features (Positive Class)')
        
        # 8. Training Data Statistics
        plt.subplot(3, 4, 8)
        stats_data = {
            'Training Samples': results['training_samples'],
            'Test Samples': results['test_samples'],
            'Vocabulary Size': results['vocabulary_size'],
            'Classes': len(self.model.classes_)
        }
        plt.text(0.1, 0.8, "Training Statistics:", fontsize=14, weight='bold')
        y_pos = 0.6
        for key, value in stats_data.items():
            plt.text(0.1, y_pos, f"{key}: {value:,}", fontsize=12)
            y_pos -= 0.1
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        
        # 9. Error Analysis
        plt.subplot(3, 4, 9)
        misclassified = y_test != y_pred
        error_by_class = pd.DataFrame({'True': y_test, 'Predicted': y_pred, 'Error': misclassified})
        error_counts = error_by_class.groupby('True')['Error'].sum()
        error_counts.plot(kind='bar', color='salmon')
        plt.title('Misclassifications by True Class')
        plt.ylabel('Number of Errors')
        plt.xticks(rotation=45)
        
        # 10. Precision-Recall by Class
        plt.subplot(3, 4, 10)
        precisions = [report[cls]['precision'] for cls in self.model.classes_]
        recalls = [report[cls]['recall'] for cls in self.model.classes_]
        plt.scatter(recalls, precisions, s=100, alpha=0.7)
        for i, cls in enumerate(self.model.classes_):
            plt.annotate(cls, (recalls[i], precisions[i]), xytext=(5, 5), textcoords='offset points')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision vs Recall by Class')
        plt.grid(True, alpha=0.3)
        
        # 11. Model Complexity Metrics
        plt.subplot(3, 4, 11)
        complexity_data = {
            'Vocabulary': results['vocabulary_size'],
            'Parameters': results['vocabulary_size'] * len(self.model.classes_),
            'Training Size': results['training_samples']
        }
        keys = list(complexity_data.keys())
        values = list(complexity_data.values())
        plt.bar(keys, values, color='lightcoral')
        plt.title('Model Complexity')
        plt.ylabel('Count (log scale)')
        plt.yscale('log')
        plt.xticks(rotation=45)
        
        # 12. Performance History (if available)
        plt.subplot(3, 4, 12)
        if len(self.model_history) > 1:
            accuracies = [h['results']['accuracy'] for h in self.model_history]
            timestamps = [h['timestamp'][:10] for h in self.model_history]  # Date only
            plt.plot(range(len(accuracies)), accuracies, 'o-', color='green')
            plt.xlabel('Model Version')
            plt.ylabel('Accuracy')
            plt.title('Model Performance History')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No history available\n(First model)', 
                    ha='center', va='center', fontsize=12)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
        
        plt.tight_layout(pad=3.0)
        
        # Save the comprehensive evaluation plot
        eval_path = os.path.join(OUTPUT_DIR, 'comprehensive_evaluation.png')
        plt.savefig(eval_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save individual plots for dashboard
        self.save_individual_evaluation_plots(y_test, y_pred, y_proba)
        
        return self.encode_image_to_base64(eval_path)
    
    def save_individual_evaluation_plots(self, y_test, y_pred, y_proba):
        """Save individual evaluation plots for dashboard display."""
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.model.classes_, 
                   yticklabels=self.model.classes_)
        plt.title('Confusion Matrix', fontsize=16, weight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Performance Metrics
        plt.figure(figsize=(10, 6))
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics_df = pd.DataFrame({
            'Precision': [report[cls]['precision'] for cls in self.model.classes_],
            'Recall': [report[cls]['recall'] for cls in self.model.classes_],
            'F1-Score': [report[cls]['f1-score'] for cls in self.model.classes_]
        }, index=self.model.classes_)
        
        metrics_df.plot(kind='bar', ax=plt.gca(), color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        plt.title('Performance Metrics by Class', fontsize=16, weight='bold')
        plt.ylabel('Score')
        plt.xlabel('Sentiment Class')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def load_or_train_model(self):
        """Load existing model or train new one with default parameters."""
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
                self.model = joblib.load(MODEL_PATH)
                self.vectorizer = joblib.load(VECTORIZER_PATH)
                logger.info("Model loaded successfully")
                
                self.model_info = {
                    'status': 'loaded',
                    'model_type': 'Naive Bayes',
                    'accuracy': getattr(self.model, 'accuracy_', 0.0),
                    'classes': self.model.classes_.tolist() if hasattr(self.model, 'classes_') else [],
                    'vocabulary_size': len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, 'vocabulary_') else 0
                }
                self.is_trained = True
                self.load_eda_results()
                
            else:
                logger.info("No existing model found. Training new model with default parameters...")
                default_params = {
                    'alpha': 1.0,
                    'max_features': 10000,
                    'ngram_range': [1, 2],
                    'balance_strategy': 'undersample_neutral',
                    'test_size': 0.2,
                    'random_state': 42
                }
                self.train_model_with_params(default_params)
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            default_params = {
                'alpha': 1.0,
                'max_features': 10000,
                'ngram_range': [1, 2],
                'balance_strategy': 'undersample_neutral',
                'test_size': 0.2,
                'random_state': 42
            }
            self.train_model_with_params(default_params)
    
    def load_eda_results(self):
        """Load EDA results from saved files."""
        try:
            eda_files = {
                'class_distribution': os.path.join(OUTPUT_DIR, 'class_distribution.png'),
                'wordcloud_positive': os.path.join(OUTPUT_DIR, 'wordcloud_positive.png'),
                'wordcloud_negative': os.path.join(OUTPUT_DIR, 'wordcloud_negative.png'),
                'wordcloud_neutral': os.path.join(OUTPUT_DIR, 'wordcloud_neutral.png'),
                'word_frequency': os.path.join(OUTPUT_DIR, 'word_frequency_analysis.png'),
                'data_quality': os.path.join(OUTPUT_DIR, 'data_quality_analysis.png'),
                'balance_comparison': os.path.join(OUTPUT_DIR, 'class_balance_comparison.png'),
                'preprocessing_comparison': os.path.join(OUTPUT_DIR, 'preprocessing_comparison.png'),
                'comprehensive_evaluation': os.path.join(OUTPUT_DIR, 'comprehensive_evaluation.png')
            }
            
            self.eda_results = {}
            for name, path in eda_files.items():
                if os.path.exists(path):
                    self.eda_results[name] = self.encode_image_to_base64(path)
                    
            # Load evaluation plots
            eval_files = {
                'confusion_matrix': os.path.join(OUTPUT_DIR, 'confusion_matrix.png'),
                'performance_metrics': os.path.join(OUTPUT_DIR, 'performance_metrics.png')
            }
            
            self.evaluation_plots = {}
            for name, path in eval_files.items():
                if os.path.exists(path):
                    self.evaluation_plots[name] = self.encode_image_to_base64(path)
                    
        except Exception as e:
            logger.error(f"Error loading EDA results: {e}")
            self.eda_results = {}
            self.evaluation_plots = {}
    
    def encode_image_to_base64(self, image_path):
        """Convert image to base64 for web display."""
        try:
            with open(image_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return None
    
    def predict_sentiment(self, text):
        """Predict sentiment for given text."""
        if not self.is_trained or not self.model or not self.vectorizer:
            return {'error': 'Model not trained or loaded'}
        
        try:
            # Preprocess text
            preprocessor = TextPreprocessor()
            processed_text = preprocessor.preprocess_texts([text])
            
            # Vectorize
            X = self.vectorizer.transform(processed_text)
            
            # Predict
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Create probability dict
            prob_dict = {}
            for i, class_name in enumerate(self.model.classes_):
                prob_dict[class_name] = float(probabilities[i])
            
            return {
                'prediction': prediction,
                'probabilities': prob_dict,
                'confidence': float(max(probabilities))
            }
            
        except Exception as e:
            logger.error(f"Error predicting sentiment: {e}")
            return {'error': str(e)}
    
    def get_dashboard_data(self):
        """Get all data needed for dashboard."""
        dashboard_data = {
            'model_info': self.model_info,
            'eda_results': self.eda_results,
            'evaluation_plots': self.evaluation_plots,
            'model_history': self.model_history,
            'best_model': self.get_best_model_params()
        }
        
        return dashboard_data


# Global analyzer instance
analyzer = EnhancedFinancialSentimentAnalyzer()


def home(request):
    """Enhanced dashboard page with comprehensive visualizations."""
    dashboard_data = analyzer.get_dashboard_data()
    
    context = {
        'title': 'Enhanced Financial Sentiment Analysis Dashboard',
        'model_info': dashboard_data['model_info'],
        'eda_results': dashboard_data['eda_results'],
        'evaluation_plots': dashboard_data['evaluation_plots'],
        'model_history': dashboard_data['model_history'],
        'best_model': dashboard_data['best_model'],
        'sample_texts': [
            'Apple stock is performing exceptionally well with strong quarterly earnings',
            'Market volatility creates uncertainty for investors as tech stocks decline',
            'Tesla earnings report shows mixed results with moderate growth',
            'Financial markets remain stable with balanced trading volume'
        ]
    }
    return render(request, 'sentiment/enhanced_dashboard.html', context)


@csrf_exempt
def retrain_model(request):
    """Endpoint to retrain model with custom parameters."""
    if request.method == 'POST':
        try:
            # Get parameters from request
            if request.content_type == 'application/json':
                data = json.loads(request.body)
            else:
                data = dict(request.POST)
                # Convert lists from form data
                for key in ['ngram_range']:
                    if key in data and isinstance(data[key], list):
                        data[key] = [int(x) for x in data[key]]
            
            # Default parameters
            default_params = {
                'alpha': 1.0,
                'max_features': 10000,
                'ngram_range': [1, 2],
                'balance_strategy': 'undersample_neutral',
                'test_size': 0.2,
                'random_state': 42,
                'vectorizer_type': 'tfidf',
                'min_df': 1,
                'max_df': 0.95,
                'fit_prior': True
            }
            
            # Update with provided parameters
            for key, value in data.items():
                if key in default_params:
                    # Convert string numbers to appropriate types
                    if key in ['alpha', 'test_size', 'max_df']:
                        default_params[key] = float(value)
                    elif key in ['max_features', 'random_state', 'min_df']:
                        default_params[key] = int(value)
                    elif key == 'fit_prior':
                        default_params[key] = str(value).lower() == 'true'
                    else:
                        default_params[key] = value
            
            # Train model with new parameters
            results = analyzer.train_model_with_params(default_params)
            
            return JsonResponse({
                'status': 'success', 
                'message': 'Model retrained successfully',
                'results': {
                    'accuracy': results['accuracy'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1_score': results['f1_score']
                },
                'parameters': default_params
            })
            
        except Exception as e:
            logger.error(f"Error retraining model: {e}")
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
def load_historical_model(request):
    """Load a specific model from history."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            model_id = data.get('model_id')
            
            if not model_id:
                return JsonResponse({'error': 'Model ID required'}, status=400)
            
            # Find model in history
            historical_model = None
            for model in analyzer.model_history:
                if model['model_id'] == int(model_id):
                    historical_model = model
                    break
            
            if not historical_model:
                return JsonResponse({'error': 'Model not found'}, status=404)
            
            # Return model parameters for user to confirm retraining
            return JsonResponse({
                'status': 'success',
                'model_info': historical_model,
                'parameters': historical_model['parameters'],
                'results': historical_model['results']
            })
            
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
def predict_api(request):
    """API endpoint for single text prediction."""
    if request.method == 'POST':
        try:
            if request.content_type == 'application/json':
                data = json.loads(request.body)
                text = data.get('text', '')
            else:
                text = request.POST.get('text', '')
            
            if not text:
                return JsonResponse({'error': 'No text provided'}, status=400)
            
            result = analyzer.predict_sentiment(text)
            
            if 'error' in result:
                return JsonResponse(result, status=500)
            else:
                return JsonResponse(result)
                
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt  
def predict_batch_api(request):
    """API endpoint for batch text prediction."""
    if request.method == 'POST':
        try:
            if request.content_type == 'application/json':
                data = json.loads(request.body)
                texts = data.get('texts', [])
            else:
                texts = request.POST.getlist('texts')
            
            if not texts:
                return JsonResponse({'error': 'No texts provided'}, status=400)
            
            results = []
            for text in texts:
                result = analyzer.predict_sentiment(text)
                if 'error' not in result:
                    results.append({
                        'text': text,
                        'prediction': result['prediction'],
                        'confidence': result['confidence'],
                        'probabilities': result['probabilities']
                    })
                else:
                    results.append({
                        'text': text,
                        'error': result['error']
                    })
            
            return JsonResponse({'results': results})
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)
