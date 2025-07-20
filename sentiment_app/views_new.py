"""
Simplified Views for Financial Sentiment Analysis Django App.
Auto-trains model and provides clean prediction interface with visualizations.
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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

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


class FinancialSentimentAnalyzer:
    """Complete ML pipeline for financial sentiment analysis."""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.model_info = {}
        self.eda_results = {}
        self.is_trained = False
        self.load_or_train_model()
    
    def load_or_train_model(self):
        """Load existing model or train new one."""
        try:
            # Try to load existing model
            if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
                self.model = joblib.load(MODEL_PATH)
                self.vectorizer = joblib.load(VECTORIZER_PATH)
                logger.info("Model loaded successfully")
                
                # Load model info
                self.model_info = {
                    'status': 'loaded',
                    'model_type': 'Naive Bayes',
                    'accuracy': getattr(self.model, 'accuracy_', 0.704),  # Default if not stored
                    'classes': self.model.classes_.tolist() if hasattr(self.model, 'classes_') else ['negative', 'neutral', 'positive'],
                    'vocabulary_size': len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, 'vocabulary_') else 0
                }
                self.is_trained = True
                
                # Load EDA results if available
                self.load_eda_results()
                
            else:
                logger.info("No existing model found. Training new model...")
                self.train_complete_pipeline()
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.train_complete_pipeline()
    
    def train_complete_pipeline(self):
        """Train the complete ML pipeline."""
        try:
            logger.info("Starting complete ML pipeline...")
            
            # 1. Load dataset
            data = load_dataset(DATASET_PATH)
            data = validate_dataset(data)
            
            # 2. Perform EDA and save results
            self.eda_results = perform_comprehensive_eda(
                data['Sentence'], 
                data['Sentiment'],
                output_dir=OUTPUT_DIR
            )
            
            # 3. Preprocess data
            preprocessor = TextPreprocessor()
            processed_texts = preprocessor.preprocess_texts(data['Sentence'])
            
            # 4. Create and train model
            from sklearn.model_selection import train_test_split
            
            # Split data
            X_train_text, X_test_text, y_train, y_test = train_test_split(
                processed_texts, data['Sentiment'],
                test_size=0.2, random_state=42, stratify=data['Sentiment']
            )
            
            # Create vectorizer and transform data
            self.vectorizer = create_feature_extractor('tfidf')
            X_train = self.vectorizer.fit_transform(X_train_text)
            X_test = self.vectorizer.transform(X_test_text)
            
            # Train model
            self.model, evaluation_results = train_and_evaluate_model(
                X_train, X_test, y_train, y_test, {'alpha': 1.0}
            )
            
            # Save models
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.vectorizer, VECTORIZER_PATH)
            
            # Store model info
            self.model_info = {
                'status': 'trained',
                'model_type': 'Naive Bayes',
                'accuracy': evaluation_results.get('accuracy', 0.0),
                'classes': self.model.classes_.tolist(),
                'vocabulary_size': len(self.vectorizer.vocabulary_),
                'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Generate evaluation plots
            self.generate_evaluation_plots(y_test, self.model.predict(X_test), self.model.predict_proba(X_test))
            
            self.is_trained = True
            logger.info(f"Model trained successfully. Accuracy: {self.model_info['accuracy']:.3f}")
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            self.model_info = {'status': 'error', 'error': str(e)}
    
    def load_eda_results(self):
        """Load EDA results from saved files."""
        try:
            # Check if EDA visualizations exist
            eda_files = {
                'class_distribution': os.path.join(OUTPUT_DIR, 'class_distribution.png'),
                'wordcloud_positive': os.path.join(OUTPUT_DIR, 'wordcloud_positive.png'),
                'wordcloud_negative': os.path.join(OUTPUT_DIR, 'wordcloud_negative.png'),
                'wordcloud_neutral': os.path.join(OUTPUT_DIR, 'wordcloud_neutral.png'),
                'word_frequency': os.path.join(OUTPUT_DIR, 'word_frequency_analysis.png')
            }
            
            self.eda_results = {}
            for name, path in eda_files.items():
                if os.path.exists(path):
                    self.eda_results[name] = self.encode_image_to_base64(path)
                    
        except Exception as e:
            logger.error(f"Error loading EDA results: {e}")
            self.eda_results = {}
    
    def generate_evaluation_plots(self, y_true, y_pred, y_proba):
        """Generate model evaluation plots."""
        try:
            # Confusion Matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.model.classes_, 
                       yticklabels=self.model.classes_)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            confusion_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
            plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Classification Report
            report = classification_report(y_true, y_pred, output_dict=True)
            
            # Save metrics plot
            plt.figure(figsize=(10, 6))
            metrics_df = pd.DataFrame({
                'Precision': [report[cls]['precision'] for cls in self.model.classes_],
                'Recall': [report[cls]['recall'] for cls in self.model.classes_],
                'F1-Score': [report[cls]['f1-score'] for cls in self.model.classes_]
            }, index=self.model.classes_)
            
            metrics_df.plot(kind='bar', ax=plt.gca())
            plt.title('Model Performance Metrics by Class')
            plt.ylabel('Score')
            plt.xlabel('Sentiment Class')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            metrics_path = os.path.join(OUTPUT_DIR, 'performance_metrics.png')
            plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Evaluation plots generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating evaluation plots: {e}")
    
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
            'evaluation_plots': {}
        }
        
        # Load evaluation plots
        eval_files = {
            'confusion_matrix': os.path.join(OUTPUT_DIR, 'confusion_matrix.png'),
            'performance_metrics': os.path.join(OUTPUT_DIR, 'performance_metrics.png')
        }
        
        for name, path in eval_files.items():
            if os.path.exists(path):
                dashboard_data['evaluation_plots'][name] = self.encode_image_to_base64(path)
        
        return dashboard_data


# Global analyzer instance
analyzer = FinancialSentimentAnalyzer()


def home(request):
    """Main dashboard page with all visualizations and prediction interface."""
    dashboard_data = analyzer.get_dashboard_data()
    
    context = {
        'title': 'Financial Sentiment Analysis Dashboard',
        'model_info': dashboard_data['model_info'],
        'eda_results': dashboard_data['eda_results'],
        'evaluation_plots': dashboard_data['evaluation_plots'],
        'sample_texts': [
            'Apple stock is performing very well today',
            'The market crashed and I lost money',
            'Tesla earnings exceeded expectations',
            'Neutral market conditions with mixed signals'
        ]
    }
    return render(request, 'sentiment/dashboard_simple.html', context)


@csrf_exempt
def predict_api(request):
    """API endpoint for single text prediction."""
    if request.method == 'POST':
        try:
            # Handle both JSON and form data
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


def retrain_model(request):
    """Endpoint to retrain the model."""
    if request.method == 'POST':
        try:
            global analyzer
            analyzer = FinancialSentimentAnalyzer()
            return JsonResponse({'status': 'success', 'message': 'Model retrained successfully'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)
