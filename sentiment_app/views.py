"""
Views for Financial Sentiment Analysis Django App.
"""

import os
import sys
import json
import joblib
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from config import MODEL_PATH, VECTORIZER_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Web-ready sentiment analyzer."""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_loaded = False
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models."""
        try:
            if os.path.exists(MODEL_PATH):
                model_data = joblib.load(MODEL_PATH)
                self.model = model_data.get('model')
                logger.info("Model loaded successfully")
            
            if os.path.exists(VECTORIZER_PATH):
                vectorizer_data = joblib.load(VECTORIZER_PATH)
                self.vectorizer = vectorizer_data.get('vectorizer')
                logger.info("Vectorizer loaded successfully")
            
            self.is_loaded = (self.model is not None and self.vectorizer is not None)
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.is_loaded = False
    
    def preprocess_text(self, text):
        """Simple text preprocessing."""
        if not text:
            return ""
        
        import re
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def predict_sentiment(self, text):
        """Predict sentiment for text."""
        if not self.is_loaded:
            return {
                'error': 'Models not loaded',
                'prediction': None,
                'confidence': 0,
                'probabilities': {}
            }
        
        if not text or not text.strip():
            return {
                'error': 'Empty text',
                'prediction': None,
                'confidence': 0,
                'probabilities': {}
            }
        
        try:
            processed_text = self.preprocess_text(text)
            if not processed_text:
                return {
                    'error': 'Text became empty after preprocessing',
                    'prediction': None,
                    'confidence': 0,
                    'probabilities': {}
                }
            
            # Vectorize and predict
            text_features = self.vectorizer.transform([processed_text])
            prediction = self.model.predict(text_features)[0]
            probabilities = self.model.predict_proba(text_features)[0]
            
            # Create response
            class_names = self.model.classes_
            prob_dict = {class_name: float(prob) for class_name, prob in zip(class_names, probabilities)}
            confidence = float(max(probabilities))
            
            return {
                'error': None,
                'prediction': str(prediction),
                'confidence': confidence,
                'probabilities': prob_dict,
                'processed_text': processed_text
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'error': f'Prediction failed: {str(e)}',
                'prediction': None,
                'confidence': 0,
                'probabilities': {}
            }
    
    def get_model_info(self):
        """Get model information."""
        if not self.is_loaded:
            return {'status': 'not_loaded'}
        
        try:
            return {
                'status': 'loaded',
                'model_type': type(self.model).__name__,
                'classes': self.model.classes_.tolist() if hasattr(self.model, 'classes_') else None,
                'vocabulary_size': len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, 'vocabulary_') else None
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}


# Global analyzer instance
analyzer = SentimentAnalyzer()


def home(request):
    """Home page."""
    context = {
        'title': 'Financial Sentiment Analysis',
        'model_info': analyzer.get_model_info()
    }
    return render(request, 'sentiment/home.html', context)


def demo(request):
    """Demo page."""
    context = {
        'title': 'Sentiment Analysis Demo',
        'sample_texts': [
            'Apple stock is performing very well today',
            'The market crashed and I lost money',
            'Tesla earnings exceeded expectations',
            'Neutral market conditions with mixed signals'
        ]
    }
    return render(request, 'sentiment/demo.html', context)


@csrf_exempt
def predict_single(request):
    """API for single text prediction."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        text = data.get('text', '').strip()
        
        if not text:
            return JsonResponse({'error': 'No text provided'}, status=400)
        
        result = analyzer.predict_sentiment(text)
        return JsonResponse(result)
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        logger.error(f"API error: {e}")
        return JsonResponse({'error': 'Internal server error'}, status=500)


@csrf_exempt
def predict_batch(request):
    """API for batch prediction."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        texts = data.get('texts', [])
        
        if not texts or not isinstance(texts, list):
            return JsonResponse({'error': 'No valid texts provided'}, status=400)
        
        if len(texts) > 100:
            return JsonResponse({'error': 'Too many texts (max 100)'}, status=400)
        
        results = []
        for text in texts:
            result = analyzer.predict_sentiment(text)
            results.append(result)
        
        return JsonResponse({'results': results})
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        logger.error(f"Batch API error: {e}")
        return JsonResponse({'error': 'Internal server error'}, status=500)


def model_info(request):
    """Model information API."""
    info = analyzer.get_model_info()
    return JsonResponse(info)
