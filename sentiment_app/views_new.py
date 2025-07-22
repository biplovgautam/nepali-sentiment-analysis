"""
Financial Sentiment Analysis Django Views
Simplified version with focus on model switching and dashboard functionality
"""

import os
import json
import joblib
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model paths
MODELS_DIR = os.path.join(settings.BASE_DIR, 'models')
ORIGINAL_MODEL_PATH = os.path.join(MODELS_DIR, 'naive_bayes_model.pkl')
ORIGINAL_VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
BALANCED_MODEL_PATH = os.path.join(MODELS_DIR, 'balanced_naive_bayes_model.pkl')
BALANCED_VECTORIZER_PATH = os.path.join(MODELS_DIR, 'balanced_tfidf_vectorizer.pkl')

# Global model storage
loaded_models = {
    'original': {'model': None, 'vectorizer': None, 'loaded': False},
    'balanced': {'model': None, 'vectorizer': None, 'loaded': False}
}

def load_model(model_type='balanced'):
    """Load the specified model and vectorizer"""
    global loaded_models
    
    try:
        if model_type == 'original':
            model_path = ORIGINAL_MODEL_PATH
            vectorizer_path = ORIGINAL_VECTORIZER_PATH
        else:
            model_path = BALANCED_MODEL_PATH
            vectorizer_path = BALANCED_VECTORIZER_PATH
        
        if not loaded_models[model_type]['loaded']:
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                loaded_models[model_type]['model'] = joblib.load(model_path)
                loaded_models[model_type]['vectorizer'] = joblib.load(vectorizer_path)
                loaded_models[model_type]['loaded'] = True
                logger.info(f"Loaded {model_type} model successfully")
            else:
                logger.error(f"Model files not found for {model_type}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"Error loading {model_type} model: {str(e)}")
        return False

def dashboard(request):
    """Main dashboard view"""
    # Try to load both models
    original_loaded = load_model('original')
    balanced_loaded = load_model('balanced')
    
    model_info = {
        'original': {
            'loaded': original_loaded,
            'accuracy': '66.3%',
            'f1_score': '0.637',
            'precision': '0.652',
            'recall': '0.663'
        },
        'balanced': {
            'loaded': balanced_loaded,
            'accuracy': '68.3%',
            'f1_score': '0.683',
            'precision': '0.687',
            'recall': '0.683'
        }
    }
    
    context = {
        'model_info': model_info,
        'default_model': 'balanced' if balanced_loaded else 'original'
    }
    
    return render(request, 'sentiment/dashboard.html', context)

@require_http_methods(["POST"])
@csrf_exempt
def predict_sentiment(request):
    """API endpoint for sentiment prediction"""
    try:
        data = json.loads(request.body)
        text = data.get('text', '').strip()
        model_type = data.get('model', 'balanced')
        
        if not text:
            return JsonResponse({
                'success': False,
                'message': 'No text provided'
            })
        
        # Load the requested model
        if not load_model(model_type):
            return JsonResponse({
                'success': False,
                'message': f'Failed to load {model_type} model'
            })
        
        # Get model and vectorizer
        model = loaded_models[model_type]['model']
        vectorizer = loaded_models[model_type]['vectorizer']
        
        # Preprocess and predict
        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)[0]
        prediction_proba = model.predict_proba(text_vectorized)[0]
        
        # Get confidence (maximum probability)
        confidence = float(max(prediction_proba))
        
        return JsonResponse({
            'success': True,
            'prediction': prediction,
            'confidence': confidence,
            'model_used': model_type,
            'probabilities': {
                'positive': float(prediction_proba[model.classes_.tolist().index('positive')] if 'positive' in model.classes_ else 0),
                'negative': float(prediction_proba[model.classes_.tolist().index('negative')] if 'negative' in model.classes_ else 0),
                'neutral': float(prediction_proba[model.classes_.tolist().index('neutral')] if 'neutral' in model.classes_ else 0)
            }
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'message': 'Invalid JSON data'
        })
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return JsonResponse({
            'success': False,
            'message': f'Prediction failed: {str(e)}'
        })

@require_http_methods(["GET"])
def model_info_api(request):
    """API endpoint to get model information"""
    try:
        model_type = request.GET.get('model', 'balanced')
        
        # Load model if not already loaded
        load_model(model_type)
        
        if loaded_models[model_type]['loaded']:
            model = loaded_models[model_type]['model']
            vectorizer = loaded_models[model_type]['vectorizer']
            
            # Get model metadata
            model_info = {
                'model_type': model_type,
                'algorithm': 'Multinomial Naive Bayes',
                'classes': model.classes_.tolist() if hasattr(model, 'classes_') else [],
                'feature_count': vectorizer.get_feature_names_out().shape[0] if hasattr(vectorizer, 'get_feature_names_out') else 'Unknown',
                'loaded': True,
                'status': 'active'
            }
            
            # Add performance metrics
            if model_type == 'original':
                model_info.update({
                    'accuracy': '66.3%',
                    'f1_score': '0.637',
                    'precision': '0.652',
                    'recall': '0.663'
                })
            else:
                model_info.update({
                    'accuracy': '68.3%',
                    'f1_score': '0.683',
                    'precision': '0.687',
                    'recall': '0.683'
                })
            
            return JsonResponse({
                'success': True,
                'model_info': model_info
            })
        else:
            return JsonResponse({
                'success': False,
                'message': f'{model_type} model not loaded'
            })
            
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return JsonResponse({
            'success': False,
            'message': f'Failed to get model info: {str(e)}'
        })

@require_http_methods(["POST"])
@csrf_exempt
def batch_predict(request):
    """API endpoint for batch sentiment prediction"""
    try:
        data = json.loads(request.body)
        texts = data.get('texts', [])
        model_type = data.get('model', 'balanced')
        
        if not texts or not isinstance(texts, list):
            return JsonResponse({
                'success': False,
                'message': 'No texts provided or invalid format'
            })
        
        # Load the requested model
        if not load_model(model_type):
            return JsonResponse({
                'success': False,
                'message': f'Failed to load {model_type} model'
            })
        
        # Get model and vectorizer
        model = loaded_models[model_type]['model']
        vectorizer = loaded_models[model_type]['vectorizer']
        
        # Process all texts
        texts_vectorized = vectorizer.transform(texts)
        predictions = model.predict(texts_vectorized)
        probabilities = model.predict_proba(texts_vectorized)
        
        # Format results
        results = []
        for i, (text, prediction, proba) in enumerate(zip(texts, predictions, probabilities)):
            results.append({
                'text': text,
                'prediction': prediction,
                'confidence': float(max(proba)),
                'probabilities': {
                    'positive': float(proba[model.classes_.tolist().index('positive')] if 'positive' in model.classes_ else 0),
                    'negative': float(proba[model.classes_.tolist().index('negative')] if 'negative' in model.classes_ else 0),
                    'neutral': float(proba[model.classes_.tolist().index('neutral')] if 'neutral' in model.classes_ else 0)
                }
            })
        
        return JsonResponse({
            'success': True,
            'results': results,
            'model_used': model_type,
            'total_processed': len(results)
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return JsonResponse({
            'success': False,
            'message': f'Batch prediction failed: {str(e)}'
        })

# Legacy views for compatibility
def ml_dashboard(request):
    """Legacy dashboard view"""
    return dashboard(request)

def home(request):
    """Legacy home view"""
    return dashboard(request)

def predict_api(request):
    """Legacy predict API"""
    return predict_sentiment(request)

def model_info(request):
    """Legacy model info"""
    return model_info_api(request)

def predict_batch_api(request):
    """Legacy batch predict API"""
    return batch_predict(request)
