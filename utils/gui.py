"""
Django web interface for Financial Sentiment Analysis.
Provides a user-friendly web interface for sentiment prediction.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import logging
from typing import Dict, Any, List
from config import MODEL_PATH, VECTORIZER_PATH, DJANGO_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Web-ready sentiment analyzer that loads pre-trained models.
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        self.model = None
        self.vectorizer = None
        self.preprocessor = None
        self.is_loaded = False
        
        # Load models on initialization
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models and vectorizer."""
        try:
            # Load model
            if os.path.exists(MODEL_PATH):
                model_data = joblib.load(MODEL_PATH)
                self.model = model_data.get('model')
                logger.info(f"Model loaded from {MODEL_PATH}")
            else:
                logger.warning(f"Model file not found at {MODEL_PATH}")
                return
            
            # Load vectorizer
            if os.path.exists(VECTORIZER_PATH):
                vectorizer_data = joblib.load(VECTORIZER_PATH)
                self.vectorizer = vectorizer_data.get('vectorizer')
                logger.info(f"Vectorizer loaded from {VECTORIZER_PATH}")
            else:
                logger.warning(f"Vectorizer file not found at {VECTORIZER_PATH}")
                return
            
            # Try to load preprocessor (optional)
            try:
                from utils.preprocess import TextPreprocessor
                self.preprocessor = TextPreprocessor()
                logger.info("Text preprocessor initialized")
            except ImportError:
                logger.warning("Could not import text preprocessor")
            
            self.is_loaded = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.is_loaded = False
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess input text for prediction.
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Preprocessed text
        """
        if not text or not text.strip():
            return ""
        
        if self.preprocessor:
            try:
                # Use the preprocessor if available
                processed_text = self.preprocessor.preprocess_text(text)
                return processed_text
            except Exception as e:
                logger.warning(f"Preprocessing failed: {e}. Using basic cleaning.")
        
        # Basic text cleaning if preprocessor not available
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a single text.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        if not self.is_loaded:
            return {
                'error': 'Models not loaded properly',
                'prediction': None,
                'confidence': 0,
                'probabilities': {}
            }
        
        if not text or not text.strip():
            return {
                'error': 'Empty text provided',
                'prediction': None,
                'confidence': 0,
                'probabilities': {}
            }
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                return {
                    'error': 'Text became empty after preprocessing',
                    'prediction': None,
                    'confidence': 0,
                    'probabilities': {}
                }
            
            # Vectorize text
            text_features = self.vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = self.model.predict(text_features)[0]
            probabilities = self.model.predict_proba(text_features)[0]
            
            # Get class names and create probability distribution
            class_names = self.model.classes_
            prob_dict = {class_name: float(prob) for class_name, prob in zip(class_names, probabilities)}
            
            # Get confidence (max probability)
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
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            List[Dict[str, Any]]: List of prediction results
        """
        results = []
        for text in texts:
            result = self.predict_sentiment(text)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Returns:
            Dict[str, Any]: Model information
        """
        if not self.is_loaded:
            return {'status': 'not_loaded', 'error': 'Models not loaded'}
        
        try:
            info = {
                'status': 'loaded',
                'model_type': type(self.model).__name__,
                'classes': self.model.classes_.tolist() if hasattr(self.model, 'classes_') else None,
                'n_features': self.model.feature_count_.shape[1] if hasattr(self.model, 'feature_count_') else None,
                'vectorizer_type': type(self.vectorizer).__name__,
                'vocabulary_size': len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, 'vocabulary_') else None
            }
            return info
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}


# Global analyzer instance
analyzer = SentimentAnalyzer()


# Django Views
def home(request):
    """Home page view."""
    context = {
        'title': 'Financial Sentiment Analysis',
        'model_info': analyzer.get_model_info()
    }
    return render(request, 'sentiment/home.html', context)


@csrf_exempt
def predict_single(request):
    """
    API endpoint for single text prediction.
    
    Expects POST request with JSON: {'text': 'input text'}
    Returns JSON with prediction results.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        text = data.get('text', '').strip()
        
        if not text:
            return JsonResponse({'error': 'No text provided'}, status=400)
        
        # Make prediction
        result = analyzer.predict_sentiment(text)
        
        return JsonResponse(result)
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        logger.error(f"API error: {e}")
        return JsonResponse({'error': 'Internal server error'}, status=500)


@csrf_exempt
def predict_batch(request):
    """
    API endpoint for batch text prediction.
    
    Expects POST request with JSON: {'texts': ['text1', 'text2', ...]}
    Returns JSON with list of prediction results.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        texts = data.get('texts', [])
        
        if not texts or not isinstance(texts, list):
            return JsonResponse({'error': 'No valid texts provided'}, status=400)
        
        if len(texts) > 100:  # Limit batch size
            return JsonResponse({'error': 'Too many texts (max 100)'}, status=400)
        
        # Make predictions
        results = analyzer.predict_batch(texts)
        
        return JsonResponse({'results': results})
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        logger.error(f"Batch API error: {e}")
        return JsonResponse({'error': 'Internal server error'}, status=500)


def model_info(request):
    """API endpoint to get model information."""
    info = analyzer.get_model_info()
    return JsonResponse(info)


def demo(request):
    """Demo page with interactive interface."""
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


# Django URL Configuration
def get_urlpatterns():
    """Get URL patterns for the sentiment analysis app."""
    from django.urls import path
    
    urlpatterns = [
        path('', home, name='home'),
        path('demo/', demo, name='demo'),
        path('api/predict/', predict_single, name='predict_single'),
        path('api/predict_batch/', predict_batch, name='predict_batch'),
        path('api/model_info/', model_info, name='model_info'),
    ]
    
    return urlpatterns


# Django Settings Configuration
def get_django_settings():
    """Get Django settings for the sentiment analysis app."""
    import os
    from pathlib import Path
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    DJANGO_SETTINGS = {
        'DEBUG': DJANGO_CONFIG.get('debug', True),
        'SECRET_KEY': 'your-secret-key-here-change-in-production',
        'ALLOWED_HOSTS': ['localhost', '127.0.0.1', '*'],
        
        'INSTALLED_APPS': [
            'django.contrib.staticfiles',
            'django.contrib.contenttypes',
            'sentiment_app',
        ],
        
        'MIDDLEWARE': [
            'django.middleware.security.SecurityMiddleware',
            'django.middleware.common.CommonMiddleware',
        ],
        
        'ROOT_URLCONF': 'sentiment_project.urls',
        
        'TEMPLATES': [
            {
                'BACKEND': 'django.template.backends.django.DjangoTemplates',
                'DIRS': [os.path.join(BASE_DIR, 'templates')],
                'APP_DIRS': True,
                'OPTIONS': {
                    'context_processors': [
                        'django.template.context_processors.debug',
                        'django.template.context_processors.request',
                    ],
                },
            },
        ],
        
        'STATIC_URL': '/static/',
        'STATICFILES_DIRS': [
            os.path.join(BASE_DIR, 'static'),
        ],
        
        'DATABASES': {
            'default': {
                'ENGINE': 'django.db.backends.sqlite3',
                'NAME': ':memory:',  # In-memory database for simplicity
            }
        },
        
        'USE_TZ': True,
        'TIME_ZONE': 'UTC',
        
        'LOGGING': {
            'version': 1,
            'disable_existing_loggers': False,
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                },
            },
            'loggers': {
                'django': {
                    'handlers': ['console'],
                    'level': 'INFO',
                },
            },
        },
    }
    
    return DJANGO_SETTINGS


# Django App Configuration
class SentimentAppConfig:
    """Django app configuration."""
    
    name = 'sentiment_app'
    verbose_name = 'Financial Sentiment Analysis'
    
    def ready(self):
        """Initialize app when Django starts."""
        # Ensure models are loaded
        global analyzer
        if not analyzer.is_loaded:
            analyzer.load_models()


# HTML Templates (as strings for simplicity)
HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .info { background: #e8f4fd; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .nav { text-align: center; margin: 20px 0; }
        .nav a { margin: 0 15px; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }
        .nav a:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        <div class="info">
            <h3>Model Status:</h3>
            {% if model_info.status == 'loaded' %}
                <p><strong>Status:</strong> ✅ Models loaded successfully</p>
                <p><strong>Model Type:</strong> {{ model_info.model_type }}</p>
                <p><strong>Classes:</strong> {{ model_info.classes|join:", " }}</p>
                <p><strong>Vocabulary Size:</strong> {{ model_info.vocabulary_size }}</p>
            {% else %}
                <p><strong>Status:</strong> ❌ Models not loaded</p>
                <p><strong>Error:</strong> {{ model_info.error }}</p>
            {% endif %}
        </div>
        
        <div class="nav">
            <a href="/demo/">Try Demo</a>
            <a href="/api/model_info/">API Info</a>
        </div>
        
        <div style="margin-top: 30px;">
            <h3>API Endpoints:</h3>
            <ul>
                <li><strong>POST /api/predict/</strong> - Single text prediction</li>
                <li><strong>POST /api/predict_batch/</strong> - Batch text predictions</li>
                <li><strong>GET /api/model_info/</strong> - Model information</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

DEMO_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .input-section { margin: 20px 0; }
        textarea { width: 100%; height: 100px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; }
        .btn { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .btn:hover { background: #0056b3; }
        .result { margin: 20px 0; padding: 15px; border-radius: 5px; }
        .result.positive { background: #d4edda; border: 1px solid #c3e6cb; }
        .result.negative { background: #f8d7da; border: 1px solid #f5c6cb; }
        .result.neutral { background: #e2e3e5; border: 1px solid #d6d8db; }
        .samples { margin: 20px 0; }
        .sample { background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; cursor: pointer; }
        .sample:hover { background: #e9ecef; }
        .back-link { text-align: center; margin: 20px 0; }
        .back-link a { color: #007bff; text-decoration: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        
        <div class="input-section">
            <label for="textInput"><strong>Enter text to analyze:</strong></label>
            <textarea id="textInput" placeholder="Enter financial text here..."></textarea>
            <br><br>
            <button class="btn" onclick="analyzeSentiment()">Analyze Sentiment</button>
        </div>
        
        <div id="result" style="display: none;"></div>
        
        <div class="samples">
            <h3>Sample Texts (click to try):</h3>
            {% for text in sample_texts %}
            <div class="sample" onclick="setSampleText('{{ text|escapejs }}')">
                {{ text }}
            </div>
            {% endfor %}
        </div>
        
        <div class="back-link">
            <a href="/">← Back to Home</a>
        </div>
    </div>
    
    <script>
        function setSampleText(text) {
            document.getElementById('textInput').value = text;
        }
        
        async function analyzeSentiment() {
            const text = document.getElementById('textInput').value.trim();
            
            if (!text) {
                alert('Please enter some text to analyze.');
                return;
            }
            
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = '<p>Analyzing...</p>';
            
            try {
                const response = await fetch('/api/predict/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    resultDiv.innerHTML = `<div class="result"><strong>Error:</strong> ${data.error}</div>`;
                } else {
                    const sentiment = data.prediction.toLowerCase();
                    const confidence = (data.confidence * 100).toFixed(1);
                    
                    let probsHtml = '<ul>';
                    for (const [sent, prob] of Object.entries(data.probabilities)) {
                        probsHtml += `<li><strong>${sent}:</strong> ${(prob * 100).toFixed(1)}%</li>`;
                    }
                    probsHtml += '</ul>';
                    
                    resultDiv.innerHTML = `
                        <div class="result ${sentiment}">
                            <h3>Prediction Results:</h3>
                            <p><strong>Sentiment:</strong> ${data.prediction}</p>
                            <p><strong>Confidence:</strong> ${confidence}%</p>
                            <p><strong>All Probabilities:</strong></p>
                            ${probsHtml}
                        </div>
                    `;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="result"><strong>Error:</strong> Failed to analyze text. ${error.message}</div>`;
            }
        }
    </script>
</body>
</html>
"""


def save_templates():
    """Save HTML templates to files."""
    templates_dir = 'templates/sentiment'
    os.makedirs(templates_dir, exist_ok=True)
    
    with open(os.path.join(templates_dir, 'home.html'), 'w') as f:
        f.write(HOME_TEMPLATE)
    
    with open(os.path.join(templates_dir, 'demo.html'), 'w') as f:
        f.write(DEMO_TEMPLATE)
    
    logger.info("HTML templates saved")


if __name__ == "__main__":
    # Save templates when module is run directly
    save_templates()
    
    print("Django GUI module ready!")
    print("To run the web server:")
    print("1. Ensure models are trained and saved")
    print("2. Run: python manage.py runserver")
    print("3. Visit: http://127.0.0.1:8000/")
