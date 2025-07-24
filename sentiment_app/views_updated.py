import os
import json
import joblib
import logging
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

# Setup logging
logger = logging.getLogger(__name__)

# Model paths - only using balanced model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BALANCED_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'balanced_naive_bayes_model.pkl')
BALANCED_VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'balanced_tfidf_vectorizer.pkl')
BALANCED_METADATA_PATH = os.path.join(BASE_DIR, 'models', 'balanced_model_metadata.json')

# Global model storage - simplified to only balanced model
loaded_model = {
    'model': None,
    'vectorizer': None,
    'metadata': None,
    'loaded': False
}


def load_model():
    """Load the balanced model and vectorizer"""
    global loaded_model
    
    try:
        if not loaded_model['loaded']:
            if os.path.exists(BALANCED_MODEL_PATH) and os.path.exists(BALANCED_VECTORIZER_PATH):
                loaded_model['model'] = joblib.load(BALANCED_MODEL_PATH)
                loaded_model['vectorizer'] = joblib.load(BALANCED_VECTORIZER_PATH)
                
                # Load metadata if available
                if os.path.exists(BALANCED_METADATA_PATH):
                    with open(BALANCED_METADATA_PATH, 'r') as f:
                        loaded_model['metadata'] = json.load(f)
                
                loaded_model['loaded'] = True
                logger.info("Loaded balanced model successfully")
            else:
                logger.error("Model files not found")
                return False
        
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False


def dashboard(request):
    """Main dashboard view"""
    # Load the balanced model
    model_loaded = load_model()
    
    model_info = {
        'loaded': model_loaded,
        'accuracy': '68.3%',
        'f1_score': '0.683',
        'precision': '0.687',
        'recall': '0.683',
        'model_type': 'Balanced Multinomial Naive Bayes',
        'feature_count': None
    }
    
    # Get feature count if model is loaded
    if model_loaded and loaded_model['vectorizer']:
        try:
            feature_names = loaded_model['vectorizer'].get_feature_names_out()
            model_info['feature_count'] = len(feature_names)
        except:
            model_info['feature_count'] = 'Unknown'
    
    context = {
        'model_info': model_info,
        'model_loaded': model_loaded
    }
    
    return render(request, 'sentiment/dashboard.html', context)


@require_http_methods(["POST"])
@csrf_exempt
def predict_sentiment(request):
    """API endpoint for sentiment prediction"""
    try:
        data = json.loads(request.body)
        text = data.get('text', '').strip()
        
        if not text:
            return JsonResponse({
                'success': False,
                'message': 'No text provided'
            })
        
        # Load the model
        if not load_model():
            return JsonResponse({
                'success': False,
                'message': 'Failed to load model'
            })
        
        # Get model and vectorizer
        model = loaded_model['model']
        vectorizer = loaded_model['vectorizer']
        
        # Preprocess and predict
        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)[0]
        prediction_proba = model.predict_proba(text_vectorized)[0]
        
        # Get confidence (maximum probability)
        confidence = float(max(prediction_proba))
        
        # Create probability mapping
        classes = model.classes_.tolist()
        probabilities = {}
        for i, class_name in enumerate(classes):
            probabilities[class_name] = float(prediction_proba[i])
        
        return JsonResponse({
            'success': True,
            'prediction': prediction,
            'confidence': confidence,
            'model_used': 'balanced',
            'probabilities': probabilities
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
        # Load model if not already loaded
        load_model()
        
        if loaded_model['loaded']:
            model = loaded_model['model']
            vectorizer = loaded_model['vectorizer']
            
            # Get model metadata
            model_info = {
                'model_type': 'Balanced Multinomial Naive Bayes',
                'algorithm': 'Multinomial Naive Bayes',
                'classes': model.classes_.tolist() if hasattr(model, 'classes_') else [],
                'feature_count': len(vectorizer.get_feature_names_out()) if hasattr(vectorizer, 'get_feature_names_out') else 'Unknown',
                'loaded': True,
                'status': 'active',
                'accuracy': '68.3%',
                'f1_score': '0.683',
                'precision': '0.687',
                'recall': '0.683'
            }
            
            # Add metadata if available
            if loaded_model['metadata']:
                model_info.update(loaded_model['metadata'])
            
            return JsonResponse({
                'success': True,
                'model_info': model_info
            })
        else:
            return JsonResponse({
                'success': False,
                'message': 'Model not loaded'
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
        
        if not texts or not isinstance(texts, list):
            return JsonResponse({
                'success': False,
                'message': 'No texts provided or invalid format'
            })
        
        # Load the model
        if not load_model():
            return JsonResponse({
                'success': False,
                'message': 'Failed to load model'
            })
        
        # Get model and vectorizer
        model = loaded_model['model']
        vectorizer = loaded_model['vectorizer']
        
        # Process all texts
        texts_vectorized = vectorizer.transform(texts)
        predictions = model.predict(texts_vectorized)
        probabilities = model.predict_proba(texts_vectorized)
        
        # Format results
        results = []
        classes = model.classes_.tolist()
        
        for i, (text, prediction, proba) in enumerate(zip(texts, predictions, probabilities)):
            # Create probability mapping
            proba_dict = {}
            for j, class_name in enumerate(classes):
                proba_dict[class_name] = float(proba[j])
            
            results.append({
                'text': text,
                'prediction': prediction,
                'confidence': float(max(proba)),
                'probabilities': proba_dict
            })
        
        return JsonResponse({
            'success': True,
            'results': results,
            'model_used': 'balanced',
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
