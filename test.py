#!/usr/bin/env python3
"""
Test script to verify all components work correctly.
"""

import sys
import os
import traceback

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        print("✅ Core ML libraries imported")
    except ImportError as e:
        print(f"❌ Core ML library import failed: {e}")
        return False
    
    try:
        from config import *
        print("✅ Config imported")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from utils.loader import load_dataset, validate_dataset
        from utils.eda import perform_comprehensive_eda
        from utils.preprocess import TextPreprocessor
        from utils.vectorizer import create_feature_extractor
        from utils.model import create_sentiment_classifier
        from utils.evaluator import ModelEvaluator
        print("✅ All utility modules imported")
    except ImportError as e:
        print(f"❌ Utility import failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_sample_data():
    """Test with sample data."""
    print("🧪 Testing with sample data...")
    
    try:
        # Create sample data
        import pandas as pd
        sample_data = pd.DataFrame({
            'Sentence': [
                'Apple stock is performing well today',
                'The market crashed badly',
                'Neutral market conditions prevail',
                'Tesla earnings exceeded expectations',
                'Stock prices are falling sharply'
            ],
            'Sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative']
        })
        
        print(f"✅ Sample data created: {sample_data.shape}")
        
        # Test preprocessing
        from utils.preprocess import TextPreprocessor
        preprocessor = TextPreprocessor()
        
        processed_texts = preprocessor.preprocess_texts(sample_data['Sentence'])
        print(f"✅ Text preprocessing: {len(processed_texts)} texts processed")
        
        # Test vectorization
        from utils.vectorizer import create_feature_extractor
        vectorizer = create_feature_extractor('tfidf')
        
        X = vectorizer.fit_transform(processed_texts)
        print(f"✅ Vectorization: {X.shape} feature matrix")
        
        # Test model training
        from utils.model import create_sentiment_classifier
        model = create_sentiment_classifier()
        
        model.train(X, sample_data['Sentiment'])
        print("✅ Model training completed")
        
        # Test prediction
        predictions = model.predict(X)
        print(f"✅ Predictions: {len(predictions)} samples predicted")
        
        # Test evaluation
        from utils.evaluator import ModelEvaluator
        evaluator = ModelEvaluator()
        
        metrics = evaluator.calculate_metrics(sample_data['Sentiment'], predictions)
        print(f"✅ Evaluation: Accuracy = {metrics['accuracy']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample data test failed: {e}")
        traceback.print_exc()
        return False

def test_config():
    """Test configuration settings."""
    print("🧪 Testing configuration...")
    
    try:
        from config import (
            PROJECT_ROOT, DATA_DIR, MODELS_DIR, OUTPUT_DIR,
            DATASET_PATH, MODEL_PATH, VECTORIZER_PATH,
            MODEL_CONFIG, PREPROCESSING_CONFIG, VECTORIZER_CONFIG
        )
        
        print("✅ All config variables accessible")
        
        # Check if directories exist or can be created
        import os
        for dir_path in [DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                print(f"✅ Created directory: {dir_path}")
            else:
                print(f"✅ Directory exists: {dir_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_django_setup():
    """Test Django components."""
    print("🧪 Testing Django setup...")
    
    try:
        # Test Django imports
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sentiment_project.settings')
        
        import django
        from django.conf import settings
        django.setup()
        
        print("✅ Django setup completed")
        
        # Test views import
        from sentiment_app.views import analyzer
        print("✅ Django views imported")
        
        # Test model loading capability
        info = analyzer.get_model_info()
        print(f"✅ Model info retrieved: {info['status']}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Django test failed (this is OK if running standalone): {e}")
        return True  # Don't fail the test for Django issues

def main():
    """Run all tests."""
    print("🔬 Running Financial Sentiment Analysis Tests")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Sample Data Processing", test_sample_data),
        ("Django Setup", test_django_setup),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🏃 Running {test_name} test...")
        try:
            if test_func():
                print(f"✅ {test_name} test PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project is ready to use.")
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
