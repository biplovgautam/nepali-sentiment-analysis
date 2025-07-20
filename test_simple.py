#!/usr/bin/env python3
"""
Simple test to verify the project works.
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_imports():
    """Test all imports."""
    print("Testing imports...")
    
    try:
        from utils.loader import load_dataset
        print("✅ loader import OK")
    except Exception as e:
        print(f"❌ loader import failed: {e}")
        return False
    
    try:
        from utils.preprocess import TextPreprocessor
        print("✅ preprocess import OK")
    except Exception as e:
        print(f"❌ preprocess import failed: {e}")
        return False
    
    try:
        from utils.vectorizer import create_feature_extractor
        print("✅ vectorizer import OK")
    except Exception as e:
        print(f"❌ vectorizer import failed: {e}")
        return False
    
    try:
        from utils.model import create_sentiment_classifier
        print("✅ model import OK")
    except Exception as e:
        print(f"❌ model import failed: {e}")
        return False
    
    return True

def test_data_loading():
    """Test data loading."""
    print("\nTesting data loading...")
    
    try:
        from utils.loader import load_dataset
        
        if not os.path.exists('data/financial_sentiment.csv'):
            print("❌ Dataset file not found")
            return False
        
        df = load_dataset('data/financial_sentiment.csv')
        print(f"✅ Dataset loaded: {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")
        
        if 'Sentiment' in df.columns:
            print(f"   Classes: {df['Sentiment'].unique()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_pipeline():
    """Test a minimal pipeline."""
    print("\nTesting minimal pipeline...")
    
    try:
        from utils.loader import load_dataset
        from utils.preprocess import TextPreprocessor
        from utils.vectorizer import create_feature_extractor
        from utils.model import create_sentiment_classifier
        from sklearn.model_selection import train_test_split
        
        # Load small sample
        df = load_dataset('data/financial_sentiment.csv')
        sample_df = df.sample(n=min(100, len(df)), random_state=42)
        print(f"   Using sample: {sample_df.shape}")
        
        # Preprocess
        preprocessor = TextPreprocessor()
        texts = preprocessor.preprocess_texts(sample_df['Sentence'].tolist())
        print("   Preprocessing OK")
        
        # Vectorize
        vectorizer = create_feature_extractor('tfidf', max_features=1000)
        X = vectorizer.fit_transform(texts)
        y = sample_df['Sentiment'].values
        print(f"   Vectorization OK: {X.shape}")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"   Data split OK: train={X_train.shape}, test={X_test.shape}")
        
        # Train
        classifier = create_sentiment_classifier()
        classifier.train(X_train, y_train)
        print("   Training OK")
        
        # Test
        accuracy = classifier.evaluate(X_test, y_test)['accuracy']
        print(f"   ✅ Pipeline test successful! Accuracy: {accuracy:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Running project tests...\n")
    
    tests = [
        ("Import test", test_imports),
        ("Data loading test", test_data_loading),
        ("Pipeline test", test_pipeline)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    print('='*50)
    
    if passed == total:
        print("🎉 All tests passed! Project is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
