#!/usr/bin/env python3
"""
Demo script for Financial Sentiment Analysis API
Shows how to use both models for predictions
"""

import requests
import json
import time

# API endpoints
BASE_URL = "http://localhost:8000"
PREDICT_URL = f"{BASE_URL}/api/predict/"
MODEL_INFO_URL = f"{BASE_URL}/api/model-info/"

def test_sentiment_prediction():
    """Test sentiment prediction with both models"""
    
    # Sample financial texts
    test_texts = [
        "The company's quarterly earnings exceeded expectations with a 15% revenue growth.",
        "Stock prices plummeted due to regulatory concerns and market volatility.",
        "The merger announcement had minimal impact on share prices this quarter.",
        "Strong performance in the tech sector boosted overall market confidence.",
        "Economic uncertainty continues to affect investor sentiment negatively."
    ]
    
    print("🧪 Testing Financial Sentiment Analysis API")
    print("=" * 60)
    
    for model_type in ['original', 'balanced']:
        print(f"\n📊 Testing {model_type.upper()} Model")
        print("-" * 40)
        
        for text in test_texts:
            try:
                # Make prediction request
                payload = {
                    "text": text,
                    "model": model_type
                }
                
                response = requests.post(
                    PREDICT_URL,
                    headers={'Content-Type': 'application/json'},
                    data=json.dumps(payload),
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result['success']:
                        sentiment = result['prediction']
                        confidence = result['confidence']
                        
                        # Color coding for terminal output
                        colors = {
                            'positive': '\033[92m',  # Green
                            'negative': '\033[91m',  # Red
                            'neutral': '\033[93m'    # Yellow
                        }
                        reset_color = '\033[0m'
                        
                        color = colors.get(sentiment, '')
                        
                        print(f"Text: {text[:50]}...")
                        print(f"Sentiment: {color}{sentiment.upper()}{reset_color} "
                              f"(Confidence: {confidence:.3f})")
                        print()
                    else:
                        print(f"❌ Prediction failed: {result['message']}")
                else:
                    print(f"❌ HTTP Error: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ Connection error: {e}")
                print("💡 Make sure the Django server is running on localhost:8000")
                return
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print()

def get_model_info():
    """Get information about available models"""
    
    print("📋 Model Information")
    print("=" * 60)
    
    for model_type in ['original', 'balanced']:
        try:
            response = requests.get(
                MODEL_INFO_URL,
                params={'model': model_type},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    info = result['model_info']
                    print(f"\n🤖 {model_type.upper()} Model:")
                    print(f"   Algorithm: {info.get('algorithm', 'N/A')}")
                    print(f"   Classes: {', '.join(info.get('classes', []))}")
                    print(f"   Accuracy: {info.get('accuracy', 'N/A')}")
                    print(f"   F1-Score: {info.get('f1_score', 'N/A')}")
                    print(f"   Status: {info.get('status', 'N/A')}")
                else:
                    print(f"❌ Failed to get {model_type} model info: {result['message']}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")
            print("💡 Make sure the Django server is running on localhost:8000")
            return
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Financial Sentiment Analysis - API Demo")
    print("Make sure the Django server is running first!")
    print()
    
    # Wait a moment for user to read
    time.sleep(2)
    
    # Test model information
    get_model_info()
    
    print("\n" + "=" * 60)
    
    # Test predictions
    test_sentiment_prediction()
    
    print("✅ Demo completed!")
    print("\n💻 To start the web interface:")
    print("   python start_server.py")
    print("   Then open: http://localhost:8000")
