#!/usr/bin/env python3
"""
Enhanced Financial Sentiment Analysis - Production Server
Run this script to start the Django application with the enhanced ML pipeline.
"""

import os
import sys
import subprocess

def main():
    """Start the Django development server."""
    
    print("🚀 Starting Enhanced Financial Sentiment Analysis Application...")
    print("=" * 60)
    
    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    # Check if virtual environment is activated
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Warning: No virtual environment detected. Consider using 'python -m venv venv'")
    
    # Check for required packages
    try:
        import django
        import sklearn
        import matplotlib
        import pandas
        import numpy
        print("✅ All required packages found")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("📥 Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Run migrations
    print("🔄 Running database migrations...")
    subprocess.run([sys.executable, "manage.py", "migrate"], check=False)
    
    # Start server
    print("🌟 Enhanced Features:")
    print("   • Advanced Data Balancing (neutral class overfitting prevention)")
    print("   • Model Parameter History & Optimization")
    print("   • Comprehensive EDA (before/after preprocessing)")
    print("   • Real-time Predictions with Confidence Scoring")
    print("   • Interactive Retraining with Custom Parameters")
    print()
    print("🌐 Starting server at: http://localhost:8000")
    print("📊 Dashboard includes:")
    print("   • Data Analysis Tab: EDA, balancing, preprocessing impact")
    print("   • Model Evaluation Tab: Confusion matrix, metrics, comprehensive analysis")
    print("   • Live Prediction Tab: Real-time sentiment analysis")
    print("   • Model History Tab: Parameter tracking, performance comparison")
    print()
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        subprocess.run([sys.executable, "manage.py", "runserver", "0.0.0.0:8000"])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()
