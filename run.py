#!/usr/bin/env python3
"""
Simple runner script for the Financial Sentiment Analysis project.
This script provides easy commands to run different parts of the project.
"""

import os
import sys
import subprocess
import argparse

def run_pipeline():
    """Run the main ML pipeline."""
    print("🚀 Running Financial Sentiment Analysis Pipeline...")
    result = subprocess.run([sys.executable, "main.py"], cwd=os.getcwd())
    return result.returncode == 0

def run_web_server():
    """Run the Django web server."""
    print("🌐 Starting Django web server...")
    print("Visit http://127.0.0.1:8000/ to use the web interface")
    
    # Check if models exist
    if not os.path.exists("models/model.pkl"):
        print("⚠️  Models not found. Please run the pipeline first with: python run.py --pipeline")
        return False
    
    result = subprocess.run([sys.executable, "manage.py", "runserver"], cwd=os.getcwd())
    return result.returncode == 0

def install_requirements():
    """Install required Python packages."""
    print("📦 Installing required packages...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    return result.returncode == 0

def setup_project():
    """Set up the project environment."""
    print("🔧 Setting up project environment...")
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        return False
    
    # Download required NLTK data
    print("📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("✅ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️  Error downloading NLTK data: {e}")
    
    # Try to download spaCy model
    print("📚 Downloading spaCy model...")
    try:
        result = subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ spaCy model downloaded")
        else:
            print("⚠️  Could not download spaCy model. Tokenization will use NLTK only.")
    except Exception as e:
        print(f"⚠️  Error downloading spaCy model: {e}")
    
    print("✅ Project setup completed!")
    return True

def run_demo():
    """Run a quick demo with sample data."""
    print("🎯 Running quick demo...")
    
    # Check if we have the dataset
    if not os.path.exists("data/financial_sentiment.csv"):
        print("❌ Dataset not found at data/financial_sentiment.csv")
        print("Please ensure you have the dataset file in the correct location.")
        return False
    
    # Run pipeline with limited data
    result = subprocess.run([
        sys.executable, "main.py", 
        "--data", "data/financial_sentiment.csv",
        "--alpha", "1.0",
        "--vectorizer", "tfidf",
        "--max-features", "5000"
    ], cwd=os.getcwd())
    
    return result.returncode == 0

def check_status():
    """Check the status of the project."""
    print("🔍 Checking project status...")
    
    # Check if dataset exists
    dataset_path = "data/financial_sentiment.csv"
    if os.path.exists(dataset_path):
        print(f"✅ Dataset found: {dataset_path}")
        # Get dataset info
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path)
            print(f"   - Shape: {df.shape}")
            if 'Sentiment' in df.columns:
                print(f"   - Classes: {df['Sentiment'].unique().tolist()}")
                print(f"   - Distribution: {df['Sentiment'].value_counts().to_dict()}")
        except Exception as e:
            print(f"   - Error reading dataset: {e}")
    else:
        print(f"❌ Dataset not found: {dataset_path}")
    
    # Check if models exist
    model_path = "models/model.pkl"
    vectorizer_path = "models/vectorizer.pkl"
    
    if os.path.exists(model_path):
        print(f"✅ Model found: {model_path}")
    else:
        print(f"❌ Model not found: {model_path}")
    
    if os.path.exists(vectorizer_path):
        print(f"✅ Vectorizer found: {vectorizer_path}")
    else:
        print(f"❌ Vectorizer not found: {vectorizer_path}")
    
    # Check output directory
    if os.path.exists("output"):
        output_files = os.listdir("output")
        print(f"✅ Output directory exists with {len(output_files)} files")
    else:
        print("❌ Output directory not found")
    
    # Check required packages
    try:
        import pandas, numpy, sklearn, nltk, matplotlib, seaborn, wordcloud, joblib
        print("✅ Core packages installed")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
    
    print("🔍 Status check completed")

def main():
    parser = argparse.ArgumentParser(
        description="Financial Sentiment Analysis Project Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --setup              # Set up the project
  python run.py --pipeline           # Run the ML pipeline
  python run.py --demo               # Run a quick demo
  python run.py --web                # Start web server
  python run.py --status             # Check project status
        """
    )
    
    parser.add_argument('--setup', action='store_true', help='Set up project environment')
    parser.add_argument('--pipeline', action='store_true', help='Run the ML pipeline')
    parser.add_argument('--web', action='store_true', help='Start the web server')
    parser.add_argument('--demo', action='store_true', help='Run a quick demo')
    parser.add_argument('--status', action='store_true', help='Check project status')
    parser.add_argument('--install', action='store_true', help='Install requirements only')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    success = True
    
    if args.install:
        success = install_requirements()
    
    if args.setup:
        success = success and setup_project()
    
    if args.status:
        check_status()
    
    if args.demo:
        success = success and run_demo()
    
    if args.pipeline:
        success = success and run_pipeline()
    
    if args.web:
        success = success and run_web_server()
    
    if success:
        print("\n✅ All operations completed successfully!")
    else:
        print("\n❌ Some operations failed. Check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
