#!/usr/bin/env python3
"""
Minimal version of the sentiment analysis pipeline that works without complex dependencies.
"""

import sys
import os

def show_differences():
    """Explain the difference between main.py and run.py"""
    print("=" * 70)
    print("DIFFERENCE BETWEEN main.py AND run.py")
    print("=" * 70)
    print()
    
    print("📄 main.py - The ML Pipeline Core:")
    print("   • Contains the complete machine learning pipeline")
    print("   • Handles data loading, preprocessing, training, evaluation")
    print("   • Command-line interface for ML operations")
    print("   • Advanced configuration and parameter tuning")
    print("   • Generates models, reports, and visualizations")
    print("   • Direct control over ML hyperparameters")
    print()
    print("   Usage examples:")
    print("   - python main.py                           # Run with defaults")
    print("   - python main.py --alpha 1.5 --max-features 5000")
    print("   - python main.py --config custom_config.json")
    print()
    
    print("🚀 run.py - The Project Manager/Runner:")
    print("   • Easy-to-use wrapper script")
    print("   • Manages different project operations")
    print("   • Handles environment setup automatically")
    print("   • Provides simple commands for common tasks")
    print("   • Status checking and health monitoring")
    print("   • Web server management")
    print()
    print("   Usage examples:")
    print("   - python run.py --setup                   # Initial setup")
    print("   - python run.py --pipeline                # Run ML pipeline")
    print("   - python run.py --web                     # Start web server")
    print("   - python run.py --demo                    # Quick demo")
    print("   - python run.py --status                  # Check status")
    print()
    
    print("🔄 Workflow Comparison:")
    print()
    print("   Direct ML Approach (main.py):")
    print("   1. python main.py --alpha 2.0 --vectorizer tfidf")
    print("   ➜ Runs ML pipeline with specific parameters")
    print()
    print("   Project Management Approach (run.py):")
    print("   1. python run.py --setup      # Set up environment")
    print("   2. python run.py --status     # Check readiness")
    print("   3. python run.py --pipeline   # Run the pipeline")
    print("   4. python run.py --web        # Start web interface")
    print()
    
    print("💡 When to use which:")
    print("   • Use main.py when:")
    print("     - You want direct control over ML parameters")
    print("     - Experimenting with different configurations")
    print("     - Integrating into other ML workflows")
    print("     - Need detailed ML pipeline control")
    print()
    print("   • Use run.py when:")
    print("     - First time setting up the project")
    print("     - Want simple, guided operations")
    print("     - Managing multiple project components")
    print("     - Need quick demos or status checks")
    print("=" * 70)

def check_project_structure():
    """Check and display project structure."""
    print("\n📁 PROJECT STRUCTURE STATUS:")
    print("-" * 50)
    
    required_files = [
        ("main.py", "ML Pipeline Core"),
        ("run.py", "Project Runner"),
        ("config.py", "Configuration"),
        ("requirements.txt", "Dependencies"),
        ("data/financial_sentiment.csv", "Dataset"),
        ("utils/loader.py", "Data Loader"),
        ("utils/preprocess.py", "Text Preprocessor"),
        ("utils/vectorizer.py", "Feature Extractor"),
        ("utils/model.py", "ML Model"),
        ("utils/evaluator.py", "Evaluator"),
        ("manage.py", "Django Manager"),
        ("sentiment_project/settings.py", "Django Settings"),
        ("sentiment_app/views.py", "Web Views")
    ]
    
    for file_path, description in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path:<35} ({description})")
        else:
            print(f"❌ {file_path:<35} ({description})")

def show_usage_guide():
    """Show how to use the project."""
    print("\n🎯 USAGE GUIDE:")
    print("-" * 50)
    print("1. First-time setup:")
    print("   python run.py --setup")
    print()
    print("2. Check everything is ready:")
    print("   python run.py --status")
    print()
    print("3. Run the ML pipeline:")
    print("   python run.py --pipeline")
    print("   # OR for custom parameters:")
    print("   python main.py --alpha 1.5 --max-features 8000")
    print()
    print("4. Start the web interface:")
    print("   python run.py --web")
    print("   # Then visit http://127.0.0.1:8000/")
    print()
    print("5. Quick demo:")
    print("   python run.py --demo")

def main():
    """Main function."""
    print("🤖 Financial Sentiment Analysis Project")
    print("Understanding main.py vs run.py")
    
    show_differences()
    check_project_structure()
    show_usage_guide()
    
    print("\n" + "=" * 70)
    print("✨ Project is ready to use!")
    print("Start with: python run.py --status")
    print("=" * 70)

if __name__ == "__main__":
    main()
