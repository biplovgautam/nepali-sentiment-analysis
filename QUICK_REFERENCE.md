# Financial Sentiment Analysis - Quick Reference

## 🔥 Key Differences: main.py vs run.py

### main.py - The ML Engine
**What it does:** Direct control of the machine learning pipeline
**When to use:** When you want to:
- Experiment with different ML parameters
- Run custom configurations
- Have fine-grained control over the pipeline
- Integrate into other ML workflows

**Examples:**
```bash
# Basic run with defaults
python main.py

# Custom parameters
python main.py --alpha 1.5 --max-features 8000 --vectorizer tfidf

# Use custom configuration file
python main.py --config my_experiment.json

# Quick test without saving models
python main.py --no-save --no-report
```

### run.py - The Project Manager
**What it does:** Easy project management and automation
**When to use:** When you want to:
- Set up the project for the first time
- Check if everything is working
- Run common operations easily
- Manage the web interface

**Examples:**
```bash
# First time setup
python run.py --setup

# Check project health
python run.py --status

# Run the pipeline with default settings
python run.py --pipeline

# Start the web interface
python run.py --web

# Quick demonstration
python run.py --demo
```

## ✅ Project Status
- **Dataset:** 5,842 financial texts (3 sentiment classes)
- **Model:** Naive Bayes with 70.4% accuracy
- **Features:** TF-IDF vectorization with 3,000 features
- **Cross-validation:** 68.4% ± 2.2% (5-fold CV)
- **Processing time:** ~24 seconds for full pipeline

## 🚀 Quick Start
1. `python run.py --status` - Check if everything is ready
2. `python main.py` - Run the ML pipeline
3. Check `output/` folder for visualizations and reports
4. Models saved in `models/` folder

## 📁 Key Output Files
- `output/class_distribution.png` - Dataset distribution
- `output/wordcloud_*.png` - Word clouds by sentiment
- `output/confusion_matrix.png` - Model performance
- `output/final_report_*.txt` - Complete analysis report
- `models/naive_bayes_model.pkl` - Trained model
- `models/vectorizer.pkl` - Feature extractor

## 🌐 Web Interface
- Start: `python run.py --web`
- Visit: `http://127.0.0.1:8000/`
- API: `/api/predict/` for single predictions
- Batch API: `/api/predict_batch/` for multiple texts

## 🔧 Advanced Usage
- Modify `config.py` for default parameters
- Create JSON config files for experiments
- Use `utils/` modules in your own scripts
- Extend with new preprocessing or models

## 🎯 Project Architecture
```
main.py           # ML Pipeline orchestrator
run.py           # Project management wrapper
config.py        # Central configuration
utils/           # Modular ML components
├── loader.py    # Data loading & validation
├── preprocess.py # Text cleaning
├── vectorizer.py # Feature extraction
├── model.py     # ML training & evaluation
├── evaluator.py # Metrics & visualization
└── gui.py       # Web integration
```

**The project is complete and working! Both main.py and run.py serve different but complementary purposes for a comprehensive ML workflow.**
