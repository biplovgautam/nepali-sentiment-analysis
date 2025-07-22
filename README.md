# Financial Text Sentiment Analysis with Supervised Learning: A Naive Bayes-Based Classifier

**STW5000CEM Introduction to Artificial Intelligence - Assignment Project**  
**Module Leader:** Er. Suman Shrestha  
**Institution:** Softwarica College of IT & E-Commerce

---

## Acknowledgments

I would like to express my sincere gratitude to **Er. Suman Shrestha**, the module leader for STW5000CEM Introduction to Artificial Intelligence. Throughout the duration of the module and even after its completion, Er. Suman Shrestha has been exceptionally supportive, providing invaluable guidance during classes and remaining available for assistance with project development, technical queries, and academic support. His expertise and dedication have been instrumental in the successful completion of this project.

---

## Project Overview

This project implements a financial sentiment analysis system using supervised learning techniques, specifically focusing on Naive Bayes classification. The system analyzes financial text data to classify sentiment as positive, negative, or neutral, providing valuable insights for financial decision-making and market analysis.

## Modern Web Interface

The project features a **responsive, modern web dashboard** built with Django and enhanced with:

### 🎨 **Design Features**
- **Financial Theme**: Gradient backgrounds with subtle patterns evoking financial markets
- **Floating Sidebar**: Non-intrusive left sidebar with model switching capabilities
- **Glass Morphism**: Modern backdrop-blur effects for cards and components
- **Responsive Design**: Optimized for desktop, tablet, and mobile devices

### 🔧 **Interactive Components**
- **Model Switcher**: Real-time switching between Original and Balanced models
- **Quick Prediction**: Instant sentiment analysis with confidence scoring
- **Performance Metrics**: Live model statistics and comparison charts
- **Dataset Overview**: Visual representation of class distribution and statistics

### 📱 **User Experience**
- **Single Page Interface**: All functionality accessible from one dashboard
- **Collapsible Sidebar**: Space-efficient design with expandable/collapsible navigation
- **Real-time Updates**: Dynamic content updates without page refreshes
- **Visual Feedback**: Color-coded results with intuitive sentiment indicators

### 🚀 **Technical Stack**
- **Frontend**: Bootstrap 5.3, Font Awesome 6.4, Google Fonts (Inter)
- **Backend**: Django 4.2.4 with simplified views and API endpoints
- **Styling**: Modern CSS with CSS Grid, Flexbox, and custom animations
- **JavaScript**: Vanilla JS for interactivity and API communication

## Dataset Information

- **Source**: [Financial Sentiment Analysis Dataset](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis) from Kaggle
- **Original Size**: 5,842 financial text samples
- **Classes**: Positive, Negative, Neutral sentiment labels
- **Domain**: Financial news, reports, and market communications
- **Format**: CSV file with 'Sentence' and 'Sentiment' columns

## Initial Data Analysis

**Data Quality Assessment:**
- Complete dataset with no null values or duplicate entries
- Average text length: 117 characters (range: 9-315)
- Rich financial terminology and numeric content requiring preprocessing

**Class Distribution Problem:**
- Severe class imbalance detected (ratio: 3.64)
- Neutral: 3,130 samples (53.6%)
- Positive: 1,852 samples (31.7%) 
- Negative: 860 samples (14.7%)

## Data Preprocessing Pipeline

**Phase 1: Initial Balancing Strategy**
- Reduced dominant classes while preserving quality
- Neutral: 3,130 → 1,200 samples (kept longest texts)
- Positive: 1,852 → 1,200 samples (kept longest texts)
- Negative: 860 samples retained (minority class)
- Result: Improved imbalance ratio from 3.64 → 1.40

**Phase 2: Text Processing**
- Numeric standardization: All numbers converted to `<NUM>` tokens
- Financial symbol preservation: Maintained $, %, EUR for domain context
- Lowercase conversion and whitespace normalization
- URL removal and special character cleaning

## Model Development and Training

**Model Architecture:**
- Algorithm: Multinomial Naive Bayes
- Vectorization: TF-IDF with stop word removal
- Features: Unigrams and bigrams (max 10,000 features)
- Training configuration: 80/20 split with stratified sampling

**Initial Model Performance:**
- Test Accuracy: 66.3%
- Macro F1-Score: 0.637
- Weighted F1-Score: 0.656
- Challenge: Poor negative sentiment detection (F1: 0.516)

## Model Improvement Strategy

**Problem Identified:**
The initial model struggled with negative sentiment classification due to limited training samples (860 vs 1,200 for other classes).

**Solution Implemented:**
- Balanced all classes to 1,200 samples each
- Duplicated longest negative sentences to reach target size
- Retrained model with fully balanced dataset
- Applied same preprocessing and vectorization pipeline

**Improved Model Performance:**
- Test Accuracy: 68.3% (+3.1% improvement)
- Macro F1-Score: 0.683 (+7.1% improvement)
- Weighted F1-Score: 0.683 (+5.0% improvement)
- Negative F1-Score: 0.714 (+38.5% improvement)

## Key Technical Decisions

**Why Save Separate Vectorizers:**
Each model uses its corresponding TF-IDF vectorizer because they were fitted on different datasets (original vs balanced), potentially resulting in different vocabularies and feature spaces.

**Stop Word Removal:**
Both models implement English stop word removal during vectorization to focus on meaningful financial terms and improve classification accuracy.

## Final Results and Artifacts

**Saved Models:**
- `naive_bayes_model.pkl` - Original model (66.3% accuracy)
- `tfidf_vectorizer.pkl` - Original vectorizer
- `balanced_naive_bayes_model.pkl` - Improved model (68.3% accuracy)
- `balanced_tfidf_vectorizer.pkl` - Balanced dataset vectorizer

**Performance Summary:**
The balanced approach demonstrates significant improvement in overall performance, particularly for negative sentiment detection, making it the recommended model for deployment.

## Conclusion

Dataset balancing through intelligent duplication of high-quality samples proves effective for improving sentiment classification performance. The 38.5% improvement in negative sentiment detection while maintaining overall accuracy demonstrates the value of addressing class imbalance in financial text analysis.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd aiassignment
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python start_server.py
   ```
   
   Or manually:
   ```bash
   python manage.py migrate
   python manage.py runserver
   ```

4. **Access the dashboard**
   Open your browser and navigate to: `http://localhost:8000`

### 🎯 Features Available

- **Model Comparison**: Switch between Original and Balanced models in real-time
- **Instant Predictions**: Test sentiment analysis with your own financial text
- **Performance Metrics**: View accuracy, F1-score, precision, and recall
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **API Endpoints**: Programmatic access for batch processing

### 📊 Model Performance

| Model | Accuracy | F1-Score | Negative F1 | Improvement |
|-------|----------|----------|-------------|-------------|
| Original | 66.3% | 0.637 | 0.516 | Baseline |
| Balanced | **68.3%** | **0.683** | **0.714** | **+7.1%** |

The balanced model shows significant improvement, especially for negative sentiment detection (+38.5% F1-score improvement).

