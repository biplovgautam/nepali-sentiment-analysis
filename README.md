# Financial Text Sentiment Analysis with Supervised Learning

**A production-ready AI system for financial sentiment classification using balanced Multinomial Naive Bayes**

**STW5000CEM Introduction to Artificial Intelligence - Assignment Project**  
**Module Leader:** Er. Suman Shrestha  
**Institution:** Softwarica College of IT & E-Commerce

---

## 🎯 Project Overview

This project implements a **production-ready financial sentiment analysis system** using supervised learning techniques, specifically focusing on Naive Bayes classification with data balancing techniques. The system analyzes financial text data to classify sentiment as positive, negative, or neutral, providing valuable insights for financial decision-making and market analysis.

### Key Achievements
- **68.3% accuracy** with balanced dataset approach
- **+38.5% improvement** in negative sentiment detection
- **Modern web interface** with real-time predictions
- **Clean, documented codebase** ready for deployment

## 🎨 Modern Web Interface

The project features a **clean, modern web dashboard** built with Django:

### Design Features
- **Financial Theme**: Professional gradient backgrounds and modern styling
- **Simplified Interface**: Focused on the balanced model for optimal performance
- **Responsive Design**: Works seamlessly across all device sizes
- **Real-time Predictions**: Instant sentiment analysis with confidence scoring

### User Experience
- **Single Model Focus**: Uses only the optimized balanced model
- **Clean Dashboard**: Streamlined interface without unnecessary complexity
- **Visual Feedback**: Color-coded results with intuitive sentiment indicators
- **Performance Metrics**: Clear display of model accuracy and statistics

### Technical Stack
- **Backend**: Django 4.2.4 with simplified views and clean API endpoints
- **Frontend**: Bootstrap 5.3, Font Awesome 6.4, modern CSS
- **Model**: Balanced Multinomial Naive Bayes (68.3% accuracy)
- **Deployment**: Production-ready configuration

## 📊 Dataset Information

- **Source**: [Financial Sentiment Analysis Dataset](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis) from Kaggle
- **Original Size**: 5,842 financial text samples  
- **Final Dataset**: 3,600 balanced samples (1,200 per class)
- **Classes**: Positive, Negative, Neutral sentiment labels
- **Domain**: Financial news, reports, and market communications
- **Format**: CSV file with 'Sentence' and 'Sentiment' columns

### Data Quality Assessment
- Complete dataset with no null values or duplicate entries
- Average text length: 117 characters (range: 9-315)
- Rich financial terminology and domain-specific content
- **Major Challenge**: Severe class imbalance (ratio: 3.64)

### Original Class Distribution
- **Neutral**: 3,130 samples (53.6%) - Dominant class
- **Positive**: 1,852 samples (31.7%) - Moderate representation  
- **Negative**: 860 samples (14.7%) - **Severely underrepresented**

## 🔧 Data Preprocessing & Model Development

### Complete ML Pipeline
The project implements a comprehensive machine learning pipeline documented in the Jupyter notebook `utils/tokenize_vectorize_train.ipynb`:

#### Phase 1: Data Balancing Strategy
1. **Smart Data Reduction**: Reduced dominant classes while preserving quality
   - Neutral: 3,130 → 1,200 samples (kept longest texts)
   - Positive: 1,852 → 1,200 samples (kept longest texts)  
   - Negative: 860 → 1,200 samples (duplicated longest texts)

2. **Quality Preservation**: Used sentence length as quality indicator
   - Longer sentences contain more contextual information
   - Better representation of financial domain language

#### Phase 2: Text Preprocessing
- **Tokenization**: Professional text cleaning and normalization
- **Numeric Standardization**: All numbers converted to `<NUM>` tokens
- **Financial Context**: Preserved $, %, EUR symbols for domain relevance
- **Case Normalization**: Lowercase conversion with whitespace cleanup

#### Phase 3: Feature Engineering
- **Vectorization**: TF-IDF with stop word removal
- **N-grams**: Unigrams and bigrams for context capture
- **Feature Limit**: Max 10,000 features for optimal performance
- **Vocabulary**: Domain-specific financial term recognition

### Model Architecture
- **Algorithm**: Multinomial Naive Bayes (optimal for text classification)
- **Training Split**: 80/20 with stratified sampling
- **Cross-validation**: Consistent performance validation
- **Hyperparameters**: Optimized for financial text domain

## 📈 Performance Results

### Balanced Model Performance (Final)
- **Test Accuracy**: 68.3%
- **Macro F1-Score**: 0.683  
- **Weighted F1-Score**: 0.683
- **Precision**: 0.687
- **Recall**: 0.683

### Class-wise Performance
- **Positive**: F1-Score 0.65 (Good performance)
- **Neutral**: F1-Score 0.69 (Best performance) 
- **Negative**: F1-Score 0.71 (Excellent improvement from 0.52)

### Key Improvements
- **Overall F1-Score**: +7.1% improvement (0.637 → 0.683)
- **Negative Detection**: +38.5% improvement (critical for financial risk)
- **Balanced Performance**: Consistent across all sentiment classes
- **Production Ready**: Stable and reliable predictions

## 🗂️ Project Structure

```
aiassignment/
├── 📊 data/
│   ├── financial_sentiment.csv              # Original dataset
│   ├── financial_sentiment_preprocessed.csv # Processed data
│   └── nepali_sentiment_dataset.csv         # Additional dataset
├── 🤖 models/
│   ├── balanced_naive_bayes_model.pkl       # Final optimized model
│   ├── balanced_tfidf_vectorizer.pkl        # Corresponding vectorizer
│   └── balanced_model_metadata.json         # Model configuration
├── 🔧 utils/
│   └── tokenize_vectorize_train.ipynb       # Complete ML pipeline
├── 🌐 sentiment_app/
│   ├── views.py                             # Simplified Django views
│   └── urls.py                              # API endpoints
├── 🎨 templates/sentiment/
│   ├── base_new.html                        # Modern base template
│   └── dashboard_new.html                   # Clean dashboard
├── 📋 requirements.txt                      # Dependencies
├── ⚙️ manage.py                            # Django management
└── 📖 README.md                            # This documentation
```

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 8GB RAM recommended

### Installation

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
   python manage.py runserver
   ```

4. **Access the dashboard**
   Open: `http://localhost:8000`

### API Usage

**Single Prediction**
```bash
curl -X POST http://localhost:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Company profits exceeded expectations this quarter"}'
```

**Batch Prediction**
```bash
curl -X POST http://localhost:8000/api/batch-predict/ \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Stock prices are rising", "Market crash expected"]}'
```

## 🔬 Technical Implementation Details

### Model Training Process
The complete training pipeline is documented in `utils/tokenize_vectorize_train.ipynb`:

1. **Data Loading & EDA**: Comprehensive dataset analysis and visualization
2. **Preprocessing**: Text cleaning, tokenization, and normalization
3. **Balancing**: Smart duplication strategy for minority classes
4. **Vectorization**: TF-IDF feature extraction with domain optimization
5. **Training**: Multinomial Naive Bayes with hyperparameter tuning
6. **Evaluation**: Cross-validation and performance metrics
7. **Serialization**: Model and vectorizer persistence

### Key Technical Decisions

**Balanced Dataset Approach**
- Used sentence length as quality indicator for duplication
- Preserved original negative samples and duplicated longest ones
- Achieved perfect class balance (1,200 samples each)

**TF-IDF Configuration**
- Max features: 10,000 (optimal performance/memory trade-off)
- N-grams: 1-2 (captures context without overfitting)
- Stop words: Removed (focuses on meaningful financial terms)

**Model Selection**
- Multinomial Naive Bayes: Proven effectiveness for text classification
- Fast inference: Sub-second prediction times
- Interpretable: Clear probability distributions for each class

### Production Optimizations
- **Model Caching**: Global model loading prevents repeated file I/O
- **Error Handling**: Comprehensive exception management
- **API Design**: RESTful endpoints with consistent JSON responses
- **Logging**: Detailed logging for monitoring and debugging

## 🎯 Use Cases & Applications

### Financial Industry Applications
- **Risk Assessment**: Early detection of negative sentiment in financial reports
- **Investment Research**: Automated sentiment scoring for stock analysis  
- **News Monitoring**: Real-time sentiment tracking of financial news
- **Customer Feedback**: Analysis of financial service reviews and feedback

### Features for Different Users
- **Analysts**: Batch processing for large document sets
- **Developers**: RESTful API for system integration
- **Researchers**: Documented pipeline for model improvement
- **Business Users**: Web interface for quick sentiment checks

## 📚 Dependencies

### Core ML Libraries
```python
scikit-learn==1.3.0    # Machine learning algorithms
pandas==2.0.3          # Data manipulation
numpy==1.24.3          # Numerical computing
joblib==1.3.1          # Model serialization
```

### Web Framework
```python
Django==4.2.4          # Web framework
django-cors-headers    # API CORS handling
```

### Data Processing
```python
matplotlib==3.7.2      # Visualization
seaborn==0.12.2        # Statistical plots
wordcloud==1.9.2       # Text visualization
```

## 🧪 Testing & Validation

### Model Validation
- **Cross-validation**: 5-fold CV for robust performance estimation
- **Test Set**: 20% holdout for unbiased evaluation
- **Stratified Sampling**: Maintains class distribution in splits

### Performance Monitoring
- **Confusion Matrix**: Detailed class-wise performance analysis
- **Classification Report**: Precision, recall, F1-score per class
- **ROC Curves**: Binary classification performance for each class

## 🔍 Future Improvements

### Model Enhancements
- **Deep Learning**: BERT/FinBERT for better contextual understanding
- **Ensemble Methods**: Combining multiple algorithms for better accuracy
- **Active Learning**: Iterative model improvement with human feedback

### Technical Upgrades
- **Caching**: Redis integration for faster API responses
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **CI/CD**: Automated testing and deployment pipeline
- **Containerization**: Docker deployment for scalability

## 🙏 Acknowledgments

I would like to express my sincere gratitude to **Er. Suman Shrestha**, the module leader for STW5000CEM Introduction to Artificial Intelligence. Throughout the duration of the module and even after its completion, Er. Suman Shrestha has been exceptionally supportive, providing invaluable guidance during classes and remaining available for assistance with project development, technical queries, and academic support. His expertise and dedication have been instrumental in the successful completion of this project.

## 📄 License

This project is developed for academic purposes as part of the STW5000CEM module at Softwarica College of IT & E-Commerce.

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please reach out through the course module or repository issues.

---

**Note**: This project demonstrates production-ready machine learning implementation with a focus on clean code, comprehensive documentation, and practical business applications in financial sentiment analysis.