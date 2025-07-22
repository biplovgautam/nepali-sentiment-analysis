# Financial Sentiment Analysis Project

## Project Overview

This project implements a comprehensive financial sentiment analysis system using machine learning techniques. The system processes financial text data, performs sentiment classification ### Expected Outcomes

### Model Performance Goals
- Overall accuracy: 80-90% (improved from previous 75-85% target due to better class balance)
- Balanced performance across all sentiment classes
- Enhanced handling of financial terminology with diversified numeric tokens
- Fast prediction times for real-time use
- Reduced overfitting due to optimal dataset size and balanceve, negative, neutral), and provides insights into financial market sentiment through natural language processing.

## Dataset Information

### Original Dataset
- **Source**: financial_sentiment.csv
- **Size**: 5,842 samples
- **Features**: Sentence (text), Sentiment (label)
- **Classes**: 
  - Neutral: 3,130 samples (53.6%)
  - Positive: 1,852 samples (31.7%)
  - Negative: 860 samples (14.7%)

### Preprocessed Dataset
- **Source**: financial_sentiment_preprocessed_improved.csv
- **Size**: 3,260 samples
- **Optimally Balanced Class Distribution**:
  - Positive: 1,200 samples (36.8%)
  - Neutral: 1,200 samples (36.8%)
  - Negative: 860 samples (26.4%)
- **Imbalance Ratio**: Significantly improved from 3.64 to 1.40 (62% improvement)

## Exploratory Data Analysis (EDA)

### Initial Analysis
The EDA revealed several key characteristics of the financial text data:

1. **Data Quality**: No null values or duplicate entries
2. **Text Length**: Average of 117 characters, ranging from 9-315 characters
3. **Class Imbalance**: Significant imbalance with neutral class dominating
4. **Financial Context**: Heavy presence of financial terms (EUR, million, company, sales)
5. **Noise Patterns**: URLs, dollar signs, hashtags, and abundant numerical data

### Visualization Insights
- Word clouds revealed distinct vocabulary patterns for each sentiment class
- Text length distributions showed neutral sentences are typically longer
- Token frequency analysis identified key financial terminology

## Data Preprocessing

### Preprocessing Steps Implemented

1. **Optimal Class Balancing**
   - Reduced neutral class from 3,130 to 1,200 samples
   - Reduced positive class from 1,852 to 1,200 samples
   - Retained longest texts (removed shortest ones for quality)
   - Kept all 860 negative samples (smallest class)
   - Achieved excellent class balance (1.40 imbalance ratio)

2. **Enhanced Financial Symbol Preservation**
   - Maintained important financial symbols: $, EUR, %
   - Preserved stock symbols (e.g., $AAPL, $TSLA)
   - Kept currency context intact

3. **Improved Numerical Normalization**
   - Replaced generic NUM token with specific token types:
     - EUR_AMOUNT: For EUR currency amounts
     - USD_AMOUNT: For USD currency amounts
     - PERCENT: For percentage values
     - YEAR: For years (temporal context)
     - LARGE_NUM: For large numbers with units
     - DECIMAL: For decimal numbers/ratios
     - NUM: For remaining standalone numbers
   - Reduced numeric token dominance from overwhelming single token to diverse, meaningful tokens

4. **Text Cleaning**
   - Converted all text to lowercase
   - Removed URLs completely
   - Cleaned extra whitespace
   - Removed unnecessary special characters while preserving financial ones

5. **Domain Knowledge Preservation**
   - Maintained financial keywords: profit, growth, sales, company
   - Preserved business terminology
   - Kept sentiment-bearing words intact

### Preprocessing Results
- Dataset optimized from 5,842 to 3,260 high-quality samples
- Excellent class balance achieved (imbalance ratio: 3.64 → 1.40, 62% improvement)
- Enhanced numeric token diversity (7 specific types vs 1 generic token)
- Preserved domain-specific context and sentiment indicators
- Reduced numeric token percentage to 7.65% while maintaining financial context

## Project Structure

```
aiassignment/
├── data/
│   ├── financial_sentiment.csv                     # Original dataset
│   ├── financial_sentiment_preprocessed_improved.csv # Final preprocessed dataset
│   ├── financial_sentiment_preprocessed.csv        # Previous preprocessing version
│   └── nepali_sentiment_dataset.csv               # Additional dataset
├── models/
│   ├── model.pkl                           # Saved models
│   ├── naive_bayes_model.pkl
│   └── vectorizer.pkl
├── utils/
│   ├── edapreprocess.ipynb                 # EDA and preprocessing notebook
│   ├── eda.py                              # EDA utilities
│   ├── preprocess.py                       # Preprocessing functions
│   ├── model.py                            # Model implementation
│   ├── vectorizer.py                       # Text vectorization
│   ├── tokenizer.py                        # Text tokenization
│   └── evaluator.py                        # Model evaluation
├── sentiment_app/                          # Django web application
├── templates/                              # HTML templates
├── main.py                                 # Main execution script
├── config.py                              # Configuration settings
└── README.md                              # Project documentation
```

## Next Steps: Model Development Pipeline

### 1. Tokenization
**Objective**: Convert preprocessed text into tokens suitable for machine learning

**Planned Implementation**:
- Use NLTK or spaCy for tokenization
- Handle financial domain-specific tokens
- Apply additional preprocessing if needed
- Remove stop words while preserving financial terms
- Implement stemming/lemmatization for token normalization

**Expected Output**: Tokenized text ready for vectorization

### 2. Vectorization
**Objective**: Transform tokens into numerical feature vectors

**Planned Approaches**:
- **TF-IDF Vectorization**: 
  - Capture term importance across documents
  - Handle financial terminology effectively
  - Set appropriate n-gram ranges (1-2 or 1-3)
  - Configure max_features based on dataset size
- **Count Vectorization**: 
  - Simple frequency-based representation
  - Good baseline for comparison
- **Advanced Options**: Word2Vec or FastText for semantic representations

**Configuration Parameters**:
- max_features: 5000-10000 (based on vocabulary size)
- ngram_range: (1, 2) for capturing phrases
- min_df: 2-5 (remove very rare terms)
- max_df: 0.95 (remove very common terms)

### 3. Naive Bayes Model Training
**Objective**: Build and train sentiment classification model

**Model Selection Rationale**:
- **Multinomial Naive Bayes**: Ideal for text classification
- **Gaussian Naive Bayes**: Alternative for normalized features
- **Complement Naive Bayes**: Good for imbalanced datasets

**Training Strategy**:
- Train-test split: 80-20 or 70-30
- Cross-validation for robust evaluation
- Hyperparameter tuning for optimal performance
- Handle class imbalance with appropriate techniques

**Expected Performance Metrics**:
- Accuracy: Target 75-85%
- Precision, Recall, F1-score for each class
- Confusion matrix analysis
- ROC-AUC for binary comparisons

### 4. Model Evaluation and Validation
**Evaluation Metrics**:
- Classification accuracy
- Precision, recall, F1-score per class
- Macro and micro-averaged metrics
- Confusion matrix analysis
- Cross-validation scores

**Validation Approach**:
- Stratified train-test split
- K-fold cross-validation (k=5)
- Performance comparison with baseline models
- Error analysis on misclassified samples

### 5. Model Deployment Considerations
**Serialization**:
- Save trained model using pickle or joblib
- Store vectorizer alongside model
- Create model loading utilities

**Web Application Integration**:
- Django-based web interface for sentiment analysis
- Real-time prediction capabilities
- User-friendly input interface
- Results visualization and interpretation

## Technical Requirements

### Dependencies
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- matplotlib, seaborn: Data visualization
- wordcloud: Text visualization
- scikit-learn: Machine learning algorithms
- nltk/spaCy: Natural language processing
- Django: Web framework (for application)

### Hardware Requirements
- Minimum 4GB RAM for dataset processing
- CPU: Multi-core processor recommended
- Storage: 1GB for datasets and models

## Usage Instructions

### Running EDA and Preprocessing
```bash
# Navigate to project directory
cd /home/biplovgautam/Desktop/aiassignment

# Run Jupyter notebook for EDA
jupyter notebook utils/edapreprocess.ipynb

# Or run preprocessing script
python utils/preprocess.py
```

### Future Model Training
```bash
# Tokenization
python utils/tokenizer.py

# Vectorization
python utils/vectorizer.py

# Model training
python utils/model.py

# Evaluation
python utils/evaluator.py
```

### Web Application
```bash
# Run Django development server
python manage.py runserver

# Access application at http://localhost:8000
```

## Expected Outcomes

### Model Performance Goals
- Overall accuracy: 75-85%
- Balanced performance across all sentiment classes
- Robust handling of financial terminology
- Fast prediction times for real-time use

### Business Value
- Automated sentiment analysis of financial news
- Market sentiment monitoring capabilities
- Support for investment decision-making
- Scalable solution for large-scale text analysis

## Contributing

To contribute to this project:
1. Follow the established code structure
2. Maintain data preprocessing pipeline integrity
3. Document all new features and modifications
4. Test changes against the validation dataset
5. Update this README with any significant changes

## License and Acknowledgments

This project is developed for educational and research purposes. Please ensure appropriate licensing for any commercial use of the financial sentiment dataset and models.
