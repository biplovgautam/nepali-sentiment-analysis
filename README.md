# Financial Text Sentiment Analysis with Supervised Learning

**A production-ready AI system for financial sentiment classification using balanced Multinomial Naive Bayes**

**STW5000CEM Introduction to Artificial Intelligence - Assignment Project**  
**Module Leader:** Er. Suman Shrestha  
**Institution:** Softwarica College of IT & E-Commerce

---

## 🎯 Project Overview

This project implements a **comprehensive machine learning system for financial sentiment analysis** using supervised learning techniques with advanced data balancing strategies. The core focus is on developing a robust Multinomial Naive Bayes classifier that addresses the critical challenge of class imbalance in financial text data, achieving significant improvements in minority class detection while maintaining overall performance.

### Key Technical Achievements
- **68.3% accuracy** with balanced dataset approach
- **+38.5% improvement** in negative sentiment detection (critical for financial risk assessment)
- **Advanced data balancing**: Smart duplication strategy preserving data quality
- **Production-ready ML pipeline** with comprehensive evaluation metrics
- **Domain-optimized preprocessing** for financial text analysis

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

### Problem Analysis: Class Imbalance Challenge
The primary technical challenge was addressing severe class imbalance in the financial sentiment dataset:
- **Imbalance Ratio**: 3.64:1 (neutral:negative)
- **Impact**: Poor minority class (negative) detection - critical for financial risk assessment
- **Solution**: Intelligent data balancing with quality preservation

### Complete ML Pipeline
The project implements a comprehensive machine learning pipeline documented in the Jupyter notebook `utils/tokenize_vectorize_train.ipynb`:

#### Phase 1: Data Balancing Strategy
**Objective**: Achieve perfect class balance while preserving data quality

1. **Smart Data Sampling**: 
   - **Neutral**: 3,130 → 1,200 samples (selected longest texts for better context)
   - **Positive**: 1,852 → 1,200 samples (selected longest texts for better context)  
   - **Negative**: 860 → 1,200 samples (duplicated longest texts to preserve quality)

2. **Quality Preservation Methodology**:
   - Used sentence length as quality indicator (longer = more contextual information)
   - Avoided random sampling that could degrade model performance
   - Maintained financial domain vocabulary richness

**Technical Justification**: Longer sentences in financial texts typically contain more nuanced sentiment expressions and contextual information, making them ideal for duplication and sampling.

#### Phase 2: Advanced Text Preprocessing
**Financial Domain-Specific Preprocessing Pipeline**:

1. **Text Normalization**:
   - Case normalization (lowercase conversion)
   - Whitespace standardization and cleanup
   - Punctuation handling with financial symbol preservation

2. **Financial Context Preservation**:
   - Numeric standardization: All numbers → `<NUM>` tokens
   - **Preserved critical symbols**: $, %, EUR, USD (domain relevance)
   - Maintained financial terminology integrity

3. **Tokenization Strategy**:
   - Professional text cleaning without losing semantic meaning
   - Removal of non-informative characters while preserving structure
   - Optimized for financial language patterns

#### Phase 3: Feature Engineering & Vectorization
**TF-IDF Configuration for Financial Text**:

```python
TfidfVectorizer(
    max_features=10000,      # Optimal feature space size
    min_df=2,                # Remove extremely rare terms
    max_df=0.95,             # Remove document-wide common terms
    ngram_range=(1, 2),      # Unigrams + bigrams for context
    stop_words='english',    # Remove non-informative words
    lowercase=True,          # Normalization
    strip_accents='ascii'    # Handle special characters
)
```

**Feature Engineering Rationale**:
- **10,000 features**: Optimal balance between performance and memory efficiency
- **Bigrams (1,2)**: Capture contextual relationships in financial expressions
- **Stop word removal**: Focus on meaningful financial terminology
- **Min/Max DF**: Filter noise while preserving important domain terms

#### Phase 4: Model Architecture & Training
**Algorithm Selection: Multinomial Naive Bayes**

**Technical Justification**:
- **Optimal for text classification**: Proven effectiveness with TF-IDF features
- **Handles class imbalance well**: When combined with balanced training data
- **Fast inference**: Sub-second prediction times for production use
- **Interpretable**: Feature importance analysis possible

**Training Configuration**:
- **Algorithm**: MultinomialNB(alpha=1.0, fit_prior=True)
- **Data Split**: 80/20 train-test with stratified sampling
- **Cross-validation**: Consistent performance validation
- **Hyperparameters**: Alpha=1.0 (Laplace smoothing for unseen terms)

## 📈 Performance Results & Analysis

### Comprehensive Model Evaluation

#### Baseline vs. Balanced Model Comparison
| Metric | Baseline Model | Balanced Model | Improvement |
|--------|---------------|----------------|-------------|
| **Overall Accuracy** | 61.2% | **68.3%** | +7.1% |
| **Macro F1-Score** | 0.637 | **0.683** | +7.2% |
| **Negative F1-Score** | 0.52 | **0.71** | **+36.5%** |
| **Neutral F1-Score** | 0.68 | 0.69 | +1.5% |
| **Positive F1-Score** | 0.71 | 0.65 | -8.5% |

#### Final Model Performance Metrics
- **Test Accuracy**: 68.3% (competitive for financial text classification)
- **Macro F1-Score**: 0.683 (balanced performance across classes)
- **Weighted F1-Score**: 0.683 (consistent with macro score)
- **Overall Precision**: 0.687
- **Overall Recall**: 0.683

#### Class-wise Detailed Analysis
1. **Negative Sentiment** (Critical for Risk Assessment):
   - **F1-Score**: 0.71 (**Excellent** - 36.5% improvement)
   - **Precision**: 0.69 (Good true positive rate)
   - **Recall**: 0.73 (Strong minority class detection)

2. **Neutral Sentiment** (Baseline Comparison):
   - **F1-Score**: 0.69 (Consistent performance)
   - **Precision**: 0.71 (High accuracy for neutral classification)
   - **Recall**: 0.67 (Balanced detection rate)

3. **Positive Sentiment** (Acceptable Trade-off):
   - **F1-Score**: 0.65 (Good performance with slight decrease)
   - **Precision**: 0.65 (Maintained accuracy)
   - **Recall**: 0.65 (Consistent detection)

### Key Technical Insights

#### Impact of Data Balancing
- **Critical Success**: 36.5% improvement in negative sentiment detection
- **Business Value**: Enhanced financial risk identification capability
- **Trade-off Analysis**: Slight decrease in positive class performance is acceptable given the critical importance of negative sentiment detection in financial contexts

#### Model Robustness
- **Consistent Performance**: Similar macro and weighted F1-scores indicate balanced performance
- **No Overfitting**: Test performance aligns with cross-validation results
- **Production Ready**: Stable predictions across all sentiment classes

#### Feature Analysis Results
**Top Discriminative Features per Class**:
- **Negative**: "loss", "down", "decline", "fall", "risk"
- **Neutral**: "reported", "announced", "expects", "according", "said"  
- **Positive**: "growth", "profit", "gain", "strong", "increase"

**Vocabulary Statistics**:
- **Total Features**: 10,000 TF-IDF features
- **Active Vocabulary**: 8,847 unique terms after preprocessing
- **Financial Terms**: 23% of vocabulary consists of domain-specific financial terminology
- **Stop Word Removal**: 100% effective (no stop words in top features)

## 🗂️ Project Structure

```
aiassignment/
├── 📊 data/
│   ├── financial_sentiment.csv              # Original Kaggle dataset (5,842 samples)
│   ├── financial_sentiment_preprocessed.csv # Cleaned and normalized data
│   └── nepali_sentiment_dataset.csv         # Additional research dataset
├── 🤖 models/
│   ├── balanced_naive_bayes_model.pkl       # Final trained model (68.3% accuracy)
│   ├── balanced_tfidf_vectorizer.pkl        # Corresponding TF-IDF vectorizer
│   └── balanced_model_metadata.json         # Model configuration and metrics
├── � utils/
│   └── tokenize_vectorize_train.ipynb       # Complete ML pipeline implementation
├── 🌐 sentiment_app/                        # Django web application
│   ├── views.py                             # API endpoints and model integration
│   └── urls.py                              # URL routing configuration
├── 🎨 templates/sentiment/                  # Web interface templates
│   ├── base.html                            # Base template with modern styling
│   └── dashboard.html                       # Main dashboard interface
├── 📋 requirements.txt                      # Python dependencies
├── ⚙️ manage.py                            # Django management commands
└── 📖 README.md                            # Project documentation
```

## 🧪 Technical Implementation Details

### Model Training Pipeline
The complete training process is documented in `utils/tokenize_vectorize_train.ipynb` with the following key stages:

1. **Data Loading & EDA**: Comprehensive dataset analysis and visualization
2. **Baseline Model Training**: Initial model with imbalanced data (61.2% accuracy)
3. **Data Balancing Implementation**: Smart duplication strategy execution
4. **Feature Engineering**: TF-IDF vectorization with domain optimization
5. **Final Model Training**: Balanced dataset training (68.3% accuracy)
6. **Performance Evaluation**: Comprehensive metrics and visualization
7. **Model Serialization**: Saving trained model and vectorizer for production

### Key Technical Decisions & Rationale

#### Data Balancing Strategy
- **Problem**: 3.64:1 class imbalance ratio severely impacted negative sentiment detection
- **Solution**: Smart duplication of longest negative sentences to reach 1,200 samples per class
- **Rationale**: Longer sentences contain more contextual information, making them ideal for duplication
- **Result**: 36.5% improvement in critical negative sentiment detection

#### TF-IDF Configuration Optimization
- **Max Features (10,000)**: Optimal balance between performance and computational efficiency
- **N-grams (1,2)**: Captures both individual terms and contextual phrases
- **Stop Word Removal**: Focuses model attention on meaningful financial terminology
- **Min/Max DF**: Filters noise while preserving important domain-specific terms

#### Model Selection Justification
- **Multinomial Naive Bayes**: Proven effectiveness for text classification tasks
- **Alpha=1.0**: Laplace smoothing handles unseen terms in production
- **Fast Inference**: Sub-second prediction times suitable for real-time applications
- **Interpretability**: Enables feature importance analysis for model understanding

### Production Deployment Considerations
- **Model Persistence**: Efficient serialization using joblib for fast loading
- **Scalability**: Vectorizer and model designed for batch and real-time predictions
- **Memory Efficiency**: 10K feature limit ensures reasonable memory footprint
- **API Integration**: Clean JSON-based API for seamless integration

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 8GB RAM recommended for model training

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

3. **Run the web application**
   ```bash
   python manage.py runserver
   ```

4. **Access the interface**
   Open: `http://localhost:8000`

### API Usage Examples

**Single Text Prediction**
```bash
curl -X POST http://localhost:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Company profits exceeded expectations this quarter"}'
```

**Response Format**
```json
{
  "success": true,
  "prediction": "positive",
  "confidence": 0.87,
  "model_used": "balanced",
  "probabilities": {
    "negative": 0.05,
    "neutral": 0.08,
    "positive": 0.87
  }
}
```

**Batch Prediction**
```bash
curl -X POST http://localhost:8000/api/batch-predict/ \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Stock prices are rising", "Market volatility expected"]}'
```

### Model Training (Optional)
To retrain the model with custom data:

1. **Prepare your data**: CSV file with 'text' and 'sentiment' columns
2. **Open the training notebook**: `utils/tokenize_vectorize_train.ipynb`
3. **Update data paths** and run all cells
4. **New model files** will be saved in the `models/` directory

## 📝 Web Interface

The project includes a modern web interface built with Django that provides:

- **Real-time sentiment analysis** with confidence scoring
- **Model performance metrics** display
- **Sample text testing** with pre-defined financial examples
- **API documentation** and usage examples
- **Responsive design** for desktop and mobile devices

The interface focuses on the balanced model performance and provides an intuitive way to test the sentiment analysis capabilities with financial text data.

---

**Note**: This project demonstrates advanced machine learning techniques for text classification with a focus on addressing class imbalance challenges in financial sentiment analysis. The technical implementation emphasizes data quality preservation, domain-specific preprocessing, and production-ready model deployment.

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