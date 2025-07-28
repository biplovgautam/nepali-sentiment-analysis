# Chapter 3: Methodology

## Dataset and Preprocessing

### Dataset Description
This project uses the Financial Sentiment Analysis dataset from Kaggle containing 5,842 financial text samples labeled with sentiment (positive, negative, neutral). The dataset includes financial news statements, market reports, and earnings communications with text lengths ranging from 9 to 315 characters.

**[Insert Image: Original dataset overview showing sample texts and labels]**

### Exploratory Data Analysis
Initial analysis revealed significant class imbalance:
- Neutral: 3,130 samples (53.6%)
- Positive: 1,852 samples (31.7%) 
- Negative: 860 samples (14.7%)
- Imbalance ratio: 3.64:1

**[Insert Image: Class distribution pie chart and bar chart comparison]**

Text analysis showed average length of 117 characters with presence of financial symbols, numbers, and domain-specific terminology.

**[Insert Image: Text length distribution histograms and word clouds by sentiment]**

### Data Preprocessing Pipeline

**Text Cleaning Function**
A preprocessing function was implemented to standardize financial text while preserving domain knowledge:

```python
def preprocess_financial_text(text):
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', '', text)
    # Replace numbers with <NUM> token
    text = re.sub(r'\b\d+\.?\d*\b', '<NUM>', text)
    # Convert to lowercase
    text = text.lower()
    # Preserve financial symbols while removing noise
    text = re.sub(r'[^\w\s$%€£¥#.,\-()]+', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

**Data Balancing Strategy**
To address class imbalance, an intelligent balancing approach was applied:
- Neutral: Reduced to 1,200 longest sentences
- Positive: Reduced to 1,200 longest sentences  
- Negative: Kept all 860 samples (minority class protection)

This achieved improved balance ratio of 1.40:1 while preserving data quality.

**[Insert Image: Before/after data balancing visualization]**

**Final Dataset Characteristics:**
- Total samples: 3,260
- Training set: 2,608 samples (80%)
- Test set: 652 samples (20%)
- Vocabulary size: Reduced from 15,847 to 12,234 tokens

## Feature Engineering and Vectorization

### TF-IDF Vectorization
Text data was converted to numerical features using Term Frequency-Inverse Document Frequency (TF-IDF) vectorization with the following configuration:

- Maximum features: 10,000 most informative terms
- N-gram range: (1,2) for unigrams and bigrams
- Minimum document frequency: 2 (removes rare terms)
- Maximum document frequency: 0.95 (removes common terms)
- Stop words: English stop words removed
- Normalization: L2 normalization applied

**[Insert Image: TF-IDF feature analysis showing top features per sentiment class]**

### Feature Analysis
The vectorization process identified key discriminative features for each sentiment class, with financial terms like currency symbols and domain-specific vocabulary showing high importance.

**[Insert Image: Top financial terms visualization by sentiment class]**

## Algorithm Selection and Implementation

### Multinomial Naive Bayes
The Multinomial Naive Bayes algorithm was selected for classification based on its effectiveness with text data and computational efficiency. The algorithm applies Bayes' theorem with conditional independence assumption between features.

**Model Configuration:**
- Alpha (smoothing): 1.0 for Laplace smoothing
- Fit prior: True to learn class distributions from data
- Training features: 8,289 TF-IDF features after preprocessing

### Algorithm Justification
Multinomial Naive Bayes was chosen over alternatives due to:
- High-dimensional sparse feature efficiency
- Good performance with limited training data
- Fast training and prediction times
- Probabilistic outputs for confidence scoring
- Proven effectiveness in text classification tasks

**[Insert Image: Algorithm comparison table or performance comparison]**

## Model Training Process

### Training Pipeline
The model training followed these steps:

1. **Data Split**: Applied stratified 80/20 train-test split
2. **Feature Extraction**: Fitted TF-IDF vectorizer on training data only
3. **Model Training**: Trained Multinomial NB with selected parameters
4. **Validation**: Evaluated performance on held-out test set
5. **Model Persistence**: Saved trained model and vectorizer

### Baseline vs Balanced Model Approach
Two models were trained for comparison:

**Baseline Model**: Trained on original imbalanced dataset
- Accuracy: 66.3%
- Poor negative sentiment detection (F1: 0.516)

**Balanced Model**: Trained on preprocessed balanced dataset  
- Accuracy: 68.3%
- Improved negative sentiment detection (F1: 0.714)

**[Insert Image: Baseline vs balanced model performance comparison]**

### Training Characteristics
- Training time: <5 seconds
- Memory usage: ~50MB for model storage
- Feature dimensionality: 8,289 TF-IDF features
- Cross-validation: 5-fold CV for model validation

**[Insert Image: Learning curves showing training progress and validation scores]**

## Model Evaluation and Analysis

### Performance Metrics
The model was evaluated using multiple classification metrics:

- **Accuracy**: Overall correctness across all classes
- **Precision**: Proportion of correct positive predictions
- **Recall**: Proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis

### Visualization and Analysis
Comprehensive visualizations were created to analyze model performance:

**[Insert Image: Model performance comparison (accuracy, F1-scores by class)]**

**[Insert Image: ROC curves for multi-class classification]**

**[Insert Image: Precision-recall curves for each sentiment class]**

**[Insert Image: Enhanced confusion matrix with percentages and counts]**

**[Insert Image: Class-specific word clouds showing meaningful terms]**

**[Insert Image: Log probability visualization showing feature weights]**

### Model Interpretability
Feature analysis revealed the most important terms for each sentiment class, with financial vocabulary and domain-specific patterns clearly identified. The log probability weights showed how different words contribute to classification decisions.

**[Insert Image: Top predictive features visualization with log probabilities]**

This methodology ensures a systematic approach to financial sentiment analysis with proper data preprocessing, appropriate algorithm selection, and comprehensive evaluation metrics.