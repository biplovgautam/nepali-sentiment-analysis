# Chapter 3: Methodology

## Dataset and Preprocessing

### Dataset Description
This project uses the Financial Sentiment Analysis dataset from Kaggle containing 5,842 financial text samples labeled with sentiment (positive, negative, neutral). The dataset includes financial news statements, market reports, and earnings communications with text lengths ranging from 9 to 315 characters.

**[Insert Image: Dataset overview showing sample texts, basic statistics, and domain analysis - from EDA notebook cell 4]**

### Exploratory Data Analysis
Initial analysis revealed significant class imbalance with neutral sentiment dominating at 53.6%, positive at 31.7%, and negative at only 14.7% (imbalance ratio: 3.64:1). This severe imbalance posed risks for minority class detection, particularly crucial for financial risk assessment.

**[Insert Image: Class distribution pie chart and bar chart comparison - from EDA notebook cell 5]**

Text analysis showed average length of 117 characters with abundant financial symbols, numeric data, and domain-specific terminology indicating rich financial context.

**[Insert Image: Text length distribution histograms and word count analysis - from EDA notebook cell 6]**

**[Insert Image: Word clouds by sentiment showing domain-specific financial vocabulary - from EDA notebook cell 8]**

### Data Preprocessing Pipeline

**Text Cleaning Function**
A preprocessing function was implemented to standardize financial text while preserving domain knowledge:

```python
def sandardize_financial_text(text):
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
An intelligent balancing approach prioritized data quality by selecting longer sentences:
- Neutral: Reduced to 1,200 longest sentences (from 3,130)
- Positive: Reduced to 1,200 longest sentences (from 1,852)
- Negative: Kept all 860 samples (minority class protection)

This achieved improved balance ratio of 1.40:1 while preserving linguistic authenticity.

**[Insert Image: Before/after data balancing comparison - from EDA notebook cell 15]**

**[Insert Image: Text length comparison showing preprocessing effects - from EDA notebook cell 16]**

**Final Dataset Characteristics:**
- Total samples: 3,260 high-quality samples
- Training set: 2,608 samples (80%)
- Test set: 652 samples (20%)
- Vocabulary reduction: 22.8% (preserved meaningful financial terms)

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

### Algorithm Comparison Analysis
A systematic evaluation of multiple machine learning algorithms was conducted for sentiment classification. The comparison considered model complexity, performance metrics, training time, and interpretability requirements.

**[Insert Image: Algorithm comparison radar chart showing performance across multiple dimensions - from EDA notebook cell 17]**

### Multinomial Naive Bayes Selection
Multinomial Naive Bayes was selected based on:
1. **Text Classification Efficiency**: Native handling of discrete features from TF-IDF
2. **Computational Performance**: O(n) time complexity for prediction
3. **Small Dataset Optimization**: Strong performance with limited training data (3,260 samples)
4. **Interpretability**: Clear probability calculations for financial decision support
5. **Robustness**: Stable performance across class imbalance scenarios

### TF-IDF Vectorization Strategy
Text representation used TF-IDF vectorization with financial-specific parameters:

```python
tfidf = TfidfVectorizer(
    max_features=10000,      # Balance performance and memory
    ngram_range=(1, 2),       # Unigrams and bigrams for context
    min_df=2,                 # Remove rare terms
    max_df=0.8,               # Remove common terms
    stop_words='english',     # Remove standard stopwords
    lowercase=True,           # Normalize case
    token_pattern=r'\b\w+\b'  # Standard word boundaries
)
```

This configuration captures both individual terms and contextual bigrams while managing vocabulary size effectively.

**[Insert Image: TF-IDF feature importance visualization showing top financial terms - from training notebook cell 12]**

## Model Training and Evaluation

### Training Configuration
Model training used stratified train-test split (80/20) with alpha smoothing parameter of 1.0:

```python
nb_model = MultinomialNB(alpha=1.0)
nb_model.fit(X_train_tfidf, y_train)
```

### Performance Metrics
The trained model achieved balanced performance across sentiment classes:

**Overall Accuracy: 68.3%**

**Class-wise Performance:**
- **Negative**: Precision=0.757, Recall=0.675, F1=0.714
- **Neutral**: Precision=0.661, Recall=0.682, F1=0.671  
- **Positive**: Precision=0.647, Recall=0.681, F1=0.664

**[Insert Image: Confusion matrix heatmap showing prediction accuracy - from training notebook cell 9]**

**[Insert Image: Classification report bar chart with precision, recall, F1 scores - from training notebook cell 10]**

### ROC and Precision-Recall Analysis
Multi-class ROC analysis demonstrated strong discriminative ability:

**[Insert Image: ROC curves for all classes showing AUC scores - from training notebook cell 13]**

**[Insert Image: Precision-Recall curves highlighting performance trade-offs - from training notebook cell 14]**

### Learning Curve Analysis
Learning curves validated model stability and identified optimal training set size:

**[Insert Image: Learning curves showing training vs validation performance - from training notebook cell 15]**

The curves indicate:
- No significant overfitting (training and validation scores converge)
- Stable performance across different training sizes
- Optimal performance achieved with current dataset size

### Feature Analysis and Interpretability

**Word Cloud Analysis**
Post-training word clouds revealed model focus on relevant financial vocabulary:

**[Insert Image: Word clouds by predicted class showing model feature priorities - from training notebook cell 16]**

**Log Probability Analysis**
Feature importance analysis through log probabilities identified key sentiment indicators:

**[Insert Image: Log probability visualization showing most discriminative features - from training notebook cell 17]**

This analysis confirms the model's reliance on appropriate financial terminology for sentiment classification.

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