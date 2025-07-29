# Chapter 3: Methodology

## 3.1 Dataset and Preprocessing

### Dataset Description
This project uses the Financial Sentiment Analysis dataset from Kaggle containing 5,842 financial text samples labeled with sentiment (positive, negative, neutral). Initial analysis revealed significant class imbalance: neutral (53.6%), positive (31.7%), and negative (14.7%).

### Data Preprocessing
A preprocessing pipeline was implemented including text cleaning, number tokenization, and class balancing. The dataset was balanced to 3,600 samples (1,200 per class) through downsampling majority classes and upsampling the minority class.

**Final Dataset:**
- Total samples: 3,600 (perfectly balanced)
- Training: 2,880 samples (80%)
- Testing: 720 samples (20%)

## 3.2 Feature Engineering

### TF-IDF Vectorization
Text was converted to numerical features using TF-IDF vectorization with parameters: max_features=10000, ngram_range=(1,2), min_df=2, max_df=0.95. This produced a training matrix of 2,880 × 8,353 features capturing both individual words and contextual bigrams.

## 3.3 Algorithm Explanation

### Multinomial Naive Bayes
Multinomial Naive Bayes applies Bayes' theorem with independence assumptions for text classification:

**Mathematical Foundation:**
```
P(class|text) = P(text|class) × P(class) / P(text)
```

The algorithm calculates probability for each word feature:
```
P(word|class) = (count(word,class) + α) / (total_words_in_class + α × vocabulary_size)
```

**Algorithm Steps:**
1. **Training**: Calculate prior probabilities P(class) and feature likelihoods P(word|class) for each sentiment class
2. **Prediction**: For new text, compute posterior probability for each class and select maximum

**Pseudo-code:**
```
FOR each class c:
    prior[c] = count(class_c) / total_documents
    FOR each word w in vocabulary:
        likelihood[c][w] = (count(w,c) + α) / (sum(words_in_c) + α × |vocab|)

FOR prediction:
    FOR each class c:
        score[c] = log(prior[c]) + Σ(count(w) × log(likelihood[c][w]))
    RETURN class with highest score
```

## 3.3 Justification for Choosing the Algorithm

### Algorithm Suitability for Financial Sentiment Analysis

**Why Multinomial Naive Bayes Fits This Problem:**

1. **Text Classification Specialization**: Multinomial Naive Bayes is specifically designed for discrete features like word counts and TF-IDF scores, making it ideal for text classification tasks.

2. **Balanced Dataset Performance**: With perfectly balanced classes (1,200 samples each), the algorithm's assumption of equal class priors aligns well with the data distribution.

3. **Small Dataset Efficiency**: With 3,600 total samples, Naive Bayes performs well on limited data due to its low parameter complexity and strong independence assumptions that prevent overfitting.

4. **Computational Efficiency**: O(n) time complexity for prediction makes it suitable for real-time financial sentiment analysis applications.

5. **Probabilistic Output**: Provides confidence scores for predictions, crucial for financial decision-making where uncertainty quantification is important.

**Comparison with Alternative Algorithm: Support Vector Machine (SVM)**

| Aspect | Multinomial Naive Bayes | Support Vector Machine |
|--------|------------------------|----------------------|
| **Training Time** | Fast (linear in features) | Slower (quadratic scaling) |
| **Prediction Speed** | Very Fast (O(n)) | Moderate (depends on support vectors) |
| **Memory Usage** | Low (stores probability tables) | Higher (stores support vectors) |
| **Interpretability** | High (clear probability weights) | Low (complex decision boundary) |
| **Small Dataset Performance** | Excellent (low parameters) | Good but may overfit |
| **Probabilistic Output** | Native probability estimates | Requires calibration |
| **Financial Domain Fit** | Strong (word-based features) | Good (general classifier) |

**Strengths in Financial Context:**
- **Interpretability**: Feature probabilities allow understanding which financial terms drive sentiment predictions
- **Speed**: Enables real-time analysis of financial news and reports
- **Probability Estimates**: Natural confidence scores for investment decision support
- **Stability**: Consistent performance across different market conditions
- **Domain Alignment**: Word-based approach suits financial text analysis

**Limitations in Current Context:**
- **Feature Independence Assumption**: Financial terms often correlate ("profit margins", "revenue growth"), but the model treats them independently
- **Limited Context**: Cannot capture complex linguistic patterns like sarcasm or conditional statements
- **Numeric Sensitivity**: Requires preprocessing to handle financial numbers effectively
- **Class Boundary Simplicity**: Linear decision boundaries may miss complex sentiment patterns

**Mitigation Strategies Implemented:**
- Used TF-IDF with bigrams to capture some contextual information
- Applied comprehensive preprocessing to standardize financial terminology
- Implemented hyperparameter tuning to optimize smoothing and feature selection
- Used balanced dataset to prevent class bias issues

The choice of Multinomial Naive Bayes is justified by its excellent performance on text classification tasks, computational efficiency suitable for financial applications, and interpretability requirements for decision support systems in finance.

## 3.4 Model Training and Hyperparameter Optimization

### 3.4.1 Baseline Model Training

First, we trained a baseline model using standard parameters to establish performance benchmarks.

**Baseline Setup:**
```python
# TF-IDF with standard parameters
tfidf_vectorizer = TfidfVectorizer(
    max_features=10000, min_df=2, max_df=0.95, 
    ngram_range=(1, 2), stop_words='english'
)

# Naive Bayes with default alpha
nb_baseline = MultinomialNB(alpha=1.0)
```

**Baseline Results:**
- Accuracy: 67.92%
- Macro F1-Score: 0.6374
- Negative F1: 0.516 (weakest performance)
- Neutral F1: 0.671
- Positive F1: 0.725

The baseline model performed reasonably well but had difficulty with negative sentiment detection.

### 3.4.2 GridSearch with Cross-Validation

We used GridSearchCV to find optimal parameters through 5-fold cross-validation.

**Parameter Grid:**
```python
param_grid = {
    'tfidf__max_features': [5000, 8000, 10000, 12000],
    'tfidf__min_df': [1, 2, 3],
    'tfidf__max_df': [0.90, 0.95, 0.98], 
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'nb__alpha': [0.1, 0.5, 1.0, 1.5, 2.0]
}
# Total: 360 parameter combinations tested
```

**Cross-Validation Process:**
- 5-fold stratified cross-validation (maintains class balance)
- Each fold: 576 samples (192 per sentiment class)
- Scoring metric: Macro F1-score for balanced evaluation
- Total model training runs: 360 × 5 = 1,800

### 3.4.3 Optimization Results

**Best Parameters Found:**
- Alpha: 1.0 (optimal smoothing)
- Max Features: 8,000 (balanced vocabulary size)
- Min DF: 2 (removes rare terms)
- Max DF: 0.90 (filters common terms)
- N-grams: (1,2) (includes bigrams for context)

**Performance Improvement:**
- CV F1-Score: 0.6990 ± 0.0127
- Test Accuracy: 67.92% → 68.06% (+0.14%)
- Test Macro F1: 0.6374 → 0.6776 (+6.31%)
- Negative F1: 0.516 → 0.714 (+38.37% - major improvement)

The optimization successfully improved negative sentiment detection while maintaining overall balanced performance across all classes.

---

# Chapter 4: Results and Evaluation

## 4.1 Final Model Performance

After hyperparameter optimization, the final model achieved the following results:

**Overall Performance:**
- **Accuracy**: 68.06% (490/720 correct predictions)
- **Macro F1-Score**: 0.6776 (average across all sentiment classes)
- **Cross-Validation F1**: 0.6990 ± 0.0127 (stable performance)

**Class-wise Performance:**
- **Negative**: Precision=0.70, Recall=0.73, F1=0.714
- **Neutral**: Precision=0.63, Recall=0.72, F1=0.671  
- **Positive**: Precision=0.74, Recall=0.60, F1=0.664

## 4.2 Baseline vs Optimized Comparison

| Model | Accuracy | Macro F1 | Negative F1 | Neutral F1 | Positive F1 |
|-------|----------|----------|-------------|-------------|-------------|
| Baseline | 67.92% | 0.6374 | 0.516 | 0.671 | 0.725 |
| Optimized | 68.06% | 0.6776 | 0.714 | 0.671 | 0.664 |
| **Improvement** | **+0.14%** | **+6.31%** | **+38.37%** | **0.00%** | **-8.41%** |

**Key Findings:**
- Significant improvement in negative sentiment detection (+38.37% F1-score)
- Overall macro F1-score improved by 6.31%
- Neutral sentiment performance maintained
- Slight decrease in positive sentiment performance, but overall better balance

## 4.3 Visualizations Generated

The notebook includes comprehensive visualizations:

1. **Confusion Matrix Analysis** - Shows prediction accuracy for each class
2. **ROC Curves** - Demonstrates discriminative ability per sentiment class
3. **Precision-Recall Curves** - Shows precision-recall trade-offs
4. **Hyperparameter Sensitivity Analysis** - Impact of different parameters
5. **Feature Importance Analysis** - Top TF-IDF features per sentiment class
6. **Learning Curves** - Model performance vs training size
7. **Word Clouds** - Most frequent words per sentiment class
8. **Probability Analysis** - Log probability weights for discriminative words

## 4.4 Model Analysis

**Strengths:**
- Strong negative sentiment detection (F1=0.714)
- Balanced performance across classes
- Fast training and prediction
- Interpretable feature weights
- Stable cross-validation performance

**Areas for Improvement:**
- Positive sentiment recall could be higher (0.60)
- Some positive cases misclassified as neutral
- Limited to bigram context

**Optimal Parameters:**
- Alpha: 1.0 (balanced smoothing)
- Max Features: 8,000 (optimal vocabulary size)
- N-grams: (1,2) (includes context)
- Min/Max DF: 2/0.90 (effective filtering)