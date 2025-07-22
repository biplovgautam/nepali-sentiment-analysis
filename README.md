# Financial Text Sentiment Analysis with Supervised Learning: A Naive Bayes-Based Classifier

**STW5000CEM Introduction to Artificial Intelligence - Assignment Project**  
**Module Leader:** Er. Suman Shrestha  
**Institution:** Softwarica College of IT & E-Commerce

---

## 🙏 Acknowledgments

I would like to express my sincere gratitude to **Er. Suman Shrestha**, the module leader for STW5000CEM Introduction to Artificial Intelligence. Throughout the duration of the module and even after its completion, Er. Suman Shrestha has been exceptionally supportive, providing invaluable guidance during classes and remaining available for assistance with project development, technical queries, and academic support. His expertise and dedication have been instrumental in the successful completion of this project.

---

## 📋 Project Overview

This project implements a **financial sentiment analysis system** using supervised learning techniques, specifically focusing on **Naive Bayes classification**. The system analyzes financial text data to classify sentiment as positive, negative, or neutral, providing valuable insights for financial decision-making and market analysis.

### 🎯 Problem Statement

Financial sentiment analysis addresses critical needs in:
- **Market Analysis**: Understanding market sentiment from financial news and reports
- **Investment Decision Support**: Extracting sentiment signals from financial communications  
- **Risk Assessment**: Identifying negative sentiment patterns indicating potential risks
- **Automated Trading**: Providing sentiment-based inputs for algorithmic trading systems

### 📚 Academic Context

This project fulfills the requirements for **STW5000CEM Introduction to Artificial Intelligence** coursework, demonstrating:
- **Classification algorithms** (Naive Bayes)
- **Data preprocessing and feature engineering**
- **Performance evaluation metrics**
- **Real-world AI problem solving**
- **Version control and project management**

---

## 📊 Current Project Status: ✅ Phase 1 Complete - EDA & Preprocessing

### 🔍 What We've Accomplished

#### 1. Comprehensive Exploratory Data Analysis (EDA)

**Dataset Overview:**
- **Source**: Financial sentiment dataset (`financial_sentiment.csv`)
- **Size**: 5,842 financial text samples
- **Classes**: Positive, Negative, Neutral sentiment labels
- **Domain**: Financial news, reports, and market communications

**Key EDA Findings:**
- ✅ **Data Quality**: No null values or duplicate entries detected
- ⚠️ **Class Imbalance**: Significant imbalance identified (Neutral: 53.6%, Positive: 31.7%, Negative: 14.7%)
- 📏 **Text Characteristics**: Average 117 characters per sentence (range: 9-315)
- 💰 **Financial Context**: Rich presence of financial terms (EUR, million, company, sales)
- 🔢 **Numeric Content**: Abundant financial figures requiring standardization

**EDA Techniques Applied:**
- **Statistical Analysis**: Descriptive statistics, null value checks, duplicate detection
- **Class Distribution Analysis**: Bar charts and pie charts for sentiment distribution
- **Text Length Analysis**: Histograms showing character and word count distributions
- **Word Cloud Visualization**: Sentiment-specific word clouds revealing key financial terminology
- **Noise Detection**: Analysis of URLs, financial symbols, and numeric patterns

#### 2. Strategic Data Preprocessing

**Problem Identification:**
- Severe class imbalance (ratio: 3.64) affecting model training potential
- Inconsistent numeric representations (raw numbers vs. standardized tokens)
- Noisy data elements (URLs, irregular punctuation)
- Need for financial domain knowledge preservation

**Preprocessing Solutions Implemented:**

**A. Intelligent Class Balancing:**
- **Strategy**: Preserve data quality while improving balance
- **Neutral Class**: Reduced from 3,130 → 1,200 samples (kept longest texts)
- **Positive Class**: Reduced from 1,852 → 1,200 samples (kept longest texts)
- **Negative Class**: Retained all 860 samples (minority class)
- **Result**: Improved imbalance ratio from 3.64 → 1.40 (62% improvement)
- **Quality Assurance**: Longer texts retained for richer information content

**B. Financial-Domain Text Preprocessing:**
```python
def preprocess_financial_text(text):
    # 1. URL Removal: Clean web links
    # 2. Numeric Standardization: Replace numbers with <NUM>
    # 3. Case Normalization: Convert to lowercase
    # 4. Financial Symbol Preservation: Keep $, %, EUR, £, ¥
    # 5. Special Character Cleaning: Remove noise, preserve meaning
    # 6. Whitespace Normalization: Clean formatting
```

**C. Preprocessing Achievements:**
- 🧹 **Text Standardization**: All numeric values converted to `<NUM>` tokens
- 💰 **Domain Preservation**: Financial symbols ($, %, EUR) maintained for context
- 📝 **Consistency**: Uniform lowercase formatting and whitespace normalization
- 🔗 **Noise Removal**: URLs and unnecessary special characters eliminated
- ✅ **Quality Validation**: No data corruption during transformation

#### 3. Post-Preprocessing Validation & Analysis

**Comparative Analysis Performed:**
- **Class Distribution Comparison**: Before/after balance visualization
- **Text Length Analysis**: Character distribution changes
- **Vocabulary Analysis**: Word count and unique vocabulary comparison
- **Numeric Standardization**: Quantification of `<NUM>` token implementation
- **Summary Statistics**: Comprehensive transformation metrics

**Validation Results:**
- ✅ **Successful Class Balancing**: 62% improvement in class balance
- ✅ **Vocabulary Optimization**: Reduced vocabulary while preserving meaning
- ✅ **Numeric Consistency**: Standardized numeric representation
- ✅ **Data Integrity**: No loss of essential financial context
- ✅ **Quality Assurance**: Final dataset ready for model training

---

## 📁 Project Structure

```
aiassignment/
├── data/
│   ├── financial_sentiment.csv                # Original dataset (5,842 samples)
│   └── financial_sentiment_preprocessed.csv   # Preprocessed dataset (3,260 samples) ✅
├── utils/
│   ├── edapreprocess.ipynb                    # ✅ COMPLETED - EDA & Preprocessing 
│   ├── tokenization.ipynb                     # 🚧 NEXT - Tokenization & Feature Engineering
│   ├── model_training.ipynb                   # 🕐 PLANNED - Naive Bayes Training
│   └── evaluation.ipynb                       # 🕐 PLANNED - Performance Evaluation
├── models/                                    # 📁 Model storage directory
├── reports/                                   # 📁 Academic reports and documentation
└── README.md                                  # 📖 This documentation
```

---

## 🚀 Next Steps: Supervised Learning Pipeline

### Phase 2: Tokenization & Feature Engineering (Next Immediate Step)
**Objective**: Transform preprocessed text into numerical features for machine learning

**Planned Implementation:**
- **Tokenization**: NLTK/spaCy-based token extraction
- **Stop Word Removal**: Domain-aware filtering (preserve financial terms)
- **Stemming/Lemmatization**: Token normalization for consistency
- **N-gram Analysis**: Capture phrase-level sentiment patterns

### Phase 3: Feature Vectorization
**Approaches:**
- **TF-IDF Vectorization**: Term frequency-inverse document frequency
  - Optimal for capturing term importance across financial documents
  - Configure n-gram ranges (1-2, 1-3) for phrase detection
  - Apply min_df/max_df filtering for vocabulary optimization
- **Count Vectorization**: Frequency-based baseline approach
- **Advanced Options**: Word2Vec/FastText for semantic representations

### Phase 4: Naive Bayes Model Training
**Algorithm Selection**: Multinomial Naive Bayes
- **Justification**: Optimal for text classification with discrete features
- **Training Strategy**: Stratified train-test split (80-20)
- **Validation**: K-fold cross-validation (k=5) for robust evaluation
- **Hyperparameter Tuning**: Alpha smoothing parameter optimization

### Phase 5: Model Evaluation & Performance Analysis
**Evaluation Metrics:**
- **Classification Accuracy**: Overall model performance
- **Precision, Recall, F1-Score**: Per-class performance analysis
- **Confusion Matrix**: Detailed classification breakdown
- **Cross-Validation Scores**: Generalization capability assessment
- **ROC-AUC Analysis**: Binary classification performance

### Phase 6: Deployment & Interface Development
**Deliverables:**
- **CLI/GUI Interface**: User-friendly sentiment analysis tool
- **Model Serialization**: Pickle/joblib model saving for deployment
- **Performance Documentation**: Comprehensive evaluation report
- **Video Demonstration**: Project walkthrough and results explanation

---

## 📈 Expected Performance Goals

### Academic Requirements Compliance
- **Algorithm Implementation**: Custom Naive Bayes with preprocessing pipeline
- **Performance Metrics**: Target 75-85% accuracy with balanced class performance
- **Code Quality**: Clean, documented, version-controlled implementation
- **Report Documentation**: 4000-word academic report with APA 7th referencing
- **Demonstration**: 10-minute video + 15-20 minute viva defense

### Technical Performance Targets
- **Classification Accuracy**: 75-85% (realistic for balanced financial sentiment data)
- **Balanced Performance**: Consistent precision/recall across all sentiment classes
- **Financial Domain Adaptation**: Effective handling of financial terminology and context
- **Computational Efficiency**: Sub-second prediction times for real-time applications
- **Generalization**: Robust performance on unseen financial text data

---

## 🛠️ Technical Implementation

### Development Environment
- **Programming Language**: Python 3.8+
- **Key Libraries**: pandas, numpy, scikit-learn, nltk, matplotlib, seaborn
- **Development Tools**: Jupyter Notebook, Git version control
- **Hardware Requirements**: 4GB+ RAM, multi-core CPU recommended

### Data Processing Pipeline
```python
# Current Implementation Status:
✅ Raw Data → EDA Analysis → Insights & Problem Identification
✅ Data Cleaning → Class Balancing → Quality Preprocessing  
✅ Validation → Comparison Analysis → Preprocessed Dataset Export
🚧 Tokenization → Feature Engineering → Model Training (Next Phase)
```

### Quality Assurance Measures
- **Data Validation**: Comprehensive before/after analysis
- **Version Control**: Git-based code management and history tracking
- **Documentation**: Inline code comments and markdown explanations
- **Testing**: Preprocessing function validation with sample data
- **Academic Standards**: APA 7th style documentation and referencing

---

## 📚 Learning Outcomes Demonstrated

### STW5000CEM Module Alignment
1. ✅ **Theoretical Understanding**: Demonstrated comprehension of AI/ML fundamentals
2. 🚧 **AI Model Implementation**: Naive Bayes classifier for real-world problem solving
3. 🚧 **Performance Evaluation**: Comprehensive metrics and validation strategies  
4. ✅ **Data Visualization**: EDA charts, comparative analysis, and result presentation
5. ✅ **Version Control**: Git-based project management and code versioning

---

## 🔄 Version Control & Project Management

**Repository Structure**: Organized codebase with clear documentation
**Commit Strategy**: Frequent commits with meaningful messages
**Branching**: Feature-based development approach
**Documentation**: Comprehensive README and inline code documentation

---

**Current Status**: ✅ EDA & Preprocessing Complete | 🚧 Ready for Tokenization Phase  
**Next Milestone**: Feature Engineering & Model Training Pipeline  
**Target Completion**: Full Sentiment Analysis System with Evaluation Report

---

*This project represents the practical application of artificial intelligence concepts learned in STW5000CEM, with special appreciation for the ongoing guidance and support provided by Er. Suman Shrestha.*
