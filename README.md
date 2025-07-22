# Financial Text Sentiment Analysis with Supervised Learning: A Naive Bayes-Based Classifier

**STW5000CEM Introduction to Artificial Intelligence - Assignment Project**  
**Module Leader:** Er. Suman Shrestha  
**Institution:** Softwarica College of IT & E-Commerce

---

##  Acknowledgments

I would like to express my sincere gratitude to **Er. Suman Shrestha**, the module leader for STW5000CEM Introduction to Artificial Intelligence. Throughout the duration of the module and even after its completion, Er. Suman Shrestha has been exceptionally supportive, providing invaluable guidance during classes and remaining available for assistance with project development, technical queries, and academic support. His expertise and dedication have been instrumental in the successful completion of this project.

---

##  Project Overview

This project implements a **financial sentiment analysis system** using supervised learning techniques, specifically focusing on **Naive Bayes classification**. The system analyzes financial text data to classify sentiment as positive, negative, or neutral, providing valuable insights for financial decision-making and market analysis.

### 📊 Dataset Information

- **Source**: [Financial Sentiment Analysis Dataset](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis) from Kaggle
- **Original Size**: 5,842 financial text samples
- **Classes**: Positive, Negative, Neutral sentiment labels
- **Domain**: Financial news, reports, and market communications
- **Format**: CSV file with 'Sentence' and 'Sentiment' columns

### 🔍 Key Findings from EDA

- **Data Quality**: No null values or duplicate entries
-  **Class Imbalance**: Significant imbalance (Neutral: 53.6%, Positive: 31.7%, Negative: 14.7%)
-  **Text Characteristics**: Average 117 characters per sentence (range: 9-315)
-  **Financial Context**: Rich presence of financial terms (EUR, million, company, sales)
-  **Numeric Content**: Abundant financial figures requiring standardization

---

##  Preprocessing Strategies Applied

### 1. Class Balancing Strategy
**Problem**: Severe class imbalance (ratio: 3.64) affecting model training
**Solution**: 
- **Neutral Class**: Reduced from 3,130 → 1,200 samples (kept longest texts for quality)
- **Positive Class**: Reduced from 1,852 → 1,200 samples (kept longest texts for quality)
- **Negative Class**: Retained all 860 samples (minority class)
- **Result**: Improved imbalance ratio from 3.64 → 1.40 (62% improvement)


### 2. Preprocessing Achievements
-  **Text Standardization**: All numeric values converted to `<NUM>` tokens
- **Domain Preservation**: Financial symbols ($, %, EUR) maintained for context
-  **Consistency**: Uniform lowercase formatting and whitespace normalization
-  **Noise Removal**: URLs and unnecessary special characters eliminated
-  **Balanced Dataset**: Final dataset with 3,260 high-quality samples

### 3. Validation Results
- **Class Balance**: 62% improvement in imbalance ratio
- **Vocabulary Optimization**: Reduced vocabulary while preserving financial terminology
- **Numeric Consistency**: Standardized numeric representation with `<NUM>` tokens
- **Data Integrity**: No loss of essential financial context during transformation

---



## 📈 Current Status

**Phase 1 Complete**:  Exploratory Data Analysis, Data Cleaning and Preprocessing
- Comprehensive EDA performed with visualizations
- Class imbalance addressed through intelligent sampling
- Text preprocessing pipeline implemented with financial domain preservation
- Dataset optimized and ready for model training

**Output**: `financial_sentiment_preprocessed.csv` with 3,260 balanced, cleaned samples ready for machine learning model training.
