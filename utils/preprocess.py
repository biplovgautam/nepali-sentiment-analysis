"""
Text preprocessing utilities for Financial Sentiment Analysis.
Handles cleaning, normalization, and preparation of text data.
"""

import pandas as pd
import numpy as np
import re
import string
from typing import List, Union
import logging
from bs4 import BeautifulSoup
from config import PREPROCESSING_CONFIG, TICKER_PATTERNS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    A comprehensive text preprocessing class for financial sentiment analysis.
    """
    
    def __init__(self, config: dict = PREPROCESSING_CONFIG):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config (dict): Preprocessing configuration parameters
        """
        self.config = config
        self.ticker_patterns = [re.compile(pattern) for pattern in TICKER_PATTERNS]
        
        # Initialize stopwords
        self._load_stopwords()
        
        logger.info("TextPreprocessor initialized successfully")
    
    def _load_stopwords(self):
        """Load stopwords from NLTK."""
        try:
            import nltk
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            self.stopwords = set(stopwords.words('english'))
            
            # Add custom financial stopwords
            financial_stopwords = {
                'stock', 'stocks', 'share', 'shares', 'market', 'trading',
                'price', 'prices', 'buy', 'sell', 'hold', 'today', 'will'
            }
            self.stopwords.update(financial_stopwords)
            
        except Exception as e:
            logger.warning(f"Could not load NLTK stopwords: {e}. Using basic stopwords.")
            self.stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def clean_html(self, text: str) -> str:
        """
        Remove HTML tags from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not text or pd.isna(text):
            return ""
        
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()
    
    def clean_urls(self, text: str) -> str:
        """
        Remove URLs from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with URLs removed
        """
        if not text or pd.isna(text):
            return ""
        
        # Remove URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, ' ', text)
        
        # Remove www links
        www_pattern = r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(www_pattern, ' ', text)
        
        return text
    
    def normalize_tickers(self, text: str) -> str:
        """
        Normalize stock ticker symbols to a standard format.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with normalized tickers
        """
        if not text or pd.isna(text):
            return ""
        
        if not self.config['normalize_tickers']:
            return text
        
        for pattern in self.ticker_patterns:
            text = pattern.sub(' TICKER ', text)
        
        return text
    
    def remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with punctuation removed
        """
        if not text or pd.isna(text):
            return ""
        
        if not self.config['remove_punctuation']:
            return text
        
        # Keep some financial punctuation that might be meaningful
        financial_punct = {'$', '%', '+', '-'}
        punctuation_to_remove = set(string.punctuation) - financial_punct
        
        translation_table = str.maketrans('', '', ''.join(punctuation_to_remove))
        return text.translate(translation_table)
    
    def remove_numbers(self, text: str) -> str:
        """
        Remove or replace numbers in text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with numbers processed
        """
        if not text or pd.isna(text):
            return ""
        
        if not self.config['remove_numbers']:
            return text
        
        # Replace numbers with a placeholder to maintain context
        # Keep percentage, currency, and decimal patterns
        text = re.sub(r'\d+\.\d+%', ' PERCENTAGE ', text)  # 15.5%
        text = re.sub(r'\$\d+\.\d+', ' CURRENCY ', text)   # $15.50
        text = re.sub(r'\d+\.\d+', ' NUMBER ', text)       # 15.5
        text = re.sub(r'\b\d+\b', ' NUMBER ', text)        # 15
        
        return text
    
    def remove_extra_whitespace(self, text: str) -> str:
        """
        Remove extra whitespace and normalize spacing.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with normalized spacing
        """
        if not text or pd.isna(text):
            return ""
        
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading and trailing whitespace
        return text.strip()
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with stopwords removed
        """
        if not text or pd.isna(text):
            return ""
        
        if not self.config['remove_stopwords']:
            return text
        
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stopwords]
        
        return ' '.join(filtered_words)
    
    def clean_financial_text(self, text: str) -> str:
        """
        Apply financial-specific text cleaning.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned financial text
        """
        if not text or pd.isna(text):
            return ""
        
        # Remove common financial noise
        text = re.sub(r'RT\s+', '', text)  # Remove retweet indicators
        text = re.sub(r'@\w+', ' MENTION ', text)  # Replace mentions
        text = re.sub(r'#\w+', ' HASHTAG ', text)  # Replace hashtags
        
        # Normalize financial terms
        financial_replacements = {
            r'\b(bull|bullish|bulls)\b': 'BULLISH',
            r'\b(bear|bearish|bears)\b': 'BEARISH',
            r'\b(up|rise|rising|gain|gains|green)\b': 'POSITIVE_MOVE',
            r'\b(down|fall|falling|drop|drops|red|decline)\b': 'NEGATIVE_MOVE',
            r'\b(buy|long|calls?)\b': 'BUY_SIGNAL',
            r'\b(sell|short|puts?)\b': 'SELL_SIGNAL'
        }
        
        for pattern, replacement in financial_replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def preprocess_single_text(self, text: str) -> str:
        """
        Apply complete preprocessing pipeline to a single text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        if not text or pd.isna(text):
            return ""
        
        # Step 1: Basic cleaning
        text = self.clean_html(text)
        text = self.clean_urls(text)
        
        # Step 2: Convert to lowercase
        text = text.lower()
        
        # Step 3: Financial-specific cleaning
        text = self.clean_financial_text(text)
        
        # Step 4: Normalize tickers
        text = self.normalize_tickers(text)
        
        # Step 5: Remove numbers
        text = self.remove_numbers(text)
        
        # Step 6: Remove punctuation
        text = self.remove_punctuation(text)
        
        # Step 7: Remove extra whitespace
        text = self.remove_extra_whitespace(text)
        
        # Step 8: Remove stopwords (optional, might be better to do after tokenization)
        if self.config['remove_stopwords']:
            text = self.remove_stopwords(text)
        
        # Step 9: Final whitespace cleanup
        text = self.remove_extra_whitespace(text)
        
        return text
    
    def preprocess_texts(self, texts: Union[List[str], pd.Series]) -> List[str]:
        """
        Preprocess a collection of texts.
        
        Args:
            texts (Union[List[str], pd.Series]): Collection of texts
            
        Returns:
            List[str]: Preprocessed texts
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        logger.info(f"Preprocessing {len(texts)} texts")
        
        preprocessed = []
        for i, text in enumerate(texts):
            processed_text = self.preprocess_single_text(text)
            preprocessed.append(processed_text)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1}/{len(texts)} texts")
        
        logger.info("Text preprocessing completed")
        return preprocessed
    
    def get_preprocessing_stats(self, original_texts: List[str], 
                              processed_texts: List[str]) -> dict:
        """
        Get statistics about the preprocessing results.
        
        Args:
            original_texts (List[str]): Original texts
            processed_texts (List[str]): Processed texts
            
        Returns:
            dict: Preprocessing statistics
        """
        original_lengths = [len(text) if text else 0 for text in original_texts]
        processed_lengths = [len(text) if text else 0 for text in processed_texts]
        
        original_word_counts = [len(text.split()) if text else 0 for text in original_texts]
        processed_word_counts = [len(text.split()) if text else 0 for text in processed_texts]
        
        stats = {
            'total_texts': len(original_texts),
            'avg_original_length': np.mean(original_lengths),
            'avg_processed_length': np.mean(processed_lengths),
            'avg_original_words': np.mean(original_word_counts),
            'avg_processed_words': np.mean(processed_word_counts),
            'length_reduction_ratio': 1 - (np.mean(processed_lengths) / np.mean(original_lengths)),
            'word_reduction_ratio': 1 - (np.mean(processed_word_counts) / np.mean(original_word_counts)),
            'empty_after_processing': sum(1 for text in processed_texts if not text or text.strip() == '')
        }
        
        return stats


def preprocess_dataset(X: pd.Series, config: dict = PREPROCESSING_CONFIG) -> tuple:
    """
    Preprocess an entire dataset of texts.
    
    Args:
        X (pd.Series): Input texts
        config (dict): Preprocessing configuration
        
    Returns:
        tuple: (preprocessed_texts, preprocessing_stats)
    """
    preprocessor = TextPreprocessor(config)
    
    logger.info(f"Starting preprocessing of {len(X)} texts")
    
    # Convert to list for processing
    original_texts = X.tolist()
    
    # Preprocess all texts
    preprocessed_texts = preprocessor.preprocess_texts(original_texts)
    
    # Get statistics
    stats = preprocessor.get_preprocessing_stats(original_texts, preprocessed_texts)
    
    logger.info("Dataset preprocessing completed")
    logger.info(f"Average length reduction: {stats['length_reduction_ratio']:.1%}")
    logger.info(f"Average word reduction: {stats['word_reduction_ratio']:.1%}")
    
    return preprocessed_texts, stats


def filter_empty_texts(X: List[str], y: pd.Series) -> tuple:
    """
    Remove empty texts after preprocessing.
    
    Args:
        X (List[str]): Preprocessed texts
        y (pd.Series): Labels
        
    Returns:
        tuple: (filtered_texts, filtered_labels)
    """
    # Find non-empty texts
    non_empty_indices = [i for i, text in enumerate(X) if text and text.strip()]
    
    # Filter texts and labels
    filtered_X = [X[i] for i in non_empty_indices]
    filtered_y = y.iloc[non_empty_indices].reset_index(drop=True)
    
    logger.info(f"Filtered out {len(X) - len(filtered_X)} empty texts")
    
    return filtered_X, filtered_y


if __name__ == "__main__":
    # Test preprocessing functions
    sample_texts = [
        "AAPL is going to the moon! 🚀 $150 target price",
        "Market is down -5% today, very bearish signal",
        "Tesla stock looks good for a swing trade, buy calls",
        "<p>Check out this link: http://example.com for more info</p>",
        "$MSFT earnings beat expectations by 15.5%"
    ]
    
    # Test single text preprocessing
    preprocessor = TextPreprocessor()
    
    print("Original vs Preprocessed texts:")
    print("=" * 60)
    
    for text in sample_texts:
        processed = preprocessor.preprocess_single_text(text)
        print(f"Original:  {text}")
        print(f"Processed: {processed}")
        print("-" * 60)