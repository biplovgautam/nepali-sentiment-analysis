"""
Tokenization and lemmatization utilities for Financial Sentiment Analysis.
Handles word tokenization, lemmatization, and stemming operations.
"""

import pandas as pd
import numpy as np
from typing import List, Union, Tuple
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextTokenizer:
    """
    A comprehensive tokenization class for financial text analysis.
    Supports both NLTK and spaCy for tokenization and normalization.
    """
    
    def __init__(self, method: str = 'nltk', use_lemmatization: bool = True):
        """
        Initialize the tokenizer.
        
        Args:
            method (str): Tokenization method ('nltk' or 'spacy')
            use_lemmatization (bool): Whether to use lemmatization (vs stemming)
        """
        self.method = method.lower()
        self.use_lemmatization = use_lemmatization
        
        # Initialize the appropriate tokenizer
        if self.method == 'nltk':
            self._init_nltk()
        elif self.method == 'spacy':
            self._init_spacy()
        else:
            raise ValueError("Method must be 'nltk' or 'spacy'")
        
        logger.info(f"TextTokenizer initialized with {self.method} method, lemmatization: {use_lemmatization}")
    
    def _init_nltk(self):
        """Initialize NLTK tokenizer and related tools."""
        try:
            import nltk
            from nltk.tokenize import word_tokenize, sent_tokenize
            from nltk.stem import WordNetLemmatizer, PorterStemmer
            from nltk.corpus import wordnet
            from nltk.tag import pos_tag
            
            # Download required NLTK data if not available
            required_downloads = ['punkt', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
            for item in required_downloads:
                try:
                    nltk.data.find(f'tokenizers/{item}' if item == 'punkt' else 
                                 f'corpora/{item}' if item in ['wordnet', 'omw-1.4'] else 
                                 f'taggers/{item}')
                except LookupError:
                    try:
                        logger.info(f"Downloading NLTK data: {item}")
                        nltk.download(item, quiet=True)
                    except Exception as e:
                        logger.warning(f"Failed to download {item}: {e}. Some features may not work.")
            
            self.word_tokenize = word_tokenize
            self.sent_tokenize = sent_tokenize
            self.pos_tag = pos_tag
            
            if self.use_lemmatization:
                self.lemmatizer = WordNetLemmatizer()
                self.normalizer = self._lemmatize_nltk
            else:
                self.stemmer = PorterStemmer()
                self.normalizer = self._stem_nltk
                
            self.wordnet = wordnet
            
        except ImportError:
            logger.error("NLTK not available. Please install nltk.")
            raise
    
    def _init_spacy(self):
        """Initialize spaCy tokenizer."""
        try:
            import spacy
            
            # Load English model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except IOError:
                logger.error("spaCy English model not found. Please install: python -m spacy download en_core_web_sm")
                raise
            
            # Disable unnecessary components for performance
            self.nlp.disable_pipes(["parser", "ner"])
            
        except ImportError:
            logger.error("spaCy not available. Please install spacy.")
            raise
    
    def _get_wordnet_pos(self, word: str, pos: str) -> str:
        """
        Convert POS tag to WordNet POS for lemmatization.
        
        Args:
            word (str): Word to get POS for
            pos (str): POS tag from NLTK
            
        Returns:
            str: WordNet POS tag
        """
        tag = pos[0].upper()
        tag_dict = {
            "J": self.wordnet.ADJ,
            "N": self.wordnet.NOUN,
            "V": self.wordnet.VERB,
            "R": self.wordnet.ADV
        }
        return tag_dict.get(tag, self.wordnet.NOUN)
    
    def _lemmatize_nltk(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens using NLTK.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Lemmatized tokens
        """
        # Get POS tags for better lemmatization
        pos_tags = self.pos_tag(tokens)
        
        lemmatized = []
        for word, pos in pos_tags:
            wordnet_pos = self._get_wordnet_pos(word, pos)
            lemma = self.lemmatizer.lemmatize(word.lower(), pos=wordnet_pos)
            lemmatized.append(lemma)
        
        return lemmatized
    
    def _stem_nltk(self, tokens: List[str]) -> List[str]:
        """
        Stem tokens using NLTK Porter Stemmer.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Stemmed tokens
        """
        return [self.stemmer.stem(token.lower()) for token in tokens]
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize a single text into words.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of tokens
        """
        if not text or pd.isna(text):
            return []
        
        if self.method == 'nltk':
            tokens = self.word_tokenize(text)
            # Filter out non-alphabetic tokens and very short tokens
            tokens = [token for token in tokens if token.isalpha() and len(token) > 1]
            return tokens
            
        elif self.method == 'spacy':
            doc = self.nlp(text)
            tokens = [token.text for token in doc if token.is_alpha and len(token.text) > 1 and not token.is_stop]
            return tokens
    
    def normalize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Normalize tokens using lemmatization or stemming.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Normalized tokens
        """
        if not tokens:
            return []
        
        if self.method == 'nltk':
            return self.normalizer(tokens)
            
        elif self.method == 'spacy':
            if self.use_lemmatization:
                # Process tokens through spaCy for lemmatization
                text = ' '.join(tokens)
                doc = self.nlp(text)
                return [token.lemma_.lower() for token in doc if token.is_alpha and len(token.lemma_) > 1]
            else:
                # Simple stemming fallback (less ideal for spaCy)
                from nltk.stem import PorterStemmer
                stemmer = PorterStemmer()
                return [stemmer.stem(token.lower()) for token in tokens]
    
    def tokenize_and_normalize(self, text: str) -> List[str]:
        """
        Complete tokenization and normalization pipeline for a single text.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: Tokenized and normalized words
        """
        if not text or pd.isna(text):
            return []
        
        # Tokenize
        tokens = self.tokenize_text(text)
        
        if not tokens:
            return []
        
        # Normalize
        normalized_tokens = self.normalize_tokens(tokens)
        
        # Final filtering
        final_tokens = [token for token in normalized_tokens if len(token) > 1 and token.isalpha()]
        
        return final_tokens
    
    def tokenize_corpus(self, texts: Union[List[str], pd.Series]) -> List[List[str]]:
        """
        Tokenize and normalize a corpus of texts.
        
        Args:
            texts (Union[List[str], pd.Series]): Collection of texts
            
        Returns:
            List[List[str]]: List of tokenized texts
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        logger.info(f"Tokenizing {len(texts)} texts using {self.method}")
        
        tokenized_corpus = []
        for i, text in enumerate(texts):
            tokens = self.tokenize_and_normalize(text)
            tokenized_corpus.append(tokens)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Tokenized {i + 1}/{len(texts)} texts")
        
        logger.info("Corpus tokenization completed")
        return tokenized_corpus
    
    def get_vocabulary(self, tokenized_corpus: List[List[str]], min_freq: int = 1) -> dict:
        """
        Extract vocabulary from tokenized corpus.
        
        Args:
            tokenized_corpus (List[List[str]]): Tokenized texts
            min_freq (int): Minimum frequency for inclusion in vocabulary
            
        Returns:
            dict: Vocabulary with word frequencies
        """
        from collections import Counter
        
        # Count all tokens
        all_tokens = []
        for tokens in tokenized_corpus:
            all_tokens.extend(tokens)
        
        # Get vocabulary with frequencies
        vocab_counts = Counter(all_tokens)
        
        # Filter by minimum frequency
        vocabulary = {word: count for word, count in vocab_counts.items() 
                     if count >= min_freq}
        
        logger.info(f"Vocabulary extracted: {len(vocabulary)} unique words")
        
        return vocabulary
    
    def filter_tokens_by_vocabulary(self, tokenized_corpus: List[List[str]], 
                                  vocabulary: set) -> List[List[str]]:
        """
        Filter tokens to keep only those in vocabulary.
        
        Args:
            tokenized_corpus (List[List[str]]): Tokenized texts
            vocabulary (set): Set of allowed words
            
        Returns:
            List[List[str]]: Filtered tokenized texts
        """
        filtered_corpus = []
        for tokens in tokenized_corpus:
            filtered_tokens = [token for token in tokens if token in vocabulary]
            filtered_corpus.append(filtered_tokens)
        
        return filtered_corpus


class AdvancedTokenizer:
    """
    Advanced tokenization with additional financial text processing features.
    """
    
    def __init__(self, base_tokenizer: TextTokenizer):
        """
        Initialize with a base tokenizer.
        
        Args:
            base_tokenizer (TextTokenizer): Base tokenizer instance
        """
        self.base_tokenizer = base_tokenizer
        
        # Financial-specific patterns
        self.financial_patterns = {
            'ticker': re.compile(r'\b[A-Z]{1,5}\b'),
            'percentage': re.compile(r'\d+(?:\.\d+)?%'),
            'currency': re.compile(r'\$\d+(?:\.\d+)?[KMB]?'),
            'number': re.compile(r'\b\d+(?:\.\d+)?\b')
        }
    
    def extract_financial_entities(self, text: str) -> dict:
        """
        Extract financial entities from text.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Extracted financial entities
        """
        entities = {}
        
        for entity_type, pattern in self.financial_patterns.items():
            matches = pattern.findall(text)
            entities[entity_type] = matches
        
        return entities
    
    def tokenize_with_entities(self, text: str) -> Tuple[List[str], dict]:
        """
        Tokenize text and extract financial entities.
        
        Args:
            text (str): Input text
            
        Returns:
            Tuple[List[str], dict]: Tokens and financial entities
        """
        tokens = self.base_tokenizer.tokenize_and_normalize(text)
        entities = self.extract_financial_entities(text)
        
        return tokens, entities


def create_tokenizer(method: str = 'nltk', use_lemmatization: bool = True) -> TextTokenizer:
    """
    Factory function to create a tokenizer instance.
    
    Args:
        method (str): Tokenization method ('nltk' or 'spacy')
        use_lemmatization (bool): Whether to use lemmatization
        
    Returns:
        TextTokenizer: Configured tokenizer instance
    """
    return TextTokenizer(method=method, use_lemmatization=use_lemmatization)


def tokenize_dataset(texts: Union[List[str], pd.Series], 
                    method: str = 'nltk', 
                    use_lemmatization: bool = True) -> Tuple[List[List[str]], dict]:
    """
    Tokenize an entire dataset and return tokens with vocabulary statistics.
    
    Args:
        texts (Union[List[str], pd.Series]): Input texts
        method (str): Tokenization method
        use_lemmatization (bool): Whether to use lemmatization
        
    Returns:
        Tuple[List[List[str]], dict]: Tokenized corpus and vocabulary stats
    """
    tokenizer = create_tokenizer(method=method, use_lemmatization=use_lemmatization)
    
    # Tokenize corpus
    tokenized_corpus = tokenizer.tokenize_corpus(texts)
    
    # Get vocabulary statistics
    vocabulary = tokenizer.get_vocabulary(tokenized_corpus)
    
    vocab_stats = {
        'total_tokens': sum(len(tokens) for tokens in tokenized_corpus),
        'unique_tokens': len(vocabulary),
        'avg_tokens_per_text': np.mean([len(tokens) for tokens in tokenized_corpus]),
        'vocab_size_by_frequency': {
            'freq_1+': len([w for w, f in vocabulary.items() if f >= 1]),
            'freq_2+': len([w for w, f in vocabulary.items() if f >= 2]),
            'freq_5+': len([w for w, f in vocabulary.items() if f >= 5]),
            'freq_10+': len([w for w, f in vocabulary.items() if f >= 10])
        }
    }
    
    logger.info(f"Tokenization completed. Vocabulary size: {vocab_stats['unique_tokens']}")
    logger.info(f"Average tokens per text: {vocab_stats['avg_tokens_per_text']:.1f}")
    
    return tokenized_corpus, vocab_stats


if __name__ == "__main__":
    # Test tokenization functions
    sample_texts = [
        "AAPL stock is bullish and gaining momentum",
        "Market crashed today, very bearish sentiment overall",
        "Tesla earnings beat expectations significantly",
        "Bought MSFT calls for 15% profit potential",
        "The market looks volatile with mixed signals"
    ]
    
    print("Testing NLTK Tokenizer with Lemmatization:")
    print("=" * 50)
    
    tokenizer_nltk = create_tokenizer(method='nltk', use_lemmatization=True)
    
    for text in sample_texts:
        tokens = tokenizer_nltk.tokenize_and_normalize(text)
        print(f"Text: {text}")
        print(f"Tokens: {tokens}")
        print("-" * 50)
    
    # Test corpus tokenization
    print("\nCorpus Tokenization Stats:")
    tokenized_corpus, stats = tokenize_dataset(sample_texts)
    for key, value in stats.items():
        print(f"{key}: {value}")