"""
Feature extraction utilities for Financial Sentiment Analysis.
Handles TF-IDF and Count vectorization with parameter optimization.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV
import joblib
import logging
from typing import List, Union, Tuple, Dict, Any
from config import VECTORIZER_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    A comprehensive feature extraction class for text classification.
    Supports both TF-IDF and Count vectorization with optimization.
    """
    
    def __init__(self, vectorizer_type: str = 'tfidf', config: dict = None):
        """
        Initialize the feature extractor.
        
        Args:
            vectorizer_type (str): Type of vectorizer ('tfidf' or 'count')
            config (dict, optional): Configuration parameters
        """
        self.vectorizer_type = vectorizer_type.lower()
        self.config = config or VECTORIZER_CONFIG.copy()
        
        # Initialize the appropriate vectorizer
        self.vectorizer = self._create_vectorizer()
        self.is_fitted = False
        
        logger.info(f"FeatureExtractor initialized with {self.vectorizer_type} vectorizer")
    
    def _create_vectorizer(self):
        """Create and configure the vectorizer based on type and config."""
        base_params = {
            'max_features': self.config.get('max_features', 10000),
            'ngram_range': self.config.get('ngram_range', (1, 2)),
            'min_df': self.config.get('min_df', 2),
            'max_df': self.config.get('max_df', 0.95),
            'lowercase': self.config.get('lowercase', True),
            'stop_words': self.config.get('stop_words', 'english')
        }
        
        if self.vectorizer_type == 'tfidf':
            # Additional TF-IDF specific parameters
            tfidf_params = base_params.copy()
            tfidf_params.update({
                'use_idf': True,
                'smooth_idf': True,
                'sublinear_tf': True,  # Use log scaling
                'norm': 'l2'  # L2 normalization
            })
            return TfidfVectorizer(**tfidf_params)
        
        elif self.vectorizer_type == 'count':
            return CountVectorizer(**base_params)
        
        else:
            raise ValueError("vectorizer_type must be 'tfidf' or 'count'")
    
    def fit(self, texts: Union[List[str], pd.Series]) -> 'FeatureExtractor':
        """
        Fit the vectorizer to the training texts.
        
        Args:
            texts (Union[List[str], pd.Series]): Training texts
            
        Returns:
            FeatureExtractor: Self for method chaining
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        logger.info(f"Fitting {self.vectorizer_type} vectorizer on {len(texts)} texts")
        
        # Filter out empty texts
        texts = [text for text in texts if text and text.strip()]
        
        self.vectorizer.fit(texts)
        self.is_fitted = True
        
        # Log vocabulary information
        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info(f"Vectorizer fitted. Vocabulary size: {vocab_size}")
        
        return self
    
    def transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Transform texts to feature vectors.
        
        Args:
            texts (Union[List[str], pd.Series]): Texts to transform
            
        Returns:
            np.ndarray: Feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transformation")
        
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        # Filter out empty texts and remember indices
        original_indices = []
        filtered_texts = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                original_indices.append(i)
                filtered_texts.append(text)
        
        logger.info(f"Transforming {len(filtered_texts)} texts to feature vectors")
        
        # Transform texts
        feature_matrix = self.vectorizer.transform(filtered_texts)
        
        # Handle empty texts by creating a sparse matrix with zeros
        if len(filtered_texts) < len(texts):
            from scipy.sparse import csr_matrix
            full_matrix = csr_matrix((len(texts), feature_matrix.shape[1]))
            
            for i, original_idx in enumerate(original_indices):
                full_matrix[original_idx] = feature_matrix[i]
            
            feature_matrix = full_matrix
        
        logger.info(f"Feature matrix shape: {feature_matrix.shape}")
        
        return feature_matrix
    
    def fit_transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Fit the vectorizer and transform texts in one step.
        
        Args:
            texts (Union[List[str], pd.Series]): Texts to fit and transform
            
        Returns:
            np.ndarray: Feature matrix
        """
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """
        Get the feature names (vocabulary) from the fitted vectorizer.
        
        Returns:
            List[str]: Feature names
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before getting feature names")
        
        if hasattr(self.vectorizer, 'get_feature_names_out'):
            return self.vectorizer.get_feature_names_out().tolist()
        else:
            return list(self.vectorizer.vocabulary_.keys())
    
    def get_top_features(self, feature_matrix: np.ndarray, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get top features by average TF-IDF/Count scores.
        
        Args:
            feature_matrix (np.ndarray): Feature matrix
            top_n (int): Number of top features to return
            
        Returns:
            List[Tuple[str, float]]: List of (feature, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before getting top features")
        
        # Calculate mean scores for each feature
        mean_scores = np.array(feature_matrix.mean(axis=0)).flatten()
        
        # Get feature names
        feature_names = self.get_feature_names()
        
        # Create feature-score pairs and sort by score
        feature_scores = list(zip(feature_names, mean_scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        return feature_scores[:top_n]
    
    def get_class_specific_features(self, X: np.ndarray, y: np.ndarray, 
                                  class_label: str, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get top features specific to a particular class.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Labels
            class_label (str): Target class label
            top_n (int): Number of top features to return
            
        Returns:
            List[Tuple[str, float]]: List of (feature, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before getting class-specific features")
        
        # Filter data for the specific class
        class_mask = (y == class_label)
        class_features = X[class_mask]
        
        if class_features.shape[0] == 0:
            logger.warning(f"No samples found for class '{class_label}'")
            return []
        
        # Calculate mean scores for the class
        mean_scores = np.array(class_features.mean(axis=0)).flatten()
        
        # Get feature names
        feature_names = self.get_feature_names()
        
        # Create feature-score pairs and sort by score
        feature_scores = list(zip(feature_names, mean_scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        return feature_scores[:top_n]
    
    def optimize_parameters(self, X_train: Union[List[str], pd.Series], 
                          y_train: np.ndarray, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Optimize vectorizer parameters using grid search.
        
        Args:
            X_train (Union[List[str], pd.Series]): Training texts
            y_train (np.ndarray): Training labels
            cv_folds (int): Number of CV folds
            
        Returns:
            Dict[str, Any]: Best parameters and scores
        """
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline
        
        # Define parameter grid based on vectorizer type
        if self.vectorizer_type == 'tfidf':
            param_grid = {
                'vectorizer__max_features': [5000, 10000, 15000],
                'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'vectorizer__min_df': [1, 2, 5],
                'vectorizer__max_df': [0.8, 0.9, 0.95],
                'vectorizer__sublinear_tf': [True, False]
            }
        else:  # count
            param_grid = {
                'vectorizer__max_features': [5000, 10000, 15000],
                'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'vectorizer__min_df': [1, 2, 5],
                'vectorizer__max_df': [0.8, 0.9, 0.95]
            }
        
        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', self._create_vectorizer()),
            ('classifier', MultinomialNB())
        ])
        
        # Grid search
        logger.info("Starting parameter optimization...")
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv_folds, 
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update vectorizer with best parameters
        best_vectorizer_params = {
            k.replace('vectorizer__', ''): v 
            for k, v in grid_search.best_params_.items() 
            if k.startswith('vectorizer__')
        }
        
        self.config.update(best_vectorizer_params)
        self.vectorizer = self._create_vectorizer()
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_vectorizer_params': best_vectorizer_params
        }
        
        logger.info(f"Optimization completed. Best CV score: {grid_search.best_score_:.4f}")
        
        return results
    
    def save_vectorizer(self, filepath: str) -> None:
        """
        Save the fitted vectorizer to disk.
        
        Args:
            filepath (str): Path to save the vectorizer
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted vectorizer")
        
        # Save both vectorizer and config
        save_data = {
            'vectorizer': self.vectorizer,
            'vectorizer_type': self.vectorizer_type,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Vectorizer saved to {filepath}")
    
    def load_vectorizer(self, filepath: str) -> 'FeatureExtractor':
        """
        Load a fitted vectorizer from disk.
        
        Args:
            filepath (str): Path to load the vectorizer from
            
        Returns:
            FeatureExtractor: Self for method chaining
        """
        save_data = joblib.load(filepath)
        
        self.vectorizer = save_data['vectorizer']
        self.vectorizer_type = save_data['vectorizer_type']
        self.config = save_data['config']
        self.is_fitted = save_data['is_fitted']
        
        logger.info(f"Vectorizer loaded from {filepath}")
        return self
    
    def get_vector_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted vectorizer.
        
        Returns:
            Dict[str, Any]: Vectorizer information
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted to get info")
        
        info = {
            'vectorizer_type': self.vectorizer_type,
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'feature_count': len(self.get_feature_names()),
            'config': self.config
        }
        
        return info


class TextAnalyzer:
    """
    Advanced text analysis using multiple vectorization techniques.
    """
    
    def __init__(self):
        """Initialize the text analyzer."""
        self.vectorizers = {}
        self.fitted_vectorizers = {}
    
    def add_vectorizer(self, name: str, vectorizer_type: str, config: dict = None) -> None:
        """
        Add a vectorizer to the analyzer.
        
        Args:
            name (str): Name for the vectorizer
            vectorizer_type (str): Type of vectorizer ('tfidf' or 'count')
            config (dict): Configuration for the vectorizer
        """
        self.vectorizers[name] = FeatureExtractor(vectorizer_type, config)
        logger.info(f"Added {vectorizer_type} vectorizer '{name}'")
    
    def fit_all(self, texts: Union[List[str], pd.Series]) -> 'TextAnalyzer':
        """
        Fit all vectorizers to the texts.
        
        Args:
            texts (Union[List[str], pd.Series]): Training texts
            
        Returns:
            TextAnalyzer: Self for method chaining
        """
        for name, vectorizer in self.vectorizers.items():
            logger.info(f"Fitting vectorizer '{name}'...")
            vectorizer.fit(texts)
            self.fitted_vectorizers[name] = vectorizer
        
        return self
    
    def transform_all(self, texts: Union[List[str], pd.Series]) -> Dict[str, np.ndarray]:
        """
        Transform texts using all fitted vectorizers.
        
        Args:
            texts (Union[List[str], pd.Series]): Texts to transform
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of feature matrices
        """
        results = {}
        for name, vectorizer in self.fitted_vectorizers.items():
            logger.info(f"Transforming with vectorizer '{name}'...")
            results[name] = vectorizer.transform(texts)
        
        return results
    
    def compare_vectorizers(self, texts: Union[List[str], pd.Series], 
                          labels: np.ndarray) -> pd.DataFrame:
        """
        Compare different vectorizers using basic classification metrics.
        
        Args:
            texts (Union[List[str], pd.Series]): Test texts
            labels (np.ndarray): True labels
            
        Returns:
            pd.DataFrame: Comparison results
        """
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import classification_report
        
        results = []
        
        for name, vectorizer in self.fitted_vectorizers.items():
            # Transform texts
            X = vectorizer.transform(texts)
            
            # Quick classification test
            classifier = MultinomialNB()
            cv_scores = cross_val_score(classifier, X, labels, cv=5, scoring='accuracy')
            
            results.append({
                'vectorizer': name,
                'mean_accuracy': cv_scores.mean(),
                'std_accuracy': cv_scores.std(),
                'vocabulary_size': len(vectorizer.get_feature_names()),
                'feature_matrix_shape': X.shape
            })
        
        return pd.DataFrame(results)


def create_feature_extractor(vectorizer_type: str = 'tfidf', 
                           config: dict = None) -> FeatureExtractor:
    """
    Factory function to create a feature extractor.
    
    Args:
        vectorizer_type (str): Type of vectorizer ('tfidf' or 'count')
        config (dict): Configuration parameters
        
    Returns:
        FeatureExtractor: Configured feature extractor
    """
    return FeatureExtractor(vectorizer_type=vectorizer_type, config=config)


def extract_features_from_texts(texts: Union[List[str], pd.Series], 
                               vectorizer_type: str = 'tfidf',
                               config: dict = None) -> Tuple[np.ndarray, FeatureExtractor]:
    """
    Extract features from texts using specified vectorizer.
    
    Args:
        texts (Union[List[str], pd.Series]): Input texts
        vectorizer_type (str): Type of vectorizer
        config (dict): Configuration parameters
        
    Returns:
        Tuple[np.ndarray, FeatureExtractor]: Feature matrix and fitted extractor
    """
    extractor = create_feature_extractor(vectorizer_type, config)
    feature_matrix = extractor.fit_transform(texts)
    
    logger.info(f"Feature extraction completed. Shape: {feature_matrix.shape}")
    
    return feature_matrix, extractor


if __name__ == "__main__":
    # Test feature extraction
    sample_texts = [
        "AAPL stock is bullish and gaining momentum in the market",
        "Market crashed today, very bearish sentiment overall",
        "Tesla earnings beat expectations significantly this quarter",
        "Bought MSFT calls for 15% profit potential",
        "The market looks volatile with mixed signals from analysts"
    ]
    
    print("Testing TF-IDF Feature Extraction:")
    print("=" * 50)
    
    # Create and fit extractor
    extractor = create_feature_extractor('tfidf')
    feature_matrix = extractor.fit_transform(sample_texts)
    
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Vocabulary size: {len(extractor.get_feature_names())}")
    
    # Get top features
    top_features = extractor.get_top_features(feature_matrix, top_n=10)
    print("\nTop 10 Features by TF-IDF Score:")
    for feature, score in top_features:
        print(f"  {feature}: {score:.4f}")
    
    # Test text analyzer
    print("\n" + "=" * 50)
    print("Testing Text Analyzer:")
    
    analyzer = TextAnalyzer()
    analyzer.add_vectorizer('tfidf', 'tfidf')
    analyzer.add_vectorizer('count', 'count')
    
    analyzer.fit_all(sample_texts)
    results = analyzer.transform_all(sample_texts)
    
    for name, matrix in results.items():
        print(f"{name} feature matrix shape: {matrix.shape}")