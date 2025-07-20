"""
Machine Learning model utilities for Financial Sentiment Analysis.
Handles model training, evaluation, and prediction.
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import logging
from typing import Dict, Tuple, List, Any, Union
from config import MODEL_CONFIG, CV_CONFIG, MODEL_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentClassifier:
    """
    A comprehensive sentiment classification system using Naive Bayes.
    """
    
    def __init__(self, model_config: dict = None):
        """
        Initialize the sentiment classifier.
        
        Args:
            model_config (dict, optional): Model configuration parameters
        """
        self.config = model_config or MODEL_CONFIG.copy()
        self.model = MultinomialNB(alpha=self.config.get('alpha', 1.0))
        self.is_trained = False
        self.feature_names = None
        self.label_mapping = None
        
        logger.info(f"SentimentClassifier initialized with alpha={self.config.get('alpha', 1.0)}")
    
    def prepare_data(self, X: Union[pd.Series, np.ndarray], y: Union[pd.Series, np.ndarray], 
                    test_size: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training by splitting into train/test sets.
        
        Args:
            X (Union[pd.Series, np.ndarray]): Features
            y (Union[pd.Series, np.ndarray]): Labels
            test_size (float, optional): Test set size
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
        """
        test_size = test_size or self.config.get('test_size', 0.2)
        random_state = self.config.get('random_state', 42)
        stratify = self.config.get('stratify', True)
        
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=stratify_param
        )
        
        logger.info(f"Data split completed. Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'SentimentClassifier':
        """
        Train the sentiment classifier.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            
        Returns:
            SentimentClassifier: Self for method chaining
        """
        logger.info("Starting model training...")
        
        # Create label mapping
        unique_labels = np.unique(y_train)
        self.label_mapping = {i: label for i, label in enumerate(unique_labels)}
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training accuracy
        train_predictions = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)
        
        logger.info(f"Model training completed. Training accuracy: {train_accuracy:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Features to predict
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X (np.ndarray): Features to predict
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        probabilities = self.model.predict_proba(X)
        return probabilities
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Evaluating model performance...")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        evaluation_results = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        logger.info(f"Model evaluation completed. Test accuracy: {accuracy:.4f}")
        
        return evaluation_results
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv_folds: int = None) -> Dict[str, Any]:
        """
        Perform cross-validation on the model.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
            cv_folds (int, optional): Number of CV folds
            
        Returns:
            Dict[str, Any]: Cross-validation results
        """
        cv_folds = cv_folds or CV_CONFIG.get('cv_folds', 5)
        scoring = CV_CONFIG.get('scoring', 'accuracy')
        n_jobs = CV_CONFIG.get('n_jobs', -1)
        
        # Check if we have enough samples for cross-validation
        unique_labels = np.unique(y)
        min_samples_per_class = np.min([np.sum(y == label) for label in unique_labels])
        
        if min_samples_per_class < cv_folds:
            logger.warning(f"Not enough samples for {cv_folds}-fold CV (min class has {min_samples_per_class} samples)")
            cv_folds = max(2, min_samples_per_class)
            logger.info(f"Reducing CV folds to {cv_folds}")
        
        if len(y) < cv_folds:
            logger.warning(f"Dataset too small for {cv_folds}-fold CV. Skipping cross-validation.")
            return {
                'cv_scores': np.array([]),
                'mean_score': 0.0,
                'std_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'skipped': True,
                'reason': 'Dataset too small'
            }
        
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        try:
            # Perform cross-validation
            cv_scores = cross_val_score(
                self.model, X, y, cv=cv_folds, 
                scoring=scoring, n_jobs=n_jobs
            )
            
            cv_results = {
                'cv_scores': cv_scores,
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'min_score': cv_scores.min(),
                'max_score': cv_scores.max(),
                'skipped': False
            }
            
            logger.info(f"Cross-validation completed. Mean {scoring}: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score'] * 2:.4f})")
            
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}. Skipping CV.")
            cv_results = {
                'cv_scores': np.array([]),
                'mean_score': 0.0,
                'std_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'skipped': True,
                'reason': str(e)
            }
        
        return cv_results
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Optimize model hyperparameters using grid search.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        logger.info("Starting hyperparameter optimization...")
        
        # Define parameter grid for Naive Bayes
        param_grid = {
            'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        }
        
        # Grid search
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.config['alpha'] = grid_search.best_params_['alpha']
        
        optimization_results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Hyperparameter optimization completed. Best score: {grid_search.best_score_:.4f}")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        return optimization_results
    
    def get_feature_importance(self, feature_names: List[str], top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get feature importance for each class.
        
        Args:
            feature_names (List[str]): List of feature names
            top_n (int): Number of top features to return per class
            
        Returns:
            Dict[str, List[Tuple[str, float]]]: Feature importance by class
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        # Get log probabilities (feature importance)
        log_probs = self.model.feature_log_prob_
        class_names = self.model.classes_
        
        feature_importance = {}
        
        for i, class_name in enumerate(class_names):
            # Get feature importance for this class
            class_importance = log_probs[i]
            
            # Create feature-importance pairs
            feature_scores = list(zip(feature_names, class_importance))
            
            # Sort by importance (descending)
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            feature_importance[class_name] = feature_scores[:top_n]
        
        return feature_importance
    
    def predict_single_text(self, text_features: np.ndarray) -> Dict[str, Any]:
        """
        Predict sentiment for a single text with detailed output.
        
        Args:
            text_features (np.ndarray): Features for single text
            
        Returns:
            Dict[str, Any]: Prediction details
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Make prediction
        prediction = self.model.predict(text_features)[0]
        probabilities = self.model.predict_proba(text_features)[0]
        
        # Create probability distribution
        class_names = self.model.classes_
        prob_distribution = dict(zip(class_names, probabilities))
        
        # Get confidence (max probability)
        confidence = max(probabilities)
        
        result = {
            'predicted_sentiment': prediction,
            'confidence': confidence,
            'probability_distribution': prob_distribution,
            'all_probabilities': probabilities.tolist()
        }
        
        return result
    
    def save_model(self, filepath: str = None) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath (str, optional): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        filepath = filepath or MODEL_PATH
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'label_mapping': self.label_mapping
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = None) -> 'SentimentClassifier':
        """
        Load a trained model from disk.
        
        Args:
            filepath (str, optional): Path to load the model from
            
        Returns:
            SentimentClassifier: Self for method chaining
        """
        filepath = filepath or MODEL_PATH
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']
        self.feature_names = model_data.get('feature_names')
        self.label_mapping = model_data.get('label_mapping')
        
        logger.info(f"Model loaded from {filepath}")
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        if not self.is_trained:
            return {'status': 'untrained'}
        
        info = {
            'status': 'trained',
            'model_type': 'MultinomialNB',
            'alpha': self.model.alpha,
            'classes': self.model.classes_.tolist(),
            'n_features': self.model.feature_count_.shape[1],
            'config': self.config
        }
        
        return info


class ModelEnsemble:
    """
    Ensemble of multiple sentiment classification models.
    """
    
    def __init__(self):
        """Initialize the model ensemble."""
        self.models = {}
        self.weights = {}
        self.is_trained = False
    
    def add_model(self, name: str, model: SentimentClassifier, weight: float = 1.0) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            name (str): Model name
            model (SentimentClassifier): Trained model
            weight (float): Model weight for ensemble voting
        """
        self.models[name] = model
        self.weights[name] = weight
        logger.info(f"Added model '{name}' to ensemble with weight {weight}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X (np.ndarray): Features to predict
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get predictions from all models
        all_predictions = {}
        for name, model in self.models.items():
            all_predictions[name] = model.predict_proba(X)
        
        # Weighted average of probabilities
        weighted_proba = np.zeros_like(list(all_predictions.values())[0])
        total_weight = sum(self.weights.values())
        
        for name, predictions in all_predictions.items():
            weight = self.weights[name] / total_weight
            weighted_proba += weight * predictions
        
        # Get final predictions
        final_predictions = np.argmax(weighted_proba, axis=1)
        
        # Convert back to class labels
        class_names = list(self.models.values())[0].model.classes_
        return class_names[final_predictions]


def create_sentiment_classifier(config: dict = None) -> SentimentClassifier:
    """
    Factory function to create a sentiment classifier.
    
    Args:
        config (dict, optional): Model configuration
        
    Returns:
        SentimentClassifier: Configured classifier
    """
    return SentimentClassifier(model_config=config)


def train_and_evaluate_model(X_train: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray, y_test: np.ndarray,
                           config: dict = None) -> Tuple[SentimentClassifier, Dict[str, Any]]:
    """
    Complete training and evaluation pipeline.
    
    Args:
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Test features  
        y_train (np.ndarray): Training labels
        y_test (np.ndarray): Test labels
        config (dict, optional): Model configuration
        
    Returns:
        Tuple[SentimentClassifier, Dict[str, Any]]: Trained model and evaluation results
    """
    # Create and train model
    classifier = create_sentiment_classifier(config)
    classifier.train(X_train, y_train)
    
    # Evaluate model
    evaluation_results = classifier.evaluate(X_test, y_test)
    
    # Cross-validation
    cv_results = classifier.cross_validate(X_train, y_train)
    evaluation_results['cross_validation'] = cv_results
    
    logger.info("Model training and evaluation pipeline completed")
    
    return classifier, evaluation_results


if __name__ == "__main__":
    # Test classifier with dummy data
    from sklearn.datasets import make_classification
    
    print("Testing Sentiment Classifier:")
    print("=" * 50)
    
    # Create dummy data
    X, y = make_classification(n_samples=1000, n_features=100, n_classes=3, 
                             n_informative=50, random_state=42)
    
    # Create classifier
    classifier = create_sentiment_classifier()
    
    # Split data
    X_train, X_test, y_train, y_test = classifier.prepare_data(X, y)
    
    # Train model
    classifier.train(X_train, y_train)
    
    # Evaluate
    results = classifier.evaluate(X_test, y_test)
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    
    # Cross-validation
    cv_results = classifier.cross_validate(X_train, y_train)
    print(f"CV Accuracy: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score'] * 2:.4f})")
    
    print("Model testing completed!")
