"""
Model evaluation and metrics utilities for Financial Sentiment Analysis.
Provides comprehensive evaluation metrics, visualization, and reporting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import learning_curve, validation_curve
import logging
from typing import Dict, List, Tuple, Any, Union
import os
from config import OUTPUT_DIR, PLOT_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics and visualizations.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the model evaluator.
        
        Args:
            output_dir (str, optional): Directory to save evaluation outputs
        """
        self.output_dir = output_dir or OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"ModelEvaluator initialized. Output directory: {self.output_dir}")
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_proba: np.ndarray = None) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_proba (np.ndarray, optional): Prediction probabilities
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        logger.info("Calculating evaluation metrics...")
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Per-class metrics
        class_report = classification_report(y_true, y_pred, output_dict=True)
        metrics['classification_report'] = class_report
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = conf_matrix
        
        # If probabilities available, calculate additional metrics
        if y_proba is not None:
            try:
                # For binary classification
                if y_proba.shape[1] == 2:
                    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                    metrics['roc_auc'] = auc(fpr, tpr)
                    
                    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
                    metrics['average_precision'] = average_precision_score(y_true, y_proba[:, 1])
                
                # For multiclass, calculate per-class ROC AUC
                else:
                    # One-vs-Rest ROC AUC for each class
                    unique_classes = np.unique(y_true)
                    class_auc = {}
                    
                    for i, class_label in enumerate(unique_classes):
                        binary_true = (y_true == class_label).astype(int)
                        if len(np.unique(binary_true)) > 1:  # Ensure both classes present
                            fpr, tpr, _ = roc_curve(binary_true, y_proba[:, i])
                            class_auc[f'class_{class_label}_auc'] = auc(fpr, tpr)
                    
                    metrics['per_class_auc'] = class_auc
                    metrics['macro_auc'] = np.mean(list(class_auc.values())) if class_auc else None
            
            except Exception as e:
                logger.warning(f"Could not calculate ROC/AUC metrics: {e}")
        
        logger.info(f"Metrics calculated. Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: List[str] = None, save_path: str = None) -> None:
        """
        Plot confusion matrix with detailed visualization.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            class_names (List[str], optional): Class names for labels
            save_path (str, optional): Path to save the plot
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Set up the plot
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Add accuracy information
        accuracy = accuracy_score(y_true, y_pred)
        plt.figtext(0.02, 0.02, f'Overall Accuracy: {accuracy:.3f}', 
                   fontsize=10, ha='left')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 class_names: List[str] = None, save_path: str = None) -> None:
        """
        Plot classification report as heatmap.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            class_names (List[str], optional): Class names
            save_path (str, optional): Path to save the plot
        """
        # Get classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Extract metrics for each class
        if class_names is None:
            class_names = [str(cls) for cls in np.unique(y_true)]
        
        # Create DataFrame for heatmap
        metrics_data = []
        for class_name in class_names:
            if str(class_name) in report:
                metrics_data.append([
                    report[str(class_name)]['precision'],
                    report[str(class_name)]['recall'],
                    report[str(class_name)]['f1-score']
                ])
        
        # Add overall metrics
        metrics_data.append([
            report['weighted avg']['precision'],
            report['weighted avg']['recall'],
            report['weighted avg']['f1-score']
        ])
        
        # Create DataFrame
        df_metrics = pd.DataFrame(
            metrics_data,
            index=class_names + ['Weighted Avg'],
            columns=['Precision', 'Recall', 'F1-Score']
        )
        
        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(df_metrics, annot=True, fmt='.3f', cmap='YlOrRd',
                   cbar_kws={'label': 'Score'})
        
        plt.title('Classification Report Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Classes', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Classification report heatmap saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, y_true: np.ndarray, y_proba: np.ndarray,
                       class_names: List[str] = None, save_path: str = None) -> None:
        """
        Plot ROC curves for multiclass classification.
        
        Args:
            y_true (np.ndarray): True labels
            y_proba (np.ndarray): Prediction probabilities
            class_names (List[str], optional): Class names
            save_path (str, optional): Path to save the plot
        """
        unique_classes = np.unique(y_true)
        n_classes = len(unique_classes)
        
        if class_names is None:
            class_names = [f'Class {cls}' for cls in unique_classes]
        
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        
        for i, (class_label, color) in enumerate(zip(unique_classes, colors)):
            # Create binary classification problem
            binary_true = (y_true == class_label).astype(int)
            
            if len(np.unique(binary_true)) > 1:  # Ensure both classes present
                fpr, tpr, _ = roc_curve(binary_true, y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, color=color, lw=2,
                        label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - One vs Rest', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def plot_learning_curves(self, estimator, X: np.ndarray, y: np.ndarray,
                           train_sizes: np.ndarray = None, save_path: str = None) -> None:
        """
        Plot learning curves to analyze training performance.
        
        Args:
            estimator: The model to evaluate
            X (np.ndarray): Features
            y (np.ndarray): Labels
            train_sizes (np.ndarray, optional): Training set sizes
            save_path (str, optional): Path to save the plot
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        logger.info("Generating learning curves...")
        
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y, train_sizes=train_sizes, cv=5,
            scoring='accuracy', n_jobs=-1, random_state=42
        )
        
        # Calculate means and standard deviations
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot learning curves
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Accuracy')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Learning Curves', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Learning curves saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_importance: Dict[str, List[Tuple[str, float]]],
                              save_path: str = None) -> None:
        """
        Plot feature importance for each class.
        
        Args:
            feature_importance (Dict[str, List[Tuple[str, float]]]): Feature importance by class
            save_path (str, optional): Path to save the plot
        """
        n_classes = len(feature_importance)
        fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 8))
        
        if n_classes == 1:
            axes = [axes]
        
        for i, (class_name, features) in enumerate(feature_importance.items()):
            # Extract feature names and scores
            feature_names = [f[0] for f in features]
            feature_scores = [f[1] for f in features]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(feature_names))
            
            axes[i].barh(y_pos, feature_scores, alpha=0.7)
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(feature_names)
            axes[i].set_xlabel('Log Probability')
            axes[i].set_title(f'Top Features - {class_name}')
            axes[i].grid(axis='x', alpha=0.3)
            
            # Invert y-axis to show most important features at top
            axes[i].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 y_proba: np.ndarray = None, class_names: List[str] = None,
                                 model_name: str = "Model") -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_proba (np.ndarray, optional): Prediction probabilities
            class_names (List[str], optional): Class names
            model_name (str): Name of the model
            
        Returns:
            str: Formatted evaluation report
        """
        logger.info("Generating comprehensive evaluation report...")
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_proba)
        
        # Start building report
        report = f"\n{'='*60}\n"
        report += f"EVALUATION REPORT: {model_name.upper()}\n"
        report += f"{'='*60}\n\n"
        
        # Basic metrics
        report += "OVERALL PERFORMANCE METRICS:\n"
        report += f"{'='*30}\n"
        report += f"Accuracy:           {metrics['accuracy']:.4f}\n"
        report += f"Precision (macro):  {metrics['precision_macro']:.4f}\n"
        report += f"Recall (macro):     {metrics['recall_macro']:.4f}\n"
        report += f"F1-Score (macro):   {metrics['f1_macro']:.4f}\n"
        report += f"Precision (weighted): {metrics['precision_weighted']:.4f}\n"
        report += f"Recall (weighted):  {metrics['recall_weighted']:.4f}\n"
        report += f"F1-Score (weighted): {metrics['f1_weighted']:.4f}\n\n"
        
        # Per-class metrics
        report += "PER-CLASS PERFORMANCE:\n"
        report += f"{'='*25}\n"
        
        class_report = metrics['classification_report']
        if class_names is None:
            class_names = [str(cls) for cls in np.unique(y_true)]
        
        for class_name in class_names:
            if str(class_name) in class_report:
                class_metrics = class_report[str(class_name)]
                report += f"\nClass '{class_name}':\n"
                report += f"  Precision: {class_metrics['precision']:.4f}\n"
                report += f"  Recall:    {class_metrics['recall']:.4f}\n"
                report += f"  F1-Score:  {class_metrics['f1-score']:.4f}\n"
                report += f"  Support:   {class_metrics['support']}\n"
        
        # Confusion matrix info
        cm = metrics['confusion_matrix']
        report += f"\nCONFUSION MATRIX:\n"
        report += f"{'='*17}\n"
        report += f"Shape: {cm.shape}\n"
        report += f"Total Predictions: {cm.sum()}\n"
        report += f"Correct Predictions: {np.trace(cm)}\n"
        
        # AUC metrics if available
        if 'macro_auc' in metrics and metrics['macro_auc'] is not None:
            report += f"\nAUC METRICS:\n"
            report += f"{'='*12}\n"
            report += f"Macro AUC: {metrics['macro_auc']:.4f}\n"
        
        report += f"\n{'='*60}\n"
        
        return report
    
    def save_evaluation_results(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_proba: np.ndarray = None, class_names: List[str] = None,
                              model_name: str = "model", create_plots: bool = True) -> str:
        """
        Save comprehensive evaluation results including plots and report.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_proba (np.ndarray, optional): Prediction probabilities
            class_names (List[str], optional): Class names
            model_name (str): Model name for file naming
            create_plots (bool): Whether to create and save plots
            
        Returns:
            str: Path to the evaluation report file
        """
        logger.info(f"Saving evaluation results for {model_name}...")
        
        # Create model-specific directory
        model_output_dir = os.path.join(self.output_dir, f"{model_name}_evaluation")
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Generate and save text report
        report = self.generate_evaluation_report(y_true, y_pred, y_proba, 
                                               class_names, model_name)
        
        report_path = os.path.join(model_output_dir, f"{model_name}_evaluation_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation report saved to {report_path}")
        
        if create_plots:
            # Save confusion matrix
            cm_path = os.path.join(model_output_dir, f"{model_name}_confusion_matrix.png")
            self.plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
            
            # Save classification report heatmap
            cr_path = os.path.join(model_output_dir, f"{model_name}_classification_report.png")
            self.plot_classification_report(y_true, y_pred, class_names, cr_path)
            
            # Save ROC curves if probabilities available
            if y_proba is not None:
                roc_path = os.path.join(model_output_dir, f"{model_name}_roc_curves.png")
                self.plot_roc_curves(y_true, y_proba, class_names, roc_path)
        
        return report_path
    
    def comprehensive_evaluation(self, model, X_test, y_test, output_dir=None):
        """
        Perform comprehensive model evaluation with visualizations.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            output_dir: Directory to save outputs
            
        Returns:
            dict: Comprehensive evaluation results
        """
        if output_dir:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("Starting comprehensive model evaluation...")
        
        # Get predictions and probabilities
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_proba)
        
        # Generate visualizations
        self.plot_confusion_matrix(y_test, y_pred, 
                                 save_path=os.path.join(self.output_dir, 'confusion_matrix.png'))
        
        self.plot_classification_report(y_test, y_pred,
                                      save_path=os.path.join(self.output_dir, 'classification_report.png'))
        
        # Generate ROC curves if we have probabilities
        if y_proba is not None:
            self.plot_roc_curves(y_test, y_proba,
                               save_path=os.path.join(self.output_dir, 'roc_curves.png'))
        
        # Save detailed evaluation report
        self.save_evaluation_report(metrics, os.path.join(self.output_dir, 'evaluation_report.txt'))
        
        logger.info("Comprehensive evaluation completed")
        return metrics


def evaluate_model_comprehensive(y_true: np.ndarray, y_pred: np.ndarray,
                               y_proba: np.ndarray = None, class_names: List[str] = None,
                               model_name: str = "Model", output_dir: str = None) -> Dict[str, Any]:
    """
    Comprehensive model evaluation with all metrics and visualizations.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_proba (np.ndarray, optional): Prediction probabilities
        class_names (List[str], optional): Class names
        model_name (str): Model name
        output_dir (str, optional): Output directory
        
    Returns:
        Dict[str, Any]: Evaluation results
    """
    evaluator = ModelEvaluator(output_dir)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred, y_proba)
    
    # Generate report
    report = evaluator.generate_evaluation_report(y_true, y_pred, y_proba, 
                                                 class_names, model_name)
    
    # Save results
    report_path = evaluator.save_evaluation_results(y_true, y_pred, y_proba,
                                                   class_names, model_name)
    
    return {
        'metrics': metrics,
        'report': report,
        'report_path': report_path,
        'evaluator': evaluator
    }


if __name__ == "__main__":
    # Test evaluator with dummy data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    
    print("Testing Model Evaluator:")
    print("=" * 50)
    
    # Create dummy data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3,
                             n_informative=15, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Evaluate
    results = evaluate_model_comprehensive(
        y_test, y_pred, y_proba, 
        class_names=['Class_0', 'Class_1', 'Class_2'],
        model_name="TestModel"
    )
    
    print("Evaluation completed!")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Report saved to: {results['report_path']}")
