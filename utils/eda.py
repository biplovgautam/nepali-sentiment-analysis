"""
Exploratory Data Analysis (EDA) utilities for Financial Sentiment Analysis.
Generates visualizations and statistical summaries.
"""

import pandas as pd
import numpy as np
import matplotlib
# Set backend before importing pyplot to avoid display issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import logging
from typing import Dict, List, Tuple, Any
import os
from config import OUTPUT_DIR, PLOT_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
        
sns.set_palette(PLOT_CONFIG['color_palette'])


def plot_class_distribution(y: pd.Series, save_path: str = None) -> None:
    """
    Plot the distribution of sentiment classes.
    
    Args:
        y (pd.Series): Target labels
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=PLOT_CONFIG['figure_size'])
    
    # Count plot
    plt.subplot(1, 2, 1)
    sentiment_counts = y.value_counts()
    colors = PLOT_CONFIG['color_palette'][:len(sentiment_counts)]
    
    bars = plt.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
    plt.title('Sentiment Class Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom')
    
    # Pie chart
    plt.subplot(1, 2, 2)
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, 
            autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Sentiment Distribution (%)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        logger.info(f"Class distribution plot saved to {save_path}")
    
    plt.show()


def print_sample_sentences(X: pd.Series, y: pd.Series, samples_per_class: int = 3) -> None:
    """
    Print sample sentences from each sentiment class.
    
    Args:
        X (pd.Series): Features (sentences)
        y (pd.Series): Labels (sentiments)
        samples_per_class (int): Number of samples to show per class
    """
    print("\n" + "="*80)
    print("SAMPLE SENTENCES FROM EACH CLASS")
    print("="*80)
    
    for sentiment in y.unique():
        sentiment_mask = y == sentiment
        sentiment_sentences = X[sentiment_mask].head(samples_per_class)
        
        print(f"\n{sentiment.upper()} SENTIMENT:")
        print("-" * 50)
        
        for i, sentence in enumerate(sentiment_sentences.values, 1):
            print(f"{i}. {sentence}")
    
    print("="*80 + "\n")


def plot_sentence_length_distribution(X: pd.Series, y: pd.Series, save_path: str = None) -> None:
    """
    Plot the distribution of sentence lengths by sentiment.
    
    Args:
        X (pd.Series): Features (sentences)
        y (pd.Series): Labels (sentiments)
        save_path (str, optional): Path to save the plot
    """
    # Calculate sentence lengths
    sentence_lengths = X.str.len()
    
    plt.figure(figsize=PLOT_CONFIG['figure_size'])
    
    # Combined histogram
    plt.subplot(2, 2, 1)
    plt.hist(sentence_lengths, bins=50, alpha=0.7, color=PLOT_CONFIG['color_palette'][0])
    plt.title('Overall Sentence Length Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Sentence Length (characters)')
    plt.ylabel('Frequency')
    
    # Box plot by sentiment
    plt.subplot(2, 2, 2)
    data_for_box = []
    labels_for_box = []
    
    for sentiment in y.unique():
        mask = y == sentiment
        lengths = sentence_lengths[mask]
        data_for_box.append(lengths)
        labels_for_box.append(sentiment)
    
    box_plot = plt.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(box_plot['boxes'], PLOT_CONFIG['color_palette']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title('Sentence Length by Sentiment', fontsize=14, fontweight='bold')
    plt.xlabel('Sentiment')
    plt.ylabel('Sentence Length (characters)')
    
    # Violin plot
    plt.subplot(2, 2, 3)
    df_temp = pd.DataFrame({'length': sentence_lengths, 'sentiment': y})
    sns.violinplot(data=df_temp, x='sentiment', y='length')
    plt.title('Sentence Length Distribution by Sentiment', fontsize=14, fontweight='bold')
    plt.xlabel('Sentiment')
    plt.ylabel('Sentence Length (characters)')
    
    # Statistics table
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    stats_data = []
    for sentiment in y.unique():
        mask = y == sentiment
        lengths = sentence_lengths[mask]
        stats_data.append([
            sentiment,
            f"{lengths.mean():.1f}",
            f"{lengths.std():.1f}",
            f"{lengths.median():.1f}",
            f"{lengths.min()}-{lengths.max()}"
        ])
    
    table = plt.table(cellText=stats_data,
                     colLabels=['Sentiment', 'Mean', 'Std', 'Median', 'Range'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    plt.title('Length Statistics by Sentiment', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        logger.info(f"Sentence length distribution plot saved to {save_path}")
    
    plt.show()


def generate_wordclouds(X: pd.Series, y: pd.Series, save_dir: str = None) -> None:
    """
    Generate word clouds for each sentiment class.
    
    Args:
        X (pd.Series): Features (sentences)
        y (pd.Series): Labels (sentiments)
        save_dir (str, optional): Directory to save word clouds
    """
    sentiments = y.unique()
    n_sentiments = len(sentiments)
    
    fig, axes = plt.subplots(1, n_sentiments, figsize=(6*n_sentiments, 6))
    
    if n_sentiments == 1:
        axes = [axes]
    
    for i, sentiment in enumerate(sentiments):
        # Get text for this sentiment
        sentiment_mask = y == sentiment
        sentiment_text = ' '.join(X[sentiment_mask].values)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=400, 
            height=400, 
            background_color='white',
            colormap=plt.cm.get_cmap('viridis'),
            max_words=100,
            relative_scaling=0.5,
            random_state=42
        ).generate(sentiment_text)
        
        # Plot
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].set_title(f'{sentiment.title()} Sentiment Word Cloud', 
                         fontsize=14, fontweight='bold')
        axes[i].axis('off')
        
        # Save individual word cloud
        if save_dir:
            individual_path = os.path.join(save_dir, f'wordcloud_{sentiment}.png')
            wordcloud.to_file(individual_path)
            logger.info(f"Word cloud for {sentiment} saved to {individual_path}")
    
    plt.tight_layout()
    
    if save_dir:
        combined_path = os.path.join(save_dir, 'wordclouds_combined.png')
        plt.savefig(combined_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        logger.info(f"Combined word clouds saved to {combined_path}")
    
    plt.show()


def plot_word_frequency_analysis(X: pd.Series, y: pd.Series, top_n: int = 20, 
                                save_path: str = None) -> None:
    """
    Plot top words frequency analysis by sentiment.
    
    Args:
        X (pd.Series): Features (sentences)
        y (pd.Series): Labels (sentiments)
        top_n (int): Number of top words to show
        save_path (str, optional): Path to save the plot
    """
    from collections import Counter
    import re
    
    sentiments = y.unique()
    n_sentiments = len(sentiments)
    
    fig, axes = plt.subplots(n_sentiments, 1, figsize=(12, 4*n_sentiments))
    
    if n_sentiments == 1:
        axes = [axes]
    
    for i, sentiment in enumerate(sentiments):
        # Get text for this sentiment
        sentiment_mask = y == sentiment
        sentiment_text = ' '.join(X[sentiment_mask].values).lower()
        
        # Simple tokenization (split by word boundaries)
        words = re.findall(r'\b[a-zA-Z]{2,}\b', sentiment_text)
        
        # Count words
        word_counts = Counter(words)
        top_words = word_counts.most_common(top_n)
        
        # Plot
        words_list, counts_list = zip(*top_words)
        
        bars = axes[i].barh(words_list[::-1], counts_list[::-1], 
                           color=PLOT_CONFIG['color_palette'][i % len(PLOT_CONFIG['color_palette'])])
        axes[i].set_title(f'Top {top_n} Words in {sentiment.title()} Sentiment', 
                         fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Frequency')
        
        # Add count labels
        for bar, count in zip(bars, counts_list[::-1]):
            axes[i].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        str(count), ha='left', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        logger.info(f"Word frequency analysis plot saved to {save_path}")
    
    plt.show()


def generate_comprehensive_eda_report(X: pd.Series, y: pd.Series, 
                                    dataset_info: Dict, output_dir: str = OUTPUT_DIR) -> None:
    """
    Generate a comprehensive EDA report with all visualizations.
    
    Args:
        X (pd.Series): Features (sentences)
        y (pd.Series): Labels (sentiments)
        dataset_info (Dict): Dataset information from loader
        output_dir (str): Directory to save all outputs
    """
    logger.info("Starting comprehensive EDA report generation")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Print basic dataset information
    print("\n" + "="*80)
    print("FINANCIAL SENTIMENT ANALYSIS - EDA REPORT")
    print("="*80)
    print(f"Total samples: {dataset_info['total_samples']}")
    print(f"Sentiment distribution: {dataset_info['sentiment_distribution']}")
    print(f"Average sentence length: {dataset_info['sentence_length_stats']['mean']:.1f} characters")
    print("="*80 + "\n")
    
    # 1. Class distribution plot
    class_dist_path = os.path.join(output_dir, 'class_distribution.png')
    plot_class_distribution(y, save_path=class_dist_path)
    
    # 2. Sample sentences
    print_sample_sentences(X, y)
    
    # 3. Sentence length distribution
    length_dist_path = os.path.join(output_dir, 'sentence_length_distribution.png')
    plot_sentence_length_distribution(X, y, save_path=length_dist_path)
    
    # 4. Word clouds
    generate_wordclouds(X, y, save_dir=output_dir)
    
    # 5. Word frequency analysis
    word_freq_path = os.path.join(output_dir, 'word_frequency_analysis.png')
    plot_word_frequency_analysis(X, y, save_path=word_freq_path)
    
    logger.info(f"EDA report generation completed. All outputs saved to {output_dir}")


def perform_comprehensive_eda(X: pd.Series, y: pd.Series, output_dir: str = OUTPUT_DIR) -> Dict[str, Any]:
    """
    Perform comprehensive EDA and return results.
    
    Args:
        X (pd.Series): Features (sentences)
        y (pd.Series): Labels (sentiments) 
        output_dir (str): Directory to save outputs
        
    Returns:
        Dict[str, Any]: EDA results and statistics
    """
    logger.info("Starting comprehensive EDA...")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get basic dataset info
    dataset_info = {
        'total_samples': len(X),
        'sentiment_distribution': y.value_counts().to_dict(),
        'sentence_length_stats': {
            'mean': X.str.len().mean(),
            'median': X.str.len().median(),
            'min': X.str.len().min(),
            'max': X.str.len().max(),
            'std': X.str.len().std()
        }
    }
    
    # Generate visualizations if output directory provided
    if output_dir:
        logger.info("Generating EDA visualizations...")
        
        # 1. Class distribution plot
        class_dist_path = os.path.join(output_dir, 'class_distribution.png')
        plot_class_distribution(y, save_path=class_dist_path)
        
        # 2. Sentence length distribution
        length_dist_path = os.path.join(output_dir, 'sentence_length_distribution.png')
        plot_sentence_length_distribution(X, y, save_path=length_dist_path)
        
        # 3. Word clouds
        generate_wordclouds(X, y, save_dir=output_dir)
        
        # 4. Word frequency analysis
        word_freq_path = os.path.join(output_dir, 'word_frequency_analysis.png')
        plot_word_frequency_analysis(X, y, save_path=word_freq_path)
    
    # Print sample sentences
    print_sample_sentences(X, y)
    
    logger.info("EDA completed successfully")
    
    return {
        'dataset_info': dataset_info,
        'visualizations_saved': output_dir is not None,
        'output_directory': output_dir
    }


if __name__ == "__main__":
    # Test EDA functions with sample data
    from utils.loader import load_and_prepare_data
    
    try:
        X, y, dataset_info = load_and_prepare_data()
        generate_comprehensive_eda_report(X, y, dataset_info)
    except Exception as e:
        print(f"Error running EDA: {e}")