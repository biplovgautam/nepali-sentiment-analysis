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


def comprehensive_eda(df, output_dir, title_prefix="Data"):
    """
    Perform comprehensive exploratory data analysis.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        output_dir (str): Directory to save plots
        title_prefix (str): Prefix for plot titles
        
    Returns:
        dict: EDA results and statistics
    """
    logger.info("Starting comprehensive EDA...")
    
    results = {}
    
    # 1. Class distribution
    class_dist_path = os.path.join(output_dir, 'class_distribution.png')
    plot_class_distribution(df['Sentiment'], class_dist_path)
    
    # 2. Text length analysis
    text_length_path = os.path.join(output_dir, 'text_length_analysis.png')
    plot_text_length_analysis(df, text_length_path, title_prefix)
    
    # 3. Word frequency analysis
    word_freq_path = os.path.join(output_dir, 'word_frequency_analysis.png')
    plot_word_frequency_analysis(df, word_freq_path, title_prefix)
    
    # 4. Word clouds for each sentiment
    wordcloud_dir = os.path.join(output_dir, 'wordclouds')
    os.makedirs(wordcloud_dir, exist_ok=True)
    generate_wordclouds(df['Sentence'], df['Sentiment'], save_dir=wordcloud_dir)
    
    # 5. Generate statistics
    results['statistics'] = generate_eda_statistics(df)
    
    # 6. Print sample sentences
    print_sample_sentences(df['Sentence'], df['Sentiment'])
    
    logger.info("Comprehensive EDA completed")
    return results


def plot_text_length_analysis(df, save_path, title_prefix="Data"):
    """
    Plot text length distribution analysis.
    
    Args:
        df (pd.DataFrame): Dataset
        save_path (str): Path to save plot
        title_prefix (str): Prefix for plot title
    """
    plt.figure(figsize=(15, 10))
    
    # Calculate text lengths
    df_copy = df.copy()
    df_copy['text_length'] = df_copy['Sentence'].str.len()
    df_copy['word_count'] = df_copy['Sentence'].str.split().str.len()
    
    # 1. Overall text length distribution
    plt.subplot(2, 3, 1)
    plt.hist(df_copy['text_length'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'{title_prefix}: Text Length Distribution')
    plt.xlabel('Character Count')
    plt.ylabel('Frequency')
    plt.axvline(df_copy['text_length'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df_copy["text_length"].mean():.1f}')
    plt.legend()
    
    # 2. Word count distribution
    plt.subplot(2, 3, 2)
    plt.hist(df_copy['word_count'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title(f'{title_prefix}: Word Count Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.axvline(df_copy['word_count'].mean(), color='red', linestyle='--',
                label=f'Mean: {df_copy["word_count"].mean():.1f}')
    plt.legend()
    
    # 3. Text length by sentiment
    plt.subplot(2, 3, 3)
    sentiments = df_copy['Sentiment'].unique()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, sentiment in enumerate(sentiments):
        sentiment_data = df_copy[df_copy['Sentiment'] == sentiment]['text_length']
        plt.hist(sentiment_data, bins=30, alpha=0.6, label=sentiment, 
                color=colors[i % len(colors)])
    
    plt.title(f'{title_prefix}: Text Length by Sentiment')
    plt.xlabel('Character Count')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 4. Box plot of text lengths by sentiment
    plt.subplot(2, 3, 4)
    sentiment_lengths = [df_copy[df_copy['Sentiment'] == s]['text_length'].values 
                        for s in sentiments]
    plt.boxplot(sentiment_lengths, labels=sentiments, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(plt.gca().patches, PLOT_CONFIG['color_palette']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title(f'{title_prefix}: Text Length Box Plot')
    plt.ylabel('Character Count')
    plt.xticks(rotation=45)
    
    # 5. Word count by sentiment
    plt.subplot(2, 3, 5)
    sentiment_words = [df_copy[df_copy['Sentiment'] == s]['word_count'].values 
                      for s in sentiments]
    plt.boxplot(sentiment_words, labels=sentiments, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(plt.gca().patches, PLOT_CONFIG['color_palette']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title(f'{title_prefix}: Word Count Box Plot')
    plt.ylabel('Word Count')
    plt.xticks(rotation=45)
    
    # 6. Summary statistics table
    plt.subplot(2, 3, 6)
    stats_text = f"{title_prefix} Statistics:\n\n"
    stats_text += f"Total samples: {len(df_copy)}\n"
    stats_text += f"Avg text length: {df_copy['text_length'].mean():.1f}\n"
    stats_text += f"Avg word count: {df_copy['word_count'].mean():.1f}\n"
    stats_text += f"Min length: {df_copy['text_length'].min()}\n"
    stats_text += f"Max length: {df_copy['text_length'].max()}\n\n"
    
    for sentiment in sentiments:
        count = len(df_copy[df_copy['Sentiment'] == sentiment])
        pct = (count / len(df_copy)) * 100
        stats_text += f"{sentiment}: {count} ({pct:.1f}%)\n"
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Text length analysis saved to {save_path}")
    plt.show()


def plot_word_frequency_analysis(df, save_path, title_prefix="Data"):
    """
    Plot word frequency analysis.
    
    Args:
        df (pd.DataFrame): Dataset
        save_path (str): Path to save plot
        title_prefix (str): Prefix for plot title
    """
    from collections import Counter
    import re
    
    plt.figure(figsize=(15, 10))
    
    # Get all words
    all_text = ' '.join(df['Sentence'].astype(str))
    words = re.findall(r'\b\w+\b', all_text.lower())
    word_freq = Counter(words)
    
    # Overall top words
    plt.subplot(2, 2, 1)
    top_words = word_freq.most_common(20)
    words_list, counts_list = zip(*top_words)
    
    plt.barh(range(len(words_list)), counts_list, color='lightblue')
    plt.yticks(range(len(words_list)), words_list)
    plt.xlabel('Frequency')
    plt.title(f'{title_prefix}: Top 20 Most Frequent Words')
    plt.gca().invert_yaxis()
    
    # Words by sentiment
    sentiments = df['Sentiment'].unique()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, sentiment in enumerate(sentiments):
        plt.subplot(2, 2, i + 2)
        sentiment_text = ' '.join(df[df['Sentiment'] == sentiment]['Sentence'].astype(str))
        sentiment_words = re.findall(r'\b\w+\b', sentiment_text.lower())
        sentiment_freq = Counter(sentiment_words)
        
        top_sentiment_words = sentiment_freq.most_common(15)
        if top_sentiment_words:
            words_list, counts_list = zip(*top_sentiment_words)
            
            plt.barh(range(len(words_list)), counts_list, color=colors[i])
            plt.yticks(range(len(words_list)), words_list)
            plt.xlabel('Frequency')
            plt.title(f'{sentiment.title()} Sentiment: Top Words')
            plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Word frequency analysis saved to {save_path}")
    plt.show()


def generate_comparison_plots(df, output_dir, original_col='Sentence', processed_col='Processed_Sentence'):
    """
    Generate before/after preprocessing comparison plots.
    
    Args:
        df (pd.DataFrame): Dataset with original and processed columns
        output_dir (str): Directory to save plots
        original_col (str): Name of original text column
        processed_col (str): Name of processed text column
    """
    plt.figure(figsize=(16, 12))
    
    # Calculate lengths
    original_lengths = df[original_col].str.len()
    processed_lengths = df[processed_col].str.len()
    
    # 1. Length comparison histogram
    plt.subplot(3, 3, 1)
    plt.hist(original_lengths, bins=50, alpha=0.6, label='Original', color='lightcoral')
    plt.hist(processed_lengths, bins=50, alpha=0.6, label='Processed', color='lightblue')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.title('Text Length: Before vs After Preprocessing')
    plt.legend()
    
    # 2. Length reduction scatter plot
    plt.subplot(3, 3, 2)
    reduction_ratio = (original_lengths - processed_lengths) / original_lengths
    plt.scatter(original_lengths, reduction_ratio, alpha=0.5, s=10)
    plt.xlabel('Original Length')
    plt.ylabel('Length Reduction Ratio')
    plt.title('Length Reduction vs Original Length')
    plt.axhline(y=reduction_ratio.mean(), color='red', linestyle='--', 
                label=f'Mean: {reduction_ratio.mean():.2%}')
    plt.legend()
    
    # 3. Word count comparison
    plt.subplot(3, 3, 3)
    original_words = df[original_col].str.split().str.len()
    processed_words = df[processed_col].str.split().str.len()
    
    plt.hist(original_words, bins=30, alpha=0.6, label='Original', color='lightcoral')
    plt.hist(processed_words, bins=30, alpha=0.6, label='Processed', color='lightblue')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.title('Word Count: Before vs After')
    plt.legend()
    
    # 4. Box plots by sentiment - Original
    plt.subplot(3, 3, 4)
    sentiments = df['Sentiment'].unique()
    original_by_sentiment = [df[df['Sentiment'] == s][original_col].str.len().values 
                           for s in sentiments]
    plt.boxplot(original_by_sentiment, labels=sentiments, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(plt.gca().patches, PLOT_CONFIG['color_palette']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel('Character Count')
    plt.title('Original Text Length by Sentiment')
    plt.xticks(rotation=45)
    
    # 5. Box plots by sentiment - Processed
    plt.subplot(3, 3, 5)
    processed_by_sentiment = [df[df['Sentiment'] == s][processed_col].str.len().values 
                            for s in sentiments]
    plt.boxplot(processed_by_sentiment, labels=sentiments, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(plt.gca().patches, PLOT_CONFIG['color_palette']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel('Character Count')
    plt.title('Processed Text Length by Sentiment')
    plt.xticks(rotation=45)
    
    # 6. Reduction ratio by sentiment
    plt.subplot(3, 3, 6)
    reduction_by_sentiment = []
    for sentiment in sentiments:
        sentiment_data = df[df['Sentiment'] == sentiment]
        orig_len = sentiment_data[original_col].str.len()
        proc_len = sentiment_data[processed_col].str.len()
        reduction = (orig_len - proc_len) / orig_len
        reduction_by_sentiment.append(reduction.values)
    
    plt.boxplot(reduction_by_sentiment, labels=sentiments)
    plt.ylabel('Length Reduction Ratio')
    plt.title('Length Reduction by Sentiment')
    plt.xticks(rotation=45)
    
    # 7-9. Sample text comparisons
    for i, sentiment in enumerate(sentiments[:3]):
        plt.subplot(3, 3, 7 + i)
        sentiment_data = df[df['Sentiment'] == sentiment].iloc[0]
        
        sample_text = f"ORIGINAL ({sentiment}):\n{sentiment_data[original_col][:100]}...\n\n"
        sample_text += f"PROCESSED:\n{sentiment_data[processed_col][:100]}..."
        
        plt.text(0.05, 0.95, sample_text, transform=plt.gca().transAxes,
                fontsize=8, verticalalignment='top', wrap=True,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        plt.axis('off')
        plt.title(f'Sample {sentiment.title()} Text')
    
    plt.tight_layout()
    
    comparison_path = os.path.join(output_dir, 'preprocessing_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    logger.info(f"Preprocessing comparison saved to {comparison_path}")
    plt.show()
    
    # Generate summary statistics
    stats = {
        'original_avg_length': original_lengths.mean(),
        'processed_avg_length': processed_lengths.mean(),
        'avg_reduction_ratio': reduction_ratio.mean(),
        'original_avg_words': original_words.mean(),
        'processed_avg_words': processed_words.mean()
    }
    
    return stats


def generate_eda_statistics(df):
    """
    Generate comprehensive statistics for the dataset.
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        dict: Statistics dictionary
    """
    stats = {
        'total_samples': len(df),
        'class_distribution': df['Sentiment'].value_counts().to_dict(),
        'avg_text_length': df['Sentence'].str.len().mean(),
        'median_text_length': df['Sentence'].str.len().median(),
        'avg_word_count': df['Sentence'].str.split().str.len().mean(),
        'unique_texts': df['Sentence'].nunique(),
        'duplicate_texts': len(df) - df['Sentence'].nunique()
    }
    
    return stats


if __name__ == "__main__":
    # Test EDA functions with sample data
    from utils.loader import load_and_prepare_data
    
    try:
        X, y, dataset_info = load_and_prepare_data()
        generate_comprehensive_eda_report(X, y, dataset_info)
    except Exception as e:
        print(f"Error running EDA: {e}")