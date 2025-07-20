"""
Django forms for Financial Sentiment Analysis web interface.
"""

from django import forms
import os
from config import DATA_DIR

class DatasetUploadForm(forms.Form):
    """Form for uploading or selecting dataset."""
    
    # Option to upload new file
    uploaded_file = forms.FileField(
        required=False,
        help_text="Upload a CSV file with 'Sentence' and 'Sentiment' columns",
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.csv'
        })
    )
    
    # Option to select existing file
    existing_file = forms.ChoiceField(
        required=False,
        help_text="Or select an existing dataset from the data directory",
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Populate existing files
        choices = [('', 'Select existing file...')]
        if os.path.exists(DATA_DIR):
            for file in os.listdir(DATA_DIR):
                if file.endswith('.csv'):
                    choices.append((file, file))
        
        self.fields['existing_file'].choices = choices
    
    def clean(self):
        cleaned_data = super().clean()
        uploaded_file = cleaned_data.get('uploaded_file')
        existing_file = cleaned_data.get('existing_file')
        
        if not uploaded_file and not existing_file:
            raise forms.ValidationError("Please either upload a file or select an existing one.")
        
        if uploaded_file and existing_file:
            raise forms.ValidationError("Please choose either upload or select, not both.")
        
        return cleaned_data


class DataFilterForm(forms.Form):
    """Form for filtering dataset samples per class."""
    
    positive_samples = forms.IntegerField(
        min_value=1,
        initial=500,
        help_text="Number of positive samples to include",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., 500'
        })
    )
    
    negative_samples = forms.IntegerField(
        min_value=1,
        initial=500,
        help_text="Number of negative samples to include",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., 500'
        })
    )
    
    neutral_samples = forms.IntegerField(
        min_value=1,
        initial=500,
        help_text="Number of neutral samples to include",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., 500'
        })
    )


class PreprocessingForm(forms.Form):
    """Form for text preprocessing options."""
    
    remove_stopwords = forms.BooleanField(
        initial=True,
        required=False,
        help_text="Remove common English stopwords",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    remove_punctuation = forms.BooleanField(
        initial=True,
        required=False,
        help_text="Remove punctuation marks",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    remove_numbers = forms.BooleanField(
        initial=True,
        required=False,
        help_text="Remove numeric characters",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    normalize_tickers = forms.BooleanField(
        initial=True,
        required=False,
        help_text="Normalize stock tickers ($AAPL -> TICKER)",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    min_length = forms.IntegerField(
        min_value=1,
        initial=3,
        help_text="Minimum text length after preprocessing",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': '3'
        })
    )
    
    max_length = forms.IntegerField(
        min_value=10,
        initial=1000,
        help_text="Maximum text length after preprocessing",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': '1000'
        })
    )


class ModelConfigForm(forms.Form):
    """Form for model configuration."""
    
    MODEL_CHOICES = [
        ('naive_bayes', 'Naive Bayes (MultinomialNB)')
    ]
    
    model_type = forms.ChoiceField(
        choices=MODEL_CHOICES,
        initial='naive_bayes',
        help_text="Choose the machine learning model",
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    alpha = forms.FloatField(
        min_value=0.001,
        max_value=10.0,
        initial=1.0,
        help_text="Smoothing parameter for Naive Bayes (0.001-10.0)",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'placeholder': '1.0'
        })
    )
    
    test_size = forms.FloatField(
        min_value=0.1,
        max_value=0.5,
        initial=0.2,
        help_text="Fraction of dataset to use for testing (0.1-0.5)",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.05',
            'placeholder': '0.2'
        })
    )
    
    cv_folds = forms.IntegerField(
        min_value=3,
        max_value=10,
        initial=5,
        help_text="Number of cross-validation folds (3-10)",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': '5'
        })
    )


class PredictionForm(forms.Form):
    """Form for single text prediction."""
    
    text_input = forms.CharField(
        max_length=1000,
        help_text="Enter financial text to analyze sentiment (max 1000 characters)",
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 4,
            'placeholder': 'e.g., Apple stock is performing well today and showing strong growth...'
        })
    )
