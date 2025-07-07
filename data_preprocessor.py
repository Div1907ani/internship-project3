"""
Data Preprocessing Module for Sentiment Analysis Project
Handles text cleaning, tokenization, and data preparation
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
from pathlib import Path
from loguru import logger

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    logger.warning("NLTK data download failed. Some features may not work.")

class TextPreprocessor:
    """Text preprocessing class"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vocabulary = {}
        self.max_sequence_length = config.get('preprocessing', {}).get('max_sequence_length', 100)
        self.vocabulary_size = config.get('preprocessing', {}).get('vocabulary_size', 10000)
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text
        """
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text: str) -> str:
        """
        Lemmatize words in text
        """
        # Use simple split instead of word_tokenize to avoid NLTK issues
        words = text.split()
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    
    def preprocess_text(self, text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
        """
        Complete text preprocessing pipeline
        """
        # Clean text
        text = self.clean_text(text)
        
        # Remove stopwords
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        # Lemmatize
        if lemmatize:
            text = self.lemmatize_text(text)
        
        return text
    
    def create_vocabulary(self, texts: List[str]) -> Dict[str, int]:
        """
        Create vocabulary from texts
        """
        word_counts = {}
        
        for text in texts:
            words = text.split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Reserve 2 spots: 0 for <UNK>, last for <PAD>
        num_words = self.vocabulary_size - 2
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        vocabulary = {'<UNK>': 0}
        for idx, (word, count) in enumerate(sorted_words[:num_words]):
            vocabulary[word] = idx + 1  # 1 to self.vocabulary_size-2
        vocabulary['<PAD>'] = self.vocabulary_size - 1  # Padding token
        
        self.vocabulary = vocabulary
        logger.info(f"Created vocabulary with {len(vocabulary)} words (UNK=0, PAD={self.vocabulary_size-1})")
        
        return vocabulary
    
    def text_to_sequence(self, text: str) -> List[int]:
        """
        Convert text to sequence of indices
        """
        words = text.split()
        sequence = [self.vocabulary.get(word, self.vocabulary['<UNK>']) for word in words[:self.max_sequence_length]]
        
        # Pad sequence
        if len(sequence) < self.max_sequence_length:
            sequence += [self.vocabulary['<PAD>']] * (self.max_sequence_length - len(sequence))
        
        return sequence
    
    def save_vocabulary(self, filepath: str) -> None:
        """
        Save vocabulary to file
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.vocabulary, f)
        logger.info(f"Saved vocabulary to {filepath}")
    
    def load_vocabulary(self, filepath: str) -> None:
        """
        Load vocabulary from file
        """
        with open(filepath, 'rb') as f:
            self.vocabulary = pickle.load(f)
        logger.info(f"Loaded vocabulary from {filepath}")

class DataPreprocessor:
    """Main data preprocessing class"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.text_preprocessor = TextPreprocessor(config)
        self.vectorizer = None
        
    def preprocess_dataset(self, df: pd.DataFrame, text_column: str = 'text', label_column: str = 'label') -> pd.DataFrame:
        """
        Preprocess entire dataset
        """
        logger.info("Starting dataset preprocessing...")
        
        # Create copy to avoid modifying original
        processed_df = df.copy()
        
        # Preprocess text
        logger.info("Preprocessing text...")
        processed_df['processed_text'] = processed_df[text_column].apply(
            lambda x: self.text_preprocessor.preprocess_text(str(x))
        )
        
        # Remove empty texts
        processed_df = processed_df[processed_df['processed_text'].str.len() > 0]
        
        # Create vocabulary
        logger.info("Creating vocabulary...")
        self.text_preprocessor.create_vocabulary(processed_df['processed_text'].tolist())
        
        # Convert to sequences
        logger.info("Converting to sequences...")
        processed_df['sequence'] = processed_df['processed_text'].apply(
            lambda x: self.text_preprocessor.text_to_sequence(x)
        )
        
        logger.info(f"Preprocessing completed. Final dataset size: {len(processed_df)}")
        
        return processed_df
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        """
        # First split: train+val and test
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df['label']
        )
        
        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size_adjusted, random_state=random_state, stratify=train_val_df['label']
        )
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def create_tfidf_features(self, df: pd.DataFrame, text_column: str = 'processed_text') -> np.ndarray:
        """
        Create TF-IDF features
        """
        logger.info("Creating TF-IDF features...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.text_preprocessor.vocabulary_size,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        tfidf_features = self.vectorizer.fit_transform(df[text_column])
        
        logger.info(f"TF-IDF features shape: {tfidf_features.shape}")
        
        return tfidf_features.toarray()
    
    def save_preprocessor(self, filepath: str) -> None:
        """
        Save preprocessor state
        """
        preprocessor_state = {
            'vocabulary': self.text_preprocessor.vocabulary,
            'vectorizer': self.vectorizer,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_state, f)
        
        logger.info(f"Saved preprocessor to {filepath}")
    
    def load_preprocessor(self, filepath: str) -> None:
        """
        Load preprocessor state
        """
        with open(filepath, 'rb') as f:
            preprocessor_state = pickle.load(f)
        
        self.text_preprocessor.vocabulary = preprocessor_state['vocabulary']
        self.vectorizer = preprocessor_state['vectorizer']
        
        logger.info(f"Loaded preprocessor from {filepath}")
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get dataset statistics
        """
        stats = {
            'total_samples': len(df),
            'positive_samples': sum(df['label'] == 1),
            'negative_samples': sum(df['label'] == 0),
            'avg_text_length': df['processed_text'].str.len().mean(),
            'min_text_length': df['processed_text'].str.len().min(),
            'max_text_length': df['processed_text'].str.len().max(),
            'vocabulary_size': len(self.text_preprocessor.vocabulary)
        }
        
        logger.info("Dataset statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        return stats 