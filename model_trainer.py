"""
Model Training Module for Sentiment Analysis Project
Supports both PyTorch and TensorFlow model training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import json
from pathlib import Path
from loguru import logger
import time

# Check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class PyTorchSentimentModel(nn.Module):
    """PyTorch LSTM-based sentiment analysis model"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 64, 
                 num_layers: int = 2, dropout: float = 0.3, padding_idx: int = 0):
        super(PyTorchSentimentModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # Take the last output
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        output = self.sigmoid(output)
        return output

class ModelTrainer:
    """Main model training class"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
    def create_pytorch_model(self, vocab_size: int) -> PyTorchSentimentModel:
        """Create PyTorch model"""
        model_config = self.config.get('models', {}).get('pytorch', {})
        vocab_size = self.config.get('preprocessing', {}).get('vocabulary_size', vocab_size)
        padding_idx = vocab_size - 1
        model = PyTorchSentimentModel(
            vocab_size=vocab_size,
            embedding_dim=model_config.get('embedding_dim', 128),
            hidden_dim=model_config.get('hidden_dim', 64),
            num_layers=model_config.get('num_layers', 2),
            dropout=model_config.get('dropout', 0.3),
            padding_idx=padding_idx
        ).to(device)
        
        logger.info(f"Created PyTorch model with {sum(p.numel() for p in model.parameters()):,} parameters (vocab_size={vocab_size}, padding_idx={padding_idx})")
        return model
    
    def prepare_pytorch_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           test_df: pd.DataFrame, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data for PyTorch training"""
        
        # Convert to tensors
        X_train = torch.tensor(train_df['sequence'].tolist(), dtype=torch.long)
        y_train = torch.tensor(train_df['label'].values, dtype=torch.long)
        
        X_val = torch.tensor(val_df['sequence'].tolist(), dtype=torch.long)
        y_val = torch.tensor(val_df['label'].values, dtype=torch.long)
        
        X_test = torch.tensor(test_df['sequence'].tolist(), dtype=torch.long)
        y_test = torch.tensor(test_df['label'].values, dtype=torch.long)
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_pytorch_model(self, model: PyTorchSentimentModel, train_loader: DataLoader, 
                           val_loader: DataLoader, epochs: int = 20, learning_rate: float = 0.001) -> Dict[str, list]:
        """Train PyTorch model"""
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = self.config.get('model_training', {}).get('early_stopping_patience', 5)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        logger.info("Starting PyTorch model training...")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y.float())
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x).squeeze()
                    loss = criterion(outputs, batch_y.float())
                    
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'models/checkpoints/best_pytorch_model.pth')
            else:
                patience_counter += 1
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - start_time
            
            logger.info(f'Epoch [{epoch+1}/{epochs}] ({epoch_time:.2f}s) - '
                       f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                       f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        model.load_state_dict(torch.load('models/checkpoints/best_pytorch_model.pth'))
        
        return history
    
    def evaluate_pytorch_model(self, model: PyTorchSentimentModel, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate PyTorch model"""
        
        model.eval()
        test_predictions = []
        test_labels = []
        test_probabilities = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()
                probabilities = outputs.cpu().numpy()
                predictions = (outputs > 0.5).float().cpu().numpy()
                
                test_predictions.extend(predictions)
                test_labels.extend(batch_y.cpu().numpy())
                test_probabilities.extend(probabilities)
        
        # Calculate metrics
        test_accuracy = sum(1 for p, l in zip(test_predictions, test_labels) if p == l) / len(test_labels)
        
        # Classification report
        report = classification_report(test_labels, test_predictions, 
                                     target_names=['Negative', 'Positive'], output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, test_predictions)
        
        results = {
            'accuracy': test_accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': test_predictions,
            'probabilities': test_probabilities,
            'true_labels': test_labels
        }
        
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info("Classification Report:")
        logger.info(classification_report(test_labels, test_predictions, target_names=['Negative', 'Positive']))
        
        return results
    
    def plot_training_history(self, history: Dict[str, list], save_path: str = None):
        """Plot training history"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history['train_acc'], label='Train Accuracy')
        ax1.plot(history['val_acc'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history['train_loss'], label='Train Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None):
        """Plot confusion matrix"""
        
        plt.figure(figsize=(8, 6))
        ConfusionMatrixDisplay(cm, display_labels=['Negative', 'Positive']).plot(cmap='Blues')
        plt.title('Confusion Matrix - Sentiment Analysis')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, model: PyTorchSentimentModel, filepath: str):
        """Save trained model"""
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch model
        torch.save(model.state_dict(), filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, model: PyTorchSentimentModel, filepath: str):
        """Load trained model"""
        
        model.load_state_dict(torch.load(filepath, map_location=device))
        model.eval()
        logger.info(f"Model loaded from {filepath}")
    
    def save_training_results(self, results: Dict[str, Any], filepath: str):
        """Save training results"""
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, dict):
                json_results[key] = value
            else:
                json_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Training results saved to {filepath}") 