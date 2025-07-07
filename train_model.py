"""
Main FastAPI Application Entry Point
Run this file to start the Sentiment Analysis API server
"""

import uvicorn
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

if __name__ == "__main__":
    # Check if model exists
    model_path = Path("models/trained/sentiment_model.pth")
    vocab_path = Path("models/trained/vocabulary.pkl")
    
    if not model_path.exists() or not vocab_path.exists():
        print("=" * 60)
        print("WARNING: Model files not found!")
        print("=" * 60)
        print("Please run the training pipeline first:")
        print("python train_model.py")
        print("\nOr the API will use a default untrained model.")
        print("=" * 60)
    
    # Start the FastAPI server
    print("Starting Sentiment Analysis API...")
    print("API will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 