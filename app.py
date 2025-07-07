"""
FastAPI Application for Sentiment Analysis Project
Provides REST API endpoints for sentiment analysis
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import torch
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.preprocessing.data_preprocessor import TextPreprocessor
from src.model_training.model_trainer import PyTorchSentimentModel
from loguru import logger

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="A comprehensive sentiment analysis API using deep learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
model = None
text_preprocessor = None
vocabulary = None
model_loaded = False

# Pydantic models for request/response
class SentimentRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for sentiment", min_length=1, max_length=1000)
    
class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probability: float
    processing_time: float
    
class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze", min_items=1, max_items=100)
    
class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    total_processing_time: float
    
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Dict[str, Any]
    timestamp: str

class ModelInfo(BaseModel):
    model_type: str
    vocabulary_size: int
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    device: str

# Load model and preprocessor
def load_model_and_preprocessor():
    """Load the trained model and preprocessor"""
    global model, text_preprocessor, vocabulary, model_loaded
    
    try:
        # Load vocabulary
        vocab_path = "models/trained/vocabulary.pkl"
        if Path(vocab_path).exists():
            import pickle
            with open(vocab_path, 'rb') as f:
                vocabulary = pickle.load(f)
            logger.info(f"Loaded vocabulary with {len(vocabulary)} words")
        else:
            logger.warning("Vocabulary file not found, using default")
            vocabulary = {'<UNK>': 0, '<PAD>': 1}
        
        # Load preprocessor with correct vocabulary size
        config = {
            'preprocessing': {
                'max_sequence_length': 100,
                'vocabulary_size': 10000  # Match the trained model
            }
        }
        text_preprocessor = TextPreprocessor(config)
        text_preprocessor.vocabulary = vocabulary
        
        # Load model with correct vocabulary size
        model_path = "models/trained/sentiment_model.pth"
        if Path(model_path).exists():
            model = PyTorchSentimentModel(
                vocab_size=10000,  # Match the trained model's vocabulary size
                embedding_dim=128,
                hidden_dim=64,
                num_layers=2,
                dropout=0.3,
                padding_idx=9999  # Match the trained model's padding index
            )
            
            # Load model weights
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            model_loaded = True
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model file not found, using default model")
            model = PyTorchSentimentModel(
                vocab_size=len(vocabulary),
                embedding_dim=128,
                hidden_dim=64,
                num_layers=2,
                dropout=0.3
            )
            model_loaded = False
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_loaded = False

# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting Sentiment Analysis API...")
    load_model_and_preprocessor()

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_info = {
        "model_type": "PyTorch LSTM",
        "vocabulary_size": len(vocabulary) if vocabulary else 0,
        "embedding_dim": 128,
        "hidden_dim": 64,
        "num_layers": 2,
        "device": str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    }
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        model_info=model_info,
        timestamp=datetime.now().isoformat()
    )

# Single text sentiment analysis
@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment of a single text"""
    
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Preprocess text
        processed_text = text_preprocessor.preprocess_text(request.text)
        
        # Convert to sequence
        sequence = text_preprocessor.text_to_sequence(processed_text)
        
        # Convert to tensor
        input_tensor = torch.tensor([sequence], dtype=torch.long)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probability = output.item()
        
        # Determine sentiment
        sentiment = "positive" if probability > 0.5 else "negative"
        confidence = probability if probability > 0.5 else 1 - probability
        
        processing_time = time.time() - start_time
        
        return SentimentResponse(
            text=request.text,
            sentiment=sentiment,
            confidence=confidence,
            probability=probability,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")

# Batch sentiment analysis
@app.post("/analyze/batch", response_model=BatchSentimentResponse)
async def analyze_sentiment_batch(request: BatchSentimentRequest):
    """Analyze sentiment of multiple texts"""
    
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    results = []
    
    try:
        for text in request.texts:
            # Preprocess text
            processed_text = text_preprocessor.preprocess_text(text)
            
            # Convert to sequence
            sequence = text_preprocessor.text_to_sequence(processed_text)
            
            # Convert to tensor
            input_tensor = torch.tensor([sequence], dtype=torch.long)
            
            # Make prediction
            with torch.no_grad():
                output = model(input_tensor)
                probability = output.item()
            
            # Determine sentiment
            sentiment = "positive" if probability > 0.5 else "negative"
            confidence = probability if probability > 0.5 else 1 - probability
            
            results.append(SentimentResponse(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                probability=probability,
                processing_time=0.0  # Individual processing time not tracked for batch
            ))
        
        total_processing_time = time.time() - start_time
        
        return BatchSentimentResponse(
            results=results,
            total_processing_time=total_processing_time
        )
        
    except Exception as e:
        logger.error(f"Error analyzing batch sentiment: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing batch sentiment: {str(e)}")

# Model information endpoint
@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    
    return ModelInfo(
        model_type="PyTorch LSTM",
        vocabulary_size=len(vocabulary) if vocabulary else 0,
        embedding_dim=128,
        hidden_dim=64,
        num_layers=2,
        device=str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    )

# Reload model endpoint
@app.post("/model/reload")
async def reload_model():
    """Reload the model and preprocessor"""
    
    try:
        load_model_and_preprocessor()
        return {"message": "Model reloaded successfully", "model_loaded": model_loaded}
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error reloading model: {str(e)}")

# Example usage endpoint
@app.get("/examples")
async def get_examples():
    """Get example texts for testing"""
    
    examples = [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "This movie is terrible, waste of time and money.",
        "Amazing performance by all actors, highly recommended!",
        "Boring plot, bad acting, disappointing overall.",
        "Great plot, excellent cinematography, wonderful soundtrack.",
        "Poor direction, confusing story, not worth watching."
    ]
    
    return {"examples": examples}

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "analyze": "/analyze",
            "batch_analyze": "/analyze/batch",
            "health": "/health",
            "model_info": "/model/info",
            "examples": "/examples"
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 