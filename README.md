# Sentiment Analysis Data Science Project

A complete end-to-end data science project that demonstrates the full pipeline from data collection and preprocessing to model training and API deployment using FastAPI.

## 🚀 Project Overview

This project implements a comprehensive sentiment analysis system with the following components:

- **Data Collection**: Synthetic data generation and web scraping capabilities
- **Data Preprocessing**: Text cleaning, tokenization, and feature engineering
- **Model Training**: PyTorch-based LSTM model for sentiment classification
- **API Development**: FastAPI web service with comprehensive endpoints
- **Model Deployment**: Docker containerization for production deployment
- **Testing**: Unit tests and API testing
- **Documentation**: Complete project documentation

## 📁 Project Structure

```
sentiment_analysis_project/
├── data/
│   ├── raw/                    # Raw data files
│   ├── processed/              # Processed data
│   └── external/               # External datasets
├── models/
│   ├── trained/                # Saved models
│   └── checkpoints/            # Model checkpoints
├── src/
│   ├── data_collection/        # Data collection scripts
│   ├── preprocessing/          # Data preprocessing
│   ├── model_training/         # Model training scripts
│   ├── api/                    # FastAPI application
│   └── utils/                  # Utility functions
├── notebooks/                  # Jupyter notebooks
├── tests/                      # Unit tests
├── config/                     # Configuration files
├── logs/                       # Application logs
├── reports/                    # Training reports and visualizations
├── requirements.txt            # Dependencies
├── main.py                     # FastAPI main application
├── train_model.py              # Model training script
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose configuration
└── README.md                   # Project documentation
```

## 🛠️ Installation

### Prerequisites

- Python 3.11+ (recommended for best compatibility)
- Docker (optional, for containerized deployment)

### Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sentiment_analysis_project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the training pipeline**
   ```bash
   python train_model.py
   ```

5. **Start the API server**
   ```bash
   python main.py
   ```

### Docker Installation

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Or build and run manually**
   ```bash
   docker build -t sentiment-analysis-api .
   docker run -p 8000:8000 sentiment-analysis-api
   ```

## 🚀 Usage

### API Endpoints

Once the server is running, the API will be available at `http://localhost:8000`

#### Available Endpoints:

- **GET /** - API information and available endpoints
- **GET /health** - Health check and model status
- **GET /docs** - Interactive API documentation (Swagger UI)
- **GET /redoc** - Alternative API documentation
- **POST /analyze** - Analyze sentiment of a single text
- **POST /analyze/batch** - Analyze sentiment of multiple texts
- **GET /model/info** - Get model information
- **POST /model/reload** - Reload the model
- **GET /examples** - Get example texts for testing

#### Example API Usage

```python
import requests

# Single text analysis
response = requests.post("http://localhost:8000/analyze", 
                        json={"text": "This movie is absolutely fantastic!"})
result = response.json()
print(f"Sentiment: {result['sentiment']}, Confidence: {result['confidence']:.3f}")

# Batch analysis
texts = [
    "This movie is absolutely fantastic!",
    "This movie is terrible, waste of time.",
    "Amazing performance by all actors!"
]

response = requests.post("http://localhost:8000/analyze/batch", 
                        json={"texts": texts})
results = response.json()
for result in results['results']:
    print(f"Text: {result['text'][:50]}...")
    print(f"Sentiment: {result['sentiment']}, Confidence: {result['confidence']:.3f}")
```

### Training Pipeline

The training pipeline can be run independently:

```bash
python train_model.py
```

This will:
1. Generate synthetic sentiment data
2. Preprocess and clean the data
3. Train a PyTorch LSTM model
4. Evaluate the model performance
5. Save the trained model and vocabulary
6. Generate training reports and visualizations

## 📊 Model Architecture

The project uses a PyTorch LSTM-based model with the following architecture:

- **Embedding Layer**: 128-dimensional word embeddings
- **LSTM Layer**: 2-layer bidirectional LSTM with 64 hidden units
- **Dropout**: 0.3 dropout for regularization
- **Output Layer**: Single neuron with sigmoid activation

### Model Performance

Typical performance metrics:
- **Accuracy**: ~85-90% on test set
- **Training Time**: ~2-5 minutes (depending on hardware)
- **Inference Time**: ~10-50ms per prediction

## 🔧 Configuration

The project uses a YAML configuration file (`config/config.yaml`) for all settings:

```yaml
# Data Collection Settings
data_collection:
  max_samples: 5000

# Data Preprocessing
preprocessing:
  max_sequence_length: 100
  vocabulary_size: 10000
  test_size: 0.2
  validation_size: 0.1

# Model Training
model_training:
  epochs: 20
  batch_size: 32
  learning_rate: 0.001
  early_stopping_patience: 5

# API Settings
api:
  host: "0.0.0.0"
  port: 8000
```

## 🧪 Testing

### Unit Tests

Run the test suite:

```bash
pytest tests/
```

### API Testing

Test the API endpoints:

```bash
# Health check
curl http://localhost:8000/health

# Single analysis
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie is fantastic!"}'

# Batch analysis
curl -X POST "http://localhost:8000/analyze/batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Great movie!", "Terrible film!"]}'
```

## 📈 Monitoring and Logging

The application includes comprehensive logging:

- **Training logs**: `logs/training.log`
- **API logs**: Console output and structured logging
- **Performance metrics**: Saved in `reports/` directory

## 🚀 Deployment

### Production Deployment

1. **Using Docker (Recommended)**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

2. **Using Gunicorn**
   ```bash
   gunicorn src.api.app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

3. **Cloud Deployment**
   - **AWS**: Deploy using ECS or Lambda
   - **Google Cloud**: Deploy using Cloud Run
   - **Azure**: Deploy using Container Instances

### Environment Variables

Set these environment variables for production:

```bash
export PYTHONPATH=/app
export LOG_LEVEL=INFO
export MODEL_PATH=models/trained/sentiment_model.pth
export VOCAB_PATH=models/trained/vocabulary.pkl
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- PyTorch for deep learning capabilities
- NLTK for natural language processing tools
- Docker for containerization

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the API documentation at `http://localhost:8000/docs`
- Review the training logs in `logs/training.log`

---

**Happy Sentiment Analysis! 🎉** 