"""
Test script for the Sentiment Analysis API
Tests all endpoints and functionality
"""

import requests
import json
import time
from typing import Dict, Any

class SentimentAPITester:
    """Test class for the Sentiment Analysis API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health(self) -> Dict[str, Any]:
        """Test health endpoint"""
        print("Testing health endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            result = response.json()
            print(f"✅ Health check passed: {result['status']}")
            print(f"   Model loaded: {result['model_loaded']}")
            return result
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return {}
    
    def test_root(self) -> Dict[str, Any]:
        """Test root endpoint"""
        print("\nTesting root endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/")
            response.raise_for_status()
            result = response.json()
            print(f"✅ Root endpoint passed: {result['message']}")
            return result
        except Exception as e:
            print(f"❌ Root endpoint failed: {e}")
            return {}
    
    def test_model_info(self) -> Dict[str, Any]:
        """Test model info endpoint"""
        print("\nTesting model info endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            response.raise_for_status()
            result = response.json()
            print(f"✅ Model info: {result['model_type']}")
            print(f"   Vocabulary size: {result['vocabulary_size']}")
            print(f"   Device: {result['device']}")
            return result
        except Exception as e:
            print(f"❌ Model info failed: {e}")
            return {}
    
    def test_examples(self) -> Dict[str, Any]:
        """Test examples endpoint"""
        print("\nTesting examples endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/examples")
            response.raise_for_status()
            result = response.json()
            print(f"✅ Examples endpoint passed: {len(result['examples'])} examples")
            return result
        except Exception as e:
            print(f"❌ Examples endpoint failed: {e}")
            return {}
    
    def test_single_analysis(self, text: str) -> Dict[str, Any]:
        """Test single text analysis"""
        print(f"\nTesting single analysis: '{text[:50]}...'")
        try:
            response = self.session.post(
                f"{self.base_url}/analyze",
                json={"text": text}
            )
            response.raise_for_status()
            result = response.json()
            print(f"✅ Analysis result: {result['sentiment']} (confidence: {result['confidence']:.3f})")
            print(f"   Processing time: {result['processing_time']:.3f}s")
            return result
        except Exception as e:
            print(f"❌ Single analysis failed: {e}")
            return {}
    
    def test_batch_analysis(self, texts: list) -> Dict[str, Any]:
        """Test batch analysis"""
        print(f"\nTesting batch analysis with {len(texts)} texts...")
        try:
            response = self.session.post(
                f"{self.base_url}/analyze/batch",
                json={"texts": texts}
            )
            response.raise_for_status()
            result = response.json()
            print(f"✅ Batch analysis completed: {len(result['results'])} results")
            print(f"   Total processing time: {result['total_processing_time']:.3f}s")
            
            # Print individual results
            for i, res in enumerate(result['results']):
                print(f"   {i+1}. '{res['text'][:30]}...' -> {res['sentiment']} ({res['confidence']:.3f})")
            
            return result
        except Exception as e:
            print(f"❌ Batch analysis failed: {e}")
            return {}
    
    def run_all_tests(self):
        """Run all tests"""
        print("=" * 60)
        print("SENTIMENT ANALYSIS API TESTING")
        print("=" * 60)
        
        # Test basic endpoints
        self.test_health()
        self.test_root()
        self.test_model_info()
        self.test_examples()
        
        # Test analysis endpoints
        test_texts = [
            "This movie is absolutely fantastic! I loved every minute of it.",
            "This movie is terrible, waste of time and money.",
            "Amazing performance by all actors, highly recommended!",
            "Boring plot, bad acting, disappointing overall.",
            "Great plot, excellent cinematography, wonderful soundtrack."
        ]
        
        # Test single analysis
        for text in test_texts[:2]:
            self.test_single_analysis(text)
        
        # Test batch analysis
        self.test_batch_analysis(test_texts)
        
        print("\n" + "=" * 60)
        print("TESTING COMPLETED")
        print("=" * 60)

def main():
    """Main test function"""
    tester = SentimentAPITester()
    
    # Wait a moment for API to be ready
    print("Waiting for API to be ready...")
    time.sleep(2)
    
    # Run all tests
    tester.run_all_tests()

if __name__ == "__main__":
    main() 