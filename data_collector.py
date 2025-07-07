"""
Data Collection Module for Sentiment Analysis Project
Collects data from various sources including web scraping and API calls
"""

import requests
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging
from loguru import logger

class DataCollector:
    """Main data collection class"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def create_sample_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        Create synthetic sentiment data for demonstration
        """
        logger.info(f"Creating {n_samples} synthetic sentiment samples...")
        
        # Sample positive and negative reviews
        positive_reviews = [
            "This movie is absolutely fantastic! I loved every minute of it.",
            "Amazing performance by all actors, highly recommended!",
            "Great plot, excellent cinematography, wonderful soundtrack.",
            "One of the best films I've ever seen, truly outstanding!",
            "Incredible storytelling, couldn't take my eyes off the screen.",
            "Perfect casting, brilliant direction, masterpiece!",
            "Absolutely loved it, will watch again and again!",
            "Outstanding movie, exceeded all my expectations!",
            "Fantastic film, highly entertaining and engaging!",
            "Brilliant masterpiece, worth every minute!",
            "Excellent movie with great acting and direction.",
            "Wonderful film that kept me engaged throughout.",
            "Superb cinematography and compelling storyline.",
            "Outstanding performances from the entire cast.",
            "A must-watch film that delivers on all fronts."
        ]
        
        negative_reviews = [
            "This movie is terrible, waste of time and money.",
            "Boring plot, bad acting, disappointing overall.",
            "Poor direction, confusing story, not worth watching.",
            "Awful movie, regretted watching it completely.",
            "Terrible acting, bad script, waste of time.",
            "Disappointing film, expected much better quality.",
            "Boring and predictable, not entertaining at all.",
            "Poor cinematography, bad soundtrack, overall failure.",
            "Terrible movie, one of the worst I've ever seen.",
            "Awful film, complete waste of time and money.",
            "Horrible acting and terrible plot development.",
            "Waste of time, completely unentertaining.",
            "Poorly written script with bad direction.",
            "Disappointing and boring throughout.",
            "One of the worst films I've ever watched."
        ]
        
        # Generate data
        np.random.seed(42)
        reviews = []
        labels = []
        sources = []
        
        for i in range(n_samples):
            if np.random.random() > 0.5:
                review = np.random.choice(positive_reviews)
                label = 1  # Positive
            else:
                review = np.random.choice(negative_reviews)
                label = 0  # Negative
            
            # Add some variation
            review = self._add_variation(review)
            reviews.append(review)
            labels.append(label)
            sources.append("synthetic")
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': reviews,
            'label': labels,
            'source': sources,
            'id': range(len(reviews))
        })
        
        logger.info(f"Created dataset with {len(df)} samples")
        logger.info(f"Positive samples: {sum(df['label'] == 1)}")
        logger.info(f"Negative samples: {sum(df['label'] == 0)}")
        
        return df
    
    def _add_variation(self, text: str) -> str:
        """Add slight variations to text to make it more realistic"""
        variations = [
            text,
            text.replace("!", "."),
            text.replace("!", "..."),
            text.lower(),
            text.capitalize()
        ]
        return np.random.choice(variations)
    
    def scrape_imdb_reviews(self, movie_ids: List[str], max_reviews: int = 1000) -> pd.DataFrame:
        """
        Scrape movie reviews from IMDB
        Note: This is a simplified version. In production, you'd need to handle rate limiting
        """
        logger.info(f"Scraping IMDB reviews for {len(movie_ids)} movies...")
        
        reviews = []
        labels = []
        
        # Setup Chrome options for headless browsing
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            
            for movie_id in movie_ids:
                try:
                    url = f"https://www.imdb.com/title/{movie_id}/reviews"
                    driver.get(url)
                    
                    # Wait for reviews to load
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "review-container"))
                    )
                    
                    # Extract reviews
                    review_elements = driver.find_elements(By.CLASS_NAME, "review-container")
                    
                    for element in review_elements[:max_reviews // len(movie_ids)]:
                        try:
                            # Extract review text
                            text_element = element.find_element(By.CLASS_NAME, "content")
                            text = text_element.text.strip()
                            
                            if len(text) > 50:  # Only keep substantial reviews
                                # Extract rating (simplified)
                                try:
                                    rating_element = element.find_element(By.CLASS_NAME, "rating-other-user-rating")
                                    rating = float(rating_element.text.split('/')[0])
                                    label = 1 if rating >= 6 else 0
                                except:
                                    label = np.random.choice([0, 1])  # Fallback
                                
                                reviews.append(text)
                                labels.append(label)
                                
                        except Exception as e:
                            logger.warning(f"Error extracting review: {e}")
                            continue
                    
                    time.sleep(2)  # Be respectful to the server
                    
                except Exception as e:
                    logger.error(f"Error scraping movie {movie_id}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error setting up webdriver: {e}")
            return pd.DataFrame()
        
        finally:
            try:
                driver.quit()
            except:
                pass
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': reviews,
            'label': labels,
            'source': 'imdb',
            'id': range(len(reviews))
        })
        
        logger.info(f"Scraped {len(df)} IMDB reviews")
        return df
    
    def collect_from_api(self, api_url: str, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Collect data from REST API
        """
        logger.info(f"Collecting data from API: {api_url}")
        
        try:
            response = self.session.get(api_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame (adjust based on API response structure)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame([data])
            
            logger.info(f"Collected {len(df)} samples from API")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting from API: {e}")
            return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, filename: str, format: str = 'csv') -> None:
        """
        Save collected data to file
        """
        output_path = Path(f"data/raw/{filename}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format == 'parquet':
            df.to_parquet(output_path, index=False)
        
        logger.info(f"Saved data to {output_path}")
    
    def load_data(self, filename: str, format: str = 'csv') -> pd.DataFrame:
        """
        Load data from file
        """
        file_path = Path(f"data/raw/{filename}")
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()
        
        try:
            if format == 'csv':
                df = pd.read_csv(file_path)
            elif format == 'json':
                df = pd.read_json(file_path)
            elif format == 'parquet':
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Loaded {len(df)} samples from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def combine_datasets(self, datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine multiple datasets
        """
        if not datasets:
            return pd.DataFrame()
        
        combined_df = pd.concat(datasets, ignore_index=True)
        combined_df['id'] = range(len(combined_df))
        
        logger.info(f"Combined {len(combined_df)} samples from {len(datasets)} datasets")
        return combined_df
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate collected data
        """
        validation_results = {
            'total_samples': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'label_distribution': df['label'].value_counts().to_dict() if 'label' in df.columns else {},
            'text_length_stats': {
                'mean': df['text'].str.len().mean(),
                'min': df['text'].str.len().min(),
                'max': df['text'].str.len().max()
            } if 'text' in df.columns else {}
        }
        
        logger.info("Data validation results:")
        for key, value in validation_results.items():
            logger.info(f"  {key}: {value}")
        
        return validation_results 