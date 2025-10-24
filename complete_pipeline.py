import pandas as pd
import numpy as np
import os
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import transformers for classification and emotion analysis
from transformers import pipeline
from tqdm import tqdm

# Basic imports for text processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Import kagglehub for data download
import kagglehub

class CompleteBookProcessingPipeline:
    """
    Complete book processing pipeline that merges all steps:
    0. Download books.csv from Kaggle
    1. Raw books.csv -> books_cleaned.csv
    2. books_cleaned.csv -> books_with_categories.csv  
    3. books_with_categories.csv -> books_with_emotions.csv
    """
    
    def __init__(self, device: str = "auto", cache_results: bool = True):
        """
        Initialize the complete processing pipeline
        """
        self.device = device
        self.cache_results = cache_results
        self.cache_dir = "processing_cache" if cache_results else None
        
        # Initialize models
        print("Initializing AI models...")
        try:
            # Text classification model
            self.text_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=device if device != "auto" else -1
            )
            print("Text classifier initialized")
            
            # Emotion analysis model
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,
                device=device if device != "auto" else -1
            )
            print("Emotion classifier initialized")
            
        except Exception as e:
            print(f"Error initializing models: {e}")
            self.text_classifier = None
            self.emotion_classifier = None
        
        # Emotion labels
        self.emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
        
        # Create cache directory
        if self.cache_results and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        print("Complete Book Processing Pipeline Initialized")
        print(f"Device: {self.device}")
        print(f"Caching: {'Enabled' if cache_results else 'Disabled'}")
    
    def step0_download_data(self, output_file: str = "books.csv") -> bool:
        """
        Step 0: Download book data from Kaggle
        """
        print("\n" + "="*60)
        print("STEP 0: DOWNLOAD DATA FROM KAGGLE")
        print("="*60)
        
        try:
            # Check if file already exists
            if os.path.exists(output_file):
                print(f"Data file {output_file} already exists, skipping download")
                return True
            
            print("Downloading book dataset from Kaggle...")
            print("Dataset: dylanjcastillo/7k-books-with-metadata")
            
            # Download the dataset
            path = kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")
            print(f"Dataset downloaded to: {path}")
            
            # Copy books.csv to current directory
            source_file = os.path.join(path, "books.csv")
            if os.path.exists(source_file):
                import shutil
                shutil.copy2(source_file, output_file)
                print(f"Data file copied to {output_file}")
                
                # Verify the file
                df = pd.read_csv(output_file)
                print(f"Downloaded {len(df)} books successfully")
                return True
            else:
                print(f"Error: books.csv not found in {path}")
                return False
                
        except Exception as e:
            print(f"Error downloading data: {e}")
            return False
    
    def step1_clean_data(self, input_file: str = "books.csv", output_file: str = "books_cleaned.csv") -> pd.DataFrame:
        """
        Step 1: Clean the raw book data (from data-exploration.ipynb approach)
        """
        print("\n" + "="*60)
        print("STEP 1: DATA CLEANING")
        print("="*60)
        
        try:
            # Load raw data
            print(f"Loading raw data from {input_file}...")
            books = pd.read_csv(input_file)
            print(f"Loaded {len(books)} books")
            
            # Create missing description flag and age of book
            books["missing_description"] = np.where(books["description"].isna(), 1, 0)
            books["age_of_book"] = 2024 - books["published_year"]
            
            # Filter books with complete essential data
            print("Filtering books with complete data...")
            initial_count = len(books)
            book_missing = books[~(books["description"].isna()) &
                                ~(books["num_pages"].isna()) &
                                ~(books["average_rating"].isna()) &
                                ~(books["published_year"].isna())]
            filtered_count = len(book_missing)
            print(f"Filtered from {initial_count} to {filtered_count} books")
            
            # Analyze word count in descriptions
            book_missing["words_in_description"] = book_missing["description"].str.split().str.len()
            
            # Filter for books with substantial descriptions (25+ words)
            book_missing_25_words = book_missing[book_missing["words_in_description"] >= 25]
            final_count = len(book_missing_25_words)
            print(f"Filtered to {final_count} books with substantial descriptions")
            
            # Create title and subtitle combination
            book_missing_25_words["title_and_subtitle"] = (
                np.where(book_missing_25_words["subtitle"].isna(), 
                         book_missing_25_words["title"],
                         book_missing_25_words[["title", "subtitle"]].astype(str).agg(": ".join, axis=1))
            )
            
            # Create tagged description for vector search
            book_missing_25_words["tagged_description"] = book_missing_25_words[["isbn13", "description"]].astype(str).agg(" ".join, axis=1)
            
            # Drop unnecessary columns
            final_books = book_missing_25_words.drop(["subtitle", "missing_description", "age_of_book", "words_in_description"], axis=1)
            
            # Save cleaned data
            final_books.to_csv(output_file, index=False)
            print(f"Cleaned data saved to {output_file}")
            print(f"Final dataset: {len(final_books)} books with {len(final_books.columns)} columns")
            
            return final_books
            
        except Exception as e:
            print(f"Error in data cleaning: {e}")
            return None
    
    def step2_add_categories(self, input_file: str = "books_cleaned.csv", output_file: str = "books_with_categories.csv") -> pd.DataFrame:
        """
        Step 2: Add enhanced categories (from text-classification.ipynb approach)
        """
        print("\n" + "="*60)
        print("STEP 2: CATEGORY ENHANCEMENT")
        print("="*60)
        
        try:
            # Load cleaned data
            print(f"Loading cleaned data from {input_file}...")
            books = pd.read_csv(input_file)
            print(f"Loaded {len(books)} books")
            
            # Category mapping from text-classification.ipynb
            category_mapping = {
                'Fiction': "Fiction",
                'Juvenile Fiction': "Children's Fiction",
                'Biography & Autobiography': "Nonfiction",
                'History': "Nonfiction",
                'Literary Criticism': "Nonfiction",
                'Philosophy': "Nonfiction",
                'Religion': "Nonfiction",
                'Comics & Graphic Novels': "Fiction",
                'Drama': "Fiction",
                'Juvenile Nonfiction': "Children's Nonfiction",
                'Science': "Nonfiction",
                'Poetry': "Fiction"
            }
            
            # Apply mapping
            books["simple_categories"] = books["categories"].map(category_mapping)
            mapped_count = books["simple_categories"].notna().sum()
            missing_count = books["simple_categories"].isna().sum()
            print(f"Mapped {mapped_count} books, {missing_count} need classification")
            
            # Classify missing categories using BART
            if self.text_classifier and missing_count > 0:
                print("Classifying missing categories using BART-large-MNLI...")
                missing_cats = books.loc[books["simple_categories"].isna(), ["isbn13", "description"]].reset_index(drop=True)
                
                fiction_categories = ["Fiction", "Nonfiction"]
                isbns = []
                predicted_cats = []
                
                # Process in batches
                batch_size = 50
                for i in tqdm(range(0, len(missing_cats), batch_size), desc="Classifying categories"):
                    batch = missing_cats.iloc[i:i+batch_size]
                    for _, row in batch.iterrows():
                        sequence = row["description"]
                        if pd.isna(sequence) or len(str(sequence)) < 10:
                            predicted_cats.append("Nonfiction")
                        else:
                            try:
                                result = self.text_classifier(str(sequence), fiction_categories)
                                max_index = np.argmax(result["scores"])
                                max_label = result["labels"][max_index]
                                predicted_cats.append(max_label)
                            except:
                                predicted_cats.append("Nonfiction")
                        isbns.append(row["isbn13"])
                
                # Merge predictions
                missing_predicted_df = pd.DataFrame({"isbn13": isbns, "predicted_categories": predicted_cats})
                books = pd.merge(books, missing_predicted_df, on="isbn13", how="left")
                books["simple_categories"] = np.where(books["simple_categories"].isna(), books["predicted_categories"], books["simple_categories"])
                books = books.drop(columns=["predicted_categories"])
            
            # Add enhanced features
            books['reading_difficulty'] = books.apply(
                lambda row: 'easy' if len(str(row.get('description', ''))) < 100 or row.get('num_pages', 0) < 200
                           else 'hard' if len(str(row.get('description', ''))) > 300 or row.get('num_pages', 0) > 500
                           else 'medium', axis=1
            )
            
            books['estimated_reading_time_hours'] = books['num_pages'].apply(
                lambda x: round((float(x) * 250) / (200 * 60), 1) if not pd.isna(x) else 0
            )
            
            books['book_age_years'] = 2024 - books['published_year']
            books['is_popular'] = books['average_rating'] >= 4.0
            books['is_recent'] = books['book_age_years'] <= 10
            books['is_classic'] = books['book_age_years'] >= 50
            
            # Save enhanced data
            books.to_csv(output_file, index=False)
            print(f"Enhanced data saved to {output_file}")
            print(f"Category distribution: {books['simple_categories'].value_counts().to_dict()}")
            
            return books
            
        except Exception as e:
            print(f"Error in category enhancement: {e}")
            return None
    
    def step3_add_emotions(self, input_file: str = "books_with_categories.csv", output_file: str = "books_with_emotions.csv") -> pd.DataFrame:
        """
        Step 3: Add emotion analysis (from sentiment-analysis.ipynb approach)
        """
        print("\n" + "="*60)
        print("STEP 3: EMOTION ANALYSIS")
        print("="*60)
        
        try:
            # Load categorized data
            print(f"Loading categorized data from {input_file}...")
            books = pd.read_csv(input_file)
            print(f"Loaded {len(books)} books")
            
            if self.emotion_classifier is None:
                print("Using keyword-based emotion analysis fallback")
                return self._add_emotions_fallback(books, output_file)
            
            # Initialize emotion columns
            for label in self.emotion_labels:
                books[f"{label}_max"] = 0.0
                books[f"{label}_mean"] = 0.0
                books[f"{label}_std"] = 0.0
                books[f"{label}_min"] = 0.0
            
            # Process books in batches
            batch_size = 100
            total_batches = (len(books) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(books))
                batch_df = books.iloc[start_idx:end_idx]
                
                print(f"Processing emotion batch {batch_idx + 1}/{total_batches}")
                
                for i in tqdm(range(len(batch_df)), desc=f"Batch {batch_idx + 1}"):
                    idx = batch_df.index[i]
                    description = batch_df["description"].iloc[i]
                    
                    if pd.isna(description) or len(str(description)) < 10:
                        continue
                    
                    try:
                        # Split description into sentences
                        sentences = str(description).split(".")
                        sentences = [s.strip() for s in sentences if s.strip()]
                        
                        if not sentences:
                            continue
                        
                        # Get emotion predictions
                        predictions = self.emotion_classifier(sentences)
                        
                        # Calculate emotion statistics
                        per_emotion_scores = {label: [] for label in self.emotion_labels}
                        
                        for prediction in predictions:
                            sorted_predictions = sorted(prediction, key=lambda x: x["label"])
                            for index, label in enumerate(self.emotion_labels):
                                per_emotion_scores[label].append(sorted_predictions[index]["score"])
                        
                        # Update DataFrame
                        for label in self.emotion_labels:
                            scores = per_emotion_scores[label]
                            books.at[idx, f"{label}_max"] = np.max(scores)
                            books.at[idx, f"{label}_mean"] = np.mean(scores)
                            books.at[idx, f"{label}_std"] = np.std(scores)
                            books.at[idx, f"{label}_min"] = np.min(scores)
                    
                    except Exception as e:
                        print(f"Error analyzing emotions for book {idx}: {e}")
                        continue
            
            # Add derived emotion features
            books['dominant_emotion'] = books.apply(
                lambda row: max(self.emotion_labels, key=lambda x: row[f"{x}_max"]), axis=1
            )
            
            books['emotion_intensity'] = books.apply(
                lambda row: sum(row[f"{label}_max"] for label in self.emotion_labels), axis=1
            )
            
            books['emotion_diversity'] = books.apply(
                lambda row: np.std([row[f"{label}_max"] for label in self.emotion_labels]), axis=1
            )
            
            positive_emotions = ['joy', 'surprise']
            negative_emotions = ['anger', 'disgust', 'fear', 'sadness']
            
            books['positive_emotion_score'] = books.apply(
                lambda row: sum(row[f"{emotion}_max"] for emotion in positive_emotions), axis=1
            )
            
            books['negative_emotion_score'] = books.apply(
                lambda row: sum(row[f"{emotion}_max"] for emotion in negative_emotions), axis=1
            )
            
            books['emotion_polarity'] = books.apply(
                lambda row: (row['positive_emotion_score'] - row['negative_emotion_score']) / 
                          (row['positive_emotion_score'] + row['negative_emotion_score']) 
                          if (row['positive_emotion_score'] + row['negative_emotion_score']) > 0 else 0, axis=1
            )
            
            books['emotion_category'] = books['emotion_polarity'].apply(
                lambda x: 'positive' if x > 0.2 else 'negative' if x < -0.2 else 'neutral'
            )
            
            # Save final dataset
            books.to_csv(output_file, index=False)
            print(f"Emotion analysis completed and saved to {output_file}")
            print(f"Emotion distribution: {books['dominant_emotion'].value_counts().to_dict()}")
            print(f"Emotion categories: {books['emotion_category'].value_counts().to_dict()}")
            
            return books
            
        except Exception as e:
            print(f"Error in emotion analysis: {e}")
            return None
    
    def _add_emotions_fallback(self, books: pd.DataFrame, output_file: str) -> pd.DataFrame:
        """Fallback emotion analysis using keywords"""
        print("Using keyword-based emotion analysis...")
        
        emotion_keywords = {
            "joy": ["happy", "joy", "cheerful", "delightful", "joyful"],
            "sadness": ["sad", "sorrowful", "melancholy", "grief", "mourning"],
            "anger": ["angry", "furious", "rage", "outrage", "wrath"],
            "fear": ["afraid", "scared", "terrified", "fearful", "anxious"],
            "surprise": ["surprising", "unexpected", "shocking", "amazing", "astonishing"],
            "disgust": ["disgusting", "revolting", "repulsive", "sickening"],
            "neutral": ["neutral", "calm", "balanced", "moderate"]
        }
        
        for label in self.emotion_labels:
            books[f"{label}_max"] = 0.0
            books[f"{label}_mean"] = 0.0
            books[f"{label}_std"] = 0.0
            books[f"{label}_min"] = 0.0
        
        def analyze_emotions_keywords(description):
            if pd.isna(description):
                return {f"{label}_max": 0.0 for label in self.emotion_labels}
            
            text_lower = str(description).lower()
            emotion_stats = {}
            
            for emotion, keywords in emotion_keywords.items():
                matches = sum(1 for keyword in keywords if keyword in text_lower)
                score = min(matches / len(keywords), 1.0)
                emotion_stats[f"{emotion}_max"] = score
                emotion_stats[f"{emotion}_mean"] = score
                emotion_stats[f"{emotion}_std"] = 0.0
                emotion_stats[f"{emotion}_min"] = score
            
            return emotion_stats
        
        emotion_results = books['description'].apply(analyze_emotions_keywords)
        
        for idx, emotion_stats in enumerate(emotion_results):
            for stat_name, stat_value in emotion_stats.items():
                books.at[idx, stat_name] = stat_value
        
        # Add derived features
        books['dominant_emotion'] = books.apply(
            lambda row: max(self.emotion_labels, key=lambda x: row[f"{x}_max"]), axis=1
        )
        
        books['emotion_intensity'] = books.apply(
            lambda row: sum(row[f"{label}_max"] for label in self.emotion_labels), axis=1
        )
        
        books['emotion_category'] = 'neutral'
        
        books.to_csv(output_file, index=False)
        print(f"Fallback emotion analysis completed and saved to {output_file}")
        
        return books
    
    
    def cleanup_unused_files(self) -> None:
        """
        Clean up unused files, keeping only essential ones
        """
        print("\n" + "="*60)
        print("CLEANUP: REMOVING UNUSED FILES")
        print("="*60)
        
        # Files to keep
        essential_files = {
            'books.csv',  # Original data
            'books_cleaned.csv',  # Step 1 output
            'books_with_categories.csv',  # Step 2 output
            'books_with_emotions.csv',  # Step 3 output
            'requirements.txt',  # Dependencies
            'README.md',  # Documentation
            'complete_pipeline.py',  # This script
            'create_web_interface.py'  # Web interface creator
        }
        
        # Files to remove
        files_to_remove = [
            'book_data_downloader.py',
            'data_cleaning.py',
            'comprehensive_book_categorizer.py',
            'improved_book_categorizer.py',
            'improved_sentiment_analyzer.py',
            'enhanced_data_preprocessing.py',
            'enhanced_sentiment_analysis.py',
            'enhanced_text_classification.py',
            'comprehensive_analysis_pipeline.py',
            'simple_vector_search.py',
            'enhanced_vector_search.py',
            'enhanced_book_recommender.py',
            'gradio-dashboard.py',
            'requirements_flask.txt',
            'requirements_enhanced.txt',
            'setup.py',
            'start_app.bat',
            'tagged_description.txt',
            'ENHANCEMENT_SUMMARY.md',
            'BOOKS_WITH_CATEGORIES_SUMMARY.md',
            'README_ENHANCED.md',
            'README_ENHANCED_COMPONENTS.md',
            'cover-not-found.jpg'
        ]
        
        removed_count = 0
        for file_path in files_to_remove:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"   Removed: {file_path}")
                    removed_count += 1
                except Exception as e:
                    print(f"   Error removing {file_path}: {e}")
        
        # Remove directories
        dirs_to_remove = ['vector_cache', 'categorization_cache', 'sentiment_cache', 'processing_cache']
        for dir_path in dirs_to_remove:
            if os.path.exists(dir_path):
                try:
                    import shutil
                    shutil.rmtree(dir_path)
                    print(f"   Removed directory: {dir_path}")
                except Exception as e:
                    print(f"   Error removing directory {dir_path}: {e}")
        
        print(f"Cleanup completed! Removed {removed_count} files")
        print("Essential files kept:")
        for file in essential_files:
            if os.path.exists(file):
                print(f"   - {file}")
    
    def run_complete_pipeline(self) -> None:
        """
        Run the complete processing pipeline
        """
        print("STARTING COMPLETE BOOK PROCESSING PIPELINE")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # Step 0: Download data from Kaggle
            if not self.step0_download_data():
                print("Pipeline failed at Step 0")
                return
            
            # Step 1: Clean data
            cleaned_data = self.step1_clean_data()
            if cleaned_data is None:
                print("Pipeline failed at Step 1")
                return
            
            # Step 2: Add categories
            categorized_data = self.step2_add_categories()
            if categorized_data is None:
                print("Pipeline failed at Step 2")
                return
            
            # Step 3: Add emotions
            emotion_data = self.step3_add_emotions()
            if emotion_data is None:
                print("Pipeline failed at Step 3")
                return
            
            # Cleanup
            self.cleanup_unused_files()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print("\n" + "="*80)
            print("COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
            print("="*80)
            print(f"Total processing time: {total_time:.2f} seconds")
            print(f"Final dataset: {len(emotion_data)} books with comprehensive features")
            print("\nEssential files:")
            print("   - books.csv (Original data)")
            print("   - books_cleaned.csv (Cleaned data)")
            print("   - books_with_categories.csv (Categorized data)")
            print("   - books_with_emotions.csv (Final dataset)")
            print("\nTo create web interface: python create_web_interface.py")
            
        except Exception as e:
            print(f"Pipeline failed with error: {e}")

# Run the complete pipeline
if __name__ == "__main__":
    pipeline = CompleteBookProcessingPipeline(device="auto", cache_results=True)
    pipeline.run_complete_pipeline()
