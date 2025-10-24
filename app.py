from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Load the enhanced book data
def load_books_data():
    try:
        df = pd.read_csv('books_with_emotions.csv')
        print(f"Loaded {len(df)} books with emotions")
        return df
    except Exception as e:
        print(f"Error loading books: {e}")
        return None

# Initialize TF-IDF vectorizer for search
def initialize_search():
    df = load_books_data()
    if df is None:
        return None, None
    
    # Prepare text data for search
    text_data = []
    for _, row in df.iterrows():
        combined_text = f"{row.get('title', '')} {row.get('authors', '')} {row.get('description', '')} {row.get('categories', '')}"
        text_data.append(combined_text)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words='english',
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.8,
        lowercase=True
    )
    
    tfidf_matrix = vectorizer.fit_transform(text_data)
    
    return df, (vectorizer, tfidf_matrix)

# Initialize data
books_df, search_components = initialize_search()

# Helper function to clean values for JSON serialization
def clean_value(value):
    if pd.isna(value) or value is None:
        if isinstance(value, (int, float)):
            return 0
        else:
            return 'Unknown'
    return value

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search_books():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        limit = data.get('limit', 20)
        category_filter = data.get('category', 'All Categories')
        emotion_filter = data.get('emotion', 'All Emotions')
        difficulty_filter = data.get('difficulty', 'All')
        
        if books_df is None:
            return jsonify([])
        
        # Start with all books
        filtered_df = books_df.copy()
        
        # Apply filters first
        if category_filter != 'All Categories':
            filtered_df = filtered_df[filtered_df['categories'] == category_filter]
        
        if emotion_filter != 'All Emotions':
            filtered_df = filtered_df[filtered_df['dominant_emotion'] == emotion_filter]
        
        if difficulty_filter != 'All':
            filtered_df = filtered_df[filtered_df['reading_difficulty'] == difficulty_filter.lower()]
        
        # If no query, return filtered results
        if not query:
            results = []
            for _, book in filtered_df.head(limit).iterrows():
                results.append({
                    'isbn13': str(book['isbn13']),
                    'title': clean_value(book.get('title', 'Unknown')),
                    'authors': clean_value(book.get('authors', 'Unknown')),
                    'description': clean_value(book.get('description', 'No description')),
                    'categories': clean_value(book.get('categories', 'Unknown')),
                    'average_rating': float(clean_value(book.get('average_rating', 0))),
                    'num_pages': int(clean_value(book.get('num_pages', 0))),
                    'published_year': int(clean_value(book.get('published_year', 0))),
                    'thumbnail': clean_value(book.get('thumbnail', '')),
                    'dominant_emotion': clean_value(book.get('dominant_emotion', 'neutral')),
                    'reading_difficulty': clean_value(book.get('reading_difficulty', 'unknown')),
                    'similarity_score': 1.0  # No search, so perfect match
                })
            return jsonify(results)
        
        # Try TF-IDF search first
        if search_components is not None:
            try:
                vectorizer, tfidf_matrix = search_components
                
                # Transform query
                query_vector = vectorizer.transform([query])
                
                # Calculate similarities
                similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
                
                # Get top results with similarity threshold
                top_indices = similarities.argsort()[-limit:][::-1]
                
                results = []
                for idx in top_indices:
                    if similarities[idx] > 0.01:  # Only include results with some similarity
                        book = books_df.iloc[idx]
                        
                        # Check if book matches filters
                        matches_filters = True
                        if category_filter != 'All Categories' and book.get('categories') != category_filter:
                            matches_filters = False
                        if emotion_filter != 'All Emotions' and book.get('dominant_emotion') != emotion_filter:
                            matches_filters = False
                        if difficulty_filter != 'All' and book.get('reading_difficulty') != difficulty_filter.lower():
                            matches_filters = False
                        
                        if matches_filters:
                            results.append({
                                'isbn13': str(book['isbn13']),
                                'title': clean_value(book.get('title', 'Unknown')),
                                'authors': clean_value(book.get('authors', 'Unknown')),
                                'description': clean_value(book.get('description', 'No description')),
                                'categories': clean_value(book.get('categories', 'Unknown')),
                                'average_rating': float(clean_value(book.get('average_rating', 0))),
                                'num_pages': int(clean_value(book.get('num_pages', 0))),
                                'published_year': int(clean_value(book.get('published_year', 0))),
                                'thumbnail': clean_value(book.get('thumbnail', '')),
                                'dominant_emotion': clean_value(book.get('dominant_emotion', 'neutral')),
                                'emotion_category': clean_value(book.get('emotion_category', 'neutral')),
                                'reading_difficulty': clean_value(book.get('reading_difficulty', 'medium')),
                                'similarity_score': float(similarities[idx])
                            })
                
                if results:
                    return jsonify(results)
            except Exception as e:
                print(f"TF-IDF search error: {e}")
        
        # Fallback to simple text search
        query_lower = query.lower()
        results = []
        
        for idx, row in filtered_df.iterrows():
            title = str(row.get('title', '')).lower()
            authors = str(row.get('authors', '')).lower()
            description = str(row.get('description', '')).lower()
            categories = str(row.get('categories', '')).lower()
            
            if (query_lower in title or 
                query_lower in authors or 
                query_lower in description or 
                query_lower in categories):
                
                # Clean NaN values
                
                results.append({
                    'isbn13': str(row['isbn13']),
                    'title': clean_value(row.get('title', 'Unknown')),
                    'authors': clean_value(row.get('authors', 'Unknown')),
                    'description': clean_value(row.get('description', 'No description')),
                    'categories': clean_value(row.get('categories', 'Unknown')),
                    'average_rating': float(clean_value(row.get('average_rating', 0))),
                    'num_pages': int(clean_value(row.get('num_pages', 0))),
                    'published_year': int(clean_value(row.get('published_year', 0))),
                    'thumbnail': clean_value(row.get('thumbnail', '')),
                    'dominant_emotion': clean_value(row.get('dominant_emotion', 'neutral')),
                    'emotion_category': clean_value(row.get('emotion_category', 'neutral')),
                    'reading_difficulty': clean_value(row.get('reading_difficulty', 'medium')),
                    'similarity_score': 1.0  # High score for exact matches
                })
                
                if len(results) >= limit:
                    break
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify([])

@app.route('/api/trending', methods=['GET'])
def get_trending():
    try:
        if books_df is None:
            return jsonify([])
        
        # Get popular recent books
        trending = books_df[
            (books_df['is_popular'] == True) & 
            (books_df['is_recent'] == True)
        ].head(10)
        
        results = []
        for _, book in trending.iterrows():
            results.append({
                'isbn13': book['isbn13'],
                'title': book.get('title_and_subtitle', book.get('title', 'Unknown')),
                'authors': book.get('authors', 'Unknown'),
                'categories': clean_value(book.get('categories', 'Unknown')),
                'average_rating': book.get('average_rating', 0),
                'thumbnail': book.get('thumbnail', ''),
                'dominant_emotion': book.get('dominant_emotion', 'neutral')
            })
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Trending error: {e}")
        return jsonify([])

@app.route('/api/categories', methods=['GET'])
def get_categories():
    try:
        if books_df is None:
            return jsonify([])
        
        categories = books_df['categories'].value_counts().head(15).to_dict()
        emotions = books_df['dominant_emotion'].value_counts().to_dict()
        
        return jsonify({
            'categories': categories,
            'emotions': emotions
        })
        
    except Exception as e:
        print(f"Categories error: {e}")
        return jsonify({})

@app.route('/api/stats', methods=['POST'])
def get_stats():
    try:
        if books_df is None:
            return jsonify({'total': 0, 'filtered': 0})
        
        data = request.get_json()
        query = data.get('query', '').strip()
        category_filter = data.get('category', 'All Categories')
        emotion_filter = data.get('emotion', 'All Emotions')
        difficulty_filter = data.get('difficulty', 'All')
        
        # Start with all books
        filtered_df = books_df.copy()
        
        # Apply category filter
        if category_filter != 'All Categories':
            filtered_df = filtered_df[filtered_df['categories'] == category_filter]
        
        # Apply emotion filter
        if emotion_filter != 'All Emotions':
            filtered_df = filtered_df[filtered_df['dominant_emotion'] == emotion_filter]
        
        # Apply difficulty filter
        if difficulty_filter != 'All':
            filtered_df = filtered_df[filtered_df['reading_difficulty'] == difficulty_filter.lower()]
        
        # Apply search query filter if provided
        if query:
            query_lower = query.lower()
            search_matches = []
            
            for idx, row in filtered_df.iterrows():
                title = str(row.get('title', '')).lower()
                authors = str(row.get('authors', '')).lower()
                description = str(row.get('description', '')).lower()
                categories = str(row.get('categories', '')).lower()
                
                if (query_lower in title or 
                    query_lower in authors or 
                    query_lower in description or 
                    query_lower in categories):
                    search_matches.append(idx)
            
            filtered_df = filtered_df.loc[search_matches]
        
        return jsonify({
            'total': len(books_df),
            'filtered': len(filtered_df),
            'total_books': len(books_df),
            'matching_criteria': len(filtered_df)
        })
        
    except Exception as e:
        print(f"Stats error: {e}")
        return jsonify({'total': 0, 'filtered': 0})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
