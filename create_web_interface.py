import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_web_interface(data_file: str = "books_with_emotions.csv") -> None:
    """
    Create web interface for the book recommender
    """
    print("="*60)
    print("CREATING WEB INTERFACE")
    print("="*60)
    
    try:
        # Check if data file exists
        if not os.path.exists(data_file):
            print(f"Error: {data_file} not found. Please run the complete pipeline first.")
            return
        
        # Create Flask app
        flask_app_content = '''from flask import Flask, render_template, request, jsonify
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search_books():
    try:
        data = request.get_json()
        query = data.get('query', '')
        limit = data.get('limit', 10)
        
        if not query or books_df is None or search_components is None:
            return jsonify([])
        
        vectorizer, tfidf_matrix = search_components
        
        # Transform query
        query_vector = vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Get top results
        top_indices = similarities.argsort()[-limit:][::-1]
        
        results = []
        for idx in top_indices:
            book = books_df.iloc[idx]
            results.append({
                'isbn13': book['isbn13'],
                'title': book.get('title_and_subtitle', book.get('title', 'Unknown')),
                'authors': book.get('authors', 'Unknown'),
                'description': book.get('description', 'No description'),
                'categories': book.get('categories', 'Unknown'),
                'simple_categories': book.get('simple_categories', 'Unknown'),
                'average_rating': book.get('average_rating', 0),
                'num_pages': book.get('num_pages', 0),
                'published_year': book.get('published_year', 0),
                'thumbnail': book.get('thumbnail', ''),
                'dominant_emotion': book.get('dominant_emotion', 'neutral'),
                'emotion_category': book.get('emotion_category', 'neutral'),
                'reading_difficulty': book.get('reading_difficulty', 'medium'),
                'similarity_score': similarities[idx]
            })
        
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
        
        categories = books_df['simple_categories'].value_counts().head(10).to_dict()
        emotions = books_df['dominant_emotion'].value_counts().to_dict()
        
        return jsonify({
            'categories': categories,
            'emotions': emotions
        })
        
    except Exception as e:
        print(f"Categories error: {e}")
        return jsonify({})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
        
        # Create templates directory
        if not os.path.exists('templates'):
            os.makedirs('templates')
        
        # Create HTML template
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Book Recommender</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        
        .header h1 {
            font-size: 3rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .search-container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        
        .search-box {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .search-input {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        .search-input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .search-btn {
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .search-btn:hover {
            transform: translateY(-2px);
        }
        
        .filters {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        .filter-select {
            padding: 10px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            background: white;
        }
        
        .results-container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .book-card {
            display: flex;
            gap: 20px;
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            margin-bottom: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .book-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        
        .book-cover {
            width: 120px;
            height: 180px;
            object-fit: cover;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        .book-info {
            flex: 1;
        }
        
        .book-title {
            font-size: 1.4rem;
            font-weight: bold;
            margin-bottom: 8px;
            color: #333;
        }
        
        .book-authors {
            color: #666;
            margin-bottom: 10px;
            font-size: 1.1rem;
        }
        
        .book-description {
            color: #555;
            line-height: 1.6;
            margin-bottom: 15px;
        }
        
        .book-meta {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        
        .meta-item {
            display: flex;
            align-items: center;
            gap: 5px;
            padding: 5px 10px;
            background: #f8f9fa;
            border-radius: 15px;
            font-size: 0.9rem;
        }
        
        .emotion-tag {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
            text-transform: uppercase;
        }
        
        .emotion-joy { background: #ffeb3b; color: #333; }
        .emotion-sadness { background: #2196f3; color: white; }
        .emotion-anger { background: #f44336; color: white; }
        .emotion-fear { background: #9c27b0; color: white; }
        .emotion-surprise { background: #ff9800; color: white; }
        .emotion-disgust { background: #4caf50; color: white; }
        .emotion-neutral { background: #9e9e9e; color: white; }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        
        .no-results {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        
        @media (max-width: 768px) {
            .book-card {
                flex-direction: column;
                text-align: center;
            }
            
            .book-cover {
                width: 100px;
                height: 150px;
                margin: 0 auto;
            }
            
            .search-box {
                flex-direction: column;
            }
            
            .filters {
                justify-content: center;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Enhanced Book Recommender</h1>
            <p>Discover books with AI-powered emotion analysis and intelligent categorization</p>
        </div>
        
        <div class="search-container">
            <div class="search-box">
                <input type="text" id="searchInput" class="search-input" placeholder="Search for books by title, author, or description...">
                <button onclick="searchBooks()" class="search-btn">Search</button>
            </div>
            
            <div class="filters">
                <select id="categoryFilter" class="filter-select">
                    <option value="">All Categories</option>
                </select>
                <select id="emotionFilter" class="filter-select">
                    <option value="">All Emotions</option>
                </select>
                <select id="difficultyFilter" class="filter-select">
                    <option value="">All Difficulties</option>
                    <option value="easy">Easy</option>
                    <option value="medium">Medium</option>
                    <option value="hard">Hard</option>
                </select>
            </div>
        </div>
        
        <div class="results-container">
            <div id="results"></div>
        </div>
    </div>

    <script>
        let allBooks = [];
        
        // Load categories and emotions
        async function loadFilters() {
            try {
                const response = await fetch('/api/categories');
                const data = await response.json();
                
                const categorySelect = document.getElementById('categoryFilter');
                const emotionSelect = document.getElementById('emotionFilter');
                
                // Add categories
                for (const [category, count] of Object.entries(data.categories)) {
                    const option = document.createElement('option');
                    option.value = category;
                    option.textContent = `${category} (${count})`;
                    categorySelect.appendChild(option);
                }
                
                // Add emotions
                for (const [emotion, count] of Object.entries(data.emotions)) {
                    const option = document.createElement('option');
                    option.value = emotion;
                    option.textContent = `${emotion} (${count})`;
                    emotionSelect.appendChild(option);
                }
            } catch (error) {
                console.error('Error loading filters:', error);
            }
        }
        
        // Search books
        async function searchBooks() {
            const query = document.getElementById('searchInput').value.trim();
            const category = document.getElementById('categoryFilter').value;
            const emotion = document.getElementById('emotionFilter').value;
            const difficulty = document.getElementById('difficultyFilter').value;
            
            if (!query) {
                alert('Please enter a search query');
                return;
            }
            
            document.getElementById('results').innerHTML = '<div class="loading">Searching for books...</div>';
            
            try {
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: query,
                        limit: 20
                    })
                });
                
                const books = await response.json();
                displayBooks(books, category, emotion, difficulty);
            } catch (error) {
                console.error('Search error:', error);
                document.getElementById('results').innerHTML = '<div class="no-results">Error searching for books</div>';
            }
        }
        
        // Display books
        function displayBooks(books, categoryFilter, emotionFilter, difficultyFilter) {
            const resultsDiv = document.getElementById('results');
            
            if (books.length === 0) {
                resultsDiv.innerHTML = '<div class="no-results">No books found matching your criteria</div>';
                return;
            }
            
            // Filter books
            let filteredBooks = books;
            if (categoryFilter) {
                filteredBooks = filteredBooks.filter(book => book.simple_categories === categoryFilter);
            }
            if (emotionFilter) {
                filteredBooks = filteredBooks.filter(book => book.dominant_emotion === emotionFilter);
            }
            if (difficultyFilter) {
                filteredBooks = filteredBooks.filter(book => book.reading_difficulty === difficultyFilter);
            }
            
            if (filteredBooks.length === 0) {
                resultsDiv.innerHTML = '<div class="no-results">No books found matching your filters</div>';
                return;
            }
            
            let html = '';
            filteredBooks.forEach(book => {
                html += `
                    <div class="book-card">
                        <img src="${book.thumbnail || 'https://via.placeholder.com/120x180?text=No+Cover'}" 
                             alt="${book.title}" class="book-cover">
                        <div class="book-info">
                            <div class="book-title">${book.title}</div>
                            <div class="book-authors">by ${book.authors}</div>
                            <div class="book-description">${book.description.substring(0, 200)}...</div>
                            <div class="book-meta">
                                <div class="meta-item">
                                    <span>Rating</span>
                                    <span>${book.average_rating.toFixed(1)}</span>
                                </div>
                                <div class="meta-item">
                                    <span>Pages</span>
                                    <span>${book.num_pages}</span>
                                </div>
                                <div class="meta-item">
                                    <span>Year</span>
                                    <span>${book.published_year}</span>
                                </div>
                                <div class="meta-item">
                                    <span>Category</span>
                                    <span>${book.simple_categories}</span>
                                </div>
                                <div class="meta-item">
                                    <span>Difficulty</span>
                                    <span>${book.reading_difficulty}</span>
                                </div>
                                <div class="emotion-tag emotion-${book.dominant_emotion}">
                                    ${book.dominant_emotion}
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            resultsDiv.innerHTML = html;
        }
        
        // Enter key search
        document.getElementById('searchInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchBooks();
            }
        });
        
        // Load filters on page load
        loadFilters();
    </script>
</body>
</html>'''
        
        # Write files
        with open('app.py', 'w', encoding='utf-8') as f:
            f.write(flask_app_content)
        
        with open('templates/index.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("Web interface created successfully!")
        print("   - app.py (Flask backend)")
        print("   - templates/index.html (Frontend)")
        print("   - Run with: python app.py")
        
    except Exception as e:
        print(f"Error creating web interface: {e}")

if __name__ == "__main__":
    create_web_interface()
