from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from models.content_based import get_content_based_recommendations
from models.collaborative import get_collaborative_recommendations, train_collaborative_model
from models.rag_model import rag_recommendation, setup_rag_model
from models.vector_db import create_faiss_index, query_faiss_index
from models.reviews import get_average_sentiment

app = Flask(__name__)

# Load data
books_df = pd.read_csv('data/books_data.csv')
books_ratings_df = pd.read_csv('data/books_rating.csv')
books_df['book_id'] = books_df.index

# Preprocess ratings data
ratings_df = books_ratings_df[['user_id', 'book_id', 'rating']]

# Add average sentiment to books_df
books_df['average_sentiment'] = books_df['book_id'].apply(lambda x: get_average_sentiment(x, books_ratings_df))

# Train collaborative filtering model
algo = train_collaborative_model(ratings_df)

# Setup RAG model
rag_tokenizer, rag_model = setup_rag_model()

# Create FAISS index for book descriptions
faiss_index = create_faiss_index(books_df)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/content', methods=['POST'])
def content_recommendations():
    title = request.form['title']
    recommendations = get_content_based_recommendations(title, books_df)
    return render_template('content_recommendations.html', title=title, recommendations=recommendations)

@app.route('/collaborative', methods=['POST'])
def collaborative_recommendations():
    user_id = request.form['user_id']
    recommendations = get_collaborative_recommendations(user_id, books_df, algo)
    return render_template('collaborative_recommendations.html', user_id=user_id, recommendations=recommendations)

@app.route('/rag', methods=['POST'])
def rag_recommendations():
    query = request.form['query']
    recommendation = rag_recommendation(query, rag_tokenizer, rag_model)
    return render_template('rag_recommendations.html', query=query, recommendation=recommendation)

@app.route('/vector', methods=['POST'])
def vector_recommendations():
    description = request.form['description']
    recommendations = query_faiss_index(description, faiss_index, books_df)
    return render_template('vector_recommendations.html', description=description, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
