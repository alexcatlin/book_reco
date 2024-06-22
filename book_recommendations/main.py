# recommend_books.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from api.google import googleBooks

ggl=googleBooks()

# Function to get book recommendations based on title
def get_recommendations(title):
        # Load the preprocessed books data
    books_df = pd.read_csv('books_data_preprocessed.csv')

    # Create a TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

    # Fit and transform the book descriptions
    tfidf_matrix = tfidf_vectorizer.fit_transform(books_df['description'])

    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    
    try:
        idx = books_df.index[books_df['title'] == title].tolist()[0]
    except IndexError:
        return "Book not found. Please try another title."
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    book_indices = [i[0] for i in sim_scores]
    return books_df.iloc[book_indices][['title', 'authors', 'average_rating']]

if __name__ == "__main__":
    user_input = input("Enter a book genre: ")
    ggl.download_and_save_books(user_input)
    
    # Get user input for the book title
    user_input = input("Enter a book title: ")
    
    # Get recommendations
    recommendations = get_recommendations(user_input)
    
    # Print the recommendations
    print("\nRecommended books:")
    if isinstance(recommendations, str):
        print(recommendations)
    else:
        for idx, row in recommendations.iterrows():
            print(f"\nTitle: {row['title']}\nAuthors: {row['authors']}\nAverage Rating: {row['average_rating']}")
