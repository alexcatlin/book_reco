#todo make google work and 
import requests
import pandas as pd
from config import GOOGLE_BOOKS_API_KEY

class googleBooks:
    def __init__(self):


        pass

    def fetch_books(self, query, max_results=40):
        url = f'https://www.googleapis.com/books/v1/volumes?q={query}&maxResults={max_results}&key={GOOGLE_BOOKS_API_KEY}'
        response = requests.get(url)
        return response.json()

    def preprocess_books(self, books_info):
        books_df = pd.DataFrame(books_info)
        
        # Preprocess the data
        books_df['description'] = books_df['description'].fillna('No description')
        books_df['authors'] = books_df['authors'].apply(lambda x: ', '.join(x) if isinstance(x, list) else 'No authors')
        books_df['categories'] = books_df['categories'].apply(lambda x: ', '.join(x) if isinstance(x, list) else 'No categories')

        return books_df

    def update_books_data(self, query):
        # Load existing data
        try:
            existing_df = pd.read_csv('books_data.csv')
        except FileNotFoundError:
            existing_df = pd.DataFrame()

        # Fetch new data
        books_data = self.fetch_books(query)
        books_info = [self.extract_book_info(book) for book in books_data.get('items', [])]

        # Preprocess new data
        new_df = self.preprocess_books(books_info)

        # Combine existing and new data
        combined_df = pd.concat([existing_df, new_df])

        # Remove duplicates
        combined_df.drop_duplicates(subset=['title', 'authors'], inplace=True)

        # Save the combined data to CSV
        combined_df.to_csv('books_data.csv', index=False)

        print(combined_df.head())
    
    
    def extract_book_info(self,book):
        volume_info = book.get('volumeInfo', {})
        return {
            'title': volume_info.get('title', 'No title'),
            'authors': volume_info.get('authors', 'No authors'),
            'description': volume_info.get('description', 'No description'),
            'categories': volume_info.get('categories', 'No categories'),
            'average_rating': volume_info.get('averageRating', 'No rating'),
            'ratings_count': volume_info.get('ratingsCount', 'No count')
        }



