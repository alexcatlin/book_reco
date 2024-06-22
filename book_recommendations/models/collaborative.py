from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

def train_collaborative_model(ratings_df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'book_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
    algo = SVD()
    algo.fit(trainset)
    predictions = algo.test(testset)
    accuracy.rmse(predictions)
    return algo

def get_collaborative_recommendations(user_id, books_df, algo, n=5):
    all_book_ids = books_df['book_id'].unique()
    predictions = [algo.predict(user_id, book_id) for book_id in all_book_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n = predictions[:n]
    recommended_books = [books_df[books_df['book_id'] == pred.iid].iloc[0] for pred in top_n]
    return recommended_books
