from textblob import TextBlob

# Function to analyze sentiment of reviews
def analyze_review_sentiment(review_text):
    analysis = TextBlob(review_text)
    return analysis.sentiment.polarity  # Returns a value between -1 (negative) and 1 (positive)

# Function to get average sentiment for a book
def get_average_sentiment(book_id, reviews_df):
    book_reviews = reviews_df[reviews_df['book_id'] == book_id]['review_text']
    if book_reviews.empty:
        return None
    sentiments = book_reviews.apply(analyze_review_sentiment)
    return sentiments.mean()
