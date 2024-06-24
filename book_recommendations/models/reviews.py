import pandas as pd
from textblob import TextBlob


# Function to analyze sentiment of reviews
def analyze_review_sentiment(review_text):
    analysis = TextBlob(review_text)
    return (
        analysis.sentiment.polarity
    )  # Returns a value between -1 (negative) and 1 (positive)


# Function to get average sentiment for a book
def get_average_sentiment(review_text):
    if pd.isna(review_text):
        return None
    analysis = TextBlob(review_text)
    return (
        analysis.sentiment.polarity
    )  # Returns a value between -1 (negative) and 1 (positive)
