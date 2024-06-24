import pandas as pd
from flask import Flask, render_template, request

import config
from models.collaborative_sklearn import get_collaborative_recommendations
from models.content_based import get_content_based_recommendations
from models.rag_model import rag_recommendation
from models.reviews import get_average_sentiment
from models.vector_db import create_faiss_index, query_faiss_index
from myapp.cleansedata import cleansedata, mergedata
from utils import fileexists
print ("line 12")
app = Flask(__name__)
print ("line 14")
NO_OF_ROWS = int(config.ROW_NUMBER_TO_TEST)
print ("line 16")
# Cleanse only if merged data does not exist

# Check if the merged file exists
if fileexists(config.MERGED_BOOKS_FILE):
    # Load the merged data
    merged_books_df = pd.read_csv(config.MERGED_BOOKS_FILE, nrows=NO_OF_ROWS)
    print ("line 23")
else:
    # Load the original data
    books_df = pd.read_csv(config.BOOKS_FILE)
    books_ratings_df = pd.read_csv(config.BOOK_RATINGS_FILE)

    books_cleansed_df, books_ratings_cleansed_df = cleansedata(
        books_ratings_df, books_df
    )
    # Merge the data and save to a new CSV file
    merged_books_df = mergedata(
        books_ratings_cleansed_df, books_cleansed_df, config.MERGED_BOOKS_FILE
    )
# limite the number of rows we play with for testing
merged_books_df = merged_books_df.head(NO_OF_ROWS)


# initializing RAG
# initializeRAG(merged_books_df)
# Preprocess ratings data
ratings_df = merged_books_df[["Title", "User_id", "Id", "review/score"]]
print(ratings_df)
# Add average sentiment to merged_books_df
merged_books_df["average_sentiment"] = merged_books_df["review/text"].apply(
    get_average_sentiment
)

print(merged_books_df["average_sentiment"])
# Setup RAG model
# rag_tokenizer, rag_model = setup_rag_model()

# Create FAISS index for book descriptions
faiss_index = create_faiss_index(merged_books_df)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/content", methods=["POST"])
def content_recommendations():
    title = request.form["title"]
    recommendations = get_content_based_recommendations(title, merged_books_df)
    return render_template(
        "content_recommendations.html", title=title, recommendations=recommendations
    )


@app.route("/collaborative", methods=["POST"])
def collaborative_recommendations():
    user_id = int(request.form["user_id"])
    recommendations = get_collaborative_recommendations(
        user_id, merged_books_df, ratings_df
    )
    return render_template(
        "collaborative_recommendations.html",
        user_id=user_id,
        recommendations=recommendations,
    )


@app.route("/rag", methods=["POST"])
def rag_recommendations():
    query = request.form["query"]
    recommendation=''
    #recommendation = rag_recommendation(query, rag_tokenizer, rag_model)
    return render_template(
        "rag_recommendations.html", query=query, recommendation=recommendation
    )


@app.route("/vector", methods=["POST"])
def vector_recommendations():
    description = request.form["description"]
    recommendations = query_faiss_index(description, faiss_index, merged_books_df)
    return render_template(
        "vector_recommendations.html",
        description=description,
        recommendations=recommendations,
    )


if __name__ == "__main__":
    app.run(debug=True)
