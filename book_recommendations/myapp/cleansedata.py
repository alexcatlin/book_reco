import numpy as np
import pandas as pd


def synthesiseratings():

    # Load the books data to get book IDs
    books_df = pd.read_csv("data/books_data.csv")
    book_ids = books_df.index.tolist()

    # Generate synthetic user ratings
    np.random.seed(42)
    user_ids = np.random.randint(1, 101, size=500)  # 100 users
    book_ids_sampled = np.random.choice(book_ids, size=500)  # Randomly chosen books
    ratings = np.random.randint(1, 6, size=500)  # Ratings between 1 and 5

    # Create a DataFrame
    ratings_df = pd.DataFrame(
        {"user_id": user_ids, "book_id": book_ids_sampled, "rating": ratings}
    )

    # Save the synthetic user ratings dataset
    ratings_df.to_csv("data/synthetic_user_ratings.csv", index=False)

    print(ratings_df.head())


def cleansedata(reviews_df, books_details_df):

    print("Reviews Dataset:")
    print(reviews_df.info())

    # Explore the Books Details dataset
    print("\nBooks Details Dataset:")
    print(books_details_df.info())
    # Check for missing values in Reviews dataset
    reviews_missing = reviews_df.isnull().sum()

    # Check for missing values in Books Details dataset
    books_missing = books_details_df.isnull().sum()

    # Display missing values
    print("Reviews Missing Values:")
    print(reviews_missing)

    print("\nBooks Details Missing Values:")
    print(books_missing)

    # Drop rows with missing 'Title' and 'User_id'
    reviews_df = reviews_df.dropna(subset=["Title", "User_id"])

    # Drop 'profileName' column, we won't be using it
    reviews_df = reviews_df.drop(columns=["profileName"])

    # Drop 'Price' column, we won't be using it
    reviews_df = reviews_df.drop(columns=["Price"])

    # Fill missing values in 'review/summary' and 'review/text' with empty strings
    reviews_df["review/summary"] = reviews_df["review/summary"].fillna("")
    reviews_df["review/text"] = reviews_df["review/text"].fillna("")

    # Display updated information about missing values
    reviews_missing_values = reviews_df.isnull().sum()
    print("Reviews Missing Values After Handling:")
    print(reviews_missing_values)

    # Drop rows with missing 'Title'
    books_details_df = books_details_df.dropna(subset=["Title"])

    # Impute missing values in 'ratingsCount' with the median
    books_details_df["ratingsCount"] = books_details_df["ratingsCount"].fillna(
        books_details_df["ratingsCount"].median()
    )

    # Fill missing values in textual columns with empty strings
    textual_columns = [
        "description",
        "authors",
        "publisher",
        "publishedDate",
        "categories",
    ]
    books_details_df[textual_columns] = books_details_df[textual_columns].fillna("")

    # Dropping columns we are not going to use
    columns_to_drop = ["image", "previewLink", "infoLink"]
    books_details_df = books_details_df.drop(columns=columns_to_drop)

    # Display updated information about missing values
    books_details_missing_values = books_details_df.isnull().sum()
    print("Books Details Missing Values After Handling:")
    print(books_details_missing_values)
    return books_details_df, reviews_df


def mergedata(reviews_df, book_details_df, output_file):
    merged_df = pd.merge(reviews_df, book_details_df, on="Title", how="left")
    merged_df.to_csv(output_file, index=False)
