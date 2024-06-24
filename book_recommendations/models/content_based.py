from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_content_based_recommendations(title, merged_books_df, n=5):
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(merged_books_df["description"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    try:
        idx = merged_books_df.index[merged_books_df["title"] == title].tolist()[0]
    except IndexError:
        return []

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : n + 1]
    book_indices = [i[0] for i in sim_scores]
    return merged_books_df.iloc[book_indices][["title", "authors", "average_rating"]]
