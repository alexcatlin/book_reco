from sklearn.metrics.pairwise import cosine_similarity


def create_user_item_matrix(ratings_df):
    return ratings_df.pivot(index="user_id", columns="book_id", values="rating").fillna(
        0
    )


def compute_cosine_similarity(user_item_matrix):
    return cosine_similarity(user_item_matrix)


def get_collaborative_recommendations(user_id, merged_books_df, ratings_df, n=5):
    user_item_matrix = create_user_item_matrix(ratings_df)
    cosine_sim_matrix = compute_cosine_similarity(user_item_matrix)

    user_index = user_item_matrix.index.tolist().index(user_id)
    similarity_scores = list(enumerate(cosine_sim_matrix[user_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1 : n + 1]

    similar_users = [i[0] for i in similarity_scores]
    similar_user_ids = user_item_matrix.index[similar_users].tolist()

    user_ratings = user_item_matrix.loc[similar_user_ids]
    mean_ratings = user_ratings.mean(axis=0)
    mean_ratings_df = mean_ratings.to_frame("mean_rating")

    recommendations = mean_ratings_df.merge(
        merged_books_df, left_index=True, right_on="book_id"
    )
    recommendations = recommendations.sort_values(by="mean_rating", ascending=False)
    return recommendations[["title", "authors", "mean_rating"]].head(n)
