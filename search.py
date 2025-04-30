def search_books(query, top_k=5):
    query_cleaned = ' '.join([word for word in query.lower().split() if word not in stop_words])
    query_vec = tfidf.transform([query_cleaned])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[-top_k:][::-1]
    return df.iloc[top_indices][['title', 'formatted_prompt', 'notes']]
