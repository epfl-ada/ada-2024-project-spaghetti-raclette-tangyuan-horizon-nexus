from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def normalize_sentiment_arc(sentiments, num_points=200):
    """
    Normalize a sentiment arc to a fixed number of points (e.g., 200).
    """
    x = np.linspace(0, len(sentiments) - 1, num_points)
    normalized_arc = np.interp(x, np.arange(len(sentiments)), sentiments)
    return normalized_arc

def prepare_genre_data_for_clustering(exploded_df, genre, normalize_arc_length=200):
    """
    Prepare sentiment arcs for a given genre for clustering.
    """
    genre_movies = exploded_df[exploded_df['genres'].str.contains(genre, na=False, case=False)]
    normalized_arcs = []

    for _, row in genre_movies.iterrows():
        try:
            movie_sentiments = eval(row['sentence_sentiments'])
            compound_scores = [sentiment['compound'] for sentiment in movie_sentiments]
            normalized_arc = normalize_sentiment_arc(compound_scores, num_points=normalize_arc_length)
            normalized_arcs.append(normalized_arc)
        except Exception:
            continue

    return np.array(normalized_arcs)

def cluster_genre_arcs(normalized_arcs, genre, n_clusters=4):
    """
    Perform clustering on sentiment arcs for a specific genre and plot barycenters.
    """
    clustering_data = normalized_arcs.reshape((normalized_arcs.shape[0], normalized_arcs.shape[1], 1))

    scaler = TimeSeriesScalerMeanVariance()
    clustering_data_scaled = scaler.fit_transform(clustering_data)

    kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", verbose=False, random_state=42)
    cluster_labels = kmeans.fit_predict(clustering_data_scaled)
    barycenters = kmeans.cluster_centers_

    plt.figure(figsize=(10, 6))
    for i, barycenter in enumerate(barycenters):
        plt.plot(barycenter.ravel(), label=f"Cluster {i + 1}", linewidth=2)
    plt.title(f"Representative Story Arcs for {genre} Movies")
    plt.xlabel("Normalized Sentence Index")
    plt.ylabel("Sentiment Score")
    plt.legend()
    plt.grid(visible=True)
    plt.tight_layout()
    plt.show()

    return cluster_labels, barycenters

def cluster_genres_arcs(movie_master_dataset, genres_to_analyze=None, n_clusters=4):
    """
    Perform clustering analysis on sentiment arcs of specified genres.
    """
    if genres_to_analyze is None:
        genres_to_analyze = ["Action", "Horror", "Drama", "Comedy"]

    vader_sentiment_path = os.path.join('data/sentence_sentimental_analysis_Vader.csv')
    vader_df = pd.read_csv(vader_sentiment_path)

    vader_df['genres'] = vader_df['genres'].fillna('')
    vader_df['genres'] = vader_df['genres'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
    exploded_df = vader_df.explode('genres')

    cluster_results = {}

    for genre in tqdm(genres_to_analyze, desc="Processing Genres"):
        print(f"Processing genre: {genre}")
        normalized_arcs = prepare_genre_data_for_clustering(exploded_df, genre)
        if len(normalized_arcs) == 0:
            print(f"No data available for {genre}. Skipping...")
            continue

        cluster_labels, barycenters = cluster_genre_arcs(normalized_arcs, genre, n_clusters)
        cluster_results[genre] = {
            "labels": cluster_labels,
            "barycenters": barycenters
        }

    print("Clustering results are ready.")
    return cluster_results
