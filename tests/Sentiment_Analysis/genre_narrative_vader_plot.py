import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def normalize_sentiment_arc(sentiments, num_points=200):
    """
    Normalize a sentiment arc to a fixed number of points (e.g., 200).
    """
    x = np.linspace(0, len(sentiments) - 1, num_points)
    normalized_arc = np.interp(x, np.arange(len(sentiments)), sentiments)
    return normalized_arc

def analyze_sentiment_arcs(movie_master_dataset):
    """
    Analyze average sentiment arcs for specific genres and return the updated movie_master_dataset.

    Args:
        movie_master_dataset (DataFrame): Main dataset containing movie metadata.

    Returns:
        DataFrame: Updated movie_master_dataset.
    """
    # Load the VADER sentiment analysis results
    vader_sentiment_path = os.path.join('data/sentence_sentimental_analysis_Vader.csv')
    vader_df = pd.read_csv(vader_sentiment_path)

    # Prepare data by splitting genres
    vader_df['genres'] = vader_df['genres'].fillna('')
    vader_df['genres'] = vader_df['genres'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
    exploded_df = vader_df.explode('genres')

    # Filter for "pure" genres
    genre_counts = exploded_df.groupby('movie_id')['genres'].count()
    pure_genre_movie_ids = genre_counts[genre_counts == 1].index
    pure_genre_df = exploded_df[exploded_df['movie_id'].isin(pure_genre_movie_ids)]

    # Define genres to analyze and limits
    genres_to_analyze = ["Action", "Horror", "Drama", "Comedy"]
    y_axis_limits = {
        "Action": (-0.3, 0.3),
        "Horror": (-0.5, 0.1),
        "Drama": (-0.1, 0.1),
        "Comedy": (-0.3, 0.3),
    }

    # Analyze sentiment arcs
    genre_sentiment_arcs = {}
    for genre in genres_to_analyze:
        genre_movies = pure_genre_df[pure_genre_df['genres'].str.contains(genre, na=False, case=False)]
        normalized_arcs = []

        for _, row in genre_movies.iterrows():
            try:
                movie_sentiments = eval(row['sentence_sentiments'])
                compound_scores = [sentiment['compound'] for sentiment in movie_sentiments]
                normalized_arc = normalize_sentiment_arc(compound_scores)
                normalized_arcs.append(normalized_arc)
            except Exception:
                continue
        
        genre_sentiment_arcs[genre] = np.mean(normalized_arcs, axis=0) if normalized_arcs else None

    # Plot sentiment arcs
    def plot_genre_sentiment_arc(genre, sentiment_arc, x_min=0, x_max=200, y_min=None, y_max=None):
        plt.figure(figsize=(10, 6))
        if sentiment_arc is not None:
            plt.plot(sentiment_arc, label=f"{genre} Genre", color='blue')
        else:
            plt.text(0.5, 0.5, f"No data available for {genre}", 
                     horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        
        plt.title(f"Average Emotional Arc for {genre} Movies")
        plt.xlabel("Normalized Sentence Index")
        plt.ylabel("Sentiment Score")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend(title="Genres")
        plt.grid(visible=True)
        plt.tight_layout()
        plt.show()

    # Plot arcs for each genre
    for genre, sentiment_arc in genre_sentiment_arcs.items():
        y_min, y_max = y_axis_limits.get(genre, (-0.5, 0.5))
        plot_genre_sentiment_arc(genre, sentiment_arc, 0, 200, y_min, y_max)

    return movie_master_dataset
