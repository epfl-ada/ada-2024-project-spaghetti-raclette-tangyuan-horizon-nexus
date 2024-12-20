import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_genre_sentiment_analysis(movie_master_dataset):
    """
    Analyze and plot average sentiment scores by genre using DistilBERT sentiment analysis.

    Parameters:
    - movie_master_dataset (DataFrame): The dataset containing movie metadata.
    - distillbert_sentiment_path (str): Path to the DistilBERT sentiment analysis CSV file.
    """
    # Load the DistilBERT sentiment analysis results
    distillbert_df = pd.read_csv('data/distillbert_sentiment_analysis.csv')

    # Calculate the average sentiment score for each movie
    average_sentiment_per_movie = distillbert_df.groupby('movie_id')['sentiment_score'].mean().reset_index()

    # Merge with the movie_master_dataset to get genres
    merged_df = pd.merge(
        average_sentiment_per_movie, 
        movie_master_dataset[['movie_id', 'genres']], 
        on='movie_id', how='inner'
    )

    # Split genres into separate rows
    merged_df['genres'] = merged_df['genres'].str.split(', ')
    merged_df = merged_df.explode('genres')

    # Count the number of sentences for each genre
    genre_sentence_count = merged_df['genres'].value_counts().reset_index()
    genre_sentence_count.columns = ['genres', 'sentence_count']

    # Select the top 20 genres by number of sentences
    top_20_genres = genre_sentence_count.head(20)

    # Merge the top 20 genres with the average sentiment scores
    average_sentiment_by_genre = merged_df.groupby('genres')['sentiment_score'].mean().reset_index()
    top_20_genres = pd.merge(top_20_genres, average_sentiment_by_genre, on='genres')

    # Plot the average sentiment by genre for the top 20 genres
    plt.figure(figsize=(14, 8))
    sns.barplot(
        x='genres', y='sentiment_score', data=top_20_genres, 
        palette='viridis', hue='genres', dodge=False, legend=False
    )
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 20 Genres by Number of Sentences and Average Sentiment (DistilBERT)', fontsize=16)
    plt.xlabel('Genre', fontsize=14)
    plt.ylabel('Average Sentiment Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
