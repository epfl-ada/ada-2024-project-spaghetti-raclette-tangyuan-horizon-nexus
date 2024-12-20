import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_vader_genre_sentiment_analysis(vader_sentiment_path):
    """
    Plots the average sentiment by genre using VADER sentiment analysis results.

    Parameters:
    - vader_sentiment_path (str): Path to the VADER sentiment analysis CSV file.
    """
    # Load the VADER sentiment analysis results
    vader_df = pd.read_csv(vader_sentiment_path)

    # Extract the compound sentiment score from the plot_sentiment column
    vader_df['plot_sentiment_compound'] = vader_df['plot_sentiment'].apply(lambda x: eval(x)['compound'])

    # Split genres into separate rows
    vader_df['genres'] = vader_df['genres'].str.split(', ')
    vader_df = vader_df.explode('genres')

    # Count the number of sentences for each genre
    genre_sentence_count = vader_df['genres'].value_counts().reset_index()
    genre_sentence_count.columns = ['genres', 'sentence_count']

    # Select the top 20 genres by number of sentences
    top_20_genres = genre_sentence_count.head(20)

    # Merge the top 20 genres with the average sentiment scores
    average_sentiment_by_genre = vader_df.groupby('genres')['plot_sentiment_compound'].mean().reset_index()
    top_20_genres = pd.merge(top_20_genres, average_sentiment_by_genre, on='genres')

    # Plot genre-based sentiment analysis for top genres
    plt.figure(figsize=(14, 8))
    sns.barplot(
        x='genres', 
        y='plot_sentiment_compound', 
        data=top_20_genres, 
        palette='Greens_d', 
        hue='genres', 
        dodge=False, 
        legend=False
    )
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Sentiment by Top 20 Genres by Number of Sentences (VADER)')
    plt.xlabel('Genre')
    plt.ylabel('Average Sentiment Score')
    plt.show()
