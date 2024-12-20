import os
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns

def plot_vader_sentiment_analysis(movie_id_example_vader, vader_sentiment_path):
    """
    Analyze and plot VADER sentiment scores for a specific movie.

    Parameters:
    - movie_id_example_vader (int): The movie ID to analyze.
    - vader_sentiment_path (str): Path to the VADER sentiment analysis CSV file.
    """

    # Load the VADER sentiment analysis results
    vader_df = pd.read_csv(vader_sentiment_path)

    # Extract the sentence sentiments for the specific movie ID
    specific_movie_df = vader_df[vader_df['movie_id'] == movie_id_example_vader].copy()

    # Convert the 'sentence_sentiments' column from string to list of dictionaries
    specific_movie_df['sentence_sentiments'] = specific_movie_df['sentence_sentiments'].apply(ast.literal_eval)

    # Flatten the list of sentence sentiments
    sentence_sentiments = specific_movie_df['sentence_sentiments'].explode().reset_index(drop=True)

    # Extract the compound sentiment score for each sentence
    compound_scores = sentence_sentiments.apply(lambda x: x['compound'])

    # Count the number of sentences
    num_sentences = len(compound_scores)
    print(f"Number of sentences for movie ID {movie_id_example_vader}: {num_sentences}")

    # Plot the compound sentence sentiments for the specific movie
    plt.figure(figsize=(12, 8))
    sns.lineplot(x=compound_scores.index, y=compound_scores, marker='o')
    plt.title(f'Sentence Sentiment for Movie ID: {movie_id_example_vader} (VADER)', fontsize=16)
    plt.xlabel('Sentence Index', fontsize=14)
    plt.ylabel('Sentiment Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Extract all sentiment scores (pos, neg, neu, compound) for each sentence
    pos_scores = sentence_sentiments.apply(lambda x: x['pos'])
    neu_scores = sentence_sentiments.apply(lambda x: x['neu'])
    neg_scores = sentence_sentiments.apply(lambda x: x['neg'])

    # Plot all sentiment scores for the specific movie
    plt.figure(figsize=(12, 8))
    sns.lineplot(x=compound_scores.index, y=compound_scores, marker='o', label='Compound', color='blue')
    sns.lineplot(x=pos_scores.index, y=pos_scores, marker='x', label='Positive', color='green')
    sns.lineplot(x=neu_scores.index, y=neu_scores, marker='s', label='Neutral', color='orange')
    sns.lineplot(x=neg_scores.index, y=neg_scores, marker='d', label='Negative', color='red')
    plt.title(f'Sentence Sentiment Scores for Movie ID: {movie_id_example_vader} (VADER)', fontsize=16)
    plt.xlabel('Sentence Index', fontsize=14)
    plt.ylabel('Sentiment Score', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
