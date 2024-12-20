import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distillbert_sentiment( movie_id_example_DistillBERT=77856):
    """
    Plot the DistilBERT sentence sentiment for a specific movie.

    Parameters:
    - data_directory (str): Path to the directory containing the sentiment CSV file.
    - movie_id_example_DistillBERT (int): The movie ID to visualize (default is 77856).
    """
    # Load the DistilBERT sentiment analysis results
    distillbert_sentiment_path = os.path.join('data/distillbert_sentiment_analysis.csv')
    distillbert_df = pd.read_csv(distillbert_sentiment_path)

    # Filter the DataFrame for the specific movie ID
    specific_movie_df = distillbert_df[distillbert_df['movie_id'] == movie_id_example_DistillBERT]

    # Count the number of sentences
    num_sentences = len(specific_movie_df)
    print(f"Number of sentences for movie ID {movie_id_example_DistillBERT}: {num_sentences}")

    # Plot the sentence sentiments for the specific movie
    plt.figure(figsize=(12, 8))
    sns.lineplot(x=specific_movie_df.index, y='sentiment_score', data=specific_movie_df, marker='o')
    plt.title(f'Sentence Sentiment for Movie ID: {movie_id_example_DistillBERT} (DistilBERT)', fontsize=16)
    plt.xlabel('Sentence Index', fontsize=14)
    plt.ylabel('Sentiment Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
