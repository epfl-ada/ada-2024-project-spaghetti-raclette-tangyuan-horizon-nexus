import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import os

def compute_sentiment_variability(exploded_df):
    """
    Computes the standard deviation of sentiment scores for each movie.

    Args:
        exploded_df (DataFrame): DataFrame containing sentence sentiment scores for each movie.

    Returns:
        DataFrame: A DataFrame with movie_id and sentiment variability.
    """
    sentiment_variability = []
    for _, row in exploded_df.iterrows():
        try:
            movie_sentiments = eval(row['sentence_sentiments'])  # Convert string to list of dicts
            compound_scores = [sentiment['compound'] for sentiment in movie_sentiments]
            std_dev = np.std(compound_scores)
            sentiment_variability.append({"movie_id": row['movie_id'], "variability": std_dev})
        except Exception:
            continue
    return pd.DataFrame(sentiment_variability)


def analyze_sentiment_var(movie_master_dataset):
    """
    Analyzes sentiment variability and returns the updated movie metadata DataFrame.

    Args:
        movie_master_dataset (DataFrame): The main movie dataset.

    Returns:
        DataFrame: Updated movie_master_dataset with sentiment variability features.
    """
    # Load the VADER sentiment analysis results
    vader_sentiment_path = os.path.join('data/sentence_sentimental_analysis_Vader.csv')
    vader_df = pd.read_csv(vader_sentiment_path)

    # Prepare data by splitting genres
    vader_df['genres'] = vader_df['genres'].fillna('')  
    vader_df['genres'] = vader_df['genres'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
    exploded_df = vader_df.explode('genres')

    # Compute variability and merge into the master dataset
    variability_df = compute_sentiment_variability(exploded_df)
    movie_master_dataset = movie_master_dataset.merge(variability_df, on="movie_id", how="left")

    # Split movies into high and low variability based on median
    median_variability = movie_master_dataset['variability'].median()
    movie_master_dataset['variability_group'] = np.where(
        movie_master_dataset['variability'] > median_variability,
        'High Variability',
        'Low Variability'
    )

    # Calculate average success for each group
    variability_success_summary = movie_master_dataset.groupby('variability_group')['success'].mean().reset_index()

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.bar(
        variability_success_summary['variability_group'],
        variability_success_summary['success'],
        color=['skyblue', 'lightcoral'],
        edgecolor='black'
    )
    plt.title("Average Success by Sentiment Variability")
    plt.xlabel("Sentiment Variability")
    plt.ylabel("Average Success")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Statistical significance test
    high_variability_success = movie_master_dataset[movie_master_dataset['variability_group'] == 'High Variability']['success']
    low_variability_success = movie_master_dataset[movie_master_dataset['variability_group'] == 'Low Variability']['success']

    t_stat, p_value = ttest_ind(high_variability_success, low_variability_success, equal_var=False)

    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4e}")

    # Remove duplicates
    movie_master_dataset = movie_master_dataset.drop_duplicates(subset='movie_id')

    # Return the updated DataFrame
    return movie_master_dataset
