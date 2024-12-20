import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
import os

def compute_shape_features(exploded_df):
    """
    Compute amplitude, slope, and peak timing features for sentiment arcs of each movie.
    """
    shape_features = []
    for _, row in exploded_df.iterrows():
        try:
            # Convert string to list of dicts if necessary
            movie_sentiments = eval(row['sentence_sentiments'])
            compound_scores = [sentiment['compound'] for sentiment in movie_sentiments]
            
            # Compute features
            amplitude = max(compound_scores) - min(compound_scores) if compound_scores else 0
            slope = (compound_scores[-1] - compound_scores[0]) / len(compound_scores) if len(compound_scores) > 1 else 0
            peak_timing = np.argmax(compound_scores) / len(compound_scores) if compound_scores else 0
            
            shape_features.append({
                "movie_id": row["movie_id"],
                "amplitude": amplitude,
                "slope": slope,
                "peak_timing": peak_timing
            })
        except Exception:
            shape_features.append({
                "movie_id": row["movie_id"],
                "amplitude": np.nan,
                "slope": np.nan,
                "peak_timing": np.nan
            })
    return pd.DataFrame(shape_features)


def perform_anova_on_feature(movie_master_dataset, feature):
    """
    Perform ANOVA on a shape-based feature (e.g., amplitude, slope, peak_timing).
    """
    try:
        # Bin the feature into quartiles
        bins = pd.qcut(movie_master_dataset[feature], q=4, labels=["Low", "Medium-Low", "Medium-High", "High"])
        movie_master_dataset[f'{feature}_bin'] = bins
        
        # Calculate average success by bin
        success_summary = movie_master_dataset.groupby(f'{feature}_bin')['success'].mean().reset_index()
        
        # Plot average success by bins
        plt.figure(figsize=(10, 6))
        plt.bar(
            success_summary[f'{feature}_bin'].astype(str),
            success_summary['success'],
            color='skyblue',
            edgecolor='black'
        )
        plt.title(f"Average Success by {feature.capitalize()} Quartiles")
        plt.xlabel(f"{feature.capitalize()} Quartiles")
        plt.ylabel("Average Success")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        # Perform ANOVA
        groups = [
            movie_master_dataset[movie_master_dataset[f'{feature}_bin'] == level]['success']
            for level in bins.unique()
        ]
        f_stat, p_value = f_oneway(*groups)
        print(f"ANOVA for {feature}: F-statistic = {f_stat:.4f}, P-value = {p_value:.4e}")
    
    except Exception as e:
        print(f"Error during ANOVA for feature: {feature}. Error: {e}")


def analyze_sentiment_shape(movie_master_dataset):
    """
    Analyze sentiment shape-based features and return the updated movie_master_dataset.

    Args:
        movie_master_dataset (DataFrame): Main dataset containing movie metadata.
        data_directory (str): Path to the data directory containing the VADER sentiment file.

    Returns:
        DataFrame: Updated movie_master_dataset with shape-based features.
    """
    # Load the VADER sentiment analysis results
    vader_sentiment_path = os.path.join('data/sentence_sentimental_analysis_Vader.csv')
    vader_df = pd.read_csv(vader_sentiment_path)

    # Prepare data by splitting genres
    vader_df['genres'] = vader_df['genres'].fillna('')
    vader_df['genres'] = vader_df['genres'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
    exploded_df = vader_df.explode('genres')

    # Compute shape-based features
    shape_features_df = compute_shape_features(exploded_df)

    # Merge features into the master dataset
    movie_master_dataset = movie_master_dataset.merge(shape_features_df, on="movie_id", how="left")

    # Perform analysis
    for feature in ["amplitude", "slope", "peak_timing"]:
        print(f"\nPerforming analysis for {feature}...")
        perform_anova_on_feature(movie_master_dataset, feature)
    
    movie_master_dataset = movie_master_dataset.drop_duplicates(subset='movie_id')
    # Return the updated DataFrame
    return movie_master_dataset
