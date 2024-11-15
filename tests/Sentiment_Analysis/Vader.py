import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Load analyzed sentiment data
def load_sentiment_data(data_directory):
    file_path = os.path.join(data_directory, 'plot_summaries_cleaned.csv')
    df_sentiment = pd.read_csv(file_path)
    
    # Perform sentiment analysis on plot summaries
    df_sentiment['sentiment'] = df_sentiment['plot_summary'].apply(lambda x: analyzer.polarity_scores(x))
    
    return df_sentiment

# Load pre-cleaned movie metadata
def load_movie_metadata(data_directory):
    movie_metadata_path = os.path.join(data_directory, 'movie_metadata_cleaned.csv')
    df_movie_metadata = pd.read_csv(movie_metadata_path)
    return df_movie_metadata

import ast

# Link metadata to sentiment data
def link_metadata(df_sentiment, df_metadata):
    linked_data = []
    for _, row in tqdm(df_sentiment.iterrows(), desc="Linking Metadata", total=len(df_sentiment)):
        movie_id = str(row['movie_id'])
        if movie_id in df_metadata['movie_id'].astype(str).values:
            metadata = df_metadata[df_metadata['movie_id'].astype(str) == movie_id].iloc[0]
            
            # Extract and concatenate genre names
            genre_dict_str = metadata['genres']
            genre_dict = ast.literal_eval(genre_dict_str)
            genre_names = list(genre_dict.values())
            genres = ', '.join(genre_names)
            
            linked_data.append({
                'movie_id': movie_id,
                'average_sentiment': row['sentiment']['compound'],
                'neg_sentiment': row['sentiment']['neg'],
                'neu_sentiment': row['sentiment']['neu'],
                'pos_sentiment': row['sentiment']['pos'],
                'genres': genres,
                'revenue': metadata['revenue'],
                'runtime': metadata['runtime']
            })
    return pd.DataFrame(linked_data)

# Analyze trends in sentiment by genre
def analyze_genre_trends(linked_data):
    # Convert columns to numeric, handling non-numeric `revenue` values
    linked_data['average_sentiment'] = pd.to_numeric(linked_data['average_sentiment'], errors='coerce')
    linked_data['neg_sentiment'] = pd.to_numeric(linked_data['neg_sentiment'], errors='coerce')
    linked_data['neu_sentiment'] = pd.to_numeric(linked_data['neu_sentiment'], errors='coerce')
    linked_data['pos_sentiment'] = pd.to_numeric(linked_data['pos_sentiment'], errors='coerce')
    linked_data['revenue'] = pd.to_numeric(linked_data['revenue'], errors='coerce')
    linked_data['runtime'] = pd.to_numeric(linked_data['runtime'], errors='coerce')

    genre_analysis = linked_data.groupby('genres').agg({
        'average_sentiment': 'mean',
        'neg_sentiment': 'mean',
        'neu_sentiment': 'mean',
        'pos_sentiment': 'mean',
        'revenue': 'mean'
    }).sort_values(by='revenue', ascending=False)

    print("\nGenre-Based Analysis:")
    print(genre_analysis)
    return genre_analysis

# Clustering of sentiment trajectories
def perform_clustering(linked_data, num_clusters=4):
    # Select only numeric columns for clustering and drop rows with NaN values
    numeric_data = linked_data[['average_sentiment', 'neg_sentiment', 'neu_sentiment', 'pos_sentiment']].dropna()
    filtered_linked_data = linked_data.loc[numeric_data.index]  # Filter linked_data to match numeric_data

    # Standardize features
    features = StandardScaler().fit_transform(numeric_data)

    # Clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    filtered_linked_data['cluster'] = labels

    # Visualization using PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=labels, cmap='viridis')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Clustering of Movies by Sentiment Trajectories')
    plt.colorbar(scatter)
    plt.show()

    print("\nCluster Analysis Summary:")
    print(filtered_linked_data.groupby('cluster').mean(numeric_only=True))
    return filtered_linked_data

def main(data_directory):
    print("Loading sentiment data...")
    # Load sentiment and metadata
    df_sentiment = load_sentiment_data(data_directory)
    print("Sentiment data loaded.")

    print("Loading movie metadata...")
    df_metadata = load_movie_metadata(data_directory)
    print("Movie metadata loaded.")

    print("Linking metadata to sentiment data...")
    # Link metadata to sentiment data
    linked_data = link_metadata(df_sentiment, df_metadata)
    print("Metadata linked to sentiment data.")

    print("Analyzing genre-based sentiment trends...")
    # Perform genre-based sentiment trend analysis
    genre_trends = analyze_genre_trends(linked_data)
    print("Genre-based sentiment trend analysis complete.")

    print("Performing clustering on sentiment features...")
    # Perform clustering on sentiment features
    clustered_data = perform_clustering(linked_data)
    print("Clustering on sentiment features complete.")

    print("Saving genre trend and cluster results...")
    # Save genre trend and cluster results
    clustered_data.to_csv(os.path.join(data_directory, 'sentiment_genre_analysis.csv'), index=False)
    print("Saved genre and cluster analysis results to sentiment_genre_analysis.csv")

current_directory = os.getcwd()
data_directory = os.path.join(current_directory, 'data')
main(data_directory)