import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load analyzed sentiment data
def load_sentiment_data(data_directory):
    file_path = os.path.join(data_directory, 'analyzed_sentiment_data.json')
    with open(file_path, 'r') as f:
        analyzed_data = json.load(f)
    return analyzed_data

# Load pre-cleaned movie metadata
def load_movie_metadata(data_directory):
    movie_metadata_path = os.path.join(data_directory, 'movie_metadata_cleaned.csv')
    df_movie_metadata = pd.read_csv(movie_metadata_path)
    return df_movie_metadata

# Link metadata to sentiment data
def link_metadata(analyzed_data, df_metadata):
    linked_data = []
    for movie_id, sentiment_info in analyzed_data.items():
        movie_id = str(movie_id)
        if movie_id in df_metadata['movie_id'].astype(str).values:
            metadata = df_metadata[df_metadata['movie_id'].astype(str) == movie_id].iloc[0]
            linked_data.append({
                'movie_id': movie_id,
                'average_sentiment': sentiment_info['average_sentiment'],
                'std_dev_sentiment': sentiment_info['std_dev_sentiment'],
                'num_peaks': sentiment_info['num_peaks'],
                'num_valleys': sentiment_info['num_valleys'],
                'genres': metadata['genres'],
                'revenue': metadata['revenue'],
                'runtime': metadata['runtime']
            })
    return pd.DataFrame(linked_data)

# Analyze trends in sentiment by genre
def analyze_genre_trends(linked_data):
    # Convert columns to numeric, handling non-numeric `revenue` values
    linked_data['average_sentiment'] = pd.to_numeric(linked_data['average_sentiment'], errors='coerce')
    linked_data['std_dev_sentiment'] = pd.to_numeric(linked_data['std_dev_sentiment'], errors='coerce')
    linked_data['num_peaks'] = pd.to_numeric(linked_data['num_peaks'], errors='coerce')
    linked_data['num_valleys'] = pd.to_numeric(linked_data['num_valleys'], errors='coerce')
    linked_data['revenue'] = pd.to_numeric(linked_data['revenue'], errors='coerce')
    linked_data['runtime'] = pd.to_numeric(linked_data['runtime'], errors='coerce')

    genre_analysis = linked_data.groupby('genres').agg({
        'average_sentiment': 'mean',
        'std_dev_sentiment': 'mean',
        'num_peaks': 'mean',
        'num_valleys': 'mean',
        'revenue': 'mean'
    }).sort_values(by='revenue', ascending=False)
    
    print("\nGenre-Based Analysis:")
    print(genre_analysis)
    return genre_analysis

# Clustering of sentiment trajectories
def perform_clustering(linked_data, num_clusters=4):
    # Select only numeric columns for clustering and drop rows with NaN values
    numeric_data = linked_data[['average_sentiment', 'std_dev_sentiment', 'num_peaks', 'num_valleys']].dropna()
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
    # Load sentiment and metadata
    analyzed_data = load_sentiment_data(data_directory)
    df_metadata = load_movie_metadata(data_directory)
    
    # Link metadata to sentiment data
    linked_data = link_metadata(analyzed_data, df_metadata)

    # Perform genre-based sentiment trend analysis
    genre_trends = analyze_genre_trends(linked_data)

    # Perform clustering on sentiment features
    clustered_data = perform_clustering(linked_data)

    # Save genre trend and cluster results
    clustered_data.to_csv(os.path.join(data_directory, 'sentiment_genre_analysis.csv'), index=False)
    print("Saved genre and cluster analysis results to sentiment_genre_analysis.csv")

current_directory = os.getcwd()
data_directory = os.path.join(current_directory, '..', 'Data')
main(data_directory)
