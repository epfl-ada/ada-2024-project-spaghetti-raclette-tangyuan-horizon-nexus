# distilBERT_basic.py

import os
import pandas as pd
import json
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from tqdm import tqdm

# Load and clean plot summaries
def clean_plot_summaries(data_directory):
    plot_summaries_path = os.path.join(data_directory, 'plot_summaries.txt')
    if not os.path.exists(plot_summaries_path):
        raise FileNotFoundError(f"The file 'plot_summaries.txt' was not found in the directory: {data_directory}")
    df_plot_summaries = pd.read_csv(plot_summaries_path, sep='\t', header=None, names=['movie_id', 'plot_summary'])
    return df_plot_summaries

# Segment each plot summary into sentences
def segment_plot_summaries(df_plot_summaries):
    df_plot_summaries['sentences'] = df_plot_summaries['plot_summary'].apply(sent_tokenize)
    return df_plot_summaries

# Analyze sentiment across sentences using DistilBERT
def analyze_sentiment(sentences):
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0)
    sentiment_scores = []
    for sentence in tqdm(sentences, desc="Analyzing Sentences", unit="sentence"):
        result = sentiment_analyzer(sentence)[0]
        score = 1 if result['label'] == 'POSITIVE' else -1  # Store as 1 for positive, -1 for negative
        sentiment_scores.append(score)
    return sentiment_scores

# Analyze all movies and save raw sentiment data
def main(data_directory, num_movies=1000):
    df_plot_summaries = clean_plot_summaries(data_directory)
    df_segmented = segment_plot_summaries(df_plot_summaries.head(num_movies))

    raw_sentiment_data = {}
    for idx, row in tqdm(df_segmented.iterrows(), desc="Processing Movies", total=len(df_segmented), unit="movie"):
        movie_id = row['movie_id']
        sentences = row['sentences']
        sentiment_scores = analyze_sentiment(sentences)
        raw_sentiment_data[movie_id] = sentiment_scores

    # Save raw sentiment data
    output_path = os.path.join(data_directory, 'raw_sentiment_data.json')
    with open(output_path, 'w') as f:
        json.dump(raw_sentiment_data, f)
    print(f"Raw sentiment data saved to {output_path}")

# Set up paths and run the main function
current_directory = os.getcwd()
data_directory = os.path.join(current_directory, '..', 'Data')
main(data_directory)
