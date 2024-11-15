import os
import pandas as pd
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from tqdm import tqdm
import torch

# Segment each plot summary into sentences
def segment_plot_summaries(df_plot_summaries):
    df_plot_summaries['sentences'] = df_plot_summaries['plot_summary'].apply(sent_tokenize)
    return df_plot_summaries

# Analyze sentiment across sentences using DistilBERT
def analyze_sentiment(sentences, sentiment_analyzer):
    sentiment_scores = []
    for sentence in sentences:
        result = sentiment_analyzer(sentence)[0]
        score = 1 if result['label'] == 'POSITIVE' else -1  # Store as 1 for positive, -1 for negative
        sentiment_scores.append((sentence, score))
    return sentiment_scores

# Analyze sentiment for all movies and save to a CSV file
def save_sentiment_analysis_csv(movie_master_path, output_path):
    if not os.path.exists(movie_master_path):
        raise FileNotFoundError(f"The file 'movie_master_dataset.csv' was not found at: {movie_master_path}")
    df_movies = pd.read_csv(movie_master_path)
    df_movies = segment_plot_summaries(df_movies)
    sentiment_data = []
    device = 0 if torch.cuda.is_available() else -1
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)
    with tqdm(total=len(df_movies), desc="Processing Movies", unit="movie") as pbar:
        for index, row in df_movies.iterrows():
            sentences = row['sentences']
            sentiment_scores = analyze_sentiment(sentences, sentiment_analyzer)
            for sentence, score in sentiment_scores:
                sentiment_data.append({
                    'movie_id': row['movie_id'],
                    'sentence': sentence,
                    'sentiment_score': score
                })
            pbar.update(1)
    # Save sentiment analysis results to a CSV file
    sentiment_df = pd.DataFrame(sentiment_data)
    sentiment_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Sentiment analysis results saved to {output_path}")

# Main function
def main():
    # Set up the correct path for the movie master dataset
    base_directory = os.getcwd()
    data_directory = os.path.join(base_directory, '..', '..', 'data')
    movie_master_path = os.path.join(data_directory, 'movie_master_dataset.csv')
    output_path = os.path.join(data_directory, 'distillbert_sentiment_analysis.csv')

    print("Starting DistilBERT sentiment analysis...")
    save_sentiment_analysis_csv(movie_master_path, output_path)
    print("Sentiment analysis complete.")

# Run the main function
if __name__ == "__main__":
    main()