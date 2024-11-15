import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Load the movie master dataset
file_path = 'data/movie_master_dataset.csv'
df_movie_master = pd.read_csv(file_path)

# Segment each plot summary into sentences
def segment_plot_summaries(df_plot_summaries):
    df_plot_summaries['sentences'] = df_plot_summaries['plot_summary'].apply(sent_tokenize)
    return df_plot_summaries

# Apply the segmentation function
df_movie_master = segment_plot_summaries(df_movie_master)

# Perform sentiment analysis on the entire plot_summary column with progress bar
tqdm.pandas(desc="Analyzing plot sentiment")
df_movie_master['plot_sentiment'] = df_movie_master['plot_summary'].progress_apply(lambda x: analyzer.polarity_scores(str(x)))

# Perform sentiment analysis on individual sentences within the plot_summary column with progress bar
def analyze_sentences(sentences):
    sentence_sentiments = [analyzer.polarity_scores(sentence) for sentence in sentences]
    return sentence_sentiments

tqdm.pandas(desc="Analyzing sentence sentiments")
df_movie_master['sentence_sentiments'] = df_movie_master['sentences'].progress_apply(lambda x: analyze_sentences(x))

# Save the results to a new CSV file in the data directory
output_file_path = 'data/sentence_sentimental_analysis_Vader.csv'
df_movie_master.to_csv(output_file_path, index=False)

print(f"Sentiment analysis completed and saved to '{output_file_path}'")