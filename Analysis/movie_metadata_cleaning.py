import os
import pandas as pd
import ast  # To safely evaluate string representations of dictionaries

# Set up path for your 'movie.metadata.tsv' file in the 'donnees' directory
current_directory = os.getcwd()
data_directory = os.path.join(current_directory, 'Data')
movie_metadata_path = os.path.join(data_directory, 'movie.metadata.tsv')

# Load the data into a DataFrame
df_movie_metadata = pd.read_csv(movie_metadata_path, sep='\t', header=None, names=[
    'movie_id', 'freebase_id', 'movie_name', 'release_date', 'revenue', 'runtime', 'languages', 'countries', 'genres'])

# Step 1: Checking for missing values (NaN)
missing_values = df_movie_metadata.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Step 2: Cleaning up 'countries', 'genres', and 'languages' columns (extracting readable values)

# Function to extract values from dictionary-like strings
def extract_values_from_dict_column(column):
    return column.apply(lambda x: ', '.join(ast.literal_eval(x).values()) if pd.notnull(x) and x.startswith('{') else x)

# Clean the 'countries', 'genres', and 'languages' columns
df_movie_metadata['countries_clean'] = extract_values_from_dict_column(df_movie_metadata['countries'])
df_movie_metadata['genres_clean'] = extract_values_from_dict_column(df_movie_metadata['genres'])

# Clean the 'languages' column by extracting only the language name (e.g., "English")
df_movie_metadata['languages_clean'] = df_movie_metadata['languages'].apply(
    lambda x: ', '.join(ast.literal_eval(x).values()).replace(' Language', '') if pd.notnull(x) and x.startswith('{') else x
)

# Step 3: Print the cleaned dataset columns for verification
print("\nCleaned 'countries_clean', 'genres_clean', and 'languages_clean' columns:")
print(df_movie_metadata[['movie_id', 'movie_name', 'countries_clean', 'genres_clean', 'languages_clean']].head())

# Step 4: Print the final version of the cleaned dataset
print("\nFinal cleaned dataset (first 5 rows):")
print(df_movie_metadata[['movie_id', 'freebase_id', 'movie_name', 'release_date', 'revenue', 'runtime', 'languages_clean', 'countries_clean', 'genres_clean']].head())

# Optional: Save cleaned DataFrame to a CSV file if needed
# df_movie_metadata.to_csv('movie_metadata_cleaned.csv', index=False)

