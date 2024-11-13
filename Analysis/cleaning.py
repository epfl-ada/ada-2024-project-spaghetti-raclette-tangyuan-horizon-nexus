import os
import pandas as pd
import ast  # To safely evaluate string representations of dictionaries

# Load and clean TV Tropes Clusters dataset
def clean_tvtropes_clusters(data_directory):
    tvtropes_clusters_path = os.path.join(data_directory, 'tvtropes.clusters.txt')
    df_tvtropes_clusters = pd.read_csv(tvtropes_clusters_path, sep='\t', header=None, names=['trope_name', 'details'], dtype=str)
    
    # Parse the details
    df_tvtropes_clusters['details_dict'] = df_tvtropes_clusters['details'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else {})
    df_tvtropes_clusters['character'] = df_tvtropes_clusters['details_dict'].apply(lambda x: x.get('char', 'Unknown'))
    df_tvtropes_clusters['movie'] = df_tvtropes_clusters['details_dict'].apply(lambda x: x.get('movie', 'Unknown'))
    df_tvtropes_clusters['movie_id'] = df_tvtropes_clusters['details_dict'].apply(lambda x: x.get('id', 'Unknown'))
    df_tvtropes_clusters['actor'] = df_tvtropes_clusters['details_dict'].apply(lambda x: x.get('actor', 'Unknown'))
    
    df_tvtropes_cleaned = df_tvtropes_clusters[['trope_name', 'character', 'movie', 'movie_id', 'actor']]
    print("\nCleaned TV Tropes dataset:")
    print(df_tvtropes_cleaned.head())
    return df_tvtropes_cleaned

# Load and clean Name Clusters dataset
def clean_name_clusters(data_directory):
    name_clusters_path = os.path.join(data_directory, 'name.clusters.txt')
    df_name_clusters = pd.read_csv(name_clusters_path, sep='\t', header=None, names=['name', 'cluster_id'], dtype=str)
    
    print("\nCleaned Name Clusters dataset:")
    print(df_name_clusters.head())
    return df_name_clusters

# Load and clean Character Metadata dataset
def clean_character_metadata(data_directory):
    character_metadata_path = os.path.join(data_directory, 'character.metadata.tsv')
    df_character_metadata = pd.read_csv(character_metadata_path, sep='\t', header=None, dtype=str, skip_blank_lines=True)
    
    def clean_row(row):
        if len(row) > 11:
            freebase_map_values = ', '.join([str(item) for item in row[10:] if pd.notnull(item)])
            return row[:10] + [freebase_map_values]
        return row[:11]
    
    df_character_metadata_cleaned = df_character_metadata.apply(lambda x: clean_row(list(x)), axis=1)
    df_character_metadata_cleaned = pd.DataFrame(df_character_metadata_cleaned.tolist(), columns=[
        'movie_id', 'freebase_id', 'release_date', 'character_name', 'actor_dob', 'actor_gender', 'actor_height', 
        'actor_ethnicity', 'actor_name', 'actor_age', 'freebase_character_map'
    ])
    
    df_character_metadata_cleaned.replace('Unknown', pd.NA, inplace=True)
    df_character_metadata_cleaned['actor_height'] = pd.to_numeric(df_character_metadata_cleaned['actor_height'], errors='coerce')
    df_character_metadata_cleaned['actor_age'] = pd.to_numeric(df_character_metadata_cleaned['actor_age'], errors='coerce')
    # Age cleaning: Replace negative values with positive and values over 125 with NaN
    df_character_metadata_cleaned['actor_age'] = df_character_metadata_cleaned['actor_age'].abs()
    df_character_metadata_cleaned.loc[df_character_metadata_cleaned['actor_age'] > 125, 'actor_age'] = pd.NA
    
    print("\nCleaned Character Metadata dataset:")
    print(df_character_metadata_cleaned.head())
    return df_character_metadata_cleaned

# Load and clean Plot Summaries dataset
def clean_plot_summaries(data_directory):
    plot_summaries_path = os.path.join(data_directory, 'plot_summaries.txt')
    df_plot_summaries = pd.read_csv(plot_summaries_path, sep='\t', header=None, names=['movie_id', 'plot_summary'])
    
    print("\nCleaned Plot Summaries dataset:")
    print(df_plot_summaries.head())
    return df_plot_summaries

# Load and clean Movie Metadata dataset
def clean_movie_metadata(data_directory):
    movie_metadata_path = os.path.join(data_directory, 'movie.metadata.tsv')
    df_movie_metadata = pd.read_csv(movie_metadata_path, sep='\t', header=None, names=[
        'movie_id', 'freebase_id', 'movie_name', 'release_date', 'revenue', 'runtime', 'languages', 'countries', 'genres'
    ])
    
    df_movie_metadata['countries_clean'] = df_movie_metadata['countries'].apply(lambda x: ', '.join(ast.literal_eval(x).values()) if pd.notnull(x) and x.startswith('{') else x)
    df_movie_metadata['genres_clean'] = df_movie_metadata['genres'].apply(lambda x: ', '.join(ast.literal_eval(x).values()) if pd.notnull(x) and x.startswith('{') else x)
    df_movie_metadata['languages_clean'] = df_movie_metadata['languages'].apply(lambda x: ', '.join(ast.literal_eval(x).values()).replace(' Language', '') if pd.notnull(x) and x.startswith('{') else x)
    
    print("\nCleaned Movie Metadata dataset:")
    print(df_movie_metadata[['movie_id', 'movie_name', 'countries_clean', 'genres_clean', 'languages_clean']].head())
    return df_movie_metadata

# Main cleaning function to process all datasets
def clean_all_datasets(data_directory):
    clean_tvtropes_clusters(data_directory)
    clean_name_clusters(data_directory)
    clean_character_metadata(data_directory)
    clean_plot_summaries(data_directory)
    clean_movie_metadata(data_directory)

# Set up the main directory and run the cleaning process
current_directory = os.getcwd()
data_directory = os.path.join(current_directory, 'Data')

clean_all_datasets(data_directory)
