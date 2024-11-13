import os
import pandas as pd
import ast

# Set up path for your 'tvtropes.clusters.txt' file in the 'donnees' directory
current_directory = os.getcwd()
data_directory = os.path.join(current_directory, 'Data')
tvtropes_clusters_path = os.path.join(data_directory, 'tvtropes.clusters.txt')

# Load the data into a DataFrame
df_tvtropes_clusters = pd.read_csv(
    tvtropes_clusters_path, 
    sep='\t', 
    header=None, 
    names=['trope_name', 'details'], 
    dtype=str
)

# Step 1: Parse the JSON-like details column
def parse_details(details):
    try:
        # Convert the string into a dictionary
        return ast.literal_eval(details)
    except (ValueError, SyntaxError):
        # If there's an error, return an empty dictionary
        return {}

# Apply the parsing function to the 'details' column
df_tvtropes_clusters['details_dict'] = df_tvtropes_clusters['details'].apply(parse_details)

# Step 2: Extract fields from the 'details_dict' column
df_tvtropes_clusters['character'] = df_tvtropes_clusters['details_dict'].apply(lambda x: x.get('char', 'Unknown'))
df_tvtropes_clusters['movie'] = df_tvtropes_clusters['details_dict'].apply(lambda x: x.get('movie', 'Unknown'))
df_tvtropes_clusters['movie_id'] = df_tvtropes_clusters['details_dict'].apply(lambda x: x.get('id', 'Unknown'))
df_tvtropes_clusters['actor'] = df_tvtropes_clusters['details_dict'].apply(lambda x: x.get('actor', 'Unknown'))

# Step 3: Drop the 'details_dict' column and keep only the relevant ones
df_tvtropes_cleaned = df_tvtropes_clusters[['trope_name', 'character', 'movie', 'movie_id', 'actor']]

# Step 4: Check for missing values
missing_values = df_tvtropes_cleaned.isnull().sum()

# Step 5: Print the missing values for each column
print("\nMissing values in each column:")
print(missing_values)

# Step 6: Preview the cleaned dataset
print("\nPreview of the cleaned TV Tropes dataset:")
print(df_tvtropes_cleaned.head(10))

# Optional: Save the cleaned DataFrame if necessary
# df_tvtropes_cleaned.to_csv('tvtropes_clusters_cleaned.csv', index=False)
