import os
import pandas as pd

# Set up path for your 'character.metadata.tsv' file in the 'donnees' directory
current_directory = os.getcwd()
data_directory = os.path.join(current_directory, 'Data')
character_metadata_path = os.path.join(data_directory, 'character.metadata.tsv')

# Load the data into a DataFrame
df_character_metadata = pd.read_csv(
    character_metadata_path, 
    sep='\t', 
    header=None, 
    dtype=str,  # Treat everything as string initially to avoid issues
    skip_blank_lines=True
)

# Step 1: Fix misaligned rows and gather the multiple Freebase character maps
def clean_row(row):
    # If the row has more than 11 columns (indicating multiple Freebase character maps)
    if len(row) > 11:
        # Convert all extra entries into strings, ignoring NaN or None values
        freebase_map_values = ', '.join([str(item) for item in row[10:] if pd.notnull(item)])  
        corrected_row = row[:10] + [freebase_map_values]  # Keep only the first 10 columns and append the combined map values
        return corrected_row
    return row[:11]  # If already aligned, ensure the row has 11 columns

# Apply cleaning function row by row
df_character_metadata_cleaned = df_character_metadata.apply(lambda x: clean_row(list(x)), axis=1)

# Step 2: Assign proper column names
df_character_metadata_cleaned = pd.DataFrame(df_character_metadata_cleaned.tolist(), columns=[
    'movie_id', 'freebase_id', 'release_date', 'character_name', 'actor_dob', 'actor_gender', 
    'actor_height', 'actor_ethnicity', 'actor_name', 'actor_age', 'freebase_character_map'
])

# Step 3: Replace 'Unknown' with NaN in the entire DataFrame
df_character_metadata_cleaned.replace('Unknown', pd.NA, inplace=True)

# Convert numeric columns to appropriate data types
df_character_metadata_cleaned['actor_height'] = pd.to_numeric(df_character_metadata_cleaned['actor_height'], errors='coerce')
df_character_metadata_cleaned['actor_age'] = pd.to_numeric(df_character_metadata_cleaned['actor_age'], errors='coerce')

# Step 4: Print missing values count for each column (NaN includes 'Unknown')
print("\nMissing values in each column:")
print(df_character_metadata_cleaned.isnull().sum())

# Step 5: Print all columns to verify the cleaned dataset
print("\n--- movie_id column ---")
print(df_character_metadata_cleaned['movie_id'].head(10))

print("\n--- freebase_id column ---")
print(df_character_metadata_cleaned['freebase_id'].head(10))

print("\n--- release_date column ---")
print(df_character_metadata_cleaned['release_date'].head(10))

print("\n--- character_name column ---")
print(df_character_metadata_cleaned['character_name'].head(10))

print("\n--- actor_dob column ---")
print(df_character_metadata_cleaned['actor_dob'].head(10))

print("\n--- actor_gender column ---")
print(df_character_metadata_cleaned['actor_gender'].head(10))

print("\n--- actor_height column ---")
print(df_character_metadata_cleaned['actor_height'].head(10))

print("\n--- actor_ethnicity column ---")
print(df_character_metadata_cleaned['actor_ethnicity'].head(10))

print("\n--- actor_name column ---")
print(df_character_metadata_cleaned['actor_name'].head(10))

print("\n--- actor_age column ---")
print(df_character_metadata_cleaned['actor_age'].head(10))

print("\n--- freebase_character_map column ---")
print(df_character_metadata_cleaned['freebase_character_map'].head(10))

# Optional: Save the cleaned DataFrame if necessary
# df_character_metadata_cleaned.to_csv('character_metadata_cleaned.csv', index=False)
