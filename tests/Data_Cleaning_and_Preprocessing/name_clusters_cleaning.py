import os
import pandas as pd

# Set up path for your 'name.clusters.txt' file in the 'donnees' directory
current_directory = os.getcwd()
data_directory = os.path.join(current_directory, 'Data')
name_clusters_path = os.path.join(data_directory, 'name.clusters.txt')

# Load the data into a DataFrame
df_name_clusters = pd.read_csv(
    name_clusters_path, 
    sep='\t', 
    header=None, 
    dtype=str,  # Treat everything as string initially to avoid issues
    skip_blank_lines=True
)

# Step 1: Assign column names (based on the expected structure, modify as needed)
# From what we've seen before, it seems like there may be two key columns: 'name' and 'cluster_id'
df_name_clusters.columns = ['name', 'cluster_id']

# Step 2: Basic Structure Verification
print(f"Total number of rows: {df_name_clusters.shape[0]}")
print(f"Total number of columns: {df_name_clusters.shape[1]}")
print(f"Column names: {df_name_clusters.columns}")

# Step 3: Checking for missing values (NaN)
missing_values = df_name_clusters.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Step 5: Print all columns to verify the cleaned dataset
print("\n--- name column ---")
print(df_name_clusters['name'].head(10))

print("\n--- cluster_id column ---")
print(df_name_clusters['cluster_id'].head(10))

# Optional: Save the cleaned DataFrame if necessary
# df_name_clusters_cleaned.to_csv('name_clusters_cleaned.csv', index=False)
