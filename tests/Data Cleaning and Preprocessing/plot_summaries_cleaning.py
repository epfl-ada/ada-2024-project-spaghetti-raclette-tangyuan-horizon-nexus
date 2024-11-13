import os
import pandas as pd

# Set up path for your 'plot_summaries.txt' file in the 'donnees' directory
current_directory = os.getcwd()
data_directory = os.path.join(current_directory, 'Data')
plot_summaries_path = os.path.join(data_directory, 'plot_summaries.txt')

# Load the data into a DataFrame
df_plot_summaries = pd.read_csv(plot_summaries_path, sep='\t', header=None, names=['movie_id', 'plot_summary'])

# Step 1: Basic Structure Verification
print(f"Total number of rows: {df_plot_summaries.shape[0]}")
print(f"Total number of columns: {df_plot_summaries.shape[1]}")
print(f"Column names: {df_plot_summaries.columns}")

# Step 2: Checking for missing values (NaN)
missing_values = df_plot_summaries.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Step 3: Check for any duplicate rows (same movie_id and plot_summary)
duplicate_rows = df_plot_summaries[df_plot_summaries.duplicated()]
print(f"\nNumber of duplicate rows: {duplicate_rows.shape[0]}")
if not duplicate_rows.empty:
    print("\nDuplicate rows:")
    print(duplicate_rows)

# Step 4: Check if movie_id is numeric and if plot_summary is a string
invalid_ids = df_plot_summaries[~df_plot_summaries['movie_id'].apply(lambda x: str(x).isdigit())]
print(f"\nNumber of invalid movie_id entries: {invalid_ids.shape[0]}")
if not invalid_ids.empty:
    print("\nInvalid movie_id entries (non-numeric IDs):")
    print(invalid_ids)

# Step 5: Preview some rows to inspect the data manually
print("\nPreview of the first few rows of the dataset:")
print(df_plot_summaries.head())

# Check for duplicate movie_id entries
duplicate_movie_ids = df_plot_summaries[df_plot_summaries.duplicated(subset='movie_id', keep=False)]

# Print the number of duplicate movie IDs
print(f"\nNumber of duplicate movie_id entries: {duplicate_movie_ids.shape[0]}")

# If there are duplicates, display them
if not duplicate_movie_ids.empty:
    print("\nDuplicate movie_id entries:")
    pd.set_option('display.max_rows', None)  # Allows printing of all rows without truncating
    print(duplicate_movie_ids)
else:
    print("All movie_id entries are unique.")
