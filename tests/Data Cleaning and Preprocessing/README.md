
## Explanation of Python Scripts

### 1. `character_metadata_cleaning.py`
- **Objective**: This script is responsible for cleaning the `character.metadata.tsv` file.
- **Key steps**:
  - It fixes misaligned rows, particularly where some rows have extra Freebase character map columns.
  - It assigns appropriate column names like `movie_id`, `freebase_id`, `release_date`, `character_name`, and so on.
  - It handles missing values by replacing `Unknown` with actual missing values (NaN) where applicable.
  - The script ensures that the `actor_height` and `actor_age` fields are numeric where possible.

### 2. `movie_metadata_cleaning.py`
- **Objective**: This script is for cleaning the `movie.metadata.tsv` file.
- **Key steps**:
  - The script parses fields like `countries`, `genres`, and `languages` that are in a dictionary format and extracts human-readable values.
  - It also handles missing values for fields like `release_date`, `revenue`, and `runtime`.
  - The `languages_clean`, `countries_clean`, and `genres_clean` columns are created to store the cleaned data.

### 3. `name_clusters_cleaning.py`
- **Objective**: This script handles the cleaning of the `name.clusters.txt` file.
- **Key steps**:
  - The script loads the `name.clusters.txt` file, assigns column names (`name`, `cluster_id`), and checks for missing values.
  - It performs basic cleaning and prints out a cleaned version of the dataset.

### 4. `plot_summaries_cleaning.py`
- **Objective**: This script is used to clean the `plot_summaries.txt` file.
- **Key steps**:
  - It loads the plot summaries dataset and checks for missing values and duplicates.
  - It ensures that all `movie_id` entries are unique and valid.
  - The cleaned dataset is printed to show the first few rows.

### 5. `tvtropes_cleaning.py`
- **Objective**: This script is used for cleaning the `tvtropes.clusters.txt` file.
- **Key steps**:
  - It parses the JSON-like details from the dataset, extracting fields like `character`, `movie`, `movie_id`, and `actor`.
  - It cleans the extracted values and creates a new DataFrame that is easier to work with.
  - The cleaned dataset is printed to show the first few rows.

### 6. `cleaning.py`
- **Objective**: This script serves as the main cleaning pipeline for all the datasets.
- **Key steps**:
  - It imports the cleaning functions from all the individual cleaning scripts and runs them in sequence.
  - This script centralizes the cleaning process, ensuring all datasets are cleaned at once and outputted in a standardized format.

## How to Run

1. Ensure all the required `.txt` and `.tsv` files are in the correct `Data` folder.
2. Run any individual script to clean a specific dataset, or run `cleaning.py` to clean all datasets at once.

Example to run all cleaning scripts:
```bash
python cleaning.py
