# Data Handling Pipelinen Draft

## 1. Raw Data Ingestion (Done)

This is the initial step where we load all the datasets needed for the project. Each dataset is imported into a pandas DataFrame to facilitate further processing.

### Datasets:
- **Movie Metadata** (`movie.metadata.tsv`): Contains information about movie titles, revenue, genres, etc.
- **Character Metadata** (`character.metadata.tsv`): Contains actor and character information.
- **Name Clusters** (`name.clusters.txt`): Maps different spellings or variations of names to a cluster.
- **TV Tropes Clusters** (`tvtropes.clusters.txt`): Links movie characters to common tropes.
- **Plot Summaries** (`plot_summaries.txt`): Summaries of movie plots for sentiment analysis.

### Key Steps:
- **Read** each dataset into a pandas DataFrame.
- **Preview** the first few rows to ensure the dataset is correctly loaded.

---

## 2. Data Cleaning & Preprocessing (Done)

At this stage, we ensure that the raw data is properly formatted and ready for analysis. This involves dealing with missing values, fixing misaligned rows, and ensuring consistency in data types.

### Steps per Dataset:

### 1. **Movie Metadata**:
   - **Clean missing values**: Handle missing values in important columns such as `revenue`.
   - **Parse fields**: Extract readable values from fields like `countries`, `genres`, and `languages`.
   - **Inflation Adjustment**: Start planning for adjusting revenue values for inflation (handled during data enrichment).

### 2. **Character Metadata**:
   - **Fix misalignments**: Ensure columns like `character_name` and `release_date` are aligned.
   - **Replace Unknown values**: Fill missing or unknown values with `NaN` to represent missing data.
   - **Convert Data Types**: Ensure numeric columns like `actor_age` and `actor_height` are converted to their proper types.

### 3. **Name Clusters**:
   - **Assign Columns**: Name the columns as `name` and `cluster_id`.
   - **Handle duplicates**: Remove any duplicate rows from the dataset.

### 4. **TV Tropes Clusters**:
   - **Parse JSON-like fields**: Extract fields such as `character`, `movie`, `actor`, and `movie_id` from the JSON-like `details` field.
   - **Check for missing values**: Replace missing entries with `NaN`.

### 5. **Plot Summaries**:
   - **Remove duplicates**: Ensure there are no duplicate rows with different plot summaries for the same movie ID.
   - **Clean plot summaries**: Remove any poorly formatted or missing summaries.

---

## 3. Data Enrichment (to be done)

In this phase, we enrich the datasets by filling in missing information or deriving new features. This step is particularly crucial for the missing **revenue** data.

### Revenue Estimation:

1. **External Data Sources**:
   - Utilize external sources such as IMDb, Box Office Mojo, or The Numbers to fill in missing revenue data for films.

2. **Predict Missing Revenue Values** if needeed:
   - For movies where external data is unavailable, create a **predictive model** to estimate revenue based on features like `runtime`, `genre`, `release date`, and `actor fame`.
   - Use **regression models** or **machine learning algorithms** for prediction.

3. **Adjust for Inflation**:
   - Standardize all revenue values by adjusting for inflation using a **Consumer Price Index (CPI)** dataset.
   - This has already been done by a group in a previous project 
   - Convert all revenues to a consistent base year (e.g., 2023 dollars) for fair comparisons.

---

## 4. Exploratory Data Analysis (EDA)

Before diving into advanced analysis, perform initial investigations to understand the structure of the data and identify key patterns.

### Key Steps:

1. **Descriptive Statistics**:
   - **Numerical fields**: Calculate means, medians, standard deviations for fields such as `revenue`, `runtime`, and `actor_age`.
   - **Categorical fields**: Count the frequency of occurrences in fields like `genres`, `countries`, and `languages`.

2. **Data Visualizations**:
   - **Revenue distribution**: Create histograms or box plots to explore the distribution of movie revenues.
   - **Genre trends**: Visualize trends in genre popularity over time.
   - **Actor fame**: Use network visualizations to explore actor influence.

---

## 5. Feature Engineering

### 1. **Actor Fame**:
   - **Social Network Analysis**: We will use **Character Metadata** and **Name Clusters** to construct an **actor co-appearance network**.
     - Nodes represent actors, and edges represent movies where actors co-starred.
     - Use metrics like **Degree Centrality** (number of connections), **Betweenness Centrality** (bridging different actors), and **Closeness Centrality** (proximity to others) to compute an **Actor Fame Score**.
     - This score will quantify an actor’s influence within the network.

### 2. **Plot Complexity**:
   - **Sentiment & Lexical Analysis**: We will apply **sentiment analysis** to movie plot summaries to gauge emotional tones.
     - Use tools like **VADER** or **TextBlob** to classify each plot summary into positive, negative, or neutral sentiment.
     - Additionally, compute **lexical diversity**, **word count**, and the presence of key emotional or complex terms to quantify **plot complexity**.
     - Combine sentiment and lexical analysis into a **Plot Complexity Score**.

---

## 6. Core Analysis: Answering Research Questions

### 1. **Actor Fame vs. Plot Complexity**:
   - **Regression Analysis**: We will use a **multiple linear regression** model to evaluate how both **Actor Fame** and **Plot Complexity** impact box office revenue.
     - Dependent variable: **Box Office Revenue** (from **Movie Metadata**).
     - Independent variables: **Actor Fame Score**, **Plot Complexity Score**, and control variables like **Genre** and **Runtime**.
     - Formula:
     ```
     Revenue ~ Actor_Fame_Score + Plot_Complexity_Score + Genre + Runtime + Release_Date
     ```

### 2. **Correlation Analysis**:
   - Investigate the correlation between **actor fame**, **plot complexity**, and **revenue** using **Pearson** or **Spearman correlation** coefficients.
   - Visualize the relationships using **heatmaps** and **scatter plots** to explore the strength of each factor's relationship with box office revenue.

### 3. **Comparing Models**:
   - **Separate Regressions**: Build two separate models—one using **Actor Fame Score** and the other using **Plot Complexity Score**.
   - Compare the **R-squared values** to determine which factor explains more variance in revenue and is thus more impactful.

---

## 7. Results and Interpretation

### 1. **Key Findings**:
   - Present the most impactful factors for box office success based on the model results.
   - Summarize which has a stronger influence—**actor fame** or **plot complexity**.

### 2. **Visualizations**:
   - Use **bar charts**, **scatter plots**, and **line graphs** to display the findings clearly.
   - Show how **revenue** varies with changes in **Actor Fame Score** and **Plot Complexity Score**.

### 3. **Answer the Main Research Question**:
   - Based on the regression and correlation analyses, conclude whether the success of a movie is more heavily influenced by **actor fame** or **plot complexity**.
   - Discuss any additional insights, such as how genre or runtime might play a role.

---