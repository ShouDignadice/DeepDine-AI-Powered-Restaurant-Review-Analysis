# DeepDine: AI-Powered Restaurant Review Analysis

DeepDine is an NLP project that analyzes Yelp restaurant reviews, groups similar customer experiences using semantic embeddings and K-Means clustering, and creates restaurant-level theme summaries.

## Current Baseline

* Approximately 20,000 Yelp restaurant reviews
* 2,000 restaurants
* Up to 10 reviews per restaurant
* Embedding model: `all-MiniLM-L6-v2`
* Embedding dimensions: 384
* Clustering algorithm: K-Means
* Current cluster count: 12

## Run the Project

Run the scripts from the project root in this order:

```bash
python src/prep_yelp_data.py
python src/clean_yelp_reviews.py
python src/embed_reviews.py
python src/cluster_reviews.py
python src/label_clusters.py
python src/summarize_restaurants.py
```

Each script uses the output created by the previous script.

## Scripts

| Script                         | Purpose                                                                         |
| ------------------------------ | ------------------------------------------------------------------------------- |
| `src/prep_yelp_data.py`        | Filters Yelp businesses to restaurants and samples approximately 20,000 reviews |
| `src/clean_yelp_reviews.py`    | Removes missing, duplicate, and very short reviews                              |
| `src/embed_reviews.py`         | Converts review text into semantic embeddings                                   |
| `src/cluster_reviews.py`       | Groups semantically similar reviews using K-Means                               |
| `src/label_clusters.py`        | Adds readable labels to the generated clusters                                  |
| `src/summarize_restaurants.py` | Creates theme summaries for each restaurant                                     |

## Data Files

The original Yelp JSON files are stored in:

```text
data/raw/
```

The preparation script creates:

```text
data/processed/yelp_restaurant_reviews.csv
```

This file contains:

```text
business_id
name
stars
text
```

The cleaning script creates:

```text
data/processed/cleaned_yelp_reviews.csv
```

This file contains the review data used to generate embeddings.

## Review Embeddings

The embeddings are stored in:

```text
data/embeddings/review_embeddings.npy
```

Each review is converted into a 384-dimensional vector.

The rows in the cleaned CSV and embedding file must remain aligned:

```text
CSV row 0 <-> embedding row 0
CSV row 1 <-> embedding row 1
CSV row 2 <-> embedding row 2
```

Regenerate the embeddings whenever the cleaned review data or embedding model changes.

## Clustering Outputs

The clustering script creates:

```text
data/processed/clustered_yelp_reviews.csv
data/processed/cluster_examples.csv
```

### `clustered_yelp_reviews.csv`

Contains every review and its assigned `cluster_id`.

The cluster ID is only a numerical group label. It does not represent review quality, sentiment, star rating, or importance.

### `cluster_examples.csv`

Contains reviews closest to the center of each cluster.

The `rank` column represents how close an example is to the cluster center:

```text
rank 1 = closest to the center
rank 2 = second closest
rank 3 = third closest
```

These examples are manually reviewed to determine what each cluster represents.

## Current Cluster Labels

The current 12-cluster baseline uses the following labels:

```python
CLUSTER_LABELS = {
    0: "Poor dine-in service and delayed food",
    1: "Pizza restaurant experiences",
    2: "Good food with inconsistent service",
    3: "Detailed food and menu evaluation",
    4: "Excellent overall dining experience",
    5: "Chinese and Thai food experiences",
    6: "Mexican restaurant experiences",
    7: "Casual American food and sandwiches",
    8: "Cafes, bakeries, and fresh food",
    9: "Extreme wait times and unfulfilled orders",
    10: "Bars, drinks, and atmosphere",
    11: "Sushi and Japanese restaurants",
}
```

The labeled reviews are saved in:

```text
data/processed/labeled_yelp_reviews.csv
```

## Restaurant Theme Summary

The final baseline output is:

```text
data/processed/restaurant_theme_summary.csv
```

This file contains:

```text
business_id
name
cluster_id
cluster_label
review_count
average_stars
total_reviews
theme_percentage
```

Each row represents one theme found in the sampled reviews for a restaurant.

Because each restaurant has approximately 10 sampled reviews, the results should be treated as sample-based insights rather than definitive business evaluations.

## Current Limitation

The model clusters complete reviews.

For example:

```text
"The service was slow, but the food was excellent."
```

This review contains both negative service feedback and positive food feedback, but K-Means assigns the entire review to only one cluster.

Star ratings are not used during clustering. The clusters are created using only the semantic review embeddings.

## Next Experiment

The next experiment will compare:

```text
12 clusters
16 clusters
20 clusters
```

The goal is to determine whether broader clusters can be separated into clearer themes such as:

* Slow service
* Rude or inattentive staff
* Delayed food
* Incorrect orders
* Food quality
* Price and value
* Atmosphere

More clusters are not automatically better. The best result should produce themes that are clear, distinct, and useful.

## Future Improvements

* Compare different cluster counts
* Test stronger embedding models
* Add sentiment analysis
* Add sentence-level clustering
* Add semantic search
* Store embeddings in a vector database
* Generate evidence-backed recommendations with OpenAI
* Build an interactive Streamlit dashboard
