# DeepDine: AI-Powered Restaurant Review Analysis

DeepDine is an NLP project that analyzes Yelp restaurant reviews to identify recurring customer experiences and create restaurant-level theme summaries.

## Process

The project uses approximately **20,000 reviews from 2,000 restaurants**, with up to 10 reviews sampled per restaurant.

The analysis follows this process:

1. Filter the Yelp dataset to restaurant businesses.
2. Sample reviews across different restaurants.
3. Remove missing, duplicate, and extremely short reviews.
4. Generate semantic embeddings using `all-MiniLM-L6-v2`.
5. Group similar reviews using K-Means clustering.
6. Review representative examples from each cluster.
7. Assign readable labels to the discovered themes.
8. Summarize theme frequency and average ratings for each restaurant.

Run the scripts from the project root in this order:

```bash
python src/prep_yelp_data.py
python src/clean_yelp_reviews.py
python src/embed_reviews.py
python src/cluster_reviews.py
python src/label_clusters.py
python src/summarize_restaurants.py
```

## Current Outcome

Each review is converted into a **384-dimensional embedding** and assigned to one of **12 clusters**.

The current clustering baseline identified themes related to:

* Poor service and delayed food
* Extreme wait times and unfulfilled orders
* Excellent overall dining experiences
* Good food with inconsistent service
* Bars, drinks, and atmosphere
* Pizza, Mexican, Chinese, Thai, sushi, café, and bakery experiences

The final restaurant summary contains:

```text
business_id
name
cluster_label
review_count
average_stars
total_reviews
theme_percentage
```

This allows the project to show which themes appear most often for each restaurant and how customers rated reviews associated with those themes.

## Current Limitation

The model clusters complete reviews.

A review such as:

```text
"The service was slow, but the food was excellent."
```

contains both negative and positive feedback, but K-Means assigns the full review to only one cluster.

The current results should therefore be treated as a baseline and as sample-based insights rather than definitive restaurant evaluations.

## Future Improvements

Planned improvements include:

* Compare 12, 16, and 20 clusters
* Test stronger embedding models
* Separate topic detection from sentiment analysis
* Split multi-topic reviews into shorter segments
* Add semantic search and a vector database
* Use OpenAI to generate evidence-backed recommendations
* Build an interactive Streamlit dashboard
