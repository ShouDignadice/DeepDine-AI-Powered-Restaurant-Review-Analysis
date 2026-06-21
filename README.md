# DeepDine: AI-Powered Restaurant Review Analysis

DeepDine is an NLP project that turns restaurant reviews into structured customer-feedback insights.

The project analyzes one restaurant at a time by cleaning review data, generating sentence embeddings, and classifying each review into predefined themes such as Food Quality, Service, Wait Time, Cleanliness, Price and Value, Order Accuracy, and Atmosphere.

## Project Goal

DeepDine helps restaurants quickly identify recurring strengths, complaints, and improvement areas from customer reviews without manually reading every review.

## Workflow

```text
Input restaurant reviews
→ Clean review data
→ Generate review embeddings
→ Classify reviews by theme
→ Create analysis-ready CSV files
```

Run the full pipeline with:

```bash
python src/main.py
```

Or pass a specific CSV file:

```bash
python src/main.py data/input/restaurant_reviews.csv
```

## Theme Classification

DeepDine uses SentenceTransformer embeddings to compare each review against predefined theme descriptions.

Each review receives:

```text
primary_theme
secondary_theme
```

The primary theme is the strongest match. The secondary theme helps capture reviews that mention more than one topic.

## Current Status

The project has been simplified to focus on the core analysis workflow. Extra generated files were removed, and the pipeline now only produces the CSV files needed for review analysis.

## Future Improvements

Possible next steps include dashboard visualizations, automatic recommendations, stronger theme definitions, sentiment summaries by theme, and support for comparing multiple restaurants.
