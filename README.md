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

## Theme Classification

DeepDine uses SentenceTransformer embeddings to compare each review against predefined theme descriptions.

Each review receives:

```text
primary_theme
secondary_theme
```

The primary theme is the strongest match. The secondary theme helps capture reviews that mention more than one topic.

## Current Results

In the current test run, DeepDine analyzed 30 restaurant reviews and classified them into recurring customer-feedback themes.

### Reviews per Theme

| Primary Theme   | Review Count | Percentage |
| --------------- | -----------: | ---------: |
| Food Quality    |           14 |     46.67% |
| Wait Time       |            5 |     16.67% |
| Atmosphere      |            4 |     13.33% |
| Cleanliness     |            4 |     13.33% |
| Price and Value |            2 |      6.67% |
| Order Accuracy  |            1 |      3.33% |

### Reviews by Theme and Rating Group

| Primary Theme   | Negative | Positive |
| --------------- | -------: | -------: |
| Food Quality    |        5 |        9 |
| Wait Time       |        3 |        2 |
| Atmosphere      |        1 |        3 |
| Cleanliness     |        3 |        1 |
| Price and Value |        1 |        1 |
| Order Accuracy  |        1 |        0 |

These results show that Food Quality was the most common theme, appearing in 46.67% of the analyzed reviews. Food Quality was also the strongest positive pattern, while Cleanliness and Wait Time appeared more often in negative reviews.

### Example Output Structure

A sample from the final analysis output includes actual classified review findings:

| business_id  | business_name | review_id | stars | review_group | primary_theme | secondary_theme | text                                                                                           |
| ------------ | ------------- | --------: | ----: | ------------ | ------------- | --------------- | ---------------------------------------------------------------------------------------------- |
| restaurant_1 | Sample Bistro |        28 |     3 | Negative     | Atmosphere    | Wait Time       | The food was good, but the restaurant was crowded and it was hard to hear people at our table. |
| restaurant_1 | Sample Bistro |         6 |     1 | Negative     | Cleanliness   | Order Accuracy  | The restaurant was dirty, the table was sticky, and my order was missing one of the sides.     |

These examples show how DeepDine keeps the original review text while adding structured analysis fields such as review group, primary theme, and secondary theme.

## Current Status

The project has been simplified to focus on the core analysis workflow. Extra generated files were removed, and the pipeline now only produces the CSV files needed for review analysis.

## Future Improvements

Possible next steps include dashboard visualizations, automatic recommendations, stronger theme definitions, sentiment summaries by theme, and support for comparing multiple restaurants.
