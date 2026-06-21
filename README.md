# DeepDine: AI-Powered Restaurant Review Analysis

DeepDine is an NLP project that turns restaurant reviews into structured customer-feedback insights.

The project analyzes one restaurant at a time by cleaning review data, generating sentence embeddings, classifying each review into predefined themes, and generating evidence-backed recommendations.

## Project Goal

DeepDine helps restaurants quickly identify recurring strengths, complaints, and improvement areas from customer reviews without manually reading every review.

## Workflow

```text
Input restaurant reviews
→ Clean review data
→ Generate review embeddings
→ Classify reviews by theme
→ Create analysis-ready CSV files
→ Generate evidence-backed recommendations
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
| Food Quality    |           13 |     43.33% |
| Order Accuracy  |            7 |     23.33% |
| Price and Value |            5 |     16.67% |
| Atmosphere      |            4 |     13.33% |
| Cleanliness     |            1 |      3.33% |

### Reviews by Theme and Rating Group

| Primary Theme   | Negative | Positive |
| --------------- | -------: | -------: |
| Food Quality    |        2 |       11 |
| Order Accuracy  |        6 |        1 |
| Price and Value |        4 |        1 |
| Atmosphere      |        1 |        3 |
| Cleanliness     |        1 |        0 |

These results show that Food Quality was the strongest positive pattern, with 11 positive reviews. Order Accuracy and Price and Value were the top improvement areas, with most reviews in those themes coming from negative customer feedback.

## Automatic Recommendations

DeepDine generates recommendation output based on the strongest positive and negative theme patterns.

### Top Strength

```text
Food Quality: 11 positive reviews
Recommendation: Food Quality is a recurring strength. Keep food preparation, taste, freshness, and portion consistency high.
```

### Top Improvement Areas

```text
Order Accuracy: 6 negative reviews, 85.71% negative within this theme
Recommendation: Improve order accuracy by checking missing items, wrong orders, substitutions, special requests, sides, sauces, and modifications before orders are completed.

Example review theme match:
Primary theme: Order Accuracy
Secondary theme: Cleanliness
Review: The restaurant was dirty, the table was sticky, and my order was missing one of the sides.
```

```text
Price and Value: 4 negative reviews, 80.00% negative within this theme
Recommendation: Review pricing, portion sizes, discounts, and perceived value to make sure customers feel the meal is worth the price.

Example review theme match:
Primary theme: Price and Value
Secondary theme: Food Quality
Review: The food tasted okay, but the prices were too high for the quality and portion size.
```

```text
Food Quality: 2 negative reviews, 15.38% negative within this theme
Recommendation: Review negative food-related feedback to identify specific issues with taste, freshness, temperature, portions, or preparation consistency.

Example review theme match:
Primary theme: Food Quality
Secondary theme: Atmosphere
Review: The restaurant looked nice, but the food was just average and the service felt rushed.
```

These recommendations are evidence-backed because each improvement area is tied to actual review patterns and a supporting customer review.

## Current Status

The project has been simplified to focus on the core analysis workflow. Extra generated files were removed, and the pipeline now produces the CSV files needed for review analysis while printing recommendation summaries in the terminal.

The recommendation system is currently rule-based and uses classified theme patterns to generate practical restaurant improvement suggestions.

## Future Improvements

Possible next steps include dashboard visualizations, refining the seven theme descriptions to improve classification accuracy, and adding aspect-level sentiment analysis to separate positive and negative themes within the same review.
