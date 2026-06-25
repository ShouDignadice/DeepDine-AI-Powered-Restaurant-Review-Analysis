# DeepDine: AI-Powered Restaurant Review Analysis

DeepDine is an NLP-powered restaurant review analysis project that transforms customer reviews into structured business insights. The project analyzes one restaurant at a time and helps identify recurring customer themes, aspect-level sentiment patterns, strengths, issue areas, and actionable recommendations.

Instead of manually reading every customer review, DeepDine turns messy review text into a cleaner analysis workflow that can support restaurant decision-making.

---

## Project Overview

Restaurant reviews are difficult to analyze because one review can contain several different opinions at the same time. A customer may praise the food, complain about the price, mention a slow service experience, and still leave either a positive or negative star rating overall.

Because of this, looking only at the full review rating does not always explain what customers actually liked or disliked.

DeepDine solves this by splitting full customer reviews into smaller aspect-level review mentions, classifying each mention into a business theme, assigning sentiment to the actual text, and displaying the findings in a Streamlit dashboard.

The project separates:

| Term             | Meaning                                                           |
| ---------------- | ----------------------------------------------------------------- |
| Full Review      | The original customer review, counted once using `review_id`      |
| Aspect Mention   | A smaller extracted part of a review focused on one topic         |
| Review Group     | Positive or Negative group based on the full review star rating   |
| Aspect Sentiment | Positive or Negative sentiment based on the actual extracted text |

This distinction is important because a negative full review can still contain a positive comment, and a positive full review can still contain a complaint.

---

## Problem Statement

Restaurants receive large amounts of unstructured customer feedback. Manually reading reviews is time-consuming, inconsistent, and difficult to summarize into clear business actions.

DeepDine addresses this problem by creating an NLP workflow that:

* Cleans restaurant review data
* Splits reviews into focused aspect-level text
* Classifies review aspects into operational themes
* Separates full review rating groups from actual text sentiment
* Generates structured analysis outputs
* Displays findings in a business-facing dashboard
* Produces automatic recommendations based on recurring customer feedback

---

## Core Features

### Review Cleaning and Processing

The project prepares restaurant review data for analysis by cleaning review text, organizing review information, and creating structured analysis-ready data.

### Aspect-Level Review Splitting

Full reviews are split into smaller review aspects so that each part of a review can be analyzed more accurately.

Example:

```text
Full Review:
"The food was great, but the service was slow and the prices were too high."

Aspect Mentions:
1. "The food was great"
2. "the service was slow"
3. "the prices were too high"
```

This allows the project to classify each part separately instead of forcing the entire review into one theme.

### Theme Classification

DeepDine classifies aspect-level review text into restaurant business themes such as:

* Food Quality
* Order Accuracy
* Price and Value
* Atmosphere
* Cleanliness
* Service, if included in the dataset/theme configuration

The classification process uses Sentence-Transformer embeddings and cosine similarity to compare review text against theme definitions.

### Sentiment Separation

DeepDine separates star-based review grouping from actual text sentiment.

For example:

```text
Review Group:
Negative, based on a 2-star rating

Aspect Sentiment:
Positive, if the extracted text says something good
```

This makes the analysis more accurate because customers often include both positive and negative feedback in the same review.

### Automatic Recommendations

The dashboard generates recommendations based on aspect-level sentiment patterns.

For example:

```text
Top aspect-level strength:
Food Quality

Recommendation:
Food quality is a recurring strength. Keep taste, freshness, preparation, temperature, and portion consistency high.
```

For improvement areas:

```text
Top aspect-level issue:
Price and Value

Recommendation:
Review whether customers feel the portion size, food quality, and overall experience match the price.
```

---

## Dashboard

DeepDine includes a Streamlit dashboard that presents the findings in a business-friendly format.

The dashboard includes:

* Overview metrics
* Full review counts
* Aspect mention counts
* Average rating
* Positive and negative aspect sentiment counts
* Full reviews by theme
* Aspect sentiment by theme
* Automatic recommendations
* Theme summary table
* Aspect-level review explorer
* Side-by-side full review and extracted aspect review comparison

The dashboard makes a clear distinction between full customer reviews and aspect-level review mentions.

---

## Story Behind the Findings

One of the most important findings from this project is that restaurant reviews are rarely simple. A review is not always fully positive or fully negative. Many reviews contain both praise and complaints in the same text.

For example, a customer might say the food tasted great but the order was wrong, or they might leave a low star rating while still mentioning that the atmosphere was good. This makes customer feedback difficult to analyze using only star ratings or full-review sentiment.

DeepDine helps solve this by breaking reviews into aspect-level mentions. Instead of asking only whether the full review is positive or negative, the system asks:

```text
What specific part of the restaurant experience is the customer talking about?
Is that specific part positive or negative?
```

This allows the dashboard to show a more realistic story.

In the sample analysis, Food Quality appeared as one of the restaurant’s strongest positive areas, but it also appeared as one of the highest negative issue areas. This means customers often praised the food, but food-related complaints were also a major part of the negative feedback.

That finding tells a stronger business story than simply saying “customers like the food” or “customers dislike the food.”

A better interpretation is:

```text
Food Quality is a high-impact area.
Customers often praise it, but negative food-related feedback suggests consistency is a problem.
The restaurant should protect what customers already like while improving consistency in taste, freshness, preparation, temperature, and portion quality.
```

This is where the project becomes useful from a business perspective. The system does not claim to perfectly judge the restaurant. Instead, it gives a general direction by showing where customer feedback is concentrated, what the restaurant is doing well, and which areas need improvement.

It is also important to note that the system is not perfect. Since full reviews are split into smaller aspect mentions, some context can be lost during the splitting process. This means the dashboard may sometimes produce false sentiment labels or classify an aspect into the wrong theme. However, the value of the system is not in judging every sentence perfectly. Its value is in revealing general patterns across many reviews, helping identify what customers consistently praise and what areas repeatedly create problems.

---

## Example Findings

In the sample analysis, Food Quality appeared as both a major strength and a major issue area. This shows that customers frequently talk about food, but their experiences are mixed depending on taste, freshness, preparation, temperature, or consistency.

This type of finding is useful because it does not simply say whether the restaurant is good or bad. Instead, it shows where customer feedback is concentrated and what the restaurant should focus on improving.

Example business interpretation:

```text
Customers frequently mention Food Quality.
Many customers praise it, but many also complain about it.
This suggests food quality is a high-impact area where consistency matters.
```

Another example is Order Accuracy. If negative aspect mentions frequently appear under Order Accuracy, the restaurant can treat this as an operational issue rather than a general customer satisfaction problem.

Example business interpretation:

```text
Customers are not only complaining about the experience overall.
They are repeatedly pointing to missing items, incorrect orders, sauces, sides, or special requests.
This suggests the restaurant should improve order-checking before completion.
```

---

## Key Design Decisions

### One Restaurant at a Time

DeepDine is designed to analyze one restaurant at a time. This keeps the findings focused and makes the recommendations more useful for a specific business.

### Aspect-Level Analysis

The project uses aspect-level splitting because full reviews often contain multiple topics. This improves the quality of theme classification and sentiment interpretation.

### Final Analysis vs. Detailed Analysis

The project separates final business-facing results from detailed model-related output.

* `final_analysis.csv` is cleaner and used for dashboard insights.
* `detailed_restaurant_analysis.csv` keeps deeper analysis details.

### Review Group vs. Sentiment

The dashboard separates star-based review grouping from actual text sentiment.

This prevents misleading cases where a negative review contains a positive comment or a positive review contains a complaint.

---

## Limitations

DeepDine is a rule-guided and embedding-based NLP project, so the results depend on the quality of the review text, the aspect-splitting process, the sentiment logic, and the theme definitions.

The system is not perfect and may produce incorrect themes or false sentiment labels. This can happen because full reviews are split into smaller aspect-level mentions before classification. When a sentence is separated from the full review, some context may be lost. For example, sarcasm, comparison, mixed wording, or references to earlier parts of the review can make the extracted aspect harder to interpret correctly.

Because of this, the dashboard should be treated as a decision-support tool, not a perfect replacement for human judgment. Its purpose is to show general patterns, recurring strengths, and likely improvement areas so restaurant owners or analysts know where to look more closely.

Current limitations include:

* Aspect splitting may not always separate review topics perfectly.
* Some aspect mentions may lose context from the original full review.
* Sentiment labels may be incorrect when the text is sarcastic, mixed, or context-dependent.
* Theme classification can mislabel ambiguous review text.
* Results depend on the quality and size of the review dataset.
* The project currently focuses on one restaurant at a time.

---

## Future Improvements

Possible future improvements include:

* Add a manually labeled training set for stronger validation
* Improve aspect-level splitting with more advanced NLP techniques
* Add neutral sentiment handling
* Add confidence thresholds for classification review
* Deploy the Streamlit dashboard online
* Add more detailed theme definitions
* Compare multiple restaurants after the single-restaurant workflow is finalized
* Add downloadable dashboard reports

---

## Project Value

DeepDine demonstrates how NLP can turn unstructured customer feedback into structured business insights.

The project combines data cleaning, NLP embeddings, cosine similarity, aspect-level analysis, sentiment interpretation, CSV reporting, and dashboard presentation to help restaurants understand what customers are saying and where improvements should be made.

The final result is not just a model output. It is a complete analytics workflow that moves from raw reviews to business-facing recommendations.

Most importantly, DeepDine shows how data can tell a business story. It reveals that restaurants can have clear strengths and weaknesses in the same area. For example, food quality can be the most praised theme while also being the largest source of negative feedback. That does not mean the food is simply good or bad. It means food quality is important to customers, and consistency should be a major focus.
