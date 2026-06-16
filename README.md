## Review Classification

DeepDine uses sentence embeddings to classify restaurant reviews into predefined operational themes. Unlike K-Means clustering, each category has a consistent business meaning across restaurants.

### Predefined Themes

* **Food Quality** — taste, freshness, temperature, preparation, portions, and presentation
* **Service** — friendliness, attentiveness, professionalism, and communication
* **Wait Time** — seating delays, slow service, delayed food, and delivery time
* **Cleanliness** — tables, bathrooms, utensils, dining areas, and hygiene
* **Price and Value** — affordability, portion value, discounts, and unexpected charges
* **Order Accuracy** — incorrect orders, missing items, substitutions, and special requests
* **Atmosphere** — noise, seating, music, comfort, parking, and crowding

Reviews with weak or overlapping theme matches are classified as **Other / Mixed**.

Reviews are also separated by rating:

* **Negative:** 1–3 stars
* **Positive:** 4–5 stars

## Sample Finding

During the initial multi-restaurant test, DeepDine analyzed 99 usable reviews for Oishii Poké.

### Top Positive Recurring Theme

Food Quality was the strongest positive recurring theme, appearing in 56 of the restaurant’s 78 positive reviews, or 71.79%. This indicates that customers frequently associate Oishii Poké with favorable food experiences, making food quality the restaurant’s primary recurring strength.

### Top Negative Recurring Theme

Food Quality was also the most common theme among negative feedback, appearing in 13 of 21 negative reviews, or 61.90%. However, these 13 reviews represent a much smaller portion of the restaurant’s total Food Quality feedback than the 56 positive reviews.

Overall, the results suggest that food is the primary reason customers discuss Oishii Poké and is generally viewed as a major strength. The smaller group of negative Food Quality reviews can be inspected separately to identify specific concerns involving preparation, taste, temperature, portions, or individual menu items.

## Project Direction

The multi-restaurant Yelp dataset is used for initial testing and validation. DeepDine’s primary goal is to analyze one restaurant at a time using approximately 50–100 or more reviews.

For each restaurant, the system will provide:

* Recurring strengths and issues
* Theme counts and percentages
* Classification confidence scores
* Representative supporting reviews
* Evidence-backed improvement recommendations

