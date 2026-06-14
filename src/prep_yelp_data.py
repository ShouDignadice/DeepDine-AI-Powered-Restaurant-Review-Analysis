from collections import defaultdict
from pathlib import Path
import json
import random

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

BUSINESSES_FILE = (
    DATA_DIR / "raw" / "yelp_academic_dataset_business.json"
)

REVIEWS_FILE = (
    DATA_DIR / "raw" / "yelp_academic_dataset_review.json"
)

OUTPUT_FILE = (
    DATA_DIR / "processed" / "yelp_restaurant_reviews.csv"
)

NUMBER_OF_RESTAURANTS = 2_000
REVIEWS_PER_RESTAURANT = 10
RANDOM_SEED = 42


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    random_generator = random.Random(RANDOM_SEED)

    businesses = pd.read_json(
        BUSINESSES_FILE,
        lines=True,
    )[["business_id", "name", "categories"]]

    restaurants = businesses.loc[
        businesses["categories"].str.contains(
            "Restaurants",
            case=False,
            na=False,
        ),
        ["business_id", "name"],
    ].drop_duplicates(subset="business_id")

    restaurant_names = dict(
        zip(
            restaurants["business_id"],
            restaurants["name"],
        )
    )

    restaurant_ids = set(restaurant_names)
    review_counts = defaultdict(int)

    # Count how many valid reviews each restaurant has.
    with REVIEWS_FILE.open("r", encoding="utf-8") as review_file:
        for line in review_file:
            review = json.loads(line)

            business_id = review.get("business_id")
            text = review.get("text")
            stars = review.get("stars")

            if business_id not in restaurant_ids:
                continue

            if not text or stars is None:
                continue

            review_counts[business_id] += 1

    qualifying_restaurants = [
        business_id
        for business_id, count in review_counts.items()
        if count >= REVIEWS_PER_RESTAURANT
    ]

    number_to_select = min(
        NUMBER_OF_RESTAURANTS,
        len(qualifying_restaurants),
    )

    selected_restaurants = set(
        random_generator.sample(
            qualifying_restaurants,
            number_to_select,
        )
    )

    selected_reviews = defaultdict(list)
    processed_counts = defaultdict(int)

    # Randomly select 10 reviews from each chosen restaurant.
    with REVIEWS_FILE.open("r", encoding="utf-8") as review_file:
        for line in review_file:
            review = json.loads(line)

            business_id = review.get("business_id")

            if business_id not in selected_restaurants:
                continue

            text = review.get("text")
            stars = review.get("stars")

            if not text or stars is None:
                continue

            processed_counts[business_id] += 1

            review_record = {
                "business_id": business_id,
                "name": restaurant_names[business_id],
                "stars": stars,
                "text": text,
            }

            restaurant_reviews = selected_reviews[business_id]

            if len(restaurant_reviews) < REVIEWS_PER_RESTAURANT:
                restaurant_reviews.append(review_record)
            else:
                random_index = random_generator.randint(
                    1,
                    processed_counts[business_id],
                )

                if random_index <= REVIEWS_PER_RESTAURANT:
                    restaurant_reviews[random_index - 1] = review_record

    records = [
        review
        for restaurant_reviews in selected_reviews.values()
        for review in restaurant_reviews
    ]

    restaurant_reviews = pd.DataFrame(
        records,
        columns=["business_id", "name", "stars", "text"],
    )

    restaurant_reviews = restaurant_reviews.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    restaurant_reviews.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Restaurant reviews saved to: {OUTPUT_FILE}")
    print(f"Total reviews: {len(restaurant_reviews):,}")
    print(
        "Total restaurants represented: "
        f"{restaurant_reviews['business_id'].nunique():,}"
    )
    print(
        f"Reviews per restaurant: {REVIEWS_PER_RESTAURANT}"
    )

if __name__ == "__main__":
    main()