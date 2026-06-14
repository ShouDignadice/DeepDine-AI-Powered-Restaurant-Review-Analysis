from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = (
    DATA_DIR / "processed" / "yelp_restaurant_reviews.csv"
)

OUTPUT_FILE = (
    DATA_DIR / "processed" / "cleaned_yelp_reviews.csv"
)

MIN_WORDS = 10


def main() -> None:
    reviews = pd.read_csv(INPUT_FILE)

    original_count = len(reviews)

    # Remove rows missing important information.
    reviews = reviews.dropna(
        subset=["business_id", "name", "stars", "text"]
    )

    # Normalize whitespace without aggressively altering the review.
    reviews["text"] = (
        reviews["text"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Remove duplicate review text.
    reviews = reviews.drop_duplicates(
        subset=["text"]
    )

    # Remove reviews that are too short to provide useful context.
    word_counts = reviews["text"].str.split().str.len()

    reviews = reviews.loc[
        word_counts >= MIN_WORDS
    ].copy()

    # Add a permanent identifier for tracking reviews later.
    reviews.insert(
        0,
        "review_id",
        range(1, len(reviews) + 1),
    )

    reviews.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Original reviews: {original_count:,}")
    print(f"Cleaned reviews: {len(reviews):,}")
    print(
        f"Removed reviews: "
        f"{original_count - len(reviews):,}"
    )
    print(
        f"Restaurants represented: "
        f"{reviews['business_id'].nunique():,}"
    )
    print(f"Cleaned data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()