from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

REVIEWS_FILE = (
    DATA_DIR / "processed" / "cleaned_yelp_reviews.csv"
)

EMBEDDINGS_FILE = (
    DATA_DIR / "embeddings" / "review_embeddings.npy"
)

THEMED_REVIEWS_FILE = (
    DATA_DIR / "processed" / "themed_yelp_reviews.csv"
)

THEME_EXAMPLES_FILE = (
    DATA_DIR / "processed" / "theme_examples.csv"
)


MODEL_NAME = "all-MiniLM-L6-v2"

EXAMPLES_PER_THEME = 5

# Minimum similarity between a review and its best theme.
MIN_THEME_SCORE = 0.30

# Minimum difference between the best and second-best themes.
MIN_SCORE_MARGIN = 0.03


THEMES = {
    "Food Quality": (
        "Comments about food taste, freshness, temperature, preparation, "
        "ingredients, portion size, presentation, or overall food quality."
    ),
    "Service": (
        "Comments about employee behavior, friendliness, attentiveness, "
        "professionalism, communication, or customer service."
    ),
    "Wait Time": (
        "Comments about slow service, long waits, delayed food, seating delays, "
        "delivery delays, or checkout time."
    ),
    "Cleanliness": (
        "Comments about dirty tables, bathrooms, dining areas, floors, "
        "utensils, hygiene, or restaurant cleanliness."
    ),
    "Price and Value": (
        "Comments about prices, affordability, expensive food, value for money, "
        "portion value, discounts, or unexpected charges."
    ),
    "Order Accuracy": (
        "Comments about incorrect orders, missing items, substitutions, "
        "delivery mistakes, or ignored special requests."
    ),
    "Atmosphere": (
        "Comments about noise, music, seating, decoration, comfort, parking, "
        "crowding, location, or the restaurant environment."
    ),
}


def validate_data(
    reviews: pd.DataFrame,
    embeddings: np.ndarray,
) -> None:
    """Validate that the review data and embeddings match."""

    required_columns = {
        "review_id",
        "business_id",
        "name",
        "stars",
        "text",
    }

    missing_columns = required_columns - set(reviews.columns)

    if missing_columns:
        raise ValueError(
            "The reviews file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if len(reviews) != len(embeddings):
        raise ValueError(
            "The number of reviews does not match the number of embeddings. "
            "Run embed_reviews.py again using the current cleaned review file."
        )

    if embeddings.ndim != 2:
        raise ValueError(
            "The embeddings file must contain a two-dimensional array."
        )

    if len(reviews) == 0:
        raise ValueError(
            "The cleaned reviews file does not contain any reviews."
        )


def classify_reviews(
    reviews: pd.DataFrame,
    review_embeddings: np.ndarray,
) -> pd.DataFrame:
    """Assign each review to the closest predefined theme."""

    print(f"Loading theme embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    theme_names = list(THEMES.keys())
    theme_descriptions = list(THEMES.values())

    theme_embeddings = model.encode(
        theme_descriptions,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if review_embeddings.shape[1] != theme_embeddings.shape[1]:
        raise ValueError(
            "The review embeddings and theme embeddings have different "
            "dimensions. Confirm that embed_reviews.py and classify_reviews.py "
            f"both use the model '{MODEL_NAME}'."
        )

    # The review embeddings and theme embeddings are normalized.
    # Their dot product is therefore equivalent to cosine similarity.
    similarity_scores = review_embeddings @ theme_embeddings.T

    ranked_theme_indices = np.argsort(
        similarity_scores,
        axis=1,
    )

    primary_indices = ranked_theme_indices[:, -1]
    secondary_indices = ranked_theme_indices[:, -2]

    row_indices = np.arange(len(reviews))

    primary_scores = similarity_scores[
        row_indices,
        primary_indices,
    ]

    secondary_scores = similarity_scores[
        row_indices,
        secondary_indices,
    ]

    score_margins = primary_scores - secondary_scores

    classified_reviews = reviews.copy()

    classified_reviews["primary_theme"] = [
        theme_names[index]
        for index in primary_indices
    ]

    classified_reviews["secondary_theme"] = [
        theme_names[index]
        for index in secondary_indices
    ]

    classified_reviews["theme_score"] = primary_scores

    classified_reviews["secondary_theme_score"] = (
        secondary_scores
    )

    classified_reviews["score_margin"] = score_margins

    # Begin with the best-matching theme.
    classified_reviews["theme"] = (
        classified_reviews["primary_theme"]
    )

    # Mark reviews as uncertain when:
    # 1. The best theme score is too low, or
    # 2. The best and second-best themes are too similar.
    classified_reviews["is_uncertain"] = (
        (
            classified_reviews["theme_score"]
            < MIN_THEME_SCORE
        )
        |
        (
            classified_reviews["score_margin"]
            < MIN_SCORE_MARGIN
        )
    )

    classified_reviews.loc[
        classified_reviews["is_uncertain"],
        "theme",
    ] = "Other / Mixed"

    # Separate low-rated issues from high-rated strengths.
    classified_reviews["review_group"] = np.where(
        classified_reviews["stars"] <= 3,
        "Negative",
        "Positive",
    )

    # Round values for cleaner CSV output.
    score_columns = [
        "theme_score",
        "secondary_theme_score",
        "score_margin",
    ]

    classified_reviews[score_columns] = (
        classified_reviews[score_columns].round(4)
    )

    return classified_reviews


def create_theme_examples(
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    """Select representative reviews for manual theme inspection."""

    examples = []

    theme_order = list(THEMES.keys()) + ["Other / Mixed"]

    for theme in theme_order:
        theme_reviews = reviews.loc[
            reviews["theme"] == theme
        ].copy()

        if theme_reviews.empty:
            continue

        if theme == "Other / Mixed":
            # Reviews with the smallest margins are the most mixed.
            theme_reviews = theme_reviews.sort_values(
                by=[
                    "score_margin",
                    "theme_score",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
        else:
            # Higher similarity scores are more representative.
            theme_reviews = theme_reviews.sort_values(
                by="theme_score",
                ascending=False,
            )

        selected_reviews = theme_reviews.head(
            EXAMPLES_PER_THEME
        )

        for rank, (_, review) in enumerate(
            selected_reviews.iterrows(),
            start=1,
        ):
            examples.append(
                {
                    "theme": theme,
                    "rank": rank,
                    "review_id": review["review_id"],
                    "business_id": review["business_id"],
                    "name": review["name"],
                    "stars": review["stars"],
                    "review_group": review["review_group"],
                    "primary_theme": review["primary_theme"],
                    "theme_score": review["theme_score"],
                    "secondary_theme": review[
                        "secondary_theme"
                    ],
                    "secondary_theme_score": review[
                        "secondary_theme_score"
                    ],
                    "score_margin": review["score_margin"],
                    "text": review["text"],
                }
            )

    return pd.DataFrame(examples)


def print_theme_summary(
    reviews: pd.DataFrame,
) -> None:
    """Display the classification results in the terminal."""

    print("\nReviews per theme:")

    theme_counts = (
        reviews["theme"]
        .value_counts()
        .rename_axis("theme")
        .reset_index(name="review_count")
    )

    theme_counts["percentage"] = (
        theme_counts["review_count"]
        / len(reviews)
        * 100
    ).round(2)

    print(
        theme_counts.to_string(
            index=False,
        )
    )

    uncertain_count = int(
        reviews["is_uncertain"].sum()
    )

    uncertain_percentage = (
        uncertain_count
        / len(reviews)
        * 100
    )

    print(
        "\nUncertain or mixed reviews: "
        f"{uncertain_count:,} "
        f"({uncertain_percentage:.2f}%)"
    )

    print("\nReviews by theme and rating group:")

    grouped_counts = pd.crosstab(
        reviews["theme"],
        reviews["review_group"],
    )

    print(grouped_counts)


def main() -> None:
    THEMED_REVIEWS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    reviews = pd.read_csv(
        REVIEWS_FILE,
    )

    review_embeddings = np.load(
        EMBEDDINGS_FILE,
    )

    validate_data(
        reviews=reviews,
        embeddings=review_embeddings,
    )

    print(f"Reviews loaded: {len(reviews):,}")
    print(
        "Embedding shape: "
        f"{review_embeddings.shape}"
    )
    print(
        "Predefined themes: "
        f"{len(THEMES)}"
    )

    classified_reviews = classify_reviews(
        reviews=reviews,
        review_embeddings=review_embeddings,
    )

    theme_examples = create_theme_examples(
        classified_reviews,
    )

    classified_reviews.to_csv(
        THEMED_REVIEWS_FILE,
        index=False,
    )

    theme_examples.to_csv(
        THEME_EXAMPLES_FILE,
        index=False,
    )

    print_theme_summary(
        classified_reviews,
    )

    print("\nTheme classification complete.")

    print(
        "Themed reviews saved to: "
        f"{THEMED_REVIEWS_FILE}"
    )

    print(
        "Theme examples saved to: "
        f"{THEME_EXAMPLES_FILE}"
    )


if __name__ == "__main__":
    main()