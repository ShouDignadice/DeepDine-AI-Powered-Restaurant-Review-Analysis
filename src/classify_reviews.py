from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

CLEANED_REVIEWS_FILE = DATA_DIR / "processed" / "cleaned_restaurant_reviews.csv"
SORTED_THEMED_REVIEWS_FILE = DATA_DIR / "processed" / "sorted_themed_restaurant_reviews.csv"
FINAL_ANALYSIS_FILE = DATA_DIR / "processed" / "final_analysis.csv"

MODEL_NAME = "all-MiniLM-L6-v2"

MIN_THEME_SCORE = 0.30
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
    """Validate that the cleaned reviews and embeddings match."""

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
            "The cleaned review data is missing required columns: "
            f"{sorted(missing_columns)}. Run clean_restaurant_reviews.py first."
        )

    if reviews.empty:
        raise ValueError(
            "The cleaned review data does not contain any reviews."
        )

    if len(reviews) != len(embeddings):
        raise ValueError(
            "The number of cleaned reviews does not match the number of "
            "review embeddings. Re-run the full pipeline from main.py."
        )

    if embeddings.ndim != 2:
        raise ValueError(
            "The review embeddings must be a two-dimensional array."
        )


def classify_reviews(
    reviews: pd.DataFrame,
    review_embeddings: np.ndarray,
    model_name: str = MODEL_NAME,
) -> pd.DataFrame:
    """Assign each review to the closest predefined primary and secondary theme."""

    validate_data(
        reviews=reviews,
        embeddings=review_embeddings,
    )

    print(f"Loading theme embedding model: {model_name}")

    model = SentenceTransformer(model_name)

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
            f"dimensions. Confirm that both steps use '{model_name}'."
        )

    similarity_scores = review_embeddings @ theme_embeddings.T

    ranked_theme_indices = np.argsort(
        similarity_scores,
        axis=1,
    )[:, ::-1]

    primary_indices = ranked_theme_indices[:, 0]
    secondary_indices = ranked_theme_indices[:, 1]

    row_indices = np.arange(len(reviews))

    primary_scores = similarity_scores[
        row_indices,
        primary_indices,
    ]

    secondary_scores = similarity_scores[
        row_indices,
        secondary_indices,
    ]

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
    classified_reviews["secondary_theme_score"] = secondary_scores
    classified_reviews["score_margin"] = primary_scores - secondary_scores

    classified_reviews["is_uncertain"] = (
        (classified_reviews["theme_score"] < MIN_THEME_SCORE)
        | (classified_reviews["score_margin"] < MIN_SCORE_MARGIN)
    )

    classified_reviews["theme_note"] = np.where(
        classified_reviews["is_uncertain"],
        "Review also closely matched the secondary theme",
        "Clear primary theme",
    )

    classified_reviews["review_group"] = np.where(
        classified_reviews["stars"] <= 3,
        "Negative",
        "Positive",
    )

    score_columns = [
        "theme_score",
        "secondary_theme_score",
        "score_margin",
    ]

    classified_reviews[score_columns] = (
        classified_reviews[score_columns].round(4)
    )

    return classified_reviews


def sort_classified_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Sort reviews so negative feedback and related themes are easier to inspect."""

    group_order = {
        "Negative": 0,
        "Positive": 1,
    }

    sorted_reviews = reviews.copy()
    sorted_reviews["review_group_order"] = (
        sorted_reviews["review_group"].map(group_order).fillna(2)
    )

    sorted_reviews = sorted_reviews.sort_values(
        by=[
            "business_id",
            "review_group_order",
            "primary_theme",
            "stars",
            "theme_score",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            False,
        ],
    ).drop(columns=["review_group_order"])

    return sorted_reviews.reset_index(drop=True)


def create_final_analysis_file(reviews: pd.DataFrame) -> pd.DataFrame:
    """Create the simplified CSV used for manual restaurant review analysis."""

    final_analysis = reviews.rename(
        columns={
            "name": "business_name",
        }
    ).copy()

    analysis_columns = [
        "business_id",
        "business_name",
        "review_id",
        "stars",
        "review_group",
        "primary_theme",
        "secondary_theme",
        "text",
    ]

    missing_columns = [
        column
        for column in analysis_columns
        if column not in final_analysis.columns
    ]

    if missing_columns:
        raise ValueError(
            "Cannot create final_analysis.csv because these columns are missing: "
            f"{missing_columns}"
        )

    final_analysis = final_analysis[analysis_columns]

    final_analysis = final_analysis.sort_values(
        by=[
            "business_id",
            "review_group",
            "primary_theme",
            "stars",
            "review_id",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)

    return final_analysis


def save_classification_outputs(
    sorted_reviews: pd.DataFrame,
    final_analysis: pd.DataFrame,
    sorted_output_file: Path = SORTED_THEMED_REVIEWS_FILE,
    final_analysis_file: Path = FINAL_ANALYSIS_FILE,
) -> None:
    """Save only the two classification analysis CSV outputs."""

    sorted_output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sorted_reviews.to_csv(
        sorted_output_file,
        index=False,
    )

    final_analysis.to_csv(
        final_analysis_file,
        index=False,
    )

def print_theme_summary(reviews: pd.DataFrame) -> None:
    """Print a short terminal summary without creating another CSV."""

    print("\nReviews per theme:")

    theme_counts = (
        reviews["primary_theme"]
        .value_counts()
        .rename_axis("primary_theme")
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

    print("\nReviews by theme and rating group:")

    grouped_counts = pd.crosstab(
        reviews["primary_theme"],
        reviews["review_group"],
    )

    print(grouped_counts)
