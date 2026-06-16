from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = (
    DATA_DIR / "processed" / "themed_yelp_reviews.csv"
)

OUTPUT_FILE = (
    DATA_DIR / "processed" / "restaurant_theme_summary.csv"
)


def validate_reviews(reviews: pd.DataFrame) -> None:
    """Validate that the themed review file has the required columns."""

    required_columns = {
        "business_id",
        "name",
        "stars",
        "text",
        "theme",
        "theme_score",
        "review_group",
    }

    missing_columns = required_columns - set(reviews.columns)

    if missing_columns:
        raise ValueError(
            "The themed review file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if reviews.empty:
        raise ValueError(
            "The themed review file does not contain any reviews."
        )


def classify_evidence_strength(
    review_count: int,
    theme_percentage: float,
) -> str:
    """Describe how strongly the reviews support a theme."""

    if review_count >= 10 and theme_percentage >= 15:
        return "Strong recurring pattern"

    if review_count >= 5 and theme_percentage >= 10:
        return "Recurring pattern"

    if review_count >= 3:
        return "Emerging pattern"

    return "Limited evidence"


def create_restaurant_summary(
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    """Create theme summaries for each restaurant and rating group."""

    restaurant_summary = (
        reviews.groupby(
            [
                "business_id",
                "name",
                "review_group",
                "theme",
            ],
            as_index=False,
        )
        .agg(
            review_count=("text", "count"),
            average_stars=("stars", "mean"),
            average_theme_score=("theme_score", "mean"),
        )
    )

    # Count the number of positive or negative reviews available
    # for each restaurant.
    group_totals = (
        reviews.groupby(
            [
                "business_id",
                "review_group",
            ],
            as_index=False,
        )
        .agg(
            total_group_reviews=("text", "count"),
        )
    )

    restaurant_summary = restaurant_summary.merge(
        group_totals,
        on=[
            "business_id",
            "review_group",
        ],
        how="left",
    )

    # Calculate the percentage within the restaurant's positive
    # or negative review group.
    restaurant_summary["theme_percentage"] = (
        restaurant_summary["review_count"]
        / restaurant_summary["total_group_reviews"]
        * 100
    )

    restaurant_summary["evidence_strength"] = (
        restaurant_summary.apply(
            lambda row: classify_evidence_strength(
                review_count=int(row["review_count"]),
                theme_percentage=float(
                    row["theme_percentage"]
                ),
            ),
            axis=1,
        )
    )

    restaurant_summary["average_stars"] = (
        restaurant_summary["average_stars"].round(2)
    )

    restaurant_summary["average_theme_score"] = (
        restaurant_summary["average_theme_score"].round(4)
    )

    restaurant_summary["theme_percentage"] = (
        restaurant_summary["theme_percentage"].round(2)
    )

    restaurant_summary = restaurant_summary.sort_values(
        by=[
            "business_id",
            "review_group",
            "review_count",
            "average_theme_score",
        ],
        ascending=[
            True,
            True,
            False,
            False,
        ],
    ).reset_index(drop=True)

    return restaurant_summary


def print_summary_results(
    restaurant_summary: pd.DataFrame,
) -> None:
    """Display basic information about the generated summary."""

    print("\nRestaurant theme summary complete.")

    print(
        f"Restaurants summarized: "
        f"{restaurant_summary['business_id'].nunique():,}"
    )

    print(
        f"Summary rows created: "
        f"{len(restaurant_summary):,}"
    )

    print("\nThemes represented:")

    print(
        restaurant_summary["theme"]
        .value_counts()
        .to_string()
    )

    print("\nPatterns by evidence strength:")

    print(
        restaurant_summary["evidence_strength"]
        .value_counts()
        .to_string()
    )


def main() -> None:
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    reviews = pd.read_csv(
        INPUT_FILE,
    )

    validate_reviews(
        reviews,
    )

    print(f"Reviews loaded: {len(reviews):,}")

    print(
        f"Restaurants loaded: "
        f"{reviews['business_id'].nunique():,}"
    )

    restaurant_summary = create_restaurant_summary(
        reviews,
    )

    restaurant_summary.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print_summary_results(
        restaurant_summary,
    )

    print(
        f"\nRestaurant summary saved to: "
        f"{OUTPUT_FILE}"
    )

if __name__ == "__main__":
    main()