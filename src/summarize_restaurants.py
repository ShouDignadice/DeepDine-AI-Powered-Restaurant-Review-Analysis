from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = (
    DATA_DIR / "processed" / "labeled_yelp_reviews.csv"
)

OUTPUT_FILE = (
    DATA_DIR / "processed" / "restaurant_theme_summary.csv"
)


def main() -> None:
    reviews = pd.read_csv(INPUT_FILE)

    restaurant_summary = (
        reviews.groupby(
            [
                "business_id",
                "name",
                "cluster_id",
                "cluster_label",
            ],
            as_index=False,
        )
        .agg(
            review_count=("text", "count"),
            average_stars=("stars", "mean"),
        )
    )

    restaurant_totals = (
        reviews.groupby("business_id")["text"]
        .count()
        .rename("total_reviews")
    )

    restaurant_summary = restaurant_summary.merge(
        restaurant_totals,
        on="business_id",
        how="left",
    )

    restaurant_summary["theme_percentage"] = (
        restaurant_summary["review_count"]
        / restaurant_summary["total_reviews"]
        * 100
    ).round(2)

    restaurant_summary["average_stars"] = (
        restaurant_summary["average_stars"].round(2)
    )

    restaurant_summary = restaurant_summary.sort_values(
        by=[
            "business_id",
            "review_count",
            "average_stars",
        ],
        ascending=[True, False, True],
    )

    restaurant_summary.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Restaurant summary saved to: {OUTPUT_FILE}")
    print(f"Summary rows: {len(restaurant_summary):,}")


if __name__ == "__main__":
    main()