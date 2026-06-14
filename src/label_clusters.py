from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = (
    DATA_DIR / "processed" / "clustered_yelp_reviews.csv"
)

OUTPUT_FILE = (
    DATA_DIR / "processed" / "labeled_yelp_reviews.csv"
)


CLUSTER_LABELS = {
    0: "Poor service and delayed food",
    1: "Pizza restaurant experiences",
    2: "Good food with inconsistent service",
    3: "Detailed food and menu feedback",
    4: "Excellent overall dining experience",
    5: "Chinese and Thai food experiences",
    6: "Mexican restaurant experiences",
    7: "Casual American food and sandwiches",
    8: "Cafes, bakeries, and fresh food",
    9: "Extreme wait times and unfulfilled orders",
    10: "Bars, drinks, and atmosphere",
    11: "Sushi and Japanese restaurant experiences",
}


def main() -> None:
    reviews = pd.read_csv(INPUT_FILE)

    reviews["cluster_label"] = reviews["cluster_id"].map(
        CLUSTER_LABELS
    )

    missing_labels = reviews.loc[
        reviews["cluster_label"].isna(),
        "cluster_id",
    ].unique()

    if len(missing_labels) > 0:
        raise ValueError(
            f"Missing labels for clusters: {missing_labels}"
        )

    reviews.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        reviews["cluster_label"]
        .value_counts()
    )

if __name__ == "__main__":
    main()