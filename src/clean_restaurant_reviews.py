from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = DATA_DIR / "input" / "restaurant_reviews.csv"
OUTPUT_FILE = DATA_DIR / "processed" / "cleaned_restaurant_reviews.csv"

MIN_WORDS = 10
DEFAULT_RESTAURANT_ID = "restaurant_1"

def clean_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "business_name",
        "review_id",
        "review_stars",
        "review_text",
    }

    missing_columns = required_columns - set(reviews.columns)

    if missing_columns:
        raise ValueError(
            "Input CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    reviews = reviews[
        [
            "business_name",
            "review_id",
            "review_stars",
            "review_text",
        ]
    ].copy()

    reviews = reviews.rename(
        columns={
            "business_name": "name",
            "review_stars": "stars",
            "review_text": "text",
        }
    )

    reviews.insert(
        1,
        "business_id",
        DEFAULT_RESTAURANT_ID,
    )

    reviews["review_id"] = (
        reviews["review_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    reviews["name"] = (
        reviews["name"]
        .fillna("Uploaded Restaurant")
        .astype(str)
        .str.strip()
    )

    reviews["stars"] = pd.to_numeric(
        reviews["stars"],
        errors="coerce",
    )

    reviews["text"] = (
        reviews["text"]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    reviews = reviews.dropna(
        subset=[
            "stars",
        ]
    )

    reviews = reviews.loc[
        reviews["review_id"].ne("")
    ].copy()

    reviews = reviews.loc[
        reviews["text"].ne("")
    ].copy()

    reviews = reviews.loc[
        reviews["stars"].between(1, 5)
    ].copy()

    reviews = reviews.drop_duplicates(
        subset=[
            "review_id",
        ]
    )

    word_counts = reviews["text"].str.split().str.len()

    reviews = reviews.loc[
        word_counts >= MIN_WORDS
    ].copy()

    reviews["stars"] = reviews["stars"].round(1)

    reviews = reviews[
        [
            "review_id",
            "business_id",
            "name",
            "stars",
            "text",
        ]
    ]

    return reviews.reset_index(drop=True)

def save_cleaned_reviews(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:

    if not input_file.exists():
        raise FileNotFoundError(
            "Input CSV not found. Save your restaurant review CSV here: "
            f"{input_file}"
        )

    reviews = pd.read_csv(input_file)
    cleaned_reviews = clean_reviews(reviews)

    cleaned_reviews.to_csv(
        output_file,
        index=False,
    )

    return cleaned_reviews