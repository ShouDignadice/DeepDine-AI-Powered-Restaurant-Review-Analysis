from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = DATA_DIR / "input" / "restaurant_reviews.csv"
OUTPUT_FILE = DATA_DIR / "processed" / "cleaned_restaurant_reviews.csv"

MIN_WORDS = 10
DEFAULT_RESTAURANT_ID = "restaurant_1"
DEFAULT_RESTAURANT_NAME = "Uploaded Restaurant"

COLUMN_ALIASES = {
    "stars": [
        "stars",
        "star",
        "rating",
        "review_rating",
        "star_rating",
    ],
    "text": [
        "text",
        "review",
        "review_text",
        "comment",
        "comments",
        "body",
    ],
    "name": [
        "name",
        "restaurant_name",
        "business_name",
        "store_name",
    ],
    "business_id": [
        "business_id",
        "restaurant_id",
        "store_id",
        "location_id",
    ],
}


def find_column(reviews: pd.DataFrame, possible_names: list[str]) -> str | None:
    """Return the first matching column name, ignoring capitalization."""

    normalized_columns = {
        column.strip().lower(): column
        for column in reviews.columns
    }

    for possible_name in possible_names:
        matching_column = normalized_columns.get(possible_name.lower())

        if matching_column is not None:
            return matching_column

    return None


def standardize_columns(reviews: pd.DataFrame) -> pd.DataFrame:
    """Rename common CSV column names into the DeepDine format."""

    column_mapping = {}

    for standard_name, aliases in COLUMN_ALIASES.items():
        matching_column = find_column(
            reviews=reviews,
            possible_names=aliases,
        )

        if matching_column is not None:
            column_mapping[matching_column] = standard_name

    reviews = reviews.rename(columns=column_mapping).copy()

    required_columns = {"stars", "text"}
    missing_columns = required_columns - set(reviews.columns)

    if missing_columns:
        raise ValueError(
            "The input CSV must contain a review text column and a star/rating "
            "column. Accepted review columns include text, review, review_text, "
            "comment, comments, or body. Accepted rating columns include stars, "
            f"rating, review_rating, or star_rating. Missing: {sorted(missing_columns)}"
        )

    if "business_id" not in reviews.columns:
        reviews["business_id"] = DEFAULT_RESTAURANT_ID

    if "name" not in reviews.columns:
        reviews["name"] = DEFAULT_RESTAURANT_NAME

    return reviews


def clean_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize one restaurant review CSV for theme analysis."""

    reviews = standardize_columns(reviews)

    reviews = reviews[
        [
            "business_id",
            "name",
            "stars",
            "text",
        ]
    ].copy()

    reviews["business_id"] = (
        reviews["business_id"]
        .fillna(DEFAULT_RESTAURANT_ID)
        .astype(str)
        .str.strip()
    )

    reviews["name"] = (
        reviews["name"]
        .fillna(DEFAULT_RESTAURANT_NAME)
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
            "business_id",
            "name",
            "stars",
        ]
    )

    reviews = reviews.loc[
        reviews["text"].ne("")
    ].copy()

    reviews = reviews.loc[
        reviews["stars"].between(1, 5)
    ].copy()

    reviews = reviews.drop_duplicates(
        subset=[
            "business_id",
            "text",
        ]
    )

    word_counts = reviews["text"].str.split().str.len()

    reviews = reviews.loc[
        word_counts >= MIN_WORDS
    ].copy()

    reviews = reviews.reset_index(drop=True)

    reviews.insert(
        0,
        "review_id",
        range(1, len(reviews) + 1),
    )

    reviews["stars"] = reviews["stars"].round(1)

    return reviews


def save_cleaned_reviews(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:

    if not input_file.exists():
        raise FileNotFoundError(
            "Input CSV not found. Save your restaurant review CSV here: "
            f"{input_file}"
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    reviews = pd.read_csv(input_file)
    cleaned_reviews = clean_reviews(reviews)

    cleaned_reviews.to_csv(
        output_file,
        index=False,
    )

    return cleaned_reviews