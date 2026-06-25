from pathlib import Path
import argparse

import classify_reviews
import clean_restaurant_reviews
import embed_reviews
import recommend_restaurant_reviews
import sentiment_reviews
import split_review_aspects


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "input" / "california_pizza_kitchen_reviews.csv"


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Run the DeepDine restaurant review analysis pipeline."
    )

    parser.add_argument(
        "input_csv",
        nargs="?",
        help=(
            "Optional path to the restaurant review CSV. If omitted, DeepDine "
            "uses data/input/restaurant_reviews.csv."
        ),
    )

    return parser.parse_args()


def resolve_input_path(input_csv: str | None) -> Path:

    if input_csv is None:
        input_file = DEFAULT_INPUT_FILE
    else:
        input_file = Path(input_csv).expanduser()

        if not input_file.exists():
            project_relative_file = PROJECT_ROOT / input_file

            if project_relative_file.exists():
                input_file = project_relative_file

    input_file = input_file.resolve()

    if not input_file.exists():
        raise FileNotFoundError(
            "Input CSV not found. Save your file at data/input/restaurant_reviews.csv "
            f"or pass a CSV path. Tried: {input_file}"
        )

    if input_file.suffix.lower() != ".csv":
        raise ValueError(
            "The input file must be a .csv file. "
            f"Received: {input_file.name}"
        )

    return input_file


def main() -> None:

    args = parse_args()
    input_file = resolve_input_path(args.input_csv)

    print("Starting DeepDine analysis...\n")

    cleaned_reviews = clean_restaurant_reviews.save_cleaned_reviews(
        input_file=input_file,
    )

    aspect_reviews = split_review_aspects.split_reviews_into_aspects(
        cleaned_reviews,
    )

    aspect_reviews = sentiment_reviews.add_aspect_sentiment(
        aspect_reviews,
    )

    aspect_embeddings = embed_reviews.create_review_embeddings(
        reviews=aspect_reviews,
    )

    classified_aspects = classify_reviews.classify_reviews(
        reviews=aspect_reviews,
        review_embeddings=aspect_embeddings,
    )

    sorted_aspects = classify_reviews.sort_classified_reviews(
        classified_aspects,
    )

    final_analysis = classify_reviews.create_final_analysis_file(
        sorted_aspects,
    )

    classify_reviews.save_classification_outputs(
        sorted_reviews=sorted_aspects,
        final_analysis=final_analysis,
    )

    classify_reviews.print_theme_summary(classified_aspects)

    recommend_restaurant_reviews.print_recommendations(classified_aspects)

if __name__ == "__main__":
    main()