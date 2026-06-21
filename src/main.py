from pathlib import Path
import argparse

import classify_reviews
import clean_restaurant_reviews
import embed_reviews


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "input" / "sample_restaurant_reviews_30.csv"


def parse_args() -> argparse.Namespace:
    """Read command-line options for the DeepDine pipeline."""

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
    """Resolve either a normal file path or a project-relative CSV path."""

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
    """Run the complete workflow and save only the three final CSV outputs."""

    args = parse_args()
    input_file = resolve_input_path(args.input_csv)

    print("Starting DeepDine analysis...\n")

    cleaned_reviews = clean_restaurant_reviews.save_cleaned_reviews(
        input_file=input_file,
    )

    print()

    review_embeddings = embed_reviews.create_review_embeddings(
        reviews=cleaned_reviews,
    )

    print()

    classified_reviews = classify_reviews.classify_reviews(
        reviews=cleaned_reviews,
        review_embeddings=review_embeddings,
    )

    sorted_reviews = classify_reviews.sort_classified_reviews(
        classified_reviews,
    )

    final_analysis = classify_reviews.create_final_analysis_file(
        sorted_reviews,
    )

    classify_reviews.save_classification_outputs(
        sorted_reviews=sorted_reviews,
        final_analysis=final_analysis,
    )

    classify_reviews.print_theme_summary(classified_reviews)

if __name__ == "__main__":
    main()