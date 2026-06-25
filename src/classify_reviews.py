from pathlib import Path
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DETAILED_ANALYSIS_FILE = DATA_DIR / "processed" / "detailed_restaurant_analysis.csv"
FINAL_ANALYSIS_FILE = DATA_DIR / "processed" / "final_analysis.csv"

MODEL_NAME = "all-MiniLM-L6-v2"

MIN_THEME_SCORE = 0.30
MIN_SCORE_MARGIN = 0.03

KEYWORD_BOOST_AMOUNT = 0.07
PHRASE_BOOST_AMOUNT = 0.10
MAX_TOTAL_KEYWORD_BOOST = 0.20


THEME_EXAMPLES = {
    "Food Quality": [
        "The food tasted good, delicious, flavorful, fresh, or well prepared.",
        "The food was bland, cold, stale, dry, undercooked, overcooked, or disappointing.",
        "The customer commented on taste, flavor, spice level, freshness, portion size, texture, or ingredients.",
        "The dish, meal, entree, rice, chicken, sauce, drink, chai, tea, dessert, or appetizer was reviewed.",
    ],
    "Service": [
        "The staff, server, cashier, waiter, waitress, employee, or manager was friendly, rude, helpful, or unprofessional.",
        "The customer commented on staff attitude, communication, attentiveness, hospitality, or problem resolution.",
        "The customer was treated well or poorly by employees.",
        "The service experience was good, bad, rude, friendly, helpful, careless, or attentive.",
    ],
    "Wait Time": [
        "The customer waited a long time for food, drinks, seating, pickup, delivery, checkout, or service.",
        "The order took too long, the restaurant was slow, or there were delays.",
        "The customer mentioned minutes, hours, waiting, speed, fast service, slow service, or timing.",
        "The food or service was quick, fast, delayed, slow, or took forever.",
    ],
    "Cleanliness": [
        "The customer commented on cleanliness, sanitation, hygiene, dirty tables, sticky surfaces, or messy dining areas.",
        "The bathroom, restroom, floor, table, utensils, trash, smell, pests, bugs, or dining area was dirty or clean.",
        "The restaurant appeared clean, dirty, sanitary, unsanitary, messy, sticky, or well maintained.",
        "The review discusses visible cleaning standards or hygiene issues.",
    ],
    "Price and Value": [
        "The customer commented on price, cost, affordability, expensive food, cheap food, or value for money.",
        "The meal was worth the price, not worth the price, overpriced, affordable, or a good deal.",
        "The review discusses portion value, discounts, overcharging, unexpected charges, or whether the meal was worth it.",
        "The customer felt the price was fair, too high, too expensive, or reasonable.",
    ],
    "Order Accuracy": [
        "The order was wrong, incorrect, missing items, missing sauces, missing sides, or incomplete.",
        "The customer received the wrong food, wrong drink, wrong order, or something different from what was ordered.",
        "Special requests, substitutions, modifications, or instructions were ignored.",
        "The restaurant forgot items, mixed up the order, or made an order accuracy mistake.",
    ],
    "Atmosphere": [
        "The customer commented on ambience, ambiance, decor, decoration, interior design, vibe, or atmosphere.",
        "The dining environment, lighting, music, seating, layout, noise level, crowding, or comfort was reviewed.",
        "The restaurant looked nice, beautiful, cozy, loud, crowded, comfortable, or well decorated.",
        "The review discusses parking, location, space, seating, environment, or overall restaurant vibe.",
    ],
}


THEME_KEYWORDS = {
    "Food Quality": {
        "food",
        "dish",
        "dishes",
        "meal",
        "meals",
        "taste",
        "tasted",
        "tastes",
        "flavor",
        "flavors",
        "fresh",
        "stale",
        "cold",
        "hot",
        "spicy",
        "bland",
        "funny",
        "portion",
        "portions",
        "ingredient",
        "ingredients",
        "cooked",
        "undercooked",
        "overcooked",
        "dry",
        "sauce",
        "rice",
        "chicken",
        "bhel",
        "chai",
        "tea",
        "drink",
        "drinks",
        "dessert",
        "appetizer",
        "delicious",
    },
    "Service": {
        "service",
        "staff",
        "server",
        "servers",
        "employee",
        "employees",
        "cashier",
        "manager",
        "waiter",
        "waitress",
        "rude",
        "friendly",
        "helpful",
        "attentive",
        "professional",
        "unprofessional",
        "polite",
        "attitude",
        "hospitality",
        "communication",
    },
    "Wait Time": {
        "wait",
        "waited",
        "waiting",
        "slow",
        "slower",
        "delay",
        "delayed",
        "minutes",
        "minute",
        "hour",
        "hours",
        "fast",
        "quick",
        "quickly",
        "speed",
        "forever",
        "late",
        "seated",
        "seating",
        "pickup",
        "delivery",
    },
    "Cleanliness": {
        "clean",
        "cleanliness",
        "dirty",
        "filthy",
        "sticky",
        "sanitary",
        "unsanitary",
        "sanitation",
        "hygiene",
        "bathroom",
        "bathrooms",
        "restroom",
        "restrooms",
        "trash",
        "garbage",
        "smell",
        "smells",
        "pest",
        "pests",
        "bug",
        "bugs",
        "floor",
        "floors",
        "table",
        "tables",
        "utensils",
        "messy",
    },
    "Price and Value": {
        "price",
        "prices",
        "priced",
        "cost",
        "costs",
        "expensive",
        "cheap",
        "overpriced",
        "affordable",
        "worth",
        "value",
        "deal",
        "deals",
        "discount",
        "discounts",
        "charge",
        "charged",
        "overcharged",
        "money",
        "reasonable",
    },
    "Order Accuracy": {
        "wrong",
        "missing",
        "incorrect",
        "forgot",
        "forgotten",
        "incomplete",
        "substitution",
        "substitutions",
        "request",
        "requests",
        "modification",
        "modifications",
    },
    "Atmosphere": {
        "ambience",
        "ambiance",
        "decor",
        "decoration",
        "decorations",
        "interior",
        "design",
        "vibe",
        "atmosphere",
        "environment",
        "music",
        "lighting",
        "seating",
        "seat",
        "seats",
        "parking",
        "location",
        "space",
        "layout",
        "comfortable",
        "cozy",
        "crowded",
        "loud",
        "noise",
        "beautiful",
    },
}


THEME_PHRASES = {
    "Food Quality": {
        "food was good",
        "food is good",
        "food was great",
        "food is great",
        "food was amazing",
        "food was delicious",
        "food tasted",
        "taste was",
        "tasted funny",
        "tastes funny",
        "sauce tasted",
        "sauce was",
        "flavor was",
        "portion size",
        "spice level",
    },
    "Service": {
        "customer service",
        "service was bad",
        "service was horrible",
        "service was great",
        "staff was rude",
        "staff were rude",
        "staff was friendly",
        "staff were friendly",
        "server was rude",
        "server was friendly",
    },
    "Wait Time": {
        "waited for",
        "waited too long",
        "long wait",
        "took too long",
        "took forever",
        "slow service",
        "quick service",
        "fast service",
        "wait time",
    },
    "Cleanliness": {
        "dirty table",
        "dirty tables",
        "sticky table",
        "sticky tables",
        "dirty bathroom",
        "dirty bathrooms",
        "clean bathroom",
        "clean bathrooms",
        "dirty floor",
        "bad smell",
    },
    "Price and Value": {
        "not worth",
        "worth the price",
        "worth it",
        "too expensive",
        "very expensive",
        "over priced",
        "good value",
        "value for money",
        "waste of money",
    },
    "Order Accuracy": {
        "wrong order",
        "missing item",
        "missing items",
        "missing sauce",
        "missing sauces",
        "wrong sauce",
        "forgot sauce",
        "forgot my sauce",
        "forgot the sauce",
        "wrong food",
        "wrong drink",
        "special request",
        "forgot my",
        "forgot the",
    },
    "Atmosphere": {
        "good ambience",
        "good ambiance",
        "nice ambience",
        "nice ambiance",
        "great atmosphere",
        "good atmosphere",
        "nice decor",
        "beautiful decor",
        "decor was",
        "decoration inside",
        "interior design",
        "place looked nice",
    },
}


def validate_data(
    reviews: pd.DataFrame,
    embeddings: np.ndarray,
) -> None:

    required_columns = {
        "review_id",
        "business_id",
        "name",
        "stars",
        "text",
        "aspect_sentiment",
    }

    missing_columns = required_columns - set(reviews.columns)

    if missing_columns:
        raise ValueError(
            "The review/aspect data is missing required columns: "
            f"{sorted(missing_columns)}. Run the full pipeline from main.py."
        )

    if reviews.empty:
        raise ValueError(
            "The review/aspect data does not contain any rows."
        )

    if len(reviews) != len(embeddings):
        raise ValueError(
            "The number of review/aspect rows does not match the number of "
            "embeddings. Re-run the full pipeline from main.py."
        )

    if embeddings.ndim != 2:
        raise ValueError(
            "The review embeddings must be a two-dimensional array."
        )


def normalize_text(text: str) -> str:

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_words(text: str) -> set[str]:

    return set(
        re.findall(
            r"\b[a-zA-Z']+\b",
            normalize_text(text),
        )
    )


def find_theme_matches(
    text: str,
    theme_name: str,
) -> list[str]:

    normalized_text = normalize_text(text)
    words = extract_words(text)

    matches = []

    for phrase in THEME_PHRASES.get(theme_name, set()):
        if phrase in normalized_text:
            matches.append(phrase)

    for keyword in THEME_KEYWORDS.get(theme_name, set()):
        if keyword in words:
            matches.append(keyword)

    return sorted(set(matches))


def create_theme_example_embeddings(
    model: SentenceTransformer,
) -> tuple[list[str], list[str], list[str], np.ndarray]:

    theme_names = list(THEME_EXAMPLES.keys())

    example_texts = []
    example_theme_names = []

    for theme_name in theme_names:
        for example in THEME_EXAMPLES[theme_name]:
            example_texts.append(example)
            example_theme_names.append(theme_name)

    example_embeddings = model.encode(
        example_texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return theme_names, example_texts, example_theme_names, example_embeddings


def calculate_semantic_theme_scores(
    review_embeddings: np.ndarray,
    example_embeddings: np.ndarray,
    theme_names: list[str],
    example_theme_names: list[str],
) -> np.ndarray:

    example_similarity_scores = review_embeddings @ example_embeddings.T

    theme_scores = np.zeros(
        (
            review_embeddings.shape[0],
            len(theme_names),
        )
    )

    for theme_index, theme_name in enumerate(theme_names):
        example_indices = [
            index
            for index, example_theme_name in enumerate(example_theme_names)
            if example_theme_name == theme_name
        ]

        theme_example_scores = example_similarity_scores[
            :,
            example_indices,
        ]

        theme_scores[:, theme_index] = theme_example_scores.max(axis=1)

    return theme_scores


def apply_keyword_guardrails(
    reviews: pd.DataFrame,
    theme_scores: np.ndarray,
    theme_names: list[str],
) -> tuple[np.ndarray, list[str]]:

    boosted_scores = theme_scores.copy()
    matched_keywords_output = []

    for row_index, text in enumerate(reviews["text"].fillna("").astype(str)):
        row_matches = []

        for theme_index, theme_name in enumerate(theme_names):
            matches = find_theme_matches(
                text=text,
                theme_name=theme_name,
            )

            if not matches:
                continue

            phrase_matches = [
                match
                for match in matches
                if " " in match
            ]

            keyword_matches = [
                match
                for match in matches
                if " " not in match
            ]

            boost = (
                len(phrase_matches) * PHRASE_BOOST_AMOUNT
                + len(keyword_matches) * KEYWORD_BOOST_AMOUNT
            )

            boost = min(
                boost,
                MAX_TOTAL_KEYWORD_BOOST,
            )

            boosted_scores[row_index, theme_index] += boost

            row_matches.append(
                f"{theme_name}: {', '.join(matches)}"
            )

        matched_keywords_output.append(
            " | ".join(row_matches)
            if row_matches
            else ""
        )

    return boosted_scores, matched_keywords_output


def classify_reviews(
    reviews: pd.DataFrame,
    review_embeddings: np.ndarray,
    model_name: str = MODEL_NAME,
) -> pd.DataFrame:

    validate_data(
        reviews=reviews,
        embeddings=review_embeddings,
    )

    model = SentenceTransformer(model_name)

    (
        theme_names,
        _,
        example_theme_names,
        example_embeddings,
    ) = create_theme_example_embeddings(model)

    if review_embeddings.shape[1] != example_embeddings.shape[1]:
        raise ValueError(
            "The review embeddings and theme embeddings have different "
            f"dimensions. Confirm that both steps use '{model_name}'."
        )

    semantic_scores = calculate_semantic_theme_scores(
        review_embeddings=review_embeddings,
        example_embeddings=example_embeddings,
        theme_names=theme_names,
        example_theme_names=example_theme_names,
    )

    boosted_scores, matched_theme_keywords = apply_keyword_guardrails(
        reviews=reviews,
        theme_scores=semantic_scores,
        theme_names=theme_names,
    )

    ranked_theme_indices = np.argsort(
        boosted_scores,
        axis=1,
    )[:, ::-1]

    primary_indices = ranked_theme_indices[:, 0]
    second_best_indices = ranked_theme_indices[:, 1]

    row_indices = np.arange(len(reviews))

    primary_scores = boosted_scores[
        row_indices,
        primary_indices,
    ]

    second_best_scores = boosted_scores[
        row_indices,
        second_best_indices,
    ]

    score_margins = primary_scores - second_best_scores

    classified_reviews = reviews.copy()

    classified_reviews["primary_theme"] = [
        theme_names[index]
        for index in primary_indices
    ]

    classified_reviews["theme_score"] = primary_scores
    classified_reviews["second_best_theme_score"] = second_best_scores
    classified_reviews["score_margin"] = score_margins

    classified_reviews["is_uncertain"] = (
        (classified_reviews["theme_score"] < MIN_THEME_SCORE)
        | (classified_reviews["score_margin"] < MIN_SCORE_MARGIN)
    )

    classified_reviews["matched_theme_keywords"] = matched_theme_keywords
    classified_reviews["classifier_method"] = (
        "theme_examples_semantic_similarity_with_keyword_guardrails"
    )

    score_columns = [
        "theme_score",
        "second_best_theme_score",
        "score_margin",
    ]

    classified_reviews[score_columns] = (
        classified_reviews[score_columns].round(4)
    )

    return classified_reviews


def sort_classified_reviews(reviews: pd.DataFrame) -> pd.DataFrame:

    required_columns = {
        "business_id",
        "aspect_sentiment",
        "primary_theme",
        "stars",
        "theme_score",
        "review_id",
        "sentence_id",
    }

    missing_columns = required_columns - set(reviews.columns)

    if missing_columns:
        raise ValueError(
            "Cannot sort classified reviews because these columns are missing: "
            f"{sorted(missing_columns)}"
        )

    sentiment_order = {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2,
    }

    sorted_reviews = reviews.copy()

    sorted_reviews["aspect_sentiment_order"] = (
        sorted_reviews["aspect_sentiment"].map(sentiment_order).fillna(3)
    )

    sorted_reviews = sorted_reviews.sort_values(
        by=[
            "business_id",
            "aspect_sentiment_order",
            "primary_theme",
            "stars",
            "theme_score",
            "review_id",
            "sentence_id",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            False,
            True,
            True,
        ],
    ).drop(columns=["aspect_sentiment_order"])

    return sorted_reviews.reset_index(drop=True)


def create_final_analysis_file(reviews: pd.DataFrame) -> pd.DataFrame:

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
        "review_rating_group",
        "sentence_id",
        "aspect_sentiment",
        "primary_theme",
        "text",
        "original_review_text",
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

    return final_analysis.reset_index(drop=True)


def save_classification_outputs(
    sorted_reviews: pd.DataFrame,
    final_analysis: pd.DataFrame,
    detailed_analysis_file: Path = DETAILED_ANALYSIS_FILE,
    final_analysis_file: Path = FINAL_ANALYSIS_FILE,
) -> None:

    detailed_analysis_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sorted_reviews.to_csv(
        detailed_analysis_file,
        index=False,
    )

    final_analysis.to_csv(
        final_analysis_file,
        index=False,
    )


def create_theme_summary_tables(
    reviews: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    theme_counts = (
        reviews["primary_theme"]
        .value_counts()
        .rename_axis("primary_theme")
        .reset_index(name="aspect_count")
    )

    theme_counts["percentage"] = (
        theme_counts["aspect_count"]
        / len(reviews)
        * 100
    ).round(2)

    grouped_counts = pd.crosstab(
        reviews["primary_theme"],
        reviews["aspect_sentiment"],
    )

    return theme_counts, grouped_counts


def print_theme_summary(
    reviews: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    return create_theme_summary_tables(reviews)