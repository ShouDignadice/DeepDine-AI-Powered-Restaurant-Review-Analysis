import re

import pandas as pd

MIN_ASPECT_WORDS = 2
MIN_ASPECT_CHARACTERS = 8

CONTRAST_CONNECTORS = [
    "but",
    "however",
    "though",
    "although",
    "yet",
    "still",
]

ASPECT_CUE_WORDS = {
    "food",
    "dish",
    "meal",
    "taste",
    "flavor",
    "fresh",
    "portion",
    "chicken",
    "rice",
    "sauce",
    "drink",
    "chai",
    "tea",
    "service",
    "staff",
    "server",
    "employee",
    "cashier",
    "waiter",
    "waitress",
    "manager",
    "wait",
    "waiting",
    "slow",
    "fast",
    "time",
    "minutes",
    "hour",
    "clean",
    "dirty",
    "bathroom",
    "table",
    "floor",
    "price",
    "value",
    "expensive",
    "cheap",
    "worth",
    "order",
    "missing",
    "wrong",
    "incorrect",
    "ambience",
    "ambiance",
    "atmosphere",
    "decor",
    "decoration",
    "music",
    "seating",
    "parking",
    "location",
}

SENTIMENT_CUE_WORDS = {
    "good",
    "great",
    "amazing",
    "excellent",
    "delicious",
    "fresh",
    "friendly",
    "helpful",
    "clean",
    "nice",
    "fast",
    "quick",
    "perfect",
    "love",
    "loved",
    "enjoyed",
    "beautiful",
    "comfortable",
    "worth",
    "recommend",
    "bad",
    "horrible",
    "terrible",
    "awful",
    "bland",
    "cold",
    "stale",
    "rude",
    "slow",
    "dirty",
    "expensive",
    "overpriced",
    "wrong",
    "missing",
    "disappointed",
    "disappointing",
    "worse",
    "worst",
    "poor",
    "mediocre",
    "avoid",
}


def clean_aspect_text(text: str) -> str:

    text = re.sub(r"\s+", " ", str(text)).strip()
    text = text.strip(" ,.!?;:-\"'()[]{}")

    return text


def is_valid_aspect(text: str) -> bool:

    cleaned_text = clean_aspect_text(text)

    if len(cleaned_text) < MIN_ASPECT_CHARACTERS:
        return False

    if len(cleaned_text.split()) < MIN_ASPECT_WORDS:
        return False

    return True


def contains_cue_word(text: str, cue_words: set[str]) -> bool:

    words = {
        word.strip(".,!?;:'\"()[]{}").lower()
        for word in str(text).split()
    }

    return bool(words & cue_words)


def split_on_sentence_endings(text: str) -> list[str]:

    parts = re.split(r"(?<=[.!?;])\s+", text)

    return [
        clean_aspect_text(part)
        for part in parts
        if is_valid_aspect(part)
    ]


def split_on_contrast_connectors(text: str) -> list[str]:

    connector_pattern = (
        r"\s*,?\s+\b("
        + "|".join(CONTRAST_CONNECTORS)
        + r")\b\s*,?\s+"
    )

    parts = re.split(
        connector_pattern,
        text,
        flags=re.IGNORECASE,
    )

    cleaned_parts = []

    for part in parts:
        cleaned_part = clean_aspect_text(part)

        if cleaned_part.lower() in CONTRAST_CONNECTORS:
            continue

        if is_valid_aspect(cleaned_part):
            cleaned_parts.append(cleaned_part)

    if not cleaned_parts:
        return [clean_aspect_text(text)]

    return cleaned_parts


def should_split_on_and(left_text: str, right_text: str) -> bool:

    left_has_aspect = contains_cue_word(left_text, ASPECT_CUE_WORDS)
    right_has_aspect = contains_cue_word(right_text, ASPECT_CUE_WORDS)

    left_has_sentiment = contains_cue_word(left_text, SENTIMENT_CUE_WORDS)
    right_has_sentiment = contains_cue_word(right_text, SENTIMENT_CUE_WORDS)

    return (
        is_valid_aspect(left_text)
        and is_valid_aspect(right_text)
        and left_has_aspect
        and right_has_aspect
        and (left_has_sentiment or right_has_sentiment)
    )


def split_on_safe_and(text: str) -> list[str]:

    parts = re.split(r"\s+\band\b\s+", text, flags=re.IGNORECASE)

    if len(parts) <= 1:
        return [clean_aspect_text(text)]

    final_parts = []
    current_part = clean_aspect_text(parts[0])

    for next_part in parts[1:]:
        next_part = clean_aspect_text(next_part)

        if should_split_on_and(current_part, next_part):
            final_parts.append(current_part)
            current_part = next_part
        else:
            current_part = f"{current_part} and {next_part}"

    if is_valid_aspect(current_part):
        final_parts.append(current_part)

    return final_parts


def split_text_into_aspects(text: str) -> list[str]:

    if not isinstance(text, str):
        return []

    cleaned_text = re.sub(r"\s+", " ", text).strip()

    if not cleaned_text:
        return []

    sentence_chunks = split_on_sentence_endings(cleaned_text)

    if not sentence_chunks and is_valid_aspect(cleaned_text):
        sentence_chunks = [clean_aspect_text(cleaned_text)]

    aspect_chunks = []

    for sentence in sentence_chunks:
        contrast_chunks = split_on_contrast_connectors(sentence)

        for contrast_chunk in contrast_chunks:
            safe_and_chunks = split_on_safe_and(contrast_chunk)

            for chunk in safe_and_chunks:
                cleaned_chunk = clean_aspect_text(chunk)

                if is_valid_aspect(cleaned_chunk):
                    aspect_chunks.append(cleaned_chunk)

    return aspect_chunks


def split_reviews_into_aspects(reviews: pd.DataFrame) -> pd.DataFrame:

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
            "Cannot split reviews into aspects because these columns are missing: "
            f"{sorted(missing_columns)}"
        )

    aspect_rows = []

    for _, review in reviews.iterrows():
        original_review_text = review["text"]
        aspects = split_text_into_aspects(original_review_text)

        review_rating_group = (
            "Negative"
            if review["stars"] <= 3
            else "Positive"
        )

        for sentence_id, aspect_text in enumerate(aspects, start=1):
            aspect_rows.append(
                {
                    "review_id": review["review_id"],
                    "business_id": review["business_id"],
                    "name": review["name"],
                    "stars": review["stars"],
                    "review_rating_group": review_rating_group,
                    "sentence_id": sentence_id,
                    "text": aspect_text,
                    "original_review_text": original_review_text,
                }
            )

    aspect_reviews = pd.DataFrame(aspect_rows)

    if aspect_reviews.empty:
        raise ValueError(
            "No sentence-level aspects were created. "
            "Check the input reviews or lower MIN_ASPECT_WORDS."
        )

    return aspect_reviews.reset_index(drop=True)