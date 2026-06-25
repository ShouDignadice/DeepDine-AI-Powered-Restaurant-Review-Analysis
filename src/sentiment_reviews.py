import pandas as pd
from transformers import pipeline

SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 32


def normalize_sentiment_label(label: str) -> str:

    cleaned_label = str(label).lower().strip()

    if "negative" in cleaned_label or cleaned_label == "label_0":
        return "Negative"

    if "neutral" in cleaned_label or cleaned_label == "label_1":
        return "Neutral"

    if "positive" in cleaned_label or cleaned_label == "label_2":
        return "Positive"

    return "Neutral"

def add_aspect_sentiment(
    reviews: pd.DataFrame,
    model_name: str = SENTIMENT_MODEL_NAME,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """Classify sentiment for each sentence/aspect using a machine learning model."""

    if "text" not in reviews.columns:
        raise ValueError(
            "Cannot classify sentiment because the dataframe is missing a text column."
        )

    if reviews.empty:
        raise ValueError(
            "Cannot classify sentiment because there are no review aspects."
        )

    print(f"Loading sentiment model: {model_name}")

    sentiment_model = pipeline(
        task="sentiment-analysis",
        model=model_name,
    )

    texts = (
        reviews["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    results = sentiment_model(
        texts,
        batch_size=batch_size,
        truncation=True,
    )

    reviews = reviews.copy()

    reviews["aspect_sentiment"] = [
        normalize_sentiment_label(result["label"])
        for result in results
    ]

    reviews["aspect_sentiment_score"] = [
        round(float(result["score"]), 4)
        for result in results
    ]

    print("Aspect sentiment classification complete.")

    return reviews