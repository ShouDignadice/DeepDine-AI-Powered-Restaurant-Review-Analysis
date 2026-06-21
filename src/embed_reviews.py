import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


def validate_reviews(reviews: pd.DataFrame) -> None:
    """Validate that reviews are ready to be embedded."""

    if "text" not in reviews.columns:
        raise ValueError(
            "The cleaned review data must contain a text column. "
            "Run clean_restaurant_reviews.py first."
        )

    if reviews.empty:
        raise ValueError(
            "The cleaned review data does not contain any reviews."
        )


def create_review_embeddings(
    reviews: pd.DataFrame,
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Convert review text into normalized sentence embeddings."""

    validate_reviews(reviews)

    print(f"Loading review embedding model: {model_name}")

    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        reviews["text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print("Review embedding step complete.")
    print(f"Embedding shape: {embeddings.shape}")

    return embeddings

if __name__ == "__main__":
    raise SystemExit(
        "embed_reviews.py is now a helper module. "
        "Run python src/main.py to execute the full DeepDine pipeline."
    )