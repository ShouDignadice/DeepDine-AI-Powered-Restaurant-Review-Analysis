from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = (
    DATA_DIR / "processed" / "cleaned_yelp_reviews.csv"
)

EMBEDDINGS_FILE = (
    DATA_DIR / "embeddings" / "review_embeddings.npy"
)

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


def main() -> None:
    reviews = pd.read_csv(INPUT_FILE)

    print(f"Reviews loaded: {len(reviews):,}")
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        reviews["text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(
        EMBEDDINGS_FILE,
        embeddings,
    )

    print("\nEmbedding generation complete.")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embeddings saved to: {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()