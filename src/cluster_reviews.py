from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

REVIEWS_FILE = (
    DATA_DIR / "processed" / "cleaned_yelp_reviews.csv"
)

EMBEDDINGS_FILE = (
    DATA_DIR / "embeddings" / "review_embeddings.npy"
)

CLUSTERED_REVIEWS_FILE = (
    DATA_DIR / "processed" / "clustered_yelp_reviews.csv"
)

CLUSTER_EXAMPLES_FILE = (
    DATA_DIR / "processed" / "cluster_examples.csv"
)

N_CLUSTERS = 12
EXAMPLES_PER_CLUSTER = 5
RANDOM_SEED = 42


def main() -> None:
    reviews = pd.read_csv(REVIEWS_FILE)
    embeddings = np.load(EMBEDDINGS_FILE)

    if len(reviews) != len(embeddings):
        raise ValueError(
            "The number of reviews does not match the number of embeddings. "
            "Regenerate the embeddings from the current cleaned CSV."
        )

    print(f"Reviews loaded: {len(reviews):,}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Creating {N_CLUSTERS} clusters...")

    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_SEED,
        n_init=10,
    )

    reviews["cluster_id"] = model.fit_predict(embeddings)

    score = silhouette_score(
        embeddings,
        reviews["cluster_id"],
        sample_size=min(5_000, len(reviews)),
        random_state=RANDOM_SEED,
    )

    reviews.to_csv(
        CLUSTERED_REVIEWS_FILE,
        index=False,
    )

    distances = model.transform(embeddings)
    examples = []

    for cluster_id in range(N_CLUSTERS):
        cluster_indices = np.where(
            reviews["cluster_id"].to_numpy() == cluster_id
        )[0]

        closest_indices = cluster_indices[
            np.argsort(
                distances[cluster_indices, cluster_id]
            )[:EXAMPLES_PER_CLUSTER]
        ]

        for rank, review_index in enumerate(
            closest_indices,
            start=1,
        ):
            review = reviews.iloc[review_index]

            examples.append(
                {
                    "cluster_id": cluster_id,
                    "rank": rank,
                    "business_id": review["business_id"],
                    "name": review["name"],
                    "stars": review["stars"],
                    "text": review["text"],
                }
            )

    cluster_examples = pd.DataFrame(examples)

    cluster_examples.to_csv(
        CLUSTER_EXAMPLES_FILE,
        index=False,
    )

    print("\nClustering complete.")
    print(f"Silhouette score: {score:.4f}")
    print("\nReviews per cluster:")
    print(
        reviews["cluster_id"]
        .value_counts()
        .sort_index()
    )
    print(f"\nClustered reviews saved to: {CLUSTERED_REVIEWS_FILE}")
    print(f"Cluster examples saved to: {CLUSTER_EXAMPLES_FILE}")

if __name__ == "__main__":
    main()