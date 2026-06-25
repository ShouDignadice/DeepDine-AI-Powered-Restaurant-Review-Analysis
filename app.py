from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# File paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
FINAL_ANALYSIS_FILE = PROJECT_ROOT / "data" / "processed" / "final_analysis.csv"


# --------------------------------------------------
# Dashboard settings
# --------------------------------------------------

st.set_page_config(
    page_title="DeepDine Review Dashboard",
    layout="wide"
)


# --------------------------------------------------
# Recommendation logic
# --------------------------------------------------

THEME_STRENGTH_RECOMMENDATIONS = {
    "Food Quality": (
        "Food quality is a recurring strength. Keep taste, freshness, preparation, "
        "temperature, and portion consistency high."
    ),
    "Order Accuracy": (
        "Order accuracy is being mentioned positively. Keep order-checking processes "
        "consistent so customers continue receiving the correct items, sauces, sides, "
        "and modifications."
    ),
    "Price and Value": (
        "Customers are responding positively to value. Continue balancing price, "
        "portion size, and food quality."
    ),
    "Atmosphere": (
        "Atmosphere is a recurring strength. Maintain the dining environment, comfort, "
        "decor, seating, and overall restaurant experience."
    ),
    "Cleanliness": (
        "Cleanliness is being noticed positively. Continue maintaining dining areas, "
        "tables, restrooms, and food preparation areas."
    ),
    "Service": (
        "Service is a recurring strength. Continue reinforcing friendly, helpful, "
        "and responsive customer interactions."
    ),
    "Customer Service": (
        "Customer service is a recurring strength. Continue reinforcing friendly, "
        "helpful, and responsive customer interactions."
    ),
}


THEME_IMPROVEMENT_RECOMMENDATIONS = {
    "Food Quality": (
        "Review recurring food complaints related to taste, freshness, temperature, "
        "preparation, portion size, and consistency."
    ),
    "Order Accuracy": (
        "Improve order-checking before completion, especially for missing items, "
        "wrong orders, sauces, sides, substitutions, and special requests."
    ),
    "Price and Value": (
        "Review whether customers feel the portion size, food quality, and overall "
        "experience match the price."
    ),
    "Atmosphere": (
        "Review the restaurant environment, including seating, decor, comfort, noise, "
        "and overall dining experience."
    ),
    "Cleanliness": (
        "Focus on dining area cleanliness, tables, restrooms, food preparation areas, "
        "and overall store presentation."
    ),
    "Service": (
        "Review service-related complaints such as staff responsiveness, wait time, "
        "communication, friendliness, and issue resolution."
    ),
    "Customer Service": (
        "Review service-related complaints such as staff responsiveness, wait time, "
        "communication, friendliness, and issue resolution."
    ),
}


# --------------------------------------------------
# Data helpers
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(FINAL_ANALYSIS_FILE)

    # Normalize column names.
    # Example: "Review Group" -> "review_group"
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    if "stars" in df.columns:
        df["stars"] = pd.to_numeric(df["stars"], errors="coerce")

    # review_group is based on the full review's star rating.
    if "review_group" not in df.columns and "stars" in df.columns:
        df["review_group"] = df["stars"].apply(
            lambda star: "Positive" if star >= 4 else "Negative"
        )

    # If no sentiment-style column exists, fall back to review_group.
    # Ideally, sentiment should be the aspect-level text sentiment.
    sentiment_like_columns = [
        "sentiment",
        "aspect_sentiment",
        "text_sentiment",
        "review_sentiment",
        "predicted_sentiment"
    ]

    has_sentiment_column = any(
        column in df.columns for column in sentiment_like_columns
    )

    if not has_sentiment_column and "review_group" in df.columns:
        df["sentiment"] = df["review_group"]

    # Normalize sentiment and review group values.
    for column in sentiment_like_columns + ["review_group"]:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype(str)
                .str.strip()
                .str.title()
                .replace("Nan", "Unknown")
            )

    return df


def get_review_id_column(df):
    possible_columns = [
        "review_id",
        "reviewid",
        "id"
    ]

    for column in possible_columns:
        if column in df.columns:
            return column

    return None


def get_full_review_text_column(df):
    possible_columns = [
        "full_review_text",
        "original_full_review",
        "original_review_text",
        "original_review",
        "full_text",
        "review_text"
    ]

    for column in possible_columns:
        if column in df.columns:
            return column

    return None


def get_aspect_text_column(df, full_review_text_column=None):
    possible_columns = [
        "aspect_text",
        "aspect_review",
        "aspect_review_text",
        "chunk_text",
        "review_chunk",
        "split_review_text",
        "sentence_text",
        "text"
    ]

    for column in possible_columns:
        if column in df.columns and column != full_review_text_column:
            return column

    return None


def get_sentiment_column(df):
    possible_columns = [
        "sentiment",
        "aspect_sentiment",
        "text_sentiment",
        "review_sentiment",
        "predicted_sentiment"
    ]

    for column in possible_columns:
        if column in df.columns:
            return column

    return None


def count_full_reviews(df, review_id_column):
    if review_id_column is not None and review_id_column in df.columns:
        return df[review_id_column].nunique()

    return len(df)


def get_full_review_level_df(df, review_id_column):
    if review_id_column is not None and review_id_column in df.columns:
        return df.drop_duplicates(subset=[review_id_column]).copy()

    return df.copy()


def create_theme_summary(df, review_id_column, sentiment_column):
    """
    Creates a summary with a clear distinction between:

    full_review_count:
        Number of unique original customer reviews that mention the theme.

    aspect_mention_count:
        Number of extracted aspect-level rows for the theme.

    positive_aspect_mentions / negative_aspect_mentions:
        Number of aspect-level rows with positive or negative sentiment.

    positive_full_reviews / negative_full_reviews:
        Number of unique original reviews containing at least one positive or
        negative aspect mention for that theme.
    """

    # Full review count by theme
    if review_id_column is not None and review_id_column in df.columns:
        full_review_counts = (
            df.groupby("primary_theme")[review_id_column]
            .nunique()
            .reset_index(name="full_review_count")
        )
    else:
        full_review_counts = (
            df.groupby("primary_theme")
            .size()
            .reset_index(name="full_review_count")
        )

    # Aspect mention count by theme
    aspect_mention_counts = (
        df.groupby("primary_theme")
        .size()
        .reset_index(name="aspect_mention_count")
    )

    # Aspect sentiment counts by theme
    aspect_sentiment_counts = (
        df.groupby(["primary_theme", sentiment_column])
        .size()
        .reset_index(name="count")
        .pivot(
            index="primary_theme",
            columns=sentiment_column,
            values="count"
        )
        .fillna(0)
        .reset_index()
    )

    if "Positive" not in aspect_sentiment_counts.columns:
        aspect_sentiment_counts["Positive"] = 0

    if "Negative" not in aspect_sentiment_counts.columns:
        aspect_sentiment_counts["Negative"] = 0

    aspect_sentiment_counts = aspect_sentiment_counts.rename(
        columns={
            "Positive": "positive_aspect_mentions",
            "Negative": "negative_aspect_mentions"
        }
    )

    # Full review sentiment counts by theme
    if review_id_column is not None and review_id_column in df.columns:
        full_review_sentiment_counts = (
            df.groupby(["primary_theme", sentiment_column])[review_id_column]
            .nunique()
            .reset_index(name="count")
            .pivot(
                index="primary_theme",
                columns=sentiment_column,
                values="count"
            )
            .fillna(0)
            .reset_index()
        )
    else:
        full_review_sentiment_counts = (
            df.groupby(["primary_theme", sentiment_column])
            .size()
            .reset_index(name="count")
            .pivot(
                index="primary_theme",
                columns=sentiment_column,
                values="count"
            )
            .fillna(0)
            .reset_index()
        )

    if "Positive" not in full_review_sentiment_counts.columns:
        full_review_sentiment_counts["Positive"] = 0

    if "Negative" not in full_review_sentiment_counts.columns:
        full_review_sentiment_counts["Negative"] = 0

    full_review_sentiment_counts = full_review_sentiment_counts.rename(
        columns={
            "Positive": "positive_full_reviews",
            "Negative": "negative_full_reviews"
        }
    )

    theme_summary = full_review_counts.merge(
        aspect_mention_counts,
        on="primary_theme",
        how="left"
    )

    theme_summary = theme_summary.merge(
        aspect_sentiment_counts[
            [
                "primary_theme",
                "positive_aspect_mentions",
                "negative_aspect_mentions"
            ]
        ],
        on="primary_theme",
        how="left"
    )

    theme_summary = theme_summary.merge(
        full_review_sentiment_counts[
            [
                "primary_theme",
                "positive_full_reviews",
                "negative_full_reviews"
            ]
        ],
        on="primary_theme",
        how="left"
    )

    numeric_columns = [
        "full_review_count",
        "aspect_mention_count",
        "positive_aspect_mentions",
        "negative_aspect_mentions",
        "positive_full_reviews",
        "negative_full_reviews"
    ]

    for column in numeric_columns:
        theme_summary[column] = theme_summary[column].fillna(0).astype(int)

    theme_summary["positive_aspect_rate"] = (
        theme_summary["positive_aspect_mentions"]
        / theme_summary["aspect_mention_count"]
        * 100
    ).round(2)

    theme_summary["negative_aspect_rate"] = (
        theme_summary["negative_aspect_mentions"]
        / theme_summary["aspect_mention_count"]
        * 100
    ).round(2)

    return theme_summary


def get_top_strength(theme_summary):
    positive_themes = theme_summary[
        theme_summary["positive_aspect_mentions"] > 0
    ]

    if positive_themes.empty:
        return None

    return positive_themes.sort_values(
        by=["positive_aspect_mentions", "positive_full_reviews"],
        ascending=False
    ).iloc[0]


def get_improvement_areas(theme_summary):
    negative_themes = theme_summary[
        theme_summary["negative_aspect_mentions"] > 0
    ]

    if negative_themes.empty:
        return negative_themes

    return negative_themes.sort_values(
        by=["negative_aspect_mentions", "negative_aspect_rate"],
        ascending=False
    )


# --------------------------------------------------
# Main dashboard
# --------------------------------------------------

st.title("DeepDine: Restaurant Review Findings Dashboard")

st.write(
    "An NLP-powered dashboard that summarizes restaurant reviews into customer "
    "themes, aspect-level sentiment patterns, and business recommendations."
)

try:
    df = load_data()

except FileNotFoundError:
    st.error(
        "final_analysis.csv was not found. Make sure the file exists at: "
        "data/processed/final_analysis.csv"
    )
    st.stop()


# --------------------------------------------------
# Validate required columns
# --------------------------------------------------

required_columns = ["stars", "review_group", "primary_theme"]

missing_columns = [
    column for column in required_columns if column not in df.columns
]

if missing_columns:
    st.error(
        "The final_analysis.csv file is missing these required columns: "
        + ", ".join(missing_columns)
    )

    st.write("Columns found in your CSV:")
    st.write(list(df.columns))

    st.stop()


review_id_column = get_review_id_column(df)
full_review_text_column = get_full_review_text_column(df)
aspect_text_column = get_aspect_text_column(df, full_review_text_column)
sentiment_column = get_sentiment_column(df)


if sentiment_column is None:
    st.warning(
        "No sentiment column was found. The dashboard will fall back to review_group."
    )
    sentiment_column = "review_group"


if review_id_column is None:
    st.warning(
        "No review_id column was found. The dashboard will count rows instead of "
        "unique full reviews. Add a review_id column for the most accurate results."
    )


if full_review_text_column is None:
    st.warning(
        "No full review text column was found. To show full reviews beside aspect "
        "reviews, add a column named full_review_text, original_review_text, "
        "or original_review."
    )


if aspect_text_column is None:
    st.warning(
        "No aspect text column was found. To show aspect reviews, add a column named "
        "aspect_text, aspect_review_text, chunk_text, or text."
    )


# --------------------------------------------------
# Explanation section
# --------------------------------------------------

st.info(
    "**How to read this dashboard:** "
    "**Full Review** means the original customer review, counted once using review_id. "
    "**Aspect Mention** means one extracted part of a review that was classified by "
    "theme and sentiment. **Review Group** comes from the overall star rating, while "
    "**Aspect Sentiment** comes from the actual text being analyzed."
)


# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------

st.sidebar.header("Filters")

if "business_name" in df.columns:
    restaurant_options = ["All"] + sorted(
        df["business_name"].dropna().unique().tolist()
    )

    selected_restaurant = st.sidebar.selectbox(
        "Restaurant",
        restaurant_options
    )
else:
    selected_restaurant = "All"

theme_options = ["All"] + sorted(
    df["primary_theme"].dropna().unique().tolist()
)

selected_theme = st.sidebar.selectbox(
    "Aspect Theme",
    theme_options
)

sentiment_options = ["All"] + sorted(
    df[sentiment_column].dropna().unique().tolist()
)

selected_sentiment = st.sidebar.selectbox(
    "Aspect Sentiment",
    sentiment_options
)


filtered_df = df.copy()

if selected_restaurant != "All" and "business_name" in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df["business_name"] == selected_restaurant
    ]

if selected_theme != "All":
    filtered_df = filtered_df[
        filtered_df["primary_theme"] == selected_theme
    ]

if selected_sentiment != "All":
    filtered_df = filtered_df[
        filtered_df[sentiment_column] == selected_sentiment
    ]


if filtered_df.empty:
    st.warning("No reviews match the selected filters.")
    st.stop()


# --------------------------------------------------
# Full review-level and aspect-level data
# --------------------------------------------------

full_review_level_df = get_full_review_level_df(
    filtered_df,
    review_id_column
)

theme_summary = create_theme_summary(
    filtered_df,
    review_id_column,
    sentiment_column
)


# --------------------------------------------------
# Overview metrics
# --------------------------------------------------

st.subheader("Overview")

total_full_reviews = count_full_reviews(filtered_df, review_id_column)
total_aspect_mentions = len(filtered_df)
average_rating = round(full_review_level_df["stars"].mean(), 2)

positive_aspect_mentions = len(
    filtered_df[filtered_df[sentiment_column] == "Positive"]
)

negative_aspect_mentions = len(
    filtered_df[filtered_df[sentiment_column] == "Negative"]
)

top_theme = (
    theme_summary
    .sort_values(by="full_review_count", ascending=False)
    .iloc[0]["primary_theme"]
)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Full Reviews", total_full_reviews)
col2.metric("Aspect Mentions", total_aspect_mentions)
col3.metric("Average Rating", average_rating)
col4.metric("Positive Aspect Mentions", positive_aspect_mentions)
col5.metric("Negative Aspect Mentions", negative_aspect_mentions)

st.caption(
    f"Top full-review theme: **{top_theme}**. "
    "Full Reviews count unique review_id values. Aspect Mentions count extracted "
    "review parts used for theme and sentiment analysis."
)

st.divider()


# --------------------------------------------------
# Full reviews by theme
# --------------------------------------------------

st.subheader("Full Reviews by Theme")

st.write(
    "This chart counts how many unique full reviews mention each theme."
)

full_review_theme_counts = (
    theme_summary
    .sort_values(by="full_review_count", ascending=False)
    .set_index("primary_theme")["full_review_count"]
)

st.bar_chart(full_review_theme_counts)

st.divider()


# --------------------------------------------------
# Aspect sentiment by theme
# --------------------------------------------------

st.subheader("Aspect Sentiment Mentions by Theme")

st.write(
    "This chart counts aspect-level positive and negative mentions within each theme. "
    "This is different from counting full reviews."
)

aspect_sentiment_by_theme = (
    theme_summary
    .set_index("primary_theme")[
        [
            "positive_aspect_mentions",
            "negative_aspect_mentions"
        ]
    ]
    .rename(
        columns={
            "positive_aspect_mentions": "Positive Aspect Mentions",
            "negative_aspect_mentions": "Negative Aspect Mentions"
        }
    )
)

st.bar_chart(aspect_sentiment_by_theme)

st.caption(
    "A single full review can contain multiple aspect mentions. For example, one "
    "review can complain about food quality while also praising atmosphere."
)

st.divider()


# --------------------------------------------------
# Automatic recommendations
# --------------------------------------------------

st.subheader("Automatic Recommendations")

st.write(
    "Recommendations are based on **aspect-level sentiment mentions**, because the "
    "goal is to identify the specific parts of reviews that customers praise or "
    "complain about. Each recommendation also shows how many **full reviews** were "
    "affected."
)

top_strength = get_top_strength(theme_summary)
improvement_areas = get_improvement_areas(theme_summary)

if top_strength is not None:
    strength_theme = top_strength["primary_theme"]
    positive_aspect_count = int(top_strength["positive_aspect_mentions"])
    positive_full_review_count = int(top_strength["positive_full_reviews"])

    st.success(
        f"Top aspect-level strength: **{strength_theme}** "
        f"({positive_aspect_count} positive aspect mentions across "
        f"{positive_full_review_count} full reviews)"
    )

    st.write(
        f"**Recommendation:** "
        f"{THEME_STRENGTH_RECOMMENDATIONS.get(strength_theme, 'Continue monitoring this area because customers are responding positively to it.')}"
    )
else:
    st.info("No positive aspect-level themes were found for the selected reviews.")

st.markdown("### Top Aspect-Level Improvement Areas")

if improvement_areas.empty:
    st.info("No negative aspect-level themes were found for the selected reviews.")
else:
    for _, row in improvement_areas.head(3).iterrows():
        theme = row["primary_theme"]
        negative_aspect_count = int(row["negative_aspect_mentions"])
        negative_full_review_count = int(row["negative_full_reviews"])
        negative_aspect_rate = row["negative_aspect_rate"]

        st.warning(
            f"**{theme}**: {negative_aspect_count} negative aspect mentions "
            f"across {negative_full_review_count} full reviews "
            f"({negative_aspect_rate}% of aspect mentions in this theme are negative)"
        )

        st.write(
            f"**Recommendation:** "
            f"{THEME_IMPROVEMENT_RECOMMENDATIONS.get(theme, 'Review this area more closely and look for recurring customer complaints.')}"
        )

st.markdown("### Theme Summary Table")

theme_summary_display = theme_summary.sort_values(
    by="full_review_count",
    ascending=False
).rename(
    columns={
        "primary_theme": "aspect_theme",
        "full_review_count": "full_reviews_mentioning_theme",
        "aspect_mention_count": "total_aspect_mentions",
        "positive_aspect_mentions": "positive_aspect_mentions",
        "negative_aspect_mentions": "negative_aspect_mentions",
        "positive_full_reviews": "full_reviews_with_positive_aspect",
        "negative_full_reviews": "full_reviews_with_negative_aspect",
        "positive_aspect_rate": "positive_aspect_rate_percent",
        "negative_aspect_rate": "negative_aspect_rate_percent"
    }
)

st.dataframe(
    theme_summary_display,
    use_container_width=True
)

st.divider()


# --------------------------------------------------
# Key findings summary
# --------------------------------------------------

st.subheader("Key Findings Summary")

if top_strength is not None and not improvement_areas.empty:
    strength_theme = top_strength["primary_theme"]
    top_issue_theme = improvement_areas.iloc[0]["primary_theme"]

    st.write(
        f"The strongest positive aspect-level feedback is related to "
        f"**{strength_theme}**, while the biggest negative aspect-level issue area "
        f"is **{top_issue_theme}**."
    )

    st.write(
        "This means the restaurant should protect the areas customers praise while "
        "using the negative aspect mentions to identify specific operational problems."
    )

elif top_strength is not None:
    strength_theme = top_strength["primary_theme"]

    st.write(
        f"The strongest positive aspect-level feedback is related to "
        f"**{strength_theme}**. No major negative aspect-level theme was found in "
        "the selected reviews."
    )

elif not improvement_areas.empty:
    top_issue_theme = improvement_areas.iloc[0]["primary_theme"]

    st.write(
        f"The biggest negative aspect-level issue area is **{top_issue_theme}**. "
        "No major positive aspect-level theme was found in the selected reviews."
    )

else:
    st.write(
        "There is not enough positive or negative aspect-level sentiment data to "
        "generate a clear summary."
    )

st.divider()


# --------------------------------------------------
# Aspect-level review explorer
# --------------------------------------------------

st.subheader("Aspect-Level Review Explorer")

st.write(
    "This table shows the aspect-level records used by the dashboard. A full review "
    "can appear more than once if it was split into multiple aspect mentions."
)

display_columns = []

for column in [
    "business_name",
    review_id_column,
    "stars",
    "review_group",
    sentiment_column,
    "primary_theme",
    aspect_text_column,
    full_review_text_column,
]:
    if column is not None and column in filtered_df.columns:
        if column not in display_columns:
            display_columns.append(column)

explorer_df = filtered_df[display_columns].copy()

rename_map = {
    "review_group": "full_review_group_by_stars",
    sentiment_column: "aspect_sentiment",
    "primary_theme": "aspect_theme"
}

if review_id_column is not None:
    rename_map[review_id_column] = "full_review_id"

if aspect_text_column is not None:
    rename_map[aspect_text_column] = "aspect_review_text"

if full_review_text_column is not None:
    rename_map[full_review_text_column] = "full_review_text"

explorer_df = explorer_df.rename(columns=rename_map)

st.dataframe(
    explorer_df,
    use_container_width=True
)

st.divider()


# --------------------------------------------------
# Sample aspect reviews with full review context
# --------------------------------------------------

if aspect_text_column is not None:
    st.subheader("Sample Aspect Reviews with Full Review Context")

    st.write(
        "This section shows the extracted aspect review beside the original full "
        "customer review so the classification can be checked in context."
    )

    sample_theme = st.selectbox(
        "Choose an aspect theme to view sample aspect reviews",
        sorted(filtered_df["primary_theme"].dropna().unique().tolist())
    )

    sample_reviews = filtered_df[
        filtered_df["primary_theme"] == sample_theme
    ].copy()

    duplicate_subset = [
        sentiment_column,
        "primary_theme",
        aspect_text_column
    ]

    if review_id_column is not None and review_id_column in sample_reviews.columns:
        duplicate_subset.insert(0, review_id_column)

    sample_reviews = sample_reviews.drop_duplicates(
        subset=duplicate_subset
    ).head(5)

    for _, row in sample_reviews.iterrows():
        stars = row["stars"]
        full_review_group = row["review_group"]
        aspect_sentiment = row[sentiment_column]
        aspect_text = row[aspect_text_column]

        if full_review_text_column is not None:
            full_review_text = row[full_review_text_column]
        else:
            full_review_text = (
                "Full review text column was not found. Add a column named "
                "full_review_text, original_review_text, or original_review to "
                "show the original full review here."
            )

        st.markdown(
            f"**Aspect Sentiment:** {aspect_sentiment} | "
            f"**Full Review Rating:** {stars} Stars | "
            f"**Full Review Group:** {full_review_group}"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Full Customer Review")
            st.write(full_review_text)

        with col2:
            st.markdown("#### Extracted Aspect Review")
            st.write(aspect_text)

        st.divider()

else:
    st.info(
        "No aspect text column was found. Add a column named aspect_text, "
        "aspect_review, aspect_review_text, chunk_text, review_chunk, text, "
        "split_review_text, or sentence_text to show sample aspect reviews."
    )