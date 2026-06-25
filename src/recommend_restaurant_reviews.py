import pandas as pd


RECOMMENDATION_RULES = {
    "Food Quality": {
        "strength": (
            "Food Quality is a recurring strength. Keep food preparation, taste, "
            "freshness, and portion consistency high."
        ),
        "improvement": (
            "Review negative food-related feedback to identify specific issues with "
            "taste, freshness, temperature, portions, or preparation consistency."
        ),
    },
    "Service": {
        "strength": (
            "Service is a recurring strength. Continue reinforcing friendly, helpful, "
            "and attentive customer interactions."
        ),
        "improvement": (
            "Improve service consistency by reviewing staff communication, attentiveness, "
            "professionalism, and issue resolution during customer interactions."
        ),
    },
    "Wait Time": {
        "strength": (
            "Wait Time feedback is mostly positive. Continue maintaining efficient service "
            "speed during busy periods."
        ),
        "improvement": (
            "Reduce delays by reviewing staffing levels, kitchen workflow, order prep time, "
            "pickup timing, and peak-hour bottlenecks."
        ),
    },
    "Cleanliness": {
        "strength": (
            "Cleanliness feedback is mostly positive. Continue maintaining visible cleaning "
            "standards across dining areas, tables, floors, and bathrooms."
        ),
        "improvement": (
            "Increase cleanliness checks during busy periods, especially for tables, floors, "
            "bathrooms, sticky surfaces, trash, and dining areas."
        ),
    },
    "Price and Value": {
        "strength": (
            "Price and Value feedback is mostly positive. Customers appear to see the meal "
            "as fair or worthwhile for the cost."
        ),
        "improvement": (
            "Review pricing, portion sizes, discounts, and perceived value to make sure "
            "customers feel the meal is worth the price."
        ),
    },
    "Order Accuracy": {
        "strength": (
            "Order Accuracy feedback is mostly positive. Continue maintaining correct orders, "
            "special requests, and complete items."
        ),
        "improvement": (
            "Improve order accuracy by checking missing items, wrong orders, substitutions, "
            "special requests, sides, sauces, and modifications before orders are completed."
        ),
    },
    "Atmosphere": {
        "strength": (
            "Atmosphere is a recurring strength. Continue maintaining a comfortable dining "
            "environment, including seating, layout, music, lighting, and overall vibe."
        ),
        "improvement": (
            "Review atmosphere-related complaints such as noise, crowding, seating comfort, "
            "parking, temperature, music, and overall dining environment."
        ),
    },
}


def validate_reviews(reviews: pd.DataFrame) -> None:

    required_columns = {
        "primary_theme",
        "aspect_sentiment",
        "stars",
        "text",
    }

    missing_columns = required_columns - set(reviews.columns)

    if missing_columns:
        raise ValueError(
            "Cannot generate recommendations because these columns are missing: "
            f"{sorted(missing_columns)}"
        )

    if reviews.empty:
        raise ValueError(
            "Cannot generate recommendations because there are no review aspects."
        )


def create_theme_summary(reviews: pd.DataFrame) -> pd.DataFrame:

    theme_summary = pd.crosstab(
        reviews["primary_theme"],
        reviews["aspect_sentiment"],
    )

    for column in ["Positive", "Neutral", "Negative"]:
        if column not in theme_summary.columns:
            theme_summary[column] = 0

    theme_summary = theme_summary.reset_index()

    theme_summary["total_aspects"] = (
        theme_summary["Positive"]
        + theme_summary["Neutral"]
        + theme_summary["Negative"]
    )

    theme_summary["negative_percentage"] = (
        theme_summary["Negative"]
        / theme_summary["total_aspects"]
        * 100
    ).round(2)

    theme_summary["positive_percentage"] = (
        theme_summary["Positive"]
        / theme_summary["total_aspects"]
        * 100
    ).round(2)

    return theme_summary.sort_values(
        by=[
            "total_aspects",
            "Negative",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)


def get_top_strength(theme_summary: pd.DataFrame) -> pd.Series | None:

    positive_themes = theme_summary.loc[
        theme_summary["Positive"] > 0
    ].copy()

    if positive_themes.empty:
        return None

    return positive_themes.sort_values(
        by=[
            "Positive",
            "positive_percentage",
            "total_aspects",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).iloc[0]


def get_improvement_areas(theme_summary: pd.DataFrame) -> pd.DataFrame:

    improvement_areas = theme_summary.loc[
        theme_summary["Negative"] > 0
    ].copy()

    if improvement_areas.empty:
        return improvement_areas

    return improvement_areas.sort_values(
        by=[
            "Negative",
            "negative_percentage",
            "total_aspects",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).head(3)


def get_top_negative_review(
    reviews: pd.DataFrame,
    theme: str,
) -> pd.Series | None:

    theme_reviews = reviews.loc[
        (reviews["primary_theme"] == theme)
        & (reviews["aspect_sentiment"] == "Negative")
    ].copy()

    if theme_reviews.empty:
        return None

    sort_columns = ["stars"]
    ascending_order = [True]

    if "aspect_sentiment_score" in theme_reviews.columns:
        sort_columns.insert(0, "aspect_sentiment_score")
        ascending_order.insert(0, False)

    if "theme_score" in theme_reviews.columns:
        sort_columns.insert(0, "theme_score")
        ascending_order.insert(0, False)

    return theme_reviews.sort_values(
        by=sort_columns,
        ascending=ascending_order,
    ).iloc[0]


def print_recommendations(reviews: pd.DataFrame) -> None:

    validate_reviews(reviews)

    theme_summary = create_theme_summary(reviews)

    print("\nAutomatic Recommendations")

    top_strength = get_top_strength(theme_summary)

    if top_strength is not None:
        strength_theme = top_strength["primary_theme"]
        strength_rule = RECOMMENDATION_RULES[strength_theme]["strength"]

        print(
            f"\nTop strength: {strength_theme} "
            f"({int(top_strength['Positive'])} positive review aspects)"
        )
        print(f"Recommendation: {strength_rule}")

    improvement_areas = get_improvement_areas(theme_summary)

    if improvement_areas.empty:
        print("\nNo major improvement areas were found from negative review aspects.")
        return

    print("\nTop improvement areas:")

    for _, row in improvement_areas.iterrows():
        theme = row["primary_theme"]
        rule = RECOMMENDATION_RULES[theme]["improvement"]

        print(
            f"\n{theme}: "
            f"{int(row['Negative'])} negative review aspects, "
            f"{row['negative_percentage']:.2f}% negative within this theme"
        )
        print(f"Recommendation: {rule}")

        top_review = get_top_negative_review(
            reviews=reviews,
            theme=theme,
        )

        if top_review is not None:
            print("\nExample negative review aspect match:")
            print(f"Primary theme: {top_review['primary_theme']}")

            if "secondary_theme" in top_review:
                print(f"Secondary theme: {top_review['secondary_theme']}")

            if "aspect_sentiment" in top_review:
                print(f"Aspect sentiment: {top_review['aspect_sentiment']}")

            print(f"Review aspect: {top_review['text']}")

            if "original_review_text" in top_review:
                print(f"Original full review: {top_review['original_review_text']}")