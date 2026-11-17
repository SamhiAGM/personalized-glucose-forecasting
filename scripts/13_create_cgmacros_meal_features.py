from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "data_audit"
)


# =========================================================
# 2. INPUT FILES
# =========================================================

FEATURE_FILE = (
    PROCESSED_DIR
    / "cgmacros_glucose_features_ready.csv"
)

HISTORY_FILE = (
    PROCESSED_DIR
    / "cgmacros_standardized_all.csv"
)


# =========================================================
# 3. OUTPUT FILES
# =========================================================

OUTPUT_FILE = (
    PROCESSED_DIR
    / "cgmacros_meal_features_ready.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "cgmacros_meal_feature_summary.csv"
)

PARTICIPANT_REPORT_FILE = (
    REPORT_DIR
    / "cgmacros_meal_feature_by_participant.csv"
)

MISSINGNESS_FILE = (
    REPORT_DIR
    / "cgmacros_meal_feature_missingness.csv"
)

DICTIONARY_FILE = (
    REPORT_DIR
    / "cgmacros_meal_feature_dictionary.csv"
)


# =========================================================
# 4. SETTINGS
# =========================================================

MEAL_WINDOW_MINUTES = 120


# =========================================================
# 5. NUTRITION SOURCE COLUMNS
# =========================================================

NUTRIENT_COLUMNS = {
    "meal_calories": "meal_calories_last_2h",
    "meal_carbs_g": "meal_carbs_last_2h_g",
    "meal_protein_g": "meal_protein_last_2h_g",
    "meal_fat_g": "meal_fat_last_2h_g",
    "meal_fiber_g": "meal_fiber_last_2h_g"
}


# =========================================================
# 6. LOAD DATA
# =========================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING STEP 13 INPUT DATA")
    print("=" * 70)

    features = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["timestamp"],
        low_memory=False
    )

    history = pd.read_csv(
        HISTORY_FILE,
        parse_dates=["timestamp"],
        low_memory=False
    )

    features["participant_id"] = (
        features["participant_id"]
        .astype(str)
    )

    history["participant_id"] = (
        history["participant_id"]
        .astype(str)
    )

    features = features.sort_values(
        [
            "participant_id",
            "timestamp"
        ],
        kind="stable"
    ).reset_index(drop=True)

    history = history.sort_values(
        [
            "participant_id",
            "timestamp"
        ],
        kind="stable"
    ).reset_index(drop=True)

    print(
        "\nFeature rows:",
        len(features)
    )

    print(
        "Feature participants:",
        features[
            "participant_id"
        ].nunique()
    )

    print(
        "Historical rows:",
        len(history)
    )

    print(
        "History participants:",
        history[
            "participant_id"
        ].nunique()
    )

    return (
        features,
        history
    )


# =========================================================
# 7. IDENTIFY MEAL EVENTS
# =========================================================

def create_meal_event_table(
    history
):

    print("\n" + "=" * 70)
    print("IDENTIFYING CGMACROS MEAL EVENTS")
    print("=" * 70)

    required_columns = [
        "participant_id",
        "timestamp",
        "meal_type",
        "meal_calories",
        "meal_carbs_g",
        "meal_protein_g",
        "meal_fat_g",
        "meal_fiber_g"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in history.columns
    ]

    if missing_columns:

        raise ValueError(
            "Required meal columns missing: "
            + str(missing_columns)
        )

    # -----------------------------------------------------
    # According to the CGMacros dictionary, Meal Type
    # indicates the start of a meal.
    #
    # Therefore Meal Type is used as the event definition.
    # -----------------------------------------------------

    meal_mask = (
        history[
            "meal_type"
        ]
        .notna()
    )

    meal_events = history.loc[
        meal_mask,
        required_columns
    ].copy()

    meal_events[
        "meal_type"
    ] = (
        meal_events[
            "meal_type"
        ]
        .astype(str)
        .str.strip()
    )

    # Convert nutrition columns to numeric again for safety
    nutrition_columns = [
        "meal_calories",
        "meal_carbs_g",
        "meal_protein_g",
        "meal_fat_g",
        "meal_fiber_g"
    ]

    for column in nutrition_columns:

        meal_events[column] = (
            pd.to_numeric(
                meal_events[column],
                errors="coerce"
            )
        )

    # Remove impossible-to-time meal events.
    # We do NOT remove them from the original source.
    invalid_meal_timestamps = (
        meal_events[
            "timestamp"
        ]
        .isna()
        .sum()
    )

    meal_events = (
        meal_events[
            meal_events[
                "timestamp"
            ].notna()
        ]
        .copy()
    )

    meal_events = meal_events.sort_values(
        [
            "participant_id",
            "timestamp"
        ],
        kind="stable"
    ).reset_index(drop=True)

    duplicate_events = (
        meal_events.duplicated(
            subset=[
                "participant_id",
                "timestamp"
            ]
        ).sum()
    )

    print(
        "\nRecorded meal events:",
        len(meal_events)
    )

    print(
        "Invalid meal timestamps:",
        invalid_meal_timestamps
    )

    print(
        "Duplicate participant/meal timestamps:",
        duplicate_events
    )

    print(
        "\nMeal types:"
    )

    print(
        meal_events[
            "meal_type"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    if duplicate_events != 0:

        raise ValueError(
            "Duplicate meal timestamps detected. "
            "Inspect them before continuing."
        )

    return meal_events


# =========================================================
# 8. COMPLETENESS-AWARE WINDOW SUM
# =========================================================

def calculate_window_sum(
    values,
    left_positions,
    right_positions
):

    values = np.asarray(
        values,
        dtype=float
    )

    # Treat NaN as zero only during temporary arithmetic.
    safe_values = np.nan_to_num(
        values,
        nan=0.0
    )

    missing_flags = (
        np.isnan(values)
        .astype(int)
    )

    prefix_sum = np.concatenate(
        [
            [0.0],
            np.cumsum(
                safe_values
            )
        ]
    )

    prefix_missing = np.concatenate(
        [
            [0],
            np.cumsum(
                missing_flags
            )
        ]
    )

    totals = (
        prefix_sum[
            right_positions
        ]
        -
        prefix_sum[
            left_positions
        ]
    )

    missing_in_window = (
        prefix_missing[
            right_positions
        ]
        -
        prefix_missing[
            left_positions
        ]
    )

    event_count = (
        right_positions
        -
        left_positions
    )

    # No meal in window genuinely means zero nutrient intake
    # from recorded meals in that window.
    totals[
        event_count == 0
    ] = 0.0

    # If a meal exists but its nutrient quantity is missing,
    # preserve uncertainty rather than pretending it was zero.
    totals[
        (event_count > 0)
        &
        (missing_in_window > 0)
    ] = np.nan

    return totals


# =========================================================
# 9. CREATE TIME-SAFE MEAL FEATURES
# =========================================================

def create_meal_features(
    features,
    meal_events
):

    print("\n" + "=" * 70)
    print("CREATING TIME-SAFE MEAL/NUTRITION FEATURES")
    print("=" * 70)

    result = features.copy()

    # -----------------------------------------------------
    # Initialize features
    # -----------------------------------------------------

    result[
        "meal_event_at_t"
    ] = 0

    result[
        "recent_meal_2h"
    ] = 0

    result[
        "meal_count_last_2h"
    ] = 0

    result[
        "time_since_last_meal_min"
    ] = np.nan

    for output_column in (
        NUTRIENT_COLUMNS.values()
    ):

        result[
            output_column
        ] = 0.0

    # -----------------------------------------------------
    # Process each participant independently
    # -----------------------------------------------------

    for participant_id, participant_rows in (
        result.groupby(
            "participant_id",
            sort=False
        )
    ):

        row_indexes = (
            participant_rows
            .sort_values(
                "timestamp"
            )
            .index
        )

        query_times = (
            result.loc[
                row_indexes,
                "timestamp"
            ]
            .to_numpy(
                dtype="datetime64[ns]"
            )
        )

        participant_meals = (
            meal_events[
                meal_events[
                    "participant_id"
                ] == participant_id
            ]
            .sort_values(
                "timestamp"
            )
            .copy()
        )

        if participant_meals.empty:
            continue

        meal_times = (
            participant_meals[
                "timestamp"
            ]
            .to_numpy(
                dtype="datetime64[ns]"
            )
        )

        # -------------------------------------------------
        # Find all meals at or before prediction time t.
        # -------------------------------------------------

        right_positions = np.searchsorted(
            meal_times,
            query_times,
            side="right"
        )

        # -------------------------------------------------
        # Two-hour historical window:
        #
        # [t - 120 minutes, t]
        # -------------------------------------------------

        window_start = (
            query_times
            -
            np.timedelta64(
                MEAL_WINDOW_MINUTES,
                "m"
            )
        )

        left_positions = np.searchsorted(
            meal_times,
            window_start,
            side="left"
        )

        meal_counts = (
            right_positions
            -
            left_positions
        )

        recent_meal = (
            meal_counts > 0
        )

        # -------------------------------------------------
        # Most recent meal at or before t
        # -------------------------------------------------

        last_positions = (
            right_positions
            - 1
        )

        has_previous_meal = (
            last_positions >= 0
        )

        time_since = np.full(
            len(query_times),
            np.nan,
            dtype=float
        )

        time_since[
            has_previous_meal
        ] = (
            (
                query_times[
                    has_previous_meal
                ]
                -
                meal_times[
                    last_positions[
                        has_previous_meal
                    ]
                ]
            )
            /
            np.timedelta64(
                1,
                "m"
            )
        ).astype(float)

        meal_at_t = (
            np.isfinite(
                time_since
            )
            &
            np.isclose(
                time_since,
                0.0
            )
        )

        # -------------------------------------------------
        # Save basic meal features
        # -------------------------------------------------

        result.loc[
            row_indexes,
            "meal_event_at_t"
        ] = (
            meal_at_t
            .astype(int)
        )

        result.loc[
            row_indexes,
            "recent_meal_2h"
        ] = (
            recent_meal
            .astype(int)
        )

        result.loc[
            row_indexes,
            "meal_count_last_2h"
        ] = (
            meal_counts
            .astype(int)
        )

        result.loc[
            row_indexes,
            "time_since_last_meal_min"
        ] = (
            time_since
        )

        # -------------------------------------------------
        # Nutrition totals from meals inside the past
        # two-hour window.
        # -------------------------------------------------

        for (
            source_column,
            output_column
        ) in NUTRIENT_COLUMNS.items():

            values = (
                participant_meals[
                    source_column
                ]
                .to_numpy(
                    dtype=float
                )
            )

            totals = (
                calculate_window_sum(
                    values,
                    left_positions,
                    right_positions
                )
            )

            result.loc[
                row_indexes,
                output_column
            ] = totals

    return result


# =========================================================
# 10. VALIDATE LEAKAGE / CONSISTENCY
# =========================================================

def validate_features(
    data
):

    print("\n" + "=" * 70)
    print("VALIDATING MEAL FEATURES")
    print("=" * 70)

    negative_time = (
        data[
            "time_since_last_meal_min"
        ]
        .dropna()
        .lt(0)
        .sum()
    )

    invalid_recent_count = (
        (
            data[
                "recent_meal_2h"
            ] == 1
        )
        &
        (
            data[
                "meal_count_last_2h"
            ] < 1
        )
    ).sum()

    invalid_no_recent_count = (
        (
            data[
                "recent_meal_2h"
            ] == 0
        )
        &
        (
            data[
                "meal_count_last_2h"
            ] != 0
        )
    ).sum()

    event_without_recent = (
        (
            data[
                "meal_event_at_t"
            ] == 1
        )
        &
        (
            data[
                "recent_meal_2h"
            ] != 1
        )
    ).sum()

    print(
        "\nNegative time-since-meal values:",
        negative_time
    )

    print(
        "Recent-meal/count inconsistencies:",
        invalid_recent_count
    )

    print(
        "No-recent/count inconsistencies:",
        invalid_no_recent_count
    )

    print(
        "Meal-at-t/recent inconsistencies:",
        event_without_recent
    )

    print(
        "Missing prediction targets:",
        data[
            "target_glucose_30min"
        ]
        .isna()
        .sum()
    )

    if negative_time != 0:

        raise ValueError(
            "Future meal information detected."
        )

    if (
        invalid_recent_count != 0
        or
        invalid_no_recent_count != 0
        or
        event_without_recent != 0
    ):

        raise ValueError(
            "Meal feature consistency "
            "validation failed."
        )

    print(
        "\nMeal feature validation passed."
    )


# =========================================================
# 11. MISSINGNESS REPORT
# =========================================================

def create_missingness_report(
    data
):

    feature_columns = [
        "meal_event_at_t",
        "recent_meal_2h",
        "time_since_last_meal_min",
        "meal_count_last_2h",
        "meal_calories_last_2h",
        "meal_carbs_last_2h_g",
        "meal_protein_last_2h_g",
        "meal_fat_last_2h_g",
        "meal_fiber_last_2h_g"
    ]

    records = []

    for column in feature_columns:

        missing = (
            data[
                column
            ]
            .isna()
            .sum()
        )

        records.append(
            {
                "feature":
                    column,

                "total_rows":
                    len(data),

                "missing_count":
                    int(
                        missing
                    ),

                "missing_percent":
                    round(
                        missing
                        / len(data)
                        * 100,
                        4
                    )
            }
        )

    return pd.DataFrame(
        records
    )


# =========================================================
# 12. PARTICIPANT REPORT
# =========================================================

def create_participant_report(
    data,
    meal_events
):

    records = []

    for participant_id, group in (
        data.groupby(
            "participant_id",
            sort=True
        )
    ):

        participant_meal_count = (
            meal_events[
                meal_events[
                    "participant_id"
                ] == participant_id
            ]
            .shape[0]
        )

        records.append(
            {
                "participant_id":
                    participant_id,

                "prediction_rows":
                    len(group),

                "recorded_meal_events":
                    participant_meal_count,

                "rows_with_recent_meal_2h":
                    int(
                        group[
                            "recent_meal_2h"
                        ].sum()
                    ),

                "rows_at_meal_event":
                    int(
                        group[
                            "meal_event_at_t"
                        ].sum()
                    ),

                "rows_without_any_prior_meal":
                    int(
                        group[
                            "time_since_last_meal_min"
                        ]
                        .isna()
                        .sum()
                    )
            }
        )

    return pd.DataFrame(
        records
    )


# =========================================================
# 13. FEATURE DICTIONARY
# =========================================================

def create_dictionary():

    return pd.DataFrame(
        [
            {
                "feature":
                    "meal_event_at_t",

                "meaning":
                    "1 if a recorded CGMacros meal begins exactly at prediction time t",

                "unit":
                    "binary",

                "uses_future_information":
                    "No"
            },

            {
                "feature":
                    "recent_meal_2h",

                "meaning":
                    "1 if at least one recorded meal occurs from t-120 minutes through t",

                "unit":
                    "binary",

                "uses_future_information":
                    "No"
            },

            {
                "feature":
                    "time_since_last_meal_min",

                "meaning":
                    "Minutes since the latest recorded meal at or before prediction time t",

                "unit":
                    "minutes",

                "uses_future_information":
                    "No"
            },

            {
                "feature":
                    "meal_count_last_2h",

                "meaning":
                    "Number of recorded meals from t-120 minutes through t",

                "unit":
                    "count",

                "uses_future_information":
                    "No"
            },

            {
                "feature":
                    "meal_calories_last_2h",

                "meaning":
                    "Sum of estimated calories from recorded meals during the previous 2 hours through t",

                "unit":
                    "kcal",

                "uses_future_information":
                    "No"
            },

            {
                "feature":
                    "meal_carbs_last_2h_g",

                "meaning":
                    "Sum of estimated carbohydrates from recorded meals during the previous 2 hours through t",

                "unit":
                    "g",

                "uses_future_information":
                    "No"
            },

            {
                "feature":
                    "meal_protein_last_2h_g",

                "meaning":
                    "Sum of estimated protein from recorded meals during the previous 2 hours through t",

                "unit":
                    "g",

                "uses_future_information":
                    "No"
            },

            {
                "feature":
                    "meal_fat_last_2h_g",

                "meaning":
                    "Sum of estimated fat from recorded meals during the previous 2 hours through t",

                "unit":
                    "g",

                "uses_future_information":
                    "No"
            },

            {
                "feature":
                    "meal_fiber_last_2h_g",

                "meaning":
                    "Sum of estimated fiber from recorded meals during the previous 2 hours through t",

                "unit":
                    "g",

                "uses_future_information":
                    "No"
            }
        ]
    )


# =========================================================
# 14. SUMMARY
# =========================================================

def create_summary(
    data,
    meal_events
):

    return pd.DataFrame(
        [
            {
                "metric":
                    "rows",

                "value":
                    len(data)
            },

            {
                "metric":
                    "participants",

                "value":
                    data[
                        "participant_id"
                    ].nunique()
            },

            {
                "metric":
                    "recorded_meal_events",

                "value":
                    len(meal_events)
            },

            {
                "metric":
                    "meal_nutrition_features_created",

                "value":
                    9
            },

            {
                "metric":
                    "rows_with_recent_meal_2h",

                "value":
                    int(
                        data[
                            "recent_meal_2h"
                        ].sum()
                    )
            },

            {
                "metric":
                    "rows_without_any_prior_meal",

                "value":
                    int(
                        data[
                            "time_since_last_meal_min"
                        ]
                        .isna()
                        .sum()
                    )
            },

            {
                "metric":
                    "missing_prediction_targets",

                "value":
                    int(
                        data[
                            "target_glucose_30min"
                        ]
                        .isna()
                        .sum()
                    )
            }
        ]
    )


# =========================================================
# 15. SAVE OUTPUTS
# =========================================================

def save_outputs(
    data,
    summary,
    participant_report,
    missingness,
    dictionary
):

    print("\n" + "=" * 70)
    print("SAVING STEP 13 OUTPUTS")
    print("=" * 70)

    data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    participant_report.to_csv(
        PARTICIPANT_REPORT_FILE,
        index=False
    )

    missingness.to_csv(
        MISSINGNESS_FILE,
        index=False
    )

    dictionary.to_csv(
        DICTIONARY_FILE,
        index=False
    )

    print("\nSaved:")

    print(OUTPUT_FILE)
    print(SUMMARY_FILE)
    print(PARTICIPANT_REPORT_FILE)
    print(MISSINGNESS_FILE)
    print(DICTIONARY_FILE)


# =========================================================
# 16. FINAL SUMMARY
# =========================================================

def print_final_summary(
    data,
    meal_events,
    missingness
):

    print("\n" + "=" * 70)
    print("STEP 13 FINAL SUMMARY")
    print("=" * 70)

    print(
        "\nRows:",
        len(data)
    )

    print(
        "Participants:",
        data[
            "participant_id"
        ].nunique()
    )

    print(
        "Recorded meal events:",
        len(meal_events)
    )

    print(
        "Meal/nutrition features created: 9"
    )

    print(
        "Rows with recent meal "
        "(last 2 hours):",
        int(
            data[
                "recent_meal_2h"
            ].sum()
        )
    )

    print(
        "Rows without any prior meal:",
        int(
            data[
                "time_since_last_meal_min"
            ]
            .isna()
            .sum()
        )
    )

    print(
        "Negative time-since-meal values:",
        int(
            data[
                "time_since_last_meal_min"
            ]
            .dropna()
            .lt(0)
            .sum()
        )
    )

    print(
        "Missing prediction targets:",
        int(
            data[
                "target_glucose_30min"
            ]
            .isna()
            .sum()
        )
    )

    print(
        "\nMeal feature missingness:"
    )

    print(
        missingness.to_string(
            index=False
        )
    )


# =========================================================
# 17. MAIN
# =========================================================

def main():

    print("=" * 70)
    print("SMART BLOOD GLUCOSE ML PROJECT")
    print("STEP 13 - CGMACROS MEAL/NUTRITION FEATURES")
    print("=" * 70)

    (
        features,
        history
    ) = load_data()

    meal_events = (
        create_meal_event_table(
            history
        )
    )

    data = (
        create_meal_features(
            features,
            meal_events
        )
    )

    validate_features(
        data
    )

    missingness = (
        create_missingness_report(
            data
        )
    )

    participant_report = (
        create_participant_report(
            data,
            meal_events
        )
    )

    dictionary = (
        create_dictionary()
    )

    summary = (
        create_summary(
            data,
            meal_events
        )
    )

    save_outputs(
        data,
        summary,
        participant_report,
        missingness,
        dictionary
    )

    print_final_summary(
        data,
        meal_events,
        missingness
    )

    print("\n" + "=" * 70)
    print("STEP 13 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":

    main()