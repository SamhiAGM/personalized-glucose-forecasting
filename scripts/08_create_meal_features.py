from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# 1. PATHS
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

INPUT_FILE = (
    PROCESSED_DIR
    / "shanghai_participant_features_ready.csv"
)

HISTORY_FILE = (
    PROCESSED_DIR
    / "shanghai_cgm_clean.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "shanghai_meal_features_ready.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "shanghai_meal_feature_summary.csv"
)

DICTIONARY_FILE = (
    REPORT_DIR
    / "shanghai_meal_feature_dictionary.csv"
)

SESSION_REPORT_FILE = (
    REPORT_DIR
    / "shanghai_meal_feature_by_session.csv"
)


# =========================================================
# 2. SETTINGS
# =========================================================

RECENT_MEAL_WINDOW_MINUTES = 120


# =========================================================
# 3. LOAD DATA
# =========================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING STEP 8 INPUT DATA")
    print("=" * 70)

    data = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamp"],
        low_memory=False
    )

    history = pd.read_csv(
        HISTORY_FILE,
        parse_dates=["timestamp"],
        low_memory=False
    )

    data["session_id"] = (
        data["session_id"]
        .astype(str)
    )

    history["session_id"] = (
        history["session_id"]
        .astype(str)
    )

    print(
        "\nFeature-ready rows:",
        len(data)
    )

    print(
        "Participants:",
        data["participant_id"].nunique()
    )

    print(
        "Sessions:",
        data["session_id"].nunique()
    )

    return data, history


# =========================================================
# 4. DETECT MEAL COLUMNS
# =========================================================

def detect_meal_columns(history):

    meal_columns = [
        column
        for column in history.columns
        if "dietary_intake"
        in column.lower()
    ]

    print("\nMeal-related source columns:")

    for column in meal_columns:
        print(
            f" - {column}"
        )

    if not meal_columns:

        raise ValueError(
            "No dietary intake columns found."
        )

    return meal_columns


# =========================================================
# 5. IDENTIFY MEAL EVENTS
# =========================================================

def nonempty_text(series):

    text = (
        series
        .astype("string")
        .str.strip()
    )

    return (
        text.notna()
        &
        (~text.str.lower().isin(
            [
                "",
                "nan",
                "none"
            ]
        ))
    )


def create_meal_event_history(
    history,
    meal_columns
):

    print("\n" + "=" * 70)
    print("IDENTIFYING RECORDED MEAL EVENTS")
    print("=" * 70)

    meal_history = history[
        [
            "session_id",
            "timestamp"
        ]
        +
        meal_columns
    ].copy()

    meal_masks = []

    for column in meal_columns:

        meal_masks.append(
            nonempty_text(
                meal_history[column]
            )
        )

    meal_event_mask = (
        pd.concat(
            meal_masks,
            axis=1
        )
        .any(axis=1)
    )

    meal_history[
        "meal_event"
    ] = (
        meal_event_mask
        .astype(int)
    )

    # -----------------------------------------------------
    # Whether descriptive meal text is available.
    #
    # This is for auditing only, not required as an
    # ML feature.
    # -----------------------------------------------------

    placeholders = {
        "data not available",
        "not available",
        "no data",
        "unknown",
        "n/a",
        "na"
    }

    description_masks = []

    for column in meal_columns:

        text = (
            meal_history[column]
            .astype("string")
            .str.strip()
        )

        meaningful = (
            text.notna()
            &
            (~text.str.lower().isin(
                placeholders
            ))
            &
            (text != "")
        )

        description_masks.append(
            meaningful
        )

    meal_history[
        "meal_description_available"
    ] = (
        pd.concat(
            description_masks,
            axis=1
        )
        .any(axis=1)
        .astype(int)
    )

    print(
        "\nRecorded meal events:",
        meal_history[
            "meal_event"
        ].sum()
    )

    print(
        "Meal events with usable description:",
        (
            meal_history[
                "meal_event"
            ]
            &
            meal_history[
                "meal_description_available"
            ]
        ).sum()
    )

    return meal_history


# =========================================================
# 6. CREATE TIME-SAFE MEAL FEATURES
# =========================================================

def create_meal_features(
    data,
    meal_history
):

    print("\n" + "=" * 70)
    print("CREATING TIME-SAFE MEAL FEATURES")
    print("=" * 70)

    result = data.copy()

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

    # Only rows representing recorded meals
    meal_events = meal_history[
        meal_history[
            "meal_event"
        ] == 1
    ][
        [
            "session_id",
            "timestamp"
        ]
    ].copy()

    meal_events = (
        meal_events
        .drop_duplicates()
        .sort_values(
            [
                "session_id",
                "timestamp"
            ]
        )
    )

    # -----------------------------------------------------
    # Process each monitoring session separately.
    #
    # This prevents meals from another participant/session
    # being attached accidentally.
    # -----------------------------------------------------

    for session_id, row_indexes in (
        result.groupby(
            "session_id",
            sort=False
        ).groups.items()
    ):

        row_indexes = list(
            row_indexes
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

        session_meals = (
            meal_events[
                meal_events[
                    "session_id"
                ] == session_id
            ]["timestamp"]
            .sort_values()
            .to_numpy(
                dtype="datetime64[ns]"
            )
        )

        if len(session_meals) == 0:
            continue

        # Number of meals at or before current time
        right_positions = (
            np.searchsorted(
                session_meals,
                query_times,
                side="right"
            )
        )

        # Beginning of the 2-hour historical window
        window_start = (
            query_times
            -
            np.timedelta64(
                RECENT_MEAL_WINDOW_MINUTES,
                "m"
            )
        )

        left_positions = (
            np.searchsorted(
                session_meals,
                window_start,
                side="left"
            )
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
        # Find the last meal at or before current time
        # -------------------------------------------------

        last_meal_position = (
            right_positions - 1
        )

        valid_last_meal = (
            last_meal_position >= 0
        )

        time_since = np.full(
            len(query_times),
            np.nan,
            dtype=float
        )

        time_since[
            valid_last_meal
        ] = (
            (
                query_times[
                    valid_last_meal
                ]
                -
                session_meals[
                    last_meal_position[
                        valid_last_meal
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
        ] = time_since

    return result


# =========================================================
# 7. VALIDATE FEATURES
# =========================================================

def validate_features(data):

    print("\n" + "=" * 70)
    print("VALIDATING MEAL FEATURES")
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
        "Sessions:",
        data[
            "session_id"
        ].nunique()
    )

    negative_time = (
        data[
            "time_since_last_meal_min"
        ]
        .dropna()
        .lt(0)
        .sum()
    )

    print(
        "Negative time-since-meal values:",
        negative_time
    )

    invalid_recent = (
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

    print(
        "Recent-meal/count inconsistencies:",
        invalid_recent
    )

    print(
        "Missing prediction targets:",
        data[
            "target_glucose_30min"
        ].isna().sum()
    )

    if negative_time != 0:

        raise ValueError(
            "Future meal information detected."
        )

    if invalid_recent != 0:

        raise ValueError(
            "Meal feature consistency check failed."
        )

    print(
        "\nMeal feature validation passed."
    )


# =========================================================
# 8. SESSION REPORT
# =========================================================

def create_session_report(
    data,
    meal_history
):

    meal_counts = (
        meal_history.groupby(
            "session_id"
        )[
            "meal_event"
        ]
        .sum()
        .rename(
            "recorded_meal_events"
        )
    )

    session_report = (
        data.groupby(
            [
                "participant_id",
                "session_id"
            ],
            as_index=False
        )
        .agg(
            rows=(
                "timestamp",
                "size"
            ),

            rows_with_recent_meal_2h=(
                "recent_meal_2h",
                "sum"
            ),

            rows_with_meal_at_t=(
                "meal_event_at_t",
                "sum"
            )
        )
    )

    session_report = (
        session_report
        .merge(
            meal_counts,
            on="session_id",
            how="left"
        )
    )

    session_report[
        "recorded_meal_events"
    ] = (
        session_report[
            "recorded_meal_events"
        ]
        .fillna(0)
        .astype(int)
    )

    return session_report


# =========================================================
# 9. FEATURE DICTIONARY
# =========================================================

def create_dictionary():

    return pd.DataFrame(
        [
            {
                "feature":
                    "meal_event_at_t",

                "meaning":
                    "1 when a recorded meal event occurs exactly at prediction time t",

                "unit":
                    "binary",

                "uses_future_information":
                    "No"
            },

            {
                "feature":
                    "recent_meal_2h",

                "meaning":
                    "1 when at least one recorded meal exists from t-120 minutes through t",

                "unit":
                    "binary",

                "uses_future_information":
                    "No"
            },

            {
                "feature":
                    "time_since_last_meal_min",

                "meaning":
                    "Minutes since most recent recorded meal at or before prediction time t",

                "unit":
                    "minutes",

                "uses_future_information":
                    "No"
            },

            {
                "feature":
                    "meal_count_last_2h",

                "meaning":
                    "Number of recorded meal events from t-120 minutes through t",

                "unit":
                    "count",

                "uses_future_information":
                    "No"
            }
        ]
    )


# =========================================================
# 10. SUMMARY
# =========================================================

def create_summary(
    data,
    meal_history
):

    no_prior_meal = (
        data[
            "time_since_last_meal_min"
        ]
        .isna()
        .sum()
    )

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
                    "sessions",
                "value":
                    data[
                        "session_id"
                    ].nunique()
            },

            {
                "metric":
                    "meal_timing_features_created",
                "value":
                    4
            },

            {
                "metric":
                    "recorded_meal_events",
                "value":
                    int(
                        meal_history[
                            "meal_event"
                        ].sum()
                    )
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
                        no_prior_meal
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
# 11. SAVE
# =========================================================

def save_outputs(
    data,
    summary,
    dictionary,
    session_report
):

    print("\n" + "=" * 70)
    print("SAVING STEP 8 OUTPUTS")
    print("=" * 70)

    data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    dictionary.to_csv(
        DICTIONARY_FILE,
        index=False
    )

    session_report.to_csv(
        SESSION_REPORT_FILE,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT_FILE)
    print(SUMMARY_FILE)
    print(DICTIONARY_FILE)
    print(SESSION_REPORT_FILE)


# =========================================================
# 12. FINAL SUMMARY
# =========================================================

def print_final_summary(
    data,
    meal_history
):

    print("\n" + "=" * 70)
    print("STEP 8 FINAL SUMMARY")
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
        "Sessions:",
        data[
            "session_id"
        ].nunique()
    )

    print(
        "Meal timing features created: 4"
    )

    print(
        "Recorded meal events:",
        int(
            meal_history[
                "meal_event"
            ].sum()
        )
    )

    print(
        "Rows with a recent meal "
        "(last 2 hours):",
        int(
            data[
                "recent_meal_2h"
            ].sum()
        )
    )

    print(
        "Rows without any prior "
        "recorded meal:",
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


# =========================================================
# 13. MAIN
# =========================================================

def main():

    print("=" * 70)
    print("SMART BLOOD GLUCOSE ML PROJECT")
    print("STEP 8 - SHANGHAI MEAL FEATURES")
    print("=" * 70)

    data, history = load_data()

    meal_columns = (
        detect_meal_columns(
            history
        )
    )

    meal_history = (
        create_meal_event_history(
            history,
            meal_columns
        )
    )

    data = create_meal_features(
        data,
        meal_history
    )

    validate_features(
        data
    )

    session_report = (
        create_session_report(
            data,
            meal_history
        )
    )

    dictionary = (
        create_dictionary()
    )

    summary = create_summary(
        data,
        meal_history
    )

    save_outputs(
        data,
        summary,
        dictionary,
        session_report
    )

    print_final_summary(
        data,
        meal_history
    )

    print("\n" + "=" * 70)
    print("STEP 8 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()