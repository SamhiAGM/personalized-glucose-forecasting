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

INPUT_FILE = (
    PROCESSED_DIR
    / "shanghai_glucose_features_ready.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "shanghai_temporal_features_ready.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "shanghai_temporal_feature_summary.csv"
)

DICTIONARY_FILE = (
    REPORT_DIR
    / "shanghai_temporal_feature_dictionary.csv"
)


# =========================================================
# 2. LOAD STEP 5 DATA
# =========================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING STEP 5 FEATURE-READY DATA")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=[
            "timestamp",
            "target_timestamp_30min"
        ],
        low_memory=False
    )

    df = df.sort_values(
        [
            "session_id",
            "timestamp"
        ]
    ).reset_index(drop=True)

    print(
        "\nRows loaded:",
        len(df)
    )

    print(
        "Participants:",
        df["participant_id"].nunique()
    )

    print(
        "Sessions:",
        df["session_id"].nunique()
    )

    print(
        "Missing timestamps:",
        df["timestamp"].isna().sum()
    )

    return df


# =========================================================
# 3. CREATE CYCLICAL TIME FEATURES
# =========================================================

def create_cyclical_time_features(df):

    print("\n" + "=" * 70)
    print("CREATING HOUR-OF-DAY CYCLICAL FEATURES")
    print("=" * 70)

    data = df.copy()

    # -----------------------------------------
    # Convert timestamp into decimal hour.
    #
    # Example:
    # 14:30 = 14.5 hours
    # -----------------------------------------

    data[
        "hour_decimal"
    ] = (
        data["timestamp"].dt.hour
        +
        data["timestamp"].dt.minute / 60.0
        +
        data["timestamp"].dt.second / 3600.0
    )

    # -----------------------------------------
    # Map the 24-hour clock onto a circle
    # -----------------------------------------

    angle = (
        2
        * np.pi
        * data["hour_decimal"]
        / 24.0
    )

    data[
        "hour_of_day_sin"
    ] = np.sin(angle)

    data[
        "hour_of_day_cos"
    ] = np.cos(angle)

    return data


# =========================================================
# 4. CREATE DAY/NIGHT INDICATOR
# =========================================================

def create_day_night_feature(data):

    print("\n" + "=" * 70)
    print("CREATING DAY/NIGHT INDICATOR")
    print("=" * 70)

    data = data.copy()

    # Project convention:
    #
    # 06:00 <= time < 18:00  -> day = 1
    # otherwise              -> night = 0

    data[
        "day_night_indicator"
    ] = (
        (
            data["hour_decimal"] >= 6
        )
        &
        (
            data["hour_decimal"] < 18
        )
    ).astype(int)

    return data


# =========================================================
# 5. CREATE ELAPSED RECORDING TIME
# =========================================================

def create_elapsed_recording_time(data):

    print("\n" + "=" * 70)
    print("CREATING ELAPSED RECORDING TIME")
    print("=" * 70)

    data = data.copy()

    # Earliest available timestamp within
    # each monitoring session

    data[
        "session_start_timestamp"
    ] = (
        data.groupby(
            "session_id"
        )["timestamp"]
        .transform("min")
    )

    # Time elapsed since session start,
    # expressed in hours

    data[
        "elapsed_recording_hours"
    ] = (
        (
            data["timestamp"]
            -
            data[
                "session_start_timestamp"
            ]
        )
        .dt.total_seconds()
        / 3600.0
    )

    return data


# =========================================================
# 6. VALIDATE TEMPORAL FEATURES
# =========================================================

def validate_features(data):

    print("\n" + "=" * 70)
    print("VALIDATING TEMPORAL FEATURES")
    print("=" * 70)

    temporal_features = [
        "hour_of_day_sin",
        "hour_of_day_cos",
        "day_night_indicator",
        "elapsed_recording_hours"
    ]

    print("\nMissing values:")

    print(
        data[
            temporal_features
        ]
        .isna()
        .sum()
    )

    # -----------------------------------------
    # Sine/cosine range validation
    # -----------------------------------------

    sin_valid = (
        data[
            "hour_of_day_sin"
        ]
        .between(-1, 1)
        .all()
    )

    cos_valid = (
        data[
            "hour_of_day_cos"
        ]
        .between(-1, 1)
        .all()
    )

    print(
        "\nSine values within [-1, 1]:",
        sin_valid
    )

    print(
        "Cosine values within [-1, 1]:",
        cos_valid
    )

    # -----------------------------------------
    # Day/night must contain only 0 or 1
    # -----------------------------------------

    valid_day_night = (
        set(
            data[
                "day_night_indicator"
            ].dropna().unique()
        )
        .issubset({0, 1})
    )

    print(
        "Day/night only contains 0/1:",
        valid_day_night
    )

    # -----------------------------------------
    # Elapsed time must never be negative
    # -----------------------------------------

    negative_elapsed = (
        data[
            "elapsed_recording_hours"
        ] < 0
    ).sum()

    print(
        "Negative elapsed-time rows:",
        negative_elapsed
    )

    # -----------------------------------------
    # Verify first retained row of each
    # session has elapsed time = 0
    # -----------------------------------------

    first_elapsed = (
        data.sort_values(
            [
                "session_id",
                "timestamp"
            ]
        )
        .groupby(
            "session_id"
        )
        .first()[
            "elapsed_recording_hours"
        ]
    )

    first_not_zero = (
        ~np.isclose(
            first_elapsed,
            0.0
        )
    ).sum()

    print(
        "Sessions whose first row is "
        "not elapsed=0:",
        first_not_zero
    )

    # -----------------------------------------
    # Ensure target remains valid
    # -----------------------------------------

    missing_targets = (
        data[
            "target_glucose_30min"
        ]
        .isna()
        .sum()
    )

    print(
        "Missing prediction targets:",
        missing_targets
    )

    if not sin_valid:
        raise ValueError(
            "Hour sine validation failed."
        )

    if not cos_valid:
        raise ValueError(
            "Hour cosine validation failed."
        )

    if not valid_day_night:
        raise ValueError(
            "Day/night validation failed."
        )

    if negative_elapsed != 0:
        raise ValueError(
            "Negative elapsed times detected."
        )

    if first_not_zero != 0:
        raise ValueError(
            "Elapsed recording time "
            "validation failed."
        )

    if missing_targets != 0:
        raise ValueError(
            "Prediction targets were lost."
        )

    print(
        "\nTemporal feature validation passed."
    )


# =========================================================
# 7. FEATURE DICTIONARY
# =========================================================

def create_feature_dictionary():

    return pd.DataFrame(
        [
            {
                "feature":
                    "hour_of_day_sin",

                "meaning":
                    "Sine encoding of time within the 24-hour day",

                "unit":
                    "unitless",

                "uses_future_information":
                    "No"
            },

            {
                "feature":
                    "hour_of_day_cos",

                "meaning":
                    "Cosine encoding of time within the 24-hour day",

                "unit":
                    "unitless",

                "uses_future_information":
                    "No"
            },

            {
                "feature":
                    "day_night_indicator",

                "meaning":
                    "1 for 06:00-17:59 and 0 otherwise",

                "unit":
                    "binary",

                "uses_future_information":
                    "No"
            },

            {
                "feature":
                    "elapsed_recording_hours",

                "meaning":
                    "Hours elapsed since the beginning of the monitoring session",

                "unit":
                    "hours",

                "uses_future_information":
                    "No"
            }
        ]
    )


# =========================================================
# 8. CREATE SUMMARY
# =========================================================

def create_summary(data):

    temporal_features = [
        "hour_of_day_sin",
        "hour_of_day_cos",
        "day_night_indicator",
        "elapsed_recording_hours"
    ]

    complete_rows = (
        data[
            temporal_features
        ]
        .notna()
        .all(axis=1)
        .sum()
    )

    return pd.DataFrame(
        [
            {
                "metric":
                    "input_rows",

                "value":
                    len(data)
            },

            {
                "metric":
                    "output_rows",

                "value":
                    len(data)
            },

            {
                "metric":
                    "rows_with_complete_temporal_features",

                "value":
                    complete_rows
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
                    "temporal_features_created",

                "value":
                    len(
                        temporal_features
                    )
            },

            {
                "metric":
                    "day_rows",

                "value":
                    int(
                        (
                            data[
                                "day_night_indicator"
                            ] == 1
                        ).sum()
                    )
            },

            {
                "metric":
                    "night_rows",

                "value":
                    int(
                        (
                            data[
                                "day_night_indicator"
                            ] == 0
                        ).sum()
                    )
            }
        ]
    )


# =========================================================
# 9. SAVE OUTPUTS
# =========================================================

def save_outputs(
    data,
    summary,
    dictionary
):

    print("\n" + "=" * 70)
    print("SAVING STEP 6 OUTPUTS")
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

    print("\nSaved:")
    print(OUTPUT_FILE)
    print(SUMMARY_FILE)
    print(DICTIONARY_FILE)


# =========================================================
# 10. FINAL SUMMARY
# =========================================================

def print_final_summary(data):

    print("\n" + "=" * 70)
    print("STEP 6 FINAL SUMMARY")
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
        "Temporal features created: 4"
    )

    print(
        "Missing hour sine:",
        data[
            "hour_of_day_sin"
        ].isna().sum()
    )

    print(
        "Missing hour cosine:",
        data[
            "hour_of_day_cos"
        ].isna().sum()
    )

    print(
        "Missing day/night:",
        data[
            "day_night_indicator"
        ].isna().sum()
    )

    print(
        "Missing elapsed time:",
        data[
            "elapsed_recording_hours"
        ].isna().sum()
    )

    print(
        "Day rows:",
        (
            data[
                "day_night_indicator"
            ] == 1
        ).sum()
    )

    print(
        "Night rows:",
        (
            data[
                "day_night_indicator"
            ] == 0
        ).sum()
    )

    print(
        "Missing prediction targets:",
        data[
            "target_glucose_30min"
        ].isna().sum()
    )


# =========================================================
# 11. MAIN
# =========================================================

def main():

    print("=" * 70)
    print("SMART BLOOD GLUCOSE ML PROJECT")
    print("STEP 6 - TEMPORAL FEATURES")
    print("=" * 70)

    data = load_data()

    data = (
        create_cyclical_time_features(
            data
        )
    )

    data = (
        create_day_night_feature(
            data
        )
    )

    data = (
        create_elapsed_recording_time(
            data
        )
    )

    validate_features(
        data
    )

    summary = create_summary(
        data
    )

    dictionary = (
        create_feature_dictionary()
    )

    save_outputs(
        data,
        summary,
        dictionary
    )

    print_final_summary(
        data
    )

    print("\n" + "=" * 70)
    print("STEP 6 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()