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

INPUT_FILE = (
    PROCESSED_DIR
    / "cgmacros_meal_features_ready.csv"
)

# Use the complete standardized timeline to determine the
# true beginning of each participant's recording.
HISTORY_FILE = (
    PROCESSED_DIR
    / "cgmacros_standardized_all.csv"
)


# =========================================================
# 3. OUTPUT FILES
# =========================================================

OUTPUT_FILE = (
    PROCESSED_DIR
    / "cgmacros_temporal_features_ready.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "cgmacros_temporal_feature_summary.csv"
)

PARTICIPANT_REPORT_FILE = (
    REPORT_DIR
    / "cgmacros_temporal_feature_by_participant.csv"
)

DICTIONARY_FILE = (
    REPORT_DIR
    / "cgmacros_temporal_feature_dictionary.csv"
)


# =========================================================
# 4. SETTINGS
# =========================================================

DAY_START_HOUR = 6
DAY_END_HOUR = 18


# =========================================================
# 5. LOAD DATA
# =========================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING STEP 14 INPUT DATA")
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

    data["participant_id"] = (
        data["participant_id"]
        .astype(str)
    )

    history["participant_id"] = (
        history["participant_id"]
        .astype(str)
    )

    data = data.sort_values(
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
        "\nInput rows:",
        len(data)
    )

    print(
        "Participants:",
        data[
            "participant_id"
        ].nunique()
    )

    print(
        "Missing timestamps:",
        data[
            "timestamp"
        ].isna().sum()
    )

    return data, history


# =========================================================
# 6. VALIDATE INPUT
# =========================================================

def validate_input(data):

    print("\n" + "=" * 70)
    print("VALIDATING STEP 14 INPUT")
    print("=" * 70)

    duplicate_count = (
        data.duplicated(
            subset=[
                "participant_id",
                "timestamp"
            ]
        ).sum()
    )

    print(
        "\nDuplicate participant/timestamps:",
        duplicate_count
    )

    print(
        "Missing prediction targets:",
        data[
            "target_glucose_30min"
        ].isna().sum()
    )

    if data["timestamp"].isna().any():

        raise ValueError(
            "Input contains invalid timestamps."
        )

    if duplicate_count != 0:

        raise ValueError(
            "Duplicate participant/timestamps detected."
        )

    if (
        data[
            "target_glucose_30min"
        ].isna().any()
    ):

        raise ValueError(
            "Prediction target is missing."
        )

    print(
        "\nInput validation passed."
    )


# =========================================================
# 7. GET PARTICIPANT RECORDING START TIMES
# =========================================================

def get_recording_start_times(history):

    print("\n" + "=" * 70)
    print("DETERMINING PARTICIPANT RECORDING START TIMES")
    print("=" * 70)

    valid_history = history[
        history[
            "timestamp"
        ].notna()
    ].copy()

    start_times = (
        valid_history
        .groupby(
            "participant_id"
        )["timestamp"]
        .min()
        .rename(
            "recording_start_timestamp"
        )
        .reset_index()
    )

    print(
        "\nParticipants with recording start:",
        start_times[
            "participant_id"
        ].nunique()
    )

    return start_times


# =========================================================
# 8. CREATE TEMPORAL FEATURES
# =========================================================

def create_temporal_features(
    data,
    start_times
):

    print("\n" + "=" * 70)
    print("CREATING TEMPORAL FEATURES")
    print("=" * 70)

    result = data.copy()

    # -----------------------------------------------------
    # Time within the day
    #
    # Use minutes so 10:30 is represented as 10.5 hours
    # rather than simply hour 10.
    # -----------------------------------------------------

    minutes_of_day = (
        result[
            "timestamp"
        ].dt.hour * 60
        +
        result[
            "timestamp"
        ].dt.minute
        +
        result[
            "timestamp"
        ].dt.second / 60.0
    )

    angle = (
        2.0
        *
        np.pi
        *
        minutes_of_day
        /
        (24.0 * 60.0)
    )

    result[
        "hour_of_day_sin"
    ] = np.sin(
        angle
    )

    result[
        "hour_of_day_cos"
    ] = np.cos(
        angle
    )

    # -----------------------------------------------------
    # Day / night
    #
    # 1 = 06:00 through 17:59
    # 0 = 18:00 through 05:59
    # -----------------------------------------------------

    result[
        "day_night_indicator"
    ] = (
        (
            result[
                "timestamp"
            ].dt.hour >= DAY_START_HOUR
        )
        &
        (
            result[
                "timestamp"
            ].dt.hour < DAY_END_HOUR
        )
    ).astype(int)

    # -----------------------------------------------------
    # Elapsed recording time
    #
    # Use each participant's earliest timestamp from the
    # full standardized dataset, not merely the first
    # feature-ready row.
    # -----------------------------------------------------

    result = result.merge(
        start_times,
        on="participant_id",
        how="left",
        validate="many_to_one"
    )

    result[
        "elapsed_recording_hours"
    ] = (
        (
            result[
                "timestamp"
            ]
            -
            result[
                "recording_start_timestamp"
            ]
        )
        .dt.total_seconds()
        /
        3600.0
    )

    return result


# =========================================================
# 9. VALIDATE TEMPORAL FEATURES
# =========================================================

def validate_temporal_features(data):

    print("\n" + "=" * 70)
    print("VALIDATING TEMPORAL FEATURES")
    print("=" * 70)

    temporal_columns = [
        "hour_of_day_sin",
        "hour_of_day_cos",
        "day_night_indicator",
        "elapsed_recording_hours"
    ]

    missing_counts = (
        data[
            temporal_columns
        ]
        .isna()
        .sum()
    )

    print(
        "\nMissing temporal features:"
    )

    print(
        missing_counts.to_string()
    )

    # -----------------------------------------------------
    # Sin/cos should remain in [-1, 1].
    # -----------------------------------------------------

    invalid_sin = (
        (
            data[
                "hour_of_day_sin"
            ] < -1
        )
        |
        (
            data[
                "hour_of_day_sin"
            ] > 1
        )
    ).sum()

    invalid_cos = (
        (
            data[
                "hour_of_day_cos"
            ] < -1
        )
        |
        (
            data[
                "hour_of_day_cos"
            ] > 1
        )
    ).sum()

    # -----------------------------------------------------
    # sin² + cos² should equal approximately 1.
    # -----------------------------------------------------

    circle_value = (
        data[
            "hour_of_day_sin"
        ] ** 2
        +
        data[
            "hour_of_day_cos"
        ] ** 2
    )

    invalid_circle = (
        ~np.isclose(
            circle_value,
            1.0,
            atol=1e-10
        )
    ).sum()

    invalid_day_night = (
        ~data[
            "day_night_indicator"
        ].isin(
            [
                0,
                1
            ]
        )
    ).sum()

    negative_elapsed = (
        data[
            "elapsed_recording_hours"
        ]
        .lt(0)
        .sum()
    )

    print(
        "\nSine values outside [-1,1]:",
        invalid_sin
    )

    print(
        "Cosine values outside [-1,1]:",
        invalid_cos
    )

    print(
        "Invalid sin/cos circle values:",
        invalid_circle
    )

    print(
        "Invalid day/night values:",
        invalid_day_night
    )

    print(
        "Negative elapsed recording times:",
        negative_elapsed
    )

    print(
        "Missing prediction targets:",
        data[
            "target_glucose_30min"
        ].isna().sum()
    )

    if (
        missing_counts.sum() != 0
        or invalid_sin != 0
        or invalid_cos != 0
        or invalid_circle != 0
        or invalid_day_night != 0
        or negative_elapsed != 0
    ):

        raise ValueError(
            "Temporal feature validation failed."
        )

    print(
        "\nTemporal feature validation passed."
    )


# =========================================================
# 10. PARTICIPANT REPORT
# =========================================================

def create_participant_report(data):

    records = []

    for participant_id, group in (
        data.groupby(
            "participant_id",
            sort=True
        )
    ):

        records.append(
            {
                "participant_id":
                    participant_id,

                "rows":
                    len(group),

                "recording_start_timestamp":
                    group[
                        "recording_start_timestamp"
                    ].iloc[0],

                "first_feature_timestamp":
                    group[
                        "timestamp"
                    ].min(),

                "last_feature_timestamp":
                    group[
                        "timestamp"
                    ].max(),

                "minimum_elapsed_hours":
                    group[
                        "elapsed_recording_hours"
                    ].min(),

                "maximum_elapsed_hours":
                    group[
                        "elapsed_recording_hours"
                    ].max(),

                "day_rows":
                    int(
                        group[
                            "day_night_indicator"
                        ].sum()
                    ),

                "night_rows":
                    int(
                        (
                            group[
                                "day_night_indicator"
                            ] == 0
                        ).sum()
                    )
            }
        )

    return pd.DataFrame(
        records
    )


# =========================================================
# 11. FEATURE DICTIONARY
# =========================================================

def create_dictionary():

    return pd.DataFrame(
        [
            {
                "feature":
                    "hour_of_day_sin",

                "meaning":
                    "Sine cyclical encoding of time within the 24-hour day",

                "unit":
                    "unitless",

                "uses_future_information":
                    "No"
            },

            {
                "feature":
                    "hour_of_day_cos",

                "meaning":
                    "Cosine cyclical encoding of time within the 24-hour day",

                "unit":
                    "unitless",

                "uses_future_information":
                    "No"
            },

            {
                "feature":
                    "day_night_indicator",

                "meaning":
                    "1 from 06:00 through 17:59 and 0 otherwise",

                "unit":
                    "binary",

                "uses_future_information":
                    "No"
            },

            {
                "feature":
                    "elapsed_recording_hours",

                "meaning":
                    "Hours elapsed since the participant's first recorded timestamp",

                "unit":
                    "hours",

                "uses_future_information":
                    "No"
            }
        ]
    )


# =========================================================
# 12. SUMMARY
# =========================================================

def create_summary(data):

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
                    "temporal_features_created",

                "value":
                    4
            },

            {
                "metric":
                    "day_rows",

                "value":
                    int(
                        data[
                            "day_night_indicator"
                        ].sum()
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
            },

            {
                "metric":
                    "missing_temporal_values",

                "value":
                    int(
                        data[
                            [
                                "hour_of_day_sin",
                                "hour_of_day_cos",
                                "day_night_indicator",
                                "elapsed_recording_hours"
                            ]
                        ]
                        .isna()
                        .sum()
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
# 13. SAVE OUTPUTS
# =========================================================

def save_outputs(
    data,
    summary,
    participant_report,
    dictionary
):

    print("\n" + "=" * 70)
    print("SAVING STEP 14 OUTPUTS")
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

    dictionary.to_csv(
        DICTIONARY_FILE,
        index=False
    )

    print("\nSaved:")

    print(OUTPUT_FILE)
    print(SUMMARY_FILE)
    print(PARTICIPANT_REPORT_FILE)
    print(DICTIONARY_FILE)


# =========================================================
# 14. FINAL SUMMARY
# =========================================================

def print_final_summary(data):

    print("\n" + "=" * 70)
    print("STEP 14 FINAL SUMMARY")
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
        int(
            data[
                "day_night_indicator"
            ].sum()
        )
    )

    print(
        "Night rows:",
        int(
            (
                data[
                    "day_night_indicator"
                ] == 0
            ).sum()
        )
    )

    print(
        "Negative elapsed recording times:",
        int(
            data[
                "elapsed_recording_hours"
            ]
            .lt(0)
            .sum()
        )
    )

    print(
        "Missing prediction targets:",
        data[
            "target_glucose_30min"
        ].isna().sum()
    )


# =========================================================
# 15. MAIN
# =========================================================

def main():

    print("=" * 70)
    print("SMART BLOOD GLUCOSE ML PROJECT")
    print("STEP 14 - CGMACROS TEMPORAL FEATURES")
    print("=" * 70)

    data, history = load_data()

    validate_input(
        data
    )

    start_times = (
        get_recording_start_times(
            history
        )
    )

    data = (
        create_temporal_features(
            data,
            start_times
        )
    )

    validate_temporal_features(
        data
    )

    participant_report = (
        create_participant_report(
            data
        )
    )

    dictionary = (
        create_dictionary()
    )

    summary = (
        create_summary(
            data
        )
    )

    save_outputs(
        data,
        summary,
        participant_report,
        dictionary
    )

    print_final_summary(
        data
    )

    print("\n" + "=" * 70)
    print("STEP 14 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":

    main()