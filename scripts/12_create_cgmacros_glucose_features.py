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

HISTORY_FILE = (
    PROCESSED_DIR
    / "cgmacros_glucose_clean.csv"
)

TARGET_READY_FILE = (
    PROCESSED_DIR
    / "cgmacros_target_ready.csv"
)


# =========================================================
# 3. OUTPUT FILES
# =========================================================

OUTPUT_FILE = (
    PROCESSED_DIR
    / "cgmacros_glucose_features_ready.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "cgmacros_glucose_feature_summary.csv"
)

PARTICIPANT_REPORT_FILE = (
    REPORT_DIR
    / "cgmacros_glucose_feature_by_participant.csv"
)

DICTIONARY_FILE = (
    REPORT_DIR
    / "cgmacros_glucose_feature_dictionary.csv"
)


# =========================================================
# 4. SETTINGS
# =========================================================

LAG_MINUTES = [
    15,
    30,
    45,
    60
]

ROLLING_WINDOW_MINUTES = 60


# =========================================================
# 5. LOAD DATA
# =========================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING STEP 12 INPUT DATA")
    print("=" * 70)

    history = pd.read_csv(
        HISTORY_FILE,
        parse_dates=["timestamp"],
        low_memory=False
    )

    target_ready = pd.read_csv(
        TARGET_READY_FILE,
        parse_dates=[
            "timestamp",
            "target_timestamp_30min"
        ],
        low_memory=False
    )

    history[
        "participant_id"
    ] = (
        history[
            "participant_id"
        ]
        .astype(str)
    )

    target_ready[
        "participant_id"
    ] = (
        target_ready[
            "participant_id"
        ]
        .astype(str)
    )

    history = history.sort_values(
        [
            "participant_id",
            "timestamp"
        ],
        kind="stable"
    ).reset_index(drop=True)

    target_ready = target_ready.sort_values(
        [
            "participant_id",
            "timestamp"
        ],
        kind="stable"
    ).reset_index(drop=True)

    print(
        "\nHistory rows:",
        len(history)
    )

    print(
        "Target-ready rows:",
        len(target_ready)
    )

    print(
        "Participants in history:",
        history[
            "participant_id"
        ].nunique()
    )

    print(
        "Participants in target-ready file:",
        target_ready[
            "participant_id"
        ].nunique()
    )

    return (
        history,
        target_ready
    )


# =========================================================
# 6. VALIDATE HISTORY
# =========================================================

def validate_history(
    history
):

    print("\n" + "=" * 70)
    print("VALIDATING GLUCOSE HISTORY")
    print("=" * 70)

    duplicates = (
        history.duplicated(
            subset=[
                "participant_id",
                "timestamp"
            ]
        ).sum()
    )

    print(
        "\nMissing timestamps:",
        history[
            "timestamp"
        ].isna().sum()
    )

    print(
        "Missing glucose:",
        history[
            "glucose_mg_dl"
        ].isna().sum()
    )

    print(
        "Duplicate participant/timestamps:",
        duplicates
    )

    if (
        history[
            "timestamp"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "History contains invalid timestamps."
        )

    if (
        history[
            "glucose_mg_dl"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "History contains missing glucose."
        )

    if duplicates != 0:

        raise ValueError(
            "History contains duplicate "
            "participant/timestamp pairs."
        )

    print(
        "\nHistory validation passed."
    )


# =========================================================
# 7. CREATE EXACT LAG FEATURES
# =========================================================

def create_lag_features(
    target_ready,
    history
):

    print("\n" + "=" * 70)
    print("CREATING EXACT GLUCOSE LAGS")
    print("=" * 70)

    data = target_ready.copy()

    # Rename primary glucose at prediction time
    data[
        "current_glucose"
    ] = (
        data[
            "glucose_mg_dl"
        ]
    )

    # -----------------------------------------------------
    # Lookup source:
    #
    # participant_id
    # timestamp
    # glucose_mg_dl
    # -----------------------------------------------------

    lookup_base = history[
        [
            "participant_id",
            "timestamp",
            "glucose_mg_dl"
        ]
    ].copy()

    for lag_minutes in LAG_MINUTES:

        print(
            f"Creating exact "
            f"{lag_minutes}-minute lag..."
        )

        lag_timestamp_column = (
            f"lag_timestamp_{lag_minutes}min"
        )

        lag_feature_column = (
            f"glucose_lag_{lag_minutes}min"
        )

        # Requested historical timestamp
        data[
            lag_timestamp_column
        ] = (
            data[
                "timestamp"
            ]
            -
            pd.Timedelta(
                minutes=lag_minutes
            )
        )

        lookup = lookup_base.rename(
            columns={
                "timestamp":
                    lag_timestamp_column,

                "glucose_mg_dl":
                    lag_feature_column
            }
        )

        data = data.merge(
            lookup,
            on=[
                "participant_id",
                lag_timestamp_column
            ],
            how="left",
            validate="many_to_one"
        )

    return data


# =========================================================
# 8. CREATE CHANGE AND SLOPE FEATURES
# =========================================================

def create_change_features(
    data
):

    print("\n" + "=" * 70)
    print("CREATING GLUCOSE CHANGE FEATURES")
    print("=" * 70)

    data = data.copy()

    data[
        "glucose_change_15min"
    ] = (
        data[
            "current_glucose"
        ]
        -
        data[
            "glucose_lag_15min"
        ]
    )

    data[
        "glucose_change_30min"
    ] = (
        data[
            "current_glucose"
        ]
        -
        data[
            "glucose_lag_30min"
        ]
    )

    data[
        "glucose_slope_15min"
    ] = (
        data[
            "glucose_change_15min"
        ]
        / 15.0
    )

    return data


# =========================================================
# 9. CREATE PAST-ONLY ROLLING FEATURES
# =========================================================

def create_rolling_features(
    data,
    history
):

    print("\n" + "=" * 70)
    print("CREATING 60-MINUTE ROLLING FEATURES")
    print("=" * 70)

    rolling_records = []

    for participant_id, group in (
        history.groupby(
            "participant_id",
            sort=False
        )
    ):

        group = (
            group[
                [
                    "timestamp",
                    "glucose_mg_dl"
                ]
            ]
            .sort_values(
                "timestamp"
            )
            .copy()
        )

        group = group.set_index(
            "timestamp"
        )

        # -------------------------------------------------
        # Past-only 60-minute window ending at time t.
        #
        # No observations after t can enter the window.
        # -------------------------------------------------

        rolling = (
            group[
                "glucose_mg_dl"
            ]
            .rolling(
                window=f"{ROLLING_WINDOW_MINUTES}min",
                closed="both",
                min_periods=2
            )
        )

        result = pd.DataFrame(
            {
                "timestamp":
                    group.index,

                "rolling_glucose_mean_60min":
                    rolling.mean().values,

                "rolling_glucose_std_60min":
                    rolling.std().values,

                "rolling_glucose_min_60min":
                    rolling.min().values,

                "rolling_glucose_max_60min":
                    rolling.max().values
            }
        )

        result[
            "participant_id"
        ] = participant_id

        rolling_records.append(
            result
        )

    rolling_features = pd.concat(
        rolling_records,
        ignore_index=True
    )

    data = data.merge(
        rolling_features,
        on=[
            "participant_id",
            "timestamp"
        ],
        how="left",
        validate="many_to_one"
    )

    return data


# =========================================================
# 10. VALIDATE FEATURE LEAKAGE
# =========================================================

def validate_time_direction(
    data
):

    print("\n" + "=" * 70)
    print("VALIDATING FEATURE TIME DIRECTION")
    print("=" * 70)

    total_future_lags = 0

    for lag_minutes in LAG_MINUTES:

        lag_timestamp_column = (
            f"lag_timestamp_{lag_minutes}min"
        )

        expected = (
            data[
                "timestamp"
            ]
            -
            pd.Timedelta(
                minutes=lag_minutes
            )
        )

        incorrect = (
            data[
                lag_timestamp_column
            ]
            != expected
        ).sum()

        total_future_lags += (
            incorrect
        )

        print(
            f"{lag_minutes}-minute "
            f"lag timestamp errors:",
            incorrect
        )

    if total_future_lags != 0:

        raise ValueError(
            "Lag timestamp validation failed."
        )

    print(
        "\nNo future information "
        "used in lag timestamps."
    )


# =========================================================
# 11. DETERMINE FEATURE-READY ROWS
# =========================================================

def create_feature_ready(
    data
):

    print("\n" + "=" * 70)
    print("CREATING FEATURE-READY DATASET")
    print("=" * 70)

    required_features = [
        "current_glucose",

        "glucose_lag_15min",
        "glucose_lag_30min",
        "glucose_lag_45min",
        "glucose_lag_60min",

        "glucose_change_15min",
        "glucose_change_30min",
        "glucose_slope_15min",

        "rolling_glucose_mean_60min",
        "rolling_glucose_std_60min",
        "rolling_glucose_min_60min",
        "rolling_glucose_max_60min",

        "target_glucose_30min"
    ]

    complete_history = (
        data[
            required_features
        ]
        .notna()
        .all(axis=1)
    )

    ready = (
        data[
            complete_history
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    rows_removed = (
        len(data)
        -
        len(ready)
    )

    availability = (
        len(ready)
        / len(data)
        * 100
    )

    print(
        "\nInput target-ready rows:",
        len(data)
    )

    print(
        "Feature-ready rows:",
        len(ready)
    )

    print(
        "Rows without complete "
        "glucose history:",
        rows_removed
    )

    print(
        "Feature availability:",
        round(
            availability,
            2
        ),
        "%"
    )

    print(
        "Missing prediction targets:",
        ready[
            "target_glucose_30min"
        ]
        .isna()
        .sum()
    )

    return ready


# =========================================================
# 12. PARTICIPANT REPORT
# =========================================================

def create_participant_report(
    data,
    ready
):

    input_counts = (
        data.groupby(
            "participant_id"
        )
        .size()
        .rename(
            "input_rows"
        )
    )

    ready_counts = (
        ready.groupby(
            "participant_id"
        )
        .size()
        .rename(
            "feature_ready_rows"
        )
    )

    report = pd.concat(
        [
            input_counts,
            ready_counts
        ],
        axis=1
    ).fillna(0)

    report[
        "feature_ready_rows"
    ] = (
        report[
            "feature_ready_rows"
        ]
        .astype(int)
    )

    report[
        "rows_removed"
    ] = (
        report[
            "input_rows"
        ]
        -
        report[
            "feature_ready_rows"
        ]
    )

    report[
        "feature_availability_percent"
    ] = (
        report[
            "feature_ready_rows"
        ]
        /
        report[
            "input_rows"
        ]
        * 100
    ).round(2)

    report = (
        report
        .reset_index()
    )

    return report


# =========================================================
# 13. FEATURE DICTIONARY
# =========================================================

def create_dictionary():

    rows = [
        {
            "feature":
                "current_glucose",
            "meaning":
                "Libre glucose at prediction time t",
            "unit":
                "mg/dL"
        },

        {
            "feature":
                "glucose_lag_15min",
            "meaning":
                "Libre glucose exactly 15 minutes before t",
            "unit":
                "mg/dL"
        },

        {
            "feature":
                "glucose_lag_30min",
            "meaning":
                "Libre glucose exactly 30 minutes before t",
            "unit":
                "mg/dL"
        },

        {
            "feature":
                "glucose_lag_45min",
            "meaning":
                "Libre glucose exactly 45 minutes before t",
            "unit":
                "mg/dL"
        },

        {
            "feature":
                "glucose_lag_60min",
            "meaning":
                "Libre glucose exactly 60 minutes before t",
            "unit":
                "mg/dL"
        },

        {
            "feature":
                "glucose_change_15min",
            "meaning":
                "Current glucose minus 15-minute lag",
            "unit":
                "mg/dL"
        },

        {
            "feature":
                "glucose_change_30min",
            "meaning":
                "Current glucose minus 30-minute lag",
            "unit":
                "mg/dL"
        },

        {
            "feature":
                "glucose_slope_15min",
            "meaning":
                "15-minute glucose change divided by 15 minutes",
            "unit":
                "mg/dL/min"
        },

        {
            "feature":
                "rolling_glucose_mean_60min",
            "meaning":
                "Mean Libre glucose during the past 60 minutes through t",
            "unit":
                "mg/dL"
        },

        {
            "feature":
                "rolling_glucose_std_60min",
            "meaning":
                "Standard deviation of Libre glucose during the past 60 minutes through t",
            "unit":
                "mg/dL"
        },

        {
            "feature":
                "rolling_glucose_min_60min",
            "meaning":
                "Minimum Libre glucose during the past 60 minutes through t",
            "unit":
                "mg/dL"
        },

        {
            "feature":
                "rolling_glucose_max_60min",
            "meaning":
                "Maximum Libre glucose during the past 60 minutes through t",
            "unit":
                "mg/dL"
        }
    ]

    dictionary = pd.DataFrame(
        rows
    )

    dictionary[
        "uses_future_information"
    ] = "No"

    return dictionary


# =========================================================
# 14. CREATE SUMMARY
# =========================================================

def create_summary(
    data,
    ready
):

    return pd.DataFrame(
        [
            {
                "metric":
                    "input_target_ready_rows",

                "value":
                    len(data)
            },

            {
                "metric":
                    "feature_ready_rows",

                "value":
                    len(ready)
            },

            {
                "metric":
                    "rows_without_complete_history",

                "value":
                    len(data)
                    -
                    len(ready)
            },

            {
                "metric":
                    "feature_availability_percent",

                "value":
                    round(
                        len(ready)
                        /
                        len(data)
                        * 100,
                        2
                    )
            },

            {
                "metric":
                    "participants_retained",

                "value":
                    ready[
                        "participant_id"
                    ].nunique()
            },

            {
                "metric":
                    "glucose_history_features_created",

                "value":
                    12
            },

            {
                "metric":
                    "missing_targets_in_ready_file",

                "value":
                    int(
                        ready[
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
    ready,
    summary,
    participant_report,
    dictionary
):

    print("\n" + "=" * 70)
    print("SAVING STEP 12 OUTPUTS")
    print("=" * 70)

    ready.to_csv(
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

    print(
        OUTPUT_FILE
    )

    print(
        SUMMARY_FILE
    )

    print(
        PARTICIPANT_REPORT_FILE
    )

    print(
        DICTIONARY_FILE
    )


# =========================================================
# 16. FINAL SUMMARY
# =========================================================

def print_final_summary(
    data,
    ready
):

    removed = (
        len(data)
        -
        len(ready)
    )

    availability = (
        len(ready)
        /
        len(data)
        * 100
    )

    print("\n" + "=" * 70)
    print("STEP 12 FINAL SUMMARY")
    print("=" * 70)

    print(
        "\nInput target-ready rows:",
        len(data)
    )

    print(
        "Feature-ready rows:",
        len(ready)
    )

    print(
        "Rows without complete history:",
        removed
    )

    print(
        "Feature availability:",
        round(
            availability,
            2
        ),
        "%"
    )

    print(
        "Participants retained:",
        ready[
            "participant_id"
        ].nunique()
    )

    print(
        "Glucose-history features created: 12"
    )

    print(
        "Missing prediction targets:",
        ready[
            "target_glucose_30min"
        ]
        .isna()
        .sum()
    )


# =========================================================
# 17. MAIN
# =========================================================

def main():

    print("=" * 70)
    print("SMART BLOOD GLUCOSE ML PROJECT")
    print("STEP 12 - CGMACROS GLUCOSE FEATURES")
    print("=" * 70)

    (
        history,
        target_ready
    ) = load_data()

    validate_history(
        history
    )

    data = create_lag_features(
        target_ready,
        history
    )

    data = create_change_features(
        data
    )

    data = create_rolling_features(
        data,
        history
    )

    validate_time_direction(
        data
    )

    ready = create_feature_ready(
        data
    )

    participant_report = (
        create_participant_report(
            data,
            ready
        )
    )

    dictionary = (
        create_dictionary()
    )

    summary = create_summary(
        data,
        ready
    )

    save_outputs(
        ready,
        summary,
        participant_report,
        dictionary
    )

    print_final_summary(
        data,
        ready
    )

    print("\n" + "=" * 70)
    print("STEP 12 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":

    main()