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
    / "cgmacros_participant_features_ready.csv"
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
    / "cgmacros_activity_features_ready.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "cgmacros_activity_feature_summary.csv"
)

MISSINGNESS_FILE = (
    REPORT_DIR
    / "cgmacros_activity_feature_missingness.csv"
)

PARTICIPANT_REPORT_FILE = (
    REPORT_DIR
    / "cgmacros_activity_feature_by_participant.csv"
)

DICTIONARY_FILE = (
    REPORT_DIR
    / "cgmacros_activity_feature_dictionary.csv"
)


# =========================================================
# 4. FEATURE LIST
# =========================================================

ACTIVITY_FEATURES = [
    "current_heart_rate_bpm",
    "heart_rate_mean_15min",
    "heart_rate_mean_60min",
    "heart_rate_std_60min",
    "current_mets",
    "mets_mean_30min",
    "current_activity_calories_last_min",
    "activity_calories_mean_30min",
    "activity_calories_mean_60min"
]


# =========================================================
# 5. LOAD DATA
# =========================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING STEP 16 INPUT DATA")
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
        features["participant_id"].astype(str)
    )

    history["participant_id"] = (
        history["participant_id"].astype(str)
    )

    features = features.sort_values(
        ["participant_id", "timestamp"],
        kind="stable"
    ).reset_index(drop=True)

    history = history.sort_values(
        ["participant_id", "timestamp"],
        kind="stable"
    ).reset_index(drop=True)

    print(
        "\nPrediction rows:",
        len(features)
    )

    print(
        "Prediction participants:",
        features["participant_id"].nunique()
    )

    print(
        "Historical rows:",
        len(history)
    )

    print(
        "History participants:",
        history["participant_id"].nunique()
    )

    return features, history


# =========================================================
# 6. VALIDATE INPUT
# =========================================================

def validate_input(features, history):

    print("\n" + "=" * 70)
    print("VALIDATING STEP 16 INPUT")
    print("=" * 70)

    required_history_columns = [
        "participant_id",
        "timestamp",
        "heart_rate_bpm",
        "activity_calories_last_min",
        "mets"
    ]

    missing_columns = [
        column
        for column in required_history_columns
        if column not in history.columns
    ]

    if missing_columns:

        raise ValueError(
            "Required standardized history columns missing: "
            f"{missing_columns}"
        )

    feature_duplicates = (
        features.duplicated(
            subset=[
                "participant_id",
                "timestamp"
            ]
        ).sum()
    )

    history_duplicates = (
        history[
            history["timestamp"].notna()
        ].duplicated(
            subset=[
                "participant_id",
                "timestamp"
            ]
        ).sum()
    )

    print(
        "\nPrediction duplicate timestamps:",
        feature_duplicates
    )

    print(
        "History duplicate timestamps:",
        history_duplicates
    )

    print(
        "Missing prediction timestamps:",
        features["timestamp"].isna().sum()
    )

    print(
        "Missing prediction targets:",
        features[
            "target_glucose_30min"
        ].isna().sum()
    )

    if feature_duplicates != 0:
        raise ValueError(
            "Prediction data contains duplicate timestamps."
        )

    if history_duplicates != 0:
        raise ValueError(
            "Historical data contains duplicate "
            "participant/timestamp pairs."
        )

    if features["timestamp"].isna().any():
        raise ValueError(
            "Prediction data contains invalid timestamps."
        )

    if features["target_glucose_30min"].isna().any():
        raise ValueError(
            "Prediction data contains missing targets."
        )

    print(
        "\nInput validation passed."
    )


# =========================================================
# 7. PREPARE ACTIVITY HISTORY
# =========================================================

def prepare_activity_history(history):

    print("\n" + "=" * 70)
    print("PREPARING ACTIVITY / HEART-RATE HISTORY")
    print("=" * 70)

    data = history[
        [
            "participant_id",
            "timestamp",
            "heart_rate_bpm",
            "activity_calories_last_min",
            "mets"
        ]
    ].copy()

    numeric_columns = [
        "heart_rate_bpm",
        "activity_calories_last_min",
        "mets"
    ]

    for column in numeric_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data[
        data["timestamp"].notna()
    ].copy()

    data = data.sort_values(
        [
            "participant_id",
            "timestamp"
        ],
        kind="stable"
    ).reset_index(drop=True)

    print(
        "\nValid HR observations:",
        data["heart_rate_bpm"].notna().sum()
    )

    print(
        "Valid MET observations:",
        data["mets"].notna().sum()
    )

    print(
        "Valid activity-calorie observations:",
        data[
            "activity_calories_last_min"
        ].notna().sum()
    )

    return data


# =========================================================
# 8. CREATE ACTIVITY FEATURES
# =========================================================

def create_activity_feature_table(history):

    print("\n" + "=" * 70)
    print("CREATING ACTIVITY / HEART-RATE FEATURES")
    print("=" * 70)

    participant_results = []

    for participant_id, group in (
        history.groupby(
            "participant_id",
            sort=True
        )
    ):

        group = group.sort_values(
            "timestamp"
        ).copy()

        group = group.set_index(
            "timestamp"
        )

        result = pd.DataFrame(
            index=group.index
        )

        # -------------------------------------------------
        # Current heart rate
        # -------------------------------------------------

        result[
            "current_heart_rate_bpm"
        ] = group[
            "heart_rate_bpm"
        ]

        # -------------------------------------------------
        # Heart-rate history
        #
        # Windows contain only observations at or before t.
        # -------------------------------------------------

        result[
            "heart_rate_mean_15min"
        ] = (
            group[
                "heart_rate_bpm"
            ]
            .rolling(
                window="15min",
                closed="both",
                min_periods=1
            )
            .mean()
        )

        result[
            "heart_rate_mean_60min"
        ] = (
            group[
                "heart_rate_bpm"
            ]
            .rolling(
                window="60min",
                closed="both",
                min_periods=1
            )
            .mean()
        )

        result[
            "heart_rate_std_60min"
        ] = (
            group[
                "heart_rate_bpm"
            ]
            .rolling(
                window="60min",
                closed="both",
                min_periods=2
            )
            .std()
        )

        # -------------------------------------------------
        # MET features
        # -------------------------------------------------

        result[
            "current_mets"
        ] = group[
            "mets"
        ]

        result[
            "mets_mean_30min"
        ] = (
            group[
                "mets"
            ]
            .rolling(
                window="30min",
                closed="both",
                min_periods=1
            )
            .mean()
        )

        # -------------------------------------------------
        # Activity calorie features
        # -------------------------------------------------

        result[
            "current_activity_calories_last_min"
        ] = group[
            "activity_calories_last_min"
        ]

        result[
            "activity_calories_mean_30min"
        ] = (
            group[
                "activity_calories_last_min"
            ]
            .rolling(
                window="30min",
                closed="both",
                min_periods=1
            )
            .mean()
        )

        result[
            "activity_calories_mean_60min"
        ] = (
            group[
                "activity_calories_last_min"
            ]
            .rolling(
                window="60min",
                closed="both",
                min_periods=1
            )
            .mean()
        )

        result = result.reset_index()

        result[
            "participant_id"
        ] = participant_id

        participant_results.append(
            result
        )

    activity_features = pd.concat(
        participant_results,
        ignore_index=True
    )

    activity_features = activity_features[
        [
            "participant_id",
            "timestamp"
        ]
        +
        ACTIVITY_FEATURES
    ]

    print(
        "\nActivity feature rows created:",
        len(activity_features)
    )

    print(
        "Participants:",
        activity_features[
            "participant_id"
        ].nunique()
    )

    return activity_features


# =========================================================
# 9. MERGE WITH MODELING ROWS
# =========================================================

def merge_features(
    features,
    activity_features
):

    print("\n" + "=" * 70)
    print("MERGING ACTIVITY FEATURES")
    print("=" * 70)

    original_rows = len(
        features
    )

    result = features.merge(
        activity_features,
        on=[
            "participant_id",
            "timestamp"
        ],
        how="left",
        validate="one_to_one"
    )

    print(
        "\nRows before merge:",
        original_rows
    )

    print(
        "Rows after merge:",
        len(result)
    )

    if len(result) != original_rows:

        raise ValueError(
            "Row count changed during activity feature merge."
        )

    return result


# =========================================================
# 10. RANGE / LEAKAGE VALIDATION
# =========================================================

def validate_activity_features(data):

    print("\n" + "=" * 70)
    print("VALIDATING ACTIVITY FEATURES")
    print("=" * 70)

    # -----------------------------------------------------
    # Data dictionary documented ranges.
    #
    # These are reported, not automatically removed.
    # -----------------------------------------------------

    hr = data[
        "current_heart_rate_bpm"
    ]

    mets = data[
        "current_mets"
    ]

    activity_calories = data[
        "current_activity_calories_last_min"
    ]

    hr_outside_range = (
        hr.notna()
        &
        (
            (hr < 30)
            |
            (hr > 176)
        )
    ).sum()

    # Raw METs were stored x10.
    # Step 10 already converted them back to actual METs.
    mets_outside_range = (
        mets.notna()
        &
        (
            (mets < 1.0)
            |
            (mets > 17.6)
        )
    ).sum()

    activity_calories_outside_range = (
        activity_calories.notna()
        &
        (
            (activity_calories < 0)
            |
            (activity_calories > 16.178)
        )
    ).sum()

    print(
        "\nHR outside documented range:",
        hr_outside_range
    )

    print(
        "METs outside documented range:",
        mets_outside_range
    )

    print(
        "Activity calories outside documented range:",
        activity_calories_outside_range
    )

    print(
        "Missing prediction targets:",
        data[
            "target_glucose_30min"
        ].isna().sum()
    )

    print(
        "Rows:",
        len(data)
    )

    print(
        "Participants:",
        data[
            "participant_id"
        ].nunique()
    )

    if data["target_glucose_30min"].isna().any():

        raise ValueError(
            "Prediction targets became missing."
        )

    print(
        "\nActivity feature validation passed."
    )

    return {
        "hr_outside_range":
            int(hr_outside_range),

        "mets_outside_range":
            int(mets_outside_range),

        "activity_calories_outside_range":
            int(activity_calories_outside_range)
    }


# =========================================================
# 11. MISSINGNESS REPORT
# =========================================================

def create_missingness_report(data):

    records = []

    participants = (
        data["participant_id"]
        .unique()
    )

    total_participants = len(
        participants
    )

    for feature in ACTIVITY_FEATURES:

        missing_rows = (
            data[feature]
            .isna()
            .sum()
        )

        participants_without_feature = 0

        for participant_id, group in (
            data.groupby(
                "participant_id",
                sort=False
            )
        ):

            if group[
                feature
            ].notna().sum() == 0:

                participants_without_feature += 1

        records.append(
            {
                "feature":
                    feature,

                "total_rows":
                    len(data),

                "missing_rows":
                    int(
                        missing_rows
                    ),

                "missing_row_percent":
                    round(
                        missing_rows
                        /
                        len(data)
                        *
                        100,
                        4
                    ),

                "participants_with_no_observations":
                    participants_without_feature,

                "total_participants":
                    total_participants
            }
        )

    return pd.DataFrame(
        records
    )


# =========================================================
# 12. PARTICIPANT REPORT
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

                "heart_rate_available_rows":
                    int(
                        group[
                            "current_heart_rate_bpm"
                        ]
                        .notna()
                        .sum()
                    ),

                "heart_rate_available_percent":
                    round(
                        group[
                            "current_heart_rate_bpm"
                        ]
                        .notna()
                        .mean()
                        * 100,
                        2
                    ),

                "mets_available_rows":
                    int(
                        group[
                            "current_mets"
                        ]
                        .notna()
                        .sum()
                    ),

                "mets_available_percent":
                    round(
                        group[
                            "current_mets"
                        ]
                        .notna()
                        .mean()
                        * 100,
                        2
                    ),

                "activity_calories_available_rows":
                    int(
                        group[
                            "current_activity_calories_last_min"
                        ]
                        .notna()
                        .sum()
                    ),

                "activity_calories_available_percent":
                    round(
                        group[
                            "current_activity_calories_last_min"
                        ]
                        .notna()
                        .mean()
                        * 100,
                        2
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

    rows = [
        {
            "feature":
                "current_heart_rate_bpm",

            "meaning":
                "Heart-rate measurement available at prediction time t",

            "unit":
                "beats/minute"
        },

        {
            "feature":
                "heart_rate_mean_15min",

            "meaning":
                "Mean heart rate from the previous 15 minutes through t",

            "unit":
                "beats/minute"
        },

        {
            "feature":
                "heart_rate_mean_60min",

            "meaning":
                "Mean heart rate from the previous 60 minutes through t",

            "unit":
                "beats/minute"
        },

        {
            "feature":
                "heart_rate_std_60min",

            "meaning":
                "Heart-rate standard deviation during the previous 60 minutes through t",

            "unit":
                "beats/minute"
        },

        {
            "feature":
                "current_mets",

            "meaning":
                "Current/recent metabolic equivalent of task estimate",

            "unit":
                "MET"
        },

        {
            "feature":
                "mets_mean_30min",

            "meaning":
                "Mean MET estimate during the previous 30 minutes through t",

            "unit":
                "MET"
        },

        {
            "feature":
                "current_activity_calories_last_min",

            "meaning":
                "Fitbit estimated activity calories for the most recent minute",

            "unit":
                "kcal"
        },

        {
            "feature":
                "activity_calories_mean_30min",

            "meaning":
                "Mean per-minute activity calorie estimate during the previous 30 minutes through t",

            "unit":
                "kcal/minute-record"
        },

        {
            "feature":
                "activity_calories_mean_60min",

            "meaning":
                "Mean per-minute activity calorie estimate during the previous 60 minutes through t",

            "unit":
                "kcal/minute-record"
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
# 14. SUMMARY REPORT
# =========================================================

def create_summary(
    data,
    missingness,
    range_results
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
                    "activity_features_created",

                "value":
                    len(
                        ACTIVITY_FEATURES
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
            },

            {
                "metric":
                    "hr_outside_documented_range",

                "value":
                    range_results[
                        "hr_outside_range"
                    ]
            },

            {
                "metric":
                    "mets_outside_documented_range",

                "value":
                    range_results[
                        "mets_outside_range"
                    ]
            },

            {
                "metric":
                    "activity_calories_outside_documented_range",

                "value":
                    range_results[
                        "activity_calories_outside_range"
                    ]
            }
        ]
    )


# =========================================================
# 15. SAVE OUTPUTS
# =========================================================

def save_outputs(
    data,
    missingness,
    participant_report,
    dictionary,
    summary
):

    print("\n" + "=" * 70)
    print("SAVING STEP 16 OUTPUTS")
    print("=" * 70)

    data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    missingness.to_csv(
        MISSINGNESS_FILE,
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

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    print("\nSaved:")

    print(OUTPUT_FILE)
    print(MISSINGNESS_FILE)
    print(PARTICIPANT_REPORT_FILE)
    print(DICTIONARY_FILE)
    print(SUMMARY_FILE)


# =========================================================
# 16. FINAL SUMMARY
# =========================================================

def print_final_summary(
    data,
    missingness,
    range_results
):

    print("\n" + "=" * 70)
    print("STEP 16 FINAL SUMMARY")
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
        "Activity/heart-rate features created:",
        len(
            ACTIVITY_FEATURES
        )
    )

    print(
        "Missing prediction targets:",
        data[
            "target_glucose_30min"
        ].isna().sum()
    )

    print(
        "\nDocumented-range audit:"
    )

    print(
        "HR outside 30-176 bpm:",
        range_results[
            "hr_outside_range"
        ]
    )

    print(
        "METs outside 1.0-17.6:",
        range_results[
            "mets_outside_range"
        ]
    )

    print(
        "Activity calories outside 0-16.178:",
        range_results[
            "activity_calories_outside_range"
        ]
    )

    print(
        "\nFeature missingness:"
    )

    print(
        missingness[
            [
                "feature",
                "missing_rows",
                "missing_row_percent",
                "participants_with_no_observations"
            ]
        ]
        .to_string(
            index=False
        )
    )


# =========================================================
# 17. MAIN
# =========================================================

def main():

    print("=" * 70)
    print("SMART BLOOD GLUCOSE ML PROJECT")
    print("STEP 16 - CGMACROS ACTIVITY / HEART-RATE FEATURES")
    print("=" * 70)

    features, history = load_data()

    validate_input(
        features,
        history
    )

    history = (
        prepare_activity_history(
            history
        )
    )

    activity_features = (
        create_activity_feature_table(
            history
        )
    )

    data = (
        merge_features(
            features,
            activity_features
        )
    )

    range_results = (
        validate_activity_features(
            data
        )
    )

    missingness = (
        create_missingness_report(
            data
        )
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
            data,
            missingness,
            range_results
        )
    )

    save_outputs(
        data,
        missingness,
        participant_report,
        dictionary,
        summary
    )

    print_final_summary(
        data,
        missingness,
        range_results
    )

    print("\n" + "=" * 70)
    print("STEP 16 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":

    main()