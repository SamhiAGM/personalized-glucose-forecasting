from pathlib import Path
import pandas as pd


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

HISTORY_FILE = (
    PROCESSED_DIR
    / "shanghai_cgm_clean.csv"
)

TARGET_FILE = (
    PROCESSED_DIR
    / "shanghai_target_ready.csv"
)

OUTPUT_ALL = (
    PROCESSED_DIR
    / "shanghai_glucose_features_all.csv"
)

OUTPUT_READY = (
    PROCESSED_DIR
    / "shanghai_glucose_features_ready.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "shanghai_glucose_feature_summary.csv"
)

MISSINGNESS_FILE = (
    REPORT_DIR
    / "shanghai_glucose_feature_missingness.csv"
)

DICTIONARY_FILE = (
    REPORT_DIR
    / "shanghai_glucose_feature_dictionary.csv"
)


# =========================================================
# 2. LOAD DATA
# =========================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING STEP 5 INPUT DATA")
    print("=" * 70)

    history = pd.read_csv(
        HISTORY_FILE,
        parse_dates=["timestamp"]
    )

    targets = pd.read_csv(
        TARGET_FILE,
        parse_dates=[
            "timestamp",
            "target_timestamp_30min"
        ]
    )

    history = history.sort_values(
        ["session_id", "timestamp"]
    ).reset_index(drop=True)

    targets = targets.sort_values(
        ["session_id", "timestamp"]
    ).reset_index(drop=True)

    print(
        "\nHistory rows:",
        len(history)
    )

    print(
        "Target-ready rows:",
        len(targets)
    )

    print(
        "Participants:",
        targets["participant_id"].nunique()
    )

    print(
        "Sessions:",
        targets["session_id"].nunique()
    )

    return history, targets


# =========================================================
# 3. CREATE EXACT TIMESTAMP LAG FEATURES
# =========================================================

def create_lag_features(
    history,
    targets
):

    print("\n" + "=" * 70)
    print("CREATING TIMESTAMP-BASED LAG FEATURES")
    print("=" * 70)

    data = targets.copy()

    # Current glucose
    data["current_glucose"] = (
        data["cgm_mg_dl"]
    )

    lag_minutes = [
        15,
        30,
        45,
        60
    ]

    history_lookup = history[
        [
            "session_id",
            "timestamp",
            "cgm_mg_dl"
        ]
    ].copy()

    for minutes in lag_minutes:

        print(
            f"Creating {minutes}-minute lag..."
        )

        lag_timestamp_column = (
            f"lag_timestamp_{minutes}min"
        )

        lag_feature_column = (
            f"glucose_lag_{minutes}min"
        )

        # Exact time we want from the past
        data[
            lag_timestamp_column
        ] = (
            data["timestamp"]
            - pd.Timedelta(
                minutes=minutes
            )
        )

        lookup = history_lookup.rename(
            columns={
                "timestamp":
                    lag_timestamp_column,

                "cgm_mg_dl":
                    lag_feature_column
            }
        )

        # Match using the SAME session
        data = data.merge(
            lookup,
            on=[
                "session_id",
                lag_timestamp_column
            ],
            how="left",
            validate="many_to_one"
        )

    return data


# =========================================================
# 4. CREATE CHANGE AND SLOPE FEATURES
# =========================================================

def create_change_features(data):

    print("\n" + "=" * 70)
    print("CREATING CHANGE AND SLOPE FEATURES")
    print("=" * 70)

    data = data.copy()

    data[
        "glucose_change_15min"
    ] = (
        data["current_glucose"]
        -
        data["glucose_lag_15min"]
    )

    data[
        "glucose_change_30min"
    ] = (
        data["current_glucose"]
        -
        data["glucose_lag_30min"]
    )

    # mg/dL per minute
    data[
        "glucose_slope_15min"
    ] = (
        data["glucose_change_15min"]
        / 15.0
    )

    return data


# =========================================================
# 5. CREATE 60-MINUTE ROLLING FEATURES
# =========================================================

def create_rolling_features(
    history
):

    print("\n" + "=" * 70)
    print("CREATING 60-MINUTE ROLLING FEATURES")
    print("=" * 70)

    session_outputs = []

    for session_id, group in history.groupby(
        "session_id",
        sort=False
    ):

        group = (
            group[
                [
                    "session_id",
                    "timestamp",
                    "cgm_mg_dl"
                ]
            ]
            .sort_values("timestamp")
            .copy()
        )

        # Timestamp becomes the rolling index.
        indexed = group.set_index(
            "timestamp"
        )

        # IMPORTANT:
        # The window uses only values from
        # t-60 minutes through current time t.
        #
        # It NEVER uses future glucose.
        rolling = (
            indexed["cgm_mg_dl"]
            .rolling(
                "60min",
                closed="both",
                min_periods=1
            )
        )

        features = pd.DataFrame(
            index=indexed.index
        )

        features[
            "rolling_glucose_mean_60min"
        ] = rolling.mean()

        features[
            "rolling_glucose_std_60min"
        ] = rolling.std(
            ddof=0
        )

        features[
            "rolling_glucose_min_60min"
        ] = rolling.min()

        features[
            "rolling_glucose_max_60min"
        ] = rolling.max()

        features[
            "rolling_glucose_count_60min"
        ] = rolling.count()

        features = (
            features
            .reset_index()
        )

        features[
            "session_id"
        ] = session_id

        session_outputs.append(
            features
        )

    rolling_features = pd.concat(
        session_outputs,
        ignore_index=True
    )

    return rolling_features


# =========================================================
# 6. MERGE ROLLING FEATURES
# =========================================================

def merge_rolling_features(
    data,
    rolling_features
):

    data = data.merge(
        rolling_features,
        on=[
            "session_id",
            "timestamp"
        ],
        how="left",
        validate="many_to_one"
    )

    return data


# =========================================================
# 7. DEFINE REQUIRED GLUCOSE FEATURES
# =========================================================

def get_required_features():

    return [
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
        "rolling_glucose_max_60min"
    ]


# =========================================================
# 8. CREATE FEATURE-READY TABLE
# =========================================================

def create_feature_ready_table(
    data
):

    print("\n" + "=" * 70)
    print("CHECKING FEATURE COMPLETENESS")
    print("=" * 70)

    required_features = (
        get_required_features()
    )

    data[
        "glucose_features_complete"
    ] = (
        data[
            required_features
        ]
        .notna()
        .all(axis=1)
        .astype(int)
    )

    ready = data[
        data[
            "glucose_features_complete"
        ] == 1
    ].copy()

    ready = ready.reset_index(
        drop=True
    )

    print(
        "\nRows before feature filtering:",
        len(data)
    )

    print(
        "Rows with complete "
        "glucose-history features:",
        len(ready)
    )

    print(
        "Rows removed because one or more "
        "history features were unavailable:",
        len(data) - len(ready)
    )

    return data, ready


# =========================================================
# 9. VALIDATE NO FUTURE LEAKAGE
# =========================================================

def validate_features(
    data,
    ready
):

    print("\n" + "=" * 70)
    print("VALIDATING GLUCOSE FEATURES")
    print("=" * 70)

    lag_minutes = [
        15,
        30,
        45,
        60
    ]

    total_wrong = 0

    for minutes in lag_minutes:

        timestamp_col = (
            f"lag_timestamp_{minutes}min"
        )

        feature_col = (
            f"glucose_lag_{minutes}min"
        )

        valid = data[
            data[feature_col].notna()
        ]

        actual_difference = (
            valid["timestamp"]
            -
            valid[timestamp_col]
        ).dt.total_seconds() / 60

        wrong = (
            actual_difference != minutes
        ).sum()

        print(
            f"{minutes}-minute lag "
            f"timing errors:",
            wrong
        )

        total_wrong += wrong

    print(
        "\nMissing target in ready table:",
        ready[
            "target_glucose_30min"
        ].isna().sum()
    )

    print(
        "Participants retained:",
        ready[
            "participant_id"
        ].nunique()
    )

    print(
        "Sessions retained:",
        ready[
            "session_id"
        ].nunique()
    )

    if total_wrong != 0:

        raise ValueError(
            "Lag timing validation failed."
        )

    print(
        "\nLag validation passed."
    )


# =========================================================
# 10. FEATURE MISSINGNESS REPORT
# =========================================================

def create_missingness_report(data):

    features = get_required_features()

    rows = []

    for feature in features:

        missing_count = (
            data[feature]
            .isna()
            .sum()
        )

        missing_percent = (
            missing_count
            / len(data)
            * 100
        )

        rows.append(
            {
                "feature":
                    feature,

                "missing_count":
                    missing_count,

                "missing_percent":
                    round(
                        missing_percent,
                        2
                    )
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# 11. FEATURE DICTIONARY
# =========================================================

def create_feature_dictionary():

    rows = [
        {
            "feature":
                "current_glucose",
            "meaning":
                "CGM glucose at prediction time t",
            "unit":
                "mg/dL"
        },
        {
            "feature":
                "glucose_lag_15min",
            "meaning":
                "CGM glucose exactly 15 minutes before t",
            "unit":
                "mg/dL"
        },
        {
            "feature":
                "glucose_lag_30min",
            "meaning":
                "CGM glucose exactly 30 minutes before t",
            "unit":
                "mg/dL"
        },
        {
            "feature":
                "glucose_lag_45min",
            "meaning":
                "CGM glucose exactly 45 minutes before t",
            "unit":
                "mg/dL"
        },
        {
            "feature":
                "glucose_lag_60min",
            "meaning":
                "CGM glucose exactly 60 minutes before t",
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
                "15-minute glucose change divided by 15",
            "unit":
                "mg/dL/min"
        },
        {
            "feature":
                "rolling_glucose_mean_60min",
            "meaning":
                "Mean glucose from t-60 min through t",
            "unit":
                "mg/dL"
        },
        {
            "feature":
                "rolling_glucose_std_60min",
            "meaning":
                "Glucose variability from t-60 min through t",
            "unit":
                "mg/dL"
        },
        {
            "feature":
                "rolling_glucose_min_60min",
            "meaning":
                "Minimum glucose from t-60 min through t",
            "unit":
                "mg/dL"
        },
        {
            "feature":
                "rolling_glucose_max_60min",
            "meaning":
                "Maximum glucose from t-60 min through t",
            "unit":
                "mg/dL"
        }
    ]

    dictionary = pd.DataFrame(rows)

    dictionary[
        "uses_future_information"
    ] = "No"

    return dictionary


# =========================================================
# 12. CREATE SUMMARY
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
                    "rows_removed_for_missing_history",
                "value":
                    len(data) - len(ready)
            },
            {
                "metric":
                    "feature_ready_percent",
                "value":
                    round(
                        len(ready)
                        / len(data)
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
                    "sessions_retained",
                "value":
                    ready[
                        "session_id"
                    ].nunique()
            },
            {
                "metric":
                    "glucose_history_features",
                "value":
                    len(
                        get_required_features()
                    )
            }
        ]
    )


# =========================================================
# 13. SAVE OUTPUTS
# =========================================================

def save_outputs(
    data,
    ready,
    summary,
    missingness,
    dictionary
):

    print("\n" + "=" * 70)
    print("SAVING STEP 5 OUTPUTS")
    print("=" * 70)

    data.to_csv(
        OUTPUT_ALL,
        index=False
    )

    ready.to_csv(
        OUTPUT_READY,
        index=False
    )

    summary.to_csv(
        SUMMARY_FILE,
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
    print(OUTPUT_ALL)
    print(OUTPUT_READY)
    print(SUMMARY_FILE)
    print(MISSINGNESS_FILE)
    print(DICTIONARY_FILE)


# =========================================================
# 14. PRINT FINAL SUMMARY
# =========================================================

def print_final_summary(
    data,
    ready
):

    print("\n" + "=" * 70)
    print("STEP 5 FINAL SUMMARY")
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
        "Rows without complete "
        "history features:",
        len(data) - len(ready)
    )

    print(
        "Feature availability:",
        f"{len(ready) / len(data) * 100:.2f}%"
    )

    print(
        "Participants retained:",
        ready[
            "participant_id"
        ].nunique()
    )

    print(
        "Sessions retained:",
        ready[
            "session_id"
        ].nunique()
    )

    print(
        "Glucose-history features created:",
        len(
            get_required_features()
        )
    )

    print(
        "Missing target values:",
        ready[
            "target_glucose_30min"
        ].isna().sum()
    )


# =========================================================
# 15. MAIN
# =========================================================

def main():

    print("=" * 70)
    print("SMART BLOOD GLUCOSE ML PROJECT")
    print("STEP 5 - GLUCOSE HISTORY FEATURES")
    print("=" * 70)

    history, targets = (
        load_data()
    )

    data = create_lag_features(
        history,
        targets
    )

    data = create_change_features(
        data
    )

    rolling_features = (
        create_rolling_features(
            history
        )
    )

    data = merge_rolling_features(
        data,
        rolling_features
    )

    data, ready = (
        create_feature_ready_table(
            data
        )
    )

    validate_features(
        data,
        ready
    )

    missingness = (
        create_missingness_report(
            data
        )
    )

    dictionary = (
        create_feature_dictionary()
    )

    summary = create_summary(
        data,
        ready
    )

    save_outputs(
        data,
        ready,
        summary,
        missingness,
        dictionary
    )

    print_final_summary(
        data,
        ready
    )

    print("\n" + "=" * 70)
    print("STEP 5 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()