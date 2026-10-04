from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# 1. PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "data_audit"

SHANGHAI_INPUT = (
    PROCESSED_DIR
    / "shanghai_meal_features_ready.csv"
)

CGMACROS_INPUT = (
    PROCESSED_DIR
    / "cgmacros_ml_ready.csv"
)

SHANGHAI_OUTPUT = (
    PROCESSED_DIR
    / "shanghai_ml_ready.csv"
)

SHANGHAI_FEATURE_LIST = (
    REPORT_DIR
    / "shanghai_final_feature_list.csv"
)

SHANGHAI_MISSINGNESS = (
    REPORT_DIR
    / "shanghai_final_feature_missingness.csv"
)

CROSS_DATASET_SUMMARY = (
    REPORT_DIR
    / "final_cross_dataset_audit_summary.csv"
)

SHARED_FEATURES_REPORT = (
    REPORT_DIR
    / "final_shared_features.csv"
)


# =========================================================
# 2. SHANGHAI FINAL APPROVED FEATURES
# =========================================================

SHANGHAI_FEATURE_GROUPS = {

    "glucose_history": [
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
    ],

    "temporal": [
        "hour_of_day_sin",
        "hour_of_day_cos",
        "day_night_indicator",
        "elapsed_recording_hours",
    ],

    "participant_clinical": [
        "participant_age",
        "participant_sex",
        "participant_bmi",
        "diabetes_duration_years",
        "hba1c_mmol_mol",
    ],

    "meal_timing": [
        "meal_event_at_t",
        "recent_meal_2h",
        "time_since_last_meal_min",
        "meal_count_last_2h",
    ],
}


SHANGHAI_FEATURES = [
    feature
    for group in SHANGHAI_FEATURE_GROUPS.values()
    for feature in group
]


# =========================================================
# 3. CGMACROS FINAL APPROVED FEATURES
# =========================================================

CGMACROS_FEATURE_GROUPS = {

    "glucose_history": [
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
    ],

    "meal_nutrition": [
        "meal_event_at_t",
        "recent_meal_2h",
        "time_since_last_meal_min",
        "meal_count_last_2h",
        "meal_calories_last_2h",
        "meal_carbs_last_2h_g",
        "meal_protein_last_2h_g",
        "meal_fat_last_2h_g",
        "meal_fiber_last_2h_g",
    ],

    "temporal": [
        "hour_of_day_sin",
        "hour_of_day_cos",
        "day_night_indicator",
        "elapsed_recording_hours",
    ],

    "participant_clinical": [
        "participant_age",
        "participant_sex_code",
        "participant_bmi",
        "hba1c_lab",
        "fasting_glucose_lab",
    ],

    "activity_heart_rate": [
        "current_heart_rate_bpm",
        "heart_rate_mean_15min",
        "heart_rate_mean_60min",
        "heart_rate_std_60min",
        "current_mets",
        "mets_mean_30min",
        "current_activity_calories_last_min",
        "activity_calories_mean_30min",
        "activity_calories_mean_60min",
    ],
}


CGMACROS_FEATURES = [
    feature
    for group in CGMACROS_FEATURE_GROUPS.values()
    for feature in group
]


TARGET = "target_glucose_30min"


# =========================================================
# 4. LOAD DATA
# =========================================================

def load_dataset(path, name):

    print("\n" + "=" * 70)
    print(f"LOADING {name}")
    print("=" * 70)

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find: {path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df["target_timestamp_30min"] = pd.to_datetime(
        df["target_timestamp_30min"],
        errors="coerce"
    )

    df["participant_id"] = (
        df["participant_id"]
        .astype(str)
    )

    print(
        "\nRows:",
        len(df)
    )

    print(
        "Participants:",
        df["participant_id"].nunique()
    )

    print(
        "Columns:",
        len(df.columns)
    )

    return df


# =========================================================
# 5. VALIDATE REQUIRED FEATURES
# =========================================================

def validate_features(
    df,
    dataset_name,
    feature_list
):

    print("\n" + "=" * 70)
    print(f"CHECKING {dataset_name} FEATURE SET")
    print("=" * 70)

    required = [
        "participant_id",
        "timestamp",
        "target_timestamp_30min",
        TARGET
    ] + feature_list

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    print(
        "\nExpected features:",
        len(feature_list)
    )

    print(
        "Missing required columns:",
        len(missing)
    )

    if missing:

        print("\nMissing:")
        for column in missing:
            print(" -", column)

        raise ValueError(
            f"{dataset_name} is missing required features."
        )

    print(
        f"{dataset_name} feature check PASSED."
    )


# =========================================================
# 6. DATASET INTEGRITY AUDIT
# =========================================================

def audit_dataset(
    df,
    name,
    feature_list
):

    print("\n" + "=" * 70)
    print(f"{name} FINAL INTEGRITY AUDIT")
    print("=" * 70)

    duplicates = (
        df.duplicated(
            subset=[
                "participant_id",
                "timestamp"
            ]
        ).sum()
    )

    missing_timestamps = (
        df["timestamp"].isna().sum()
    )

    missing_target_timestamps = (
        df[
            "target_timestamp_30min"
        ].isna().sum()
    )

    missing_targets = (
        df[TARGET].isna().sum()
    )

    horizon = (
        (
            df["target_timestamp_30min"]
            -
            df["timestamp"]
        )
        .dt.total_seconds()
        /
        60.0
    )

    incorrect_horizon = (
        ~np.isclose(
            horizon,
            30.0
        )
    ).sum()

    rows_with_missing_features = (
        df[
            feature_list
        ]
        .isna()
        .any(axis=1)
        .sum()
    )

    complete_rows = (
        df[
            feature_list
        ]
        .notna()
        .all(axis=1)
        .sum()
    )

    target_in_features = (
        TARGET in feature_list
    )

    metadata_in_features = [
        column
        for column in [
            "participant_id",
            "timestamp",
            "target_timestamp_30min",
            "source_file"
        ]
        if column in feature_list
    ]

    future_named_features = [
        feature
        for feature in feature_list
        if (
            "future" in feature.lower()
            or
            feature.startswith("target_")
        )
    ]

    print(
        "\nRows:",
        len(df)
    )

    print(
        "Participants:",
        df["participant_id"].nunique()
    )

    print(
        "Approved model features:",
        len(feature_list)
    )

    print(
        "Duplicate participant/timestamps:",
        duplicates
    )

    print(
        "Missing timestamps:",
        missing_timestamps
    )

    print(
        "Missing target timestamps:",
        missing_target_timestamps
    )

    print(
        "Missing prediction targets:",
        missing_targets
    )

    print(
        "Incorrect 30-minute target horizons:",
        incorrect_horizon
    )

    print(
        "Rows with any missing input feature:",
        rows_with_missing_features
    )

    print(
        "Rows with all input features present:",
        complete_rows
    )

    print(
        "Target included in X:",
        target_in_features
    )

    print(
        "Metadata included in X:",
        metadata_in_features
    )

    print(
        "Future/target-like feature names:",
        future_named_features
    )

    critical_failures = (
        duplicates
        + missing_timestamps
        + missing_target_timestamps
        + missing_targets
        + incorrect_horizon
    )

    if target_in_features:
        critical_failures += 1

    if metadata_in_features:
        critical_failures += 1

    if future_named_features:
        critical_failures += 1

    if critical_failures != 0:

        raise ValueError(
            f"{name} failed final integrity audit."
        )

    print(
        f"\n{name} FINAL AUDIT PASSED."
    )

    return {
        "dataset": name,
        "rows": len(df),
        "participants": df[
            "participant_id"
        ].nunique(),
        "approved_features": len(feature_list),
        "duplicates": int(duplicates),
        "missing_targets": int(missing_targets),
        "incorrect_horizons": int(incorrect_horizon),
        "rows_with_missing_features":
            int(rows_with_missing_features),
        "complete_feature_rows":
            int(complete_rows),
        "status": "PASS"
    }


# =========================================================
# 7. CREATE SHANGHAI ML-READY FILE
# =========================================================

def create_shanghai_ml_ready(df):

    print("\n" + "=" * 70)
    print("CREATING SHANGHAI FINAL ML-READY TABLE")
    print("=" * 70)

    columns = (
        [
            "participant_id",
            "timestamp",
            "target_timestamp_30min"
        ]
        +
        SHANGHAI_FEATURES
        +
        [TARGET]
    )

    output = df[
        columns
    ].copy()

    output.to_csv(
        SHANGHAI_OUTPUT,
        index=False
    )

    print(
        "\nRows:",
        len(output)
    )

    print(
        "Participants:",
        output[
            "participant_id"
        ].nunique()
    )

    print(
        "Model features:",
        len(SHANGHAI_FEATURES)
    )

    print(
        "Total columns:",
        len(output.columns)
    )

    print(
        "Saved:",
        SHANGHAI_OUTPUT
    )

    return output


# =========================================================
# 8. SHANGHAI FEATURE LIST
# =========================================================

def create_shanghai_feature_list():

    records = []

    number = 1

    for group_name, features in (
        SHANGHAI_FEATURE_GROUPS.items()
    ):

        for feature in features:

            records.append(
                {
                    "feature_number": number,
                    "feature_group": group_name,
                    "feature": feature,
                    "model_role": "input_X",
                    "uses_future_information": "No"
                }
            )

            number += 1

    report = pd.DataFrame(
        records
    )

    report.to_csv(
        SHANGHAI_FEATURE_LIST,
        index=False
    )

    return report


# =========================================================
# 9. SHANGHAI MISSINGNESS
# =========================================================

def create_shanghai_missingness(df):

    records = []

    for group_name, features in (
        SHANGHAI_FEATURE_GROUPS.items()
    ):

        for feature in features:

            missing = (
                df[feature]
                .isna()
                .sum()
            )

            participant_missing = 0

            for _, participant in (
                df.groupby(
                    "participant_id",
                    sort=False
                )
            ):

                if (
                    participant[
                        feature
                    ]
                    .notna()
                    .sum()
                    == 0
                ):
                    participant_missing += 1

            records.append(
                {
                    "feature_group":
                        group_name,

                    "feature":
                        feature,

                    "missing_rows":
                        int(missing),

                    "missing_row_percent":
                        round(
                            missing
                            /
                            len(df)
                            *
                            100,
                            4
                        ),

                    "participants_with_no_observations":
                        participant_missing
                }
            )

    report = pd.DataFrame(
        records
    )

    report.to_csv(
        SHANGHAI_MISSINGNESS,
        index=False
    )

    return report


# =========================================================
# 10. CROSS-DATASET FEATURE COMPARISON
# =========================================================

def compare_features():

    shanghai_set = set(
        SHANGHAI_FEATURES
    )

    cgmacros_set = set(
        CGMACROS_FEATURES
    )

    all_features = sorted(
        shanghai_set
        |
        cgmacros_set
    )

    records = []

    for feature in all_features:

        records.append(
            {
                "feature": feature,
                "in_shanghai":
                    feature in shanghai_set,
                "in_cgmacros":
                    feature in cgmacros_set,
                "shared_exact_name":
                    (
                        feature in shanghai_set
                        and
                        feature in cgmacros_set
                    )
            }
        )

    comparison = pd.DataFrame(
        records
    )

    comparison.to_csv(
        SHARED_FEATURES_REPORT,
        index=False
    )

    return comparison


# =========================================================
# 11. SAVE CROSS-DATASET SUMMARY
# =========================================================

def save_cross_summary(
    shanghai_result,
    cgmacros_result,
    comparison
):

    shared_count = (
        comparison[
            "shared_exact_name"
        ].sum()
    )

    summary = pd.DataFrame(
        [
            shanghai_result,
            cgmacros_result
        ]
    )

    summary[
        "shared_exact_feature_names"
    ] = int(
        shared_count
    )

    summary.to_csv(
        CROSS_DATASET_SUMMARY,
        index=False
    )

    return summary, int(shared_count)


# =========================================================
# 12. FINAL CONSOLE SUMMARY
# =========================================================

def print_final_summary(
    shanghai,
    cgmacros,
    shared_count
):

    print("\n" + "=" * 70)
    print("STEP 18 FINAL CROSS-DATASET SUMMARY")
    print("=" * 70)

    print("\nSHANGHAI T2DM")
    print(
        "Rows:",
        shanghai["rows"]
    )
    print(
        "Participants:",
        shanghai["participants"]
    )
    print(
        "Approved model features:",
        shanghai["approved_features"]
    )
    print(
        "Duplicate participant/timestamps:",
        shanghai["duplicates"]
    )
    print(
        "Missing prediction targets:",
        shanghai["missing_targets"]
    )
    print(
        "Incorrect target horizons:",
        shanghai["incorrect_horizons"]
    )
    print(
        "Audit status:",
        shanghai["status"]
    )

    print("\nCGMACROS")
    print(
        "Rows:",
        cgmacros["rows"]
    )
    print(
        "Participants:",
        cgmacros["participants"]
    )
    print(
        "Approved model features:",
        cgmacros["approved_features"]
    )
    print(
        "Duplicate participant/timestamps:",
        cgmacros["duplicates"]
    )
    print(
        "Missing prediction targets:",
        cgmacros["missing_targets"]
    )
    print(
        "Incorrect target horizons:",
        cgmacros["incorrect_horizons"]
    )
    print(
        "Audit status:",
        cgmacros["status"]
    )

    print(
        "\nExact feature names shared by both datasets:",
        shared_count
    )

    print("\nFINAL HANDOFF STATUS:")
    print(
        "Shanghai T2DM: READY FOR MODELING"
    )
    print(
        "CGMacros:      READY FOR MODELING"
    )

    print("\nIMPORTANT:")
    print(
        "Do NOT randomly split rows."
    )
    print(
        "Split by participant ID."
    )
    print(
        "Fit imputation and scaling on training participants only."
    )
    print(
        "Do not directly concatenate the two datasets "
        "until their feature definitions and units are harmonized."
    )


# =========================================================
# 13. MAIN
# =========================================================

def main():

    print("=" * 70)
    print("SMART BLOOD GLUCOSE PREDICTION")
    print("STEP 18 - FINAL CROSS-DATASET AUDIT")
    print("=" * 70)

    shanghai = load_dataset(
        SHANGHAI_INPUT,
        "SHANGHAI T2DM"
    )

    cgmacros = load_dataset(
        CGMACROS_INPUT,
        "CGMACROS"
    )

    validate_features(
        shanghai,
        "SHANGHAI T2DM",
        SHANGHAI_FEATURES
    )

    validate_features(
        cgmacros,
        "CGMACROS",
        CGMACROS_FEATURES
    )

    shanghai_result = audit_dataset(
        shanghai,
        "SHANGHAI T2DM",
        SHANGHAI_FEATURES
    )

    cgmacros_result = audit_dataset(
        cgmacros,
        "CGMACROS",
        CGMACROS_FEATURES
    )

    create_shanghai_ml_ready(
        shanghai
    )

    create_shanghai_feature_list()

    create_shanghai_missingness(
        shanghai
    )

    comparison = compare_features()

    summary, shared_count = (
        save_cross_summary(
            shanghai_result,
            cgmacros_result,
            comparison
        )
    )

    print_final_summary(
        shanghai_result,
        cgmacros_result,
        shared_count
    )

    print("\n" + "=" * 70)
    print("STEP 18 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()