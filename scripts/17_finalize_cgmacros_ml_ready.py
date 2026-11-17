from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# 1. PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "data_audit"

INPUT_FILE = (
    PROCESSED_DIR
    / "cgmacros_activity_features_ready.csv"
)

HISTORY_FILE = (
    PROCESSED_DIR
    / "cgmacros_standardized_all.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "cgmacros_ml_ready.csv"
)

FEATURE_LIST_FILE = (
    REPORT_DIR
    / "cgmacros_final_feature_list.csv"
)

MISSINGNESS_FILE = (
    REPORT_DIR
    / "cgmacros_final_feature_missingness.csv"
)

QUALITY_FLAGS_FILE = (
    REPORT_DIR
    / "cgmacros_final_quality_flags.csv"
)

PARTICIPANT_SUMMARY_FILE = (
    REPORT_DIR
    / "cgmacros_final_participant_summary.csv"
)

AUDIT_SUMMARY_FILE = (
    REPORT_DIR
    / "cgmacros_final_audit_summary.csv"
)


# =========================================================
# 2. FINAL APPROVED MODEL FEATURES
# =========================================================

FEATURE_GROUPS = {

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


MODEL_FEATURES = [
    feature
    for features in FEATURE_GROUPS.values()
    for feature in features
]


TARGET_COLUMN = "target_glucose_30min"


# =========================================================
# 3. LOAD DATA
# =========================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING STEP 17 DATA")
    print("=" * 70)

    data = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    history = pd.read_csv(
        HISTORY_FILE,
        low_memory=False
    )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce"
    )

    data["target_timestamp_30min"] = pd.to_datetime(
        data["target_timestamp_30min"],
        errors="coerce"
    )

    history["timestamp"] = pd.to_datetime(
        history["timestamp"],
        errors="coerce"
    )

    data["participant_id"] = (
        data["participant_id"]
        .astype(str)
    )

    history["participant_id"] = (
        history["participant_id"]
        .astype(str)
    )

    print(
        "\nRows:",
        len(data)
    )

    print(
        "Participants:",
        data["participant_id"].nunique()
    )

    print(
        "Approved model features:",
        len(MODEL_FEATURES)
    )

    return data, history


# =========================================================
# 4. CHECK REQUIRED COLUMNS
# =========================================================

def validate_required_columns(data):

    print("\n" + "=" * 70)
    print("CHECKING REQUIRED COLUMNS")
    print("=" * 70)

    required = [
        "participant_id",
        "timestamp",
        "target_timestamp_30min",
        TARGET_COLUMN,
    ] + MODEL_FEATURES

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    print(
        "\nMissing required columns:",
        len(missing)
    )

    if missing:

        print(missing)

        raise ValueError(
            "Final dataset is missing required columns."
        )

    if len(MODEL_FEATURES) != 39:

        raise ValueError(
            f"Expected 39 approved model features, "
            f"found {len(MODEL_FEATURES)}."
        )

    print(
        "All 39 approved features are present."
    )


# =========================================================
# 5. CLEAN CONFIRMED INVALID MET VALUES
# =========================================================

def repair_mets(data, history):

    print("\n" + "=" * 70)
    print("HANDLING DOCUMENTED MET RANGE ISSUE")
    print("=" * 70)

    met_history = history[
        [
            "participant_id",
            "timestamp",
            "mets"
        ]
    ].copy()

    met_history["mets"] = pd.to_numeric(
        met_history["mets"],
        errors="coerce"
    )

    invalid_mask = (
        met_history["mets"].notna()
        &
        (
            (met_history["mets"] < 1.0)
            |
            (met_history["mets"] > 17.6)
        )
    )

    invalid_mets = met_history.loc[
        invalid_mask,
        [
            "participant_id",
            "timestamp",
            "mets"
        ]
    ].copy()

    print(
        "\nInvalid source MET observations:",
        len(invalid_mets)
    )

    if not invalid_mets.empty:

        print(
            invalid_mets.to_string(
                index=False
            )
        )

    # Do not modify raw/source data on disk.
    # Only treat documented invalid values as missing
    # in this final modeling copy.
    met_history.loc[
        invalid_mask,
        "mets"
    ] = np.nan

    met_feature_parts = []

    for participant_id, group in (
        met_history.groupby(
            "participant_id",
            sort=False
        )
    ):

        group = (
            group[
                group["timestamp"].notna()
            ]
            .sort_values("timestamp")
            .copy()
        )

        group = group.set_index(
            "timestamp"
        )

        result = pd.DataFrame(
            index=group.index
        )

        result[
            "current_mets_final"
        ] = group["mets"]

        result[
            "mets_mean_30min_final"
        ] = (
            group["mets"]
            .rolling(
                window="30min",
                closed="both",
                min_periods=1
            )
            .mean()
        )

        result = result.reset_index()

        result[
            "participant_id"
        ] = participant_id

        met_feature_parts.append(
            result
        )

    met_features = pd.concat(
        met_feature_parts,
        ignore_index=True
    )

    before_current = (
        data["current_mets"].copy()
    )

    before_mean = (
        data["mets_mean_30min"].copy()
    )

    result = data.merge(
        met_features,
        on=[
            "participant_id",
            "timestamp"
        ],
        how="left",
        validate="one_to_one"
    )

    changed_current = (
        ~np.isclose(
            before_current.to_numpy(
                dtype=float
            ),
            result[
                "current_mets_final"
            ].to_numpy(
                dtype=float
            ),
            equal_nan=True
        )
    ).sum()

    changed_mean = (
        ~np.isclose(
            before_mean.to_numpy(
                dtype=float
            ),
            result[
                "mets_mean_30min_final"
            ].to_numpy(
                dtype=float
            ),
            equal_nan=True
        )
    ).sum()

    result["current_mets"] = (
        result[
            "current_mets_final"
        ]
    )

    result["mets_mean_30min"] = (
        result[
            "mets_mean_30min_final"
        ]
    )

    result = result.drop(
        columns=[
            "current_mets_final",
            "mets_mean_30min_final"
        ]
    )

    print(
        "\nRows where current MET changed:",
        int(changed_current)
    )

    print(
        "Rows where rolling MET mean changed:",
        int(changed_mean)
    )

    return (
        result,
        invalid_mets,
        int(changed_current),
        int(changed_mean)
    )


# =========================================================
# 6. CORE INTEGRITY + LEAKAGE AUDIT
# =========================================================

def run_integrity_audit(data):

    print("\n" + "=" * 70)
    print("RUNNING FINAL INTEGRITY / LEAKAGE AUDIT")
    print("=" * 70)

    duplicate_rows = (
        data.duplicated(
            subset=[
                "participant_id",
                "timestamp"
            ]
        ).sum()
    )

    missing_timestamp = (
        data["timestamp"]
        .isna()
        .sum()
    )

    missing_target_timestamp = (
        data[
            "target_timestamp_30min"
        ]
        .isna()
        .sum()
    )

    missing_targets = (
        data[TARGET_COLUMN]
        .isna()
        .sum()
    )

    target_horizon_minutes = (
        (
            data[
                "target_timestamp_30min"
            ]
            -
            data["timestamp"]
        )
        .dt.total_seconds()
        /
        60.0
    )

    incorrect_horizon = (
        ~np.isclose(
            target_horizon_minutes,
            30.0
        )
    ).sum()

    feature_target_overlap = (
        TARGET_COLUMN
        in MODEL_FEATURES
    )

    metadata_in_features = [
        column
        for column in [
            "participant_id",
            "timestamp",
            "target_timestamp_30min",
            "source_file"
        ]
        if column in MODEL_FEATURES
    ]

    helper_or_future_columns = [
        feature
        for feature in MODEL_FEATURES
        if (
            "target_" in feature
            or
            "future" in feature.lower()
        )
    ]

    final_invalid_mets = (
        data["current_mets"].notna()
        &
        (
            (data["current_mets"] < 1.0)
            |
            (data["current_mets"] > 17.6)
        )
    ).sum()

    print(
        "\nDuplicate participant/timestamps:",
        duplicate_rows
    )

    print(
        "Missing timestamps:",
        missing_timestamp
    )

    print(
        "Missing target timestamps:",
        missing_target_timestamp
    )

    print(
        "Missing prediction targets:",
        missing_targets
    )

    print(
        "Incorrect prediction horizons:",
        incorrect_horizon
    )

    print(
        "Target accidentally included as feature:",
        feature_target_overlap
    )

    print(
        "Metadata accidentally included as model feature:",
        metadata_in_features
    )

    print(
        "Future/target-like model features:",
        helper_or_future_columns
    )

    print(
        "Invalid MET values remaining:",
        final_invalid_mets
    )

    if duplicate_rows != 0:
        raise ValueError(
            "Duplicate modeling rows detected."
        )

    if missing_timestamp != 0:
        raise ValueError(
            "Missing prediction timestamps detected."
        )

    if missing_target_timestamp != 0:
        raise ValueError(
            "Missing target timestamps detected."
        )

    if missing_targets != 0:
        raise ValueError(
            "Missing prediction targets detected."
        )

    if incorrect_horizon != 0:
        raise ValueError(
            "Prediction horizon is not exactly 30 minutes."
        )

    if feature_target_overlap:
        raise ValueError(
            "Target leakage: target is present in feature list."
        )

    if metadata_in_features:
        raise ValueError(
            "Metadata columns are incorrectly included as features."
        )

    if helper_or_future_columns:
        raise ValueError(
            "Possible future-information feature detected."
        )

    if final_invalid_mets != 0:
        raise ValueError(
            "Invalid MET values remain after final cleaning."
        )

    print(
        "\nCore integrity and leakage audit PASSED."
    )

    return {
        "duplicate_rows":
            int(duplicate_rows),

        "missing_timestamp":
            int(missing_timestamp),

        "missing_target_timestamp":
            int(missing_target_timestamp),

        "missing_targets":
            int(missing_targets),

        "incorrect_horizon":
            int(incorrect_horizon),

        "invalid_mets_remaining":
            int(final_invalid_mets)
    }


# =========================================================
# 7. CREATE FINAL FEATURE MISSINGNESS REPORT
# =========================================================

def create_missingness_report(data):

    print("\n" + "=" * 70)
    print("CREATING FINAL FEATURE MISSINGNESS REPORT")
    print("=" * 70)

    records = []

    total_participants = (
        data[
            "participant_id"
        ].nunique()
    )

    for group_name, features in (
        FEATURE_GROUPS.items()
    ):

        for feature in features:

            missing_rows = (
                data[feature]
                .isna()
                .sum()
            )

            participants_without_feature = 0

            for _, participant_data in (
                data.groupby(
                    "participant_id",
                    sort=False
                )
            ):

                if (
                    participant_data[
                        feature
                    ]
                    .notna()
                    .sum()
                    == 0
                ):

                    participants_without_feature += 1

            records.append(
                {
                    "feature_group":
                        group_name,

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
                        int(
                            participants_without_feature
                        ),

                    "total_participants":
                        int(
                            total_participants
                        )
                }
            )

    missingness = pd.DataFrame(
        records
    )

    return missingness


# =========================================================
# 8. CREATE QUALITY FLAGS REPORT
# =========================================================

def create_quality_flags(
    data,
    invalid_mets
):

    flags = []

    # -----------------------------------------------------
    # Confirmed invalid MET source values
    # -----------------------------------------------------

    for _, row in invalid_mets.iterrows():

        flags.append(
            {
                "issue_type":
                    "MET outside documented range",

                "participant_id":
                    row["participant_id"],

                "timestamp":
                    row["timestamp"],

                "value":
                    row["mets"],

                "action":
                    "Preserved in source data; treated as missing in final ML-ready features"
            }
        )

    # -----------------------------------------------------
    # Activity-calorie boundary findings
    # -----------------------------------------------------

    activity_mask = (
        data[
            "current_activity_calories_last_min"
        ].notna()
        &
        (
            (
                data[
                    "current_activity_calories_last_min"
                ]
                < 0
            )
            |
            (
                data[
                    "current_activity_calories_last_min"
                ]
                > 16.178
            )
        )
    )

    for _, row in (
        data.loc[
            activity_mask,
            [
                "participant_id",
                "timestamp",
                "current_activity_calories_last_min"
            ]
        ]
        .iterrows()
    ):

        flags.append(
            {
                "issue_type":
                    "Activity calories marginally outside documented range",

                "participant_id":
                    row["participant_id"],

                "timestamp":
                    row["timestamp"],

                "value":
                    row[
                        "current_activity_calories_last_min"
                    ],

                "action":
                    "Retained and documented as boundary/precision observation"
            }
        )

    # -----------------------------------------------------
    # Current glucose outside documented sensor range
    # -----------------------------------------------------

    glucose_mask = (
        data[
            "current_glucose"
        ].notna()
        &
        (
            (
                data[
                    "current_glucose"
                ] < 40
            )
            |
            (
                data[
                    "current_glucose"
                ] > 400
            )
        )
    )

    for _, row in (
        data.loc[
            glucose_mask,
            [
                "participant_id",
                "timestamp",
                "current_glucose"
            ]
        ]
        .iterrows()
    ):

        flags.append(
            {
                "issue_type":
                    "Current Libre glucose outside documented range",

                "participant_id":
                    row["participant_id"],

                "timestamp":
                    row["timestamp"],

                "value":
                    row["current_glucose"],

                "action":
                    "Retained and documented; no manual clipping"
            }
        )

    # -----------------------------------------------------
    # Target glucose outside documented sensor range
    # -----------------------------------------------------

    target_mask = (
        data[
            TARGET_COLUMN
        ].notna()
        &
        (
            (
                data[
                    TARGET_COLUMN
                ] < 40
            )
            |
            (
                data[
                    TARGET_COLUMN
                ] > 400
            )
        )
    )

    for _, row in (
        data.loc[
            target_mask,
            [
                "participant_id",
                "timestamp",
                TARGET_COLUMN
            ]
        ]
        .iterrows()
    ):

        flags.append(
            {
                "issue_type":
                    "Target Libre glucose outside documented range",

                "participant_id":
                    row["participant_id"],

                "timestamp":
                    row["timestamp"],

                "value":
                    row[
                        TARGET_COLUMN
                    ],

                "action":
                    "Retained and documented; no manual clipping"
            }
        )

    return pd.DataFrame(
        flags,
        columns=[
            "issue_type",
            "participant_id",
            "timestamp",
            "value",
            "action"
        ]
    )


# =========================================================
# 9. CREATE PARTICIPANT SUMMARY
# =========================================================

def create_participant_summary(data):

    records = []

    for participant_id, group in (
        data.groupby(
            "participant_id",
            sort=True
        )
    ):

        feature_missing_cells = (
            group[
                MODEL_FEATURES
            ]
            .isna()
            .sum()
            .sum()
        )

        complete_rows = (
            group[
                MODEL_FEATURES
            ]
            .notna()
            .all(axis=1)
            .sum()
        )

        records.append(
            {
                "participant_id":
                    participant_id,

                "rows":
                    len(group),

                "first_timestamp":
                    group[
                        "timestamp"
                    ].min(),

                "last_timestamp":
                    group[
                        "timestamp"
                    ].max(),

                "complete_feature_rows":
                    int(
                        complete_rows
                    ),

                "complete_feature_row_percent":
                    round(
                        complete_rows
                        /
                        len(group)
                        *
                        100,
                        2
                    ),

                "missing_feature_cells":
                    int(
                        feature_missing_cells
                    )
            }
        )

    return pd.DataFrame(
        records
    )


# =========================================================
# 10. CREATE FEATURE LIST
# =========================================================

def create_feature_list():

    records = []

    position = 1

    for group_name, features in (
        FEATURE_GROUPS.items()
    ):

        for feature in features:

            records.append(
                {
                    "feature_number":
                        position,

                    "feature_group":
                        group_name,

                    "feature":
                        feature,

                    "model_role":
                        "input_X",

                    "uses_future_information":
                        "No"
                }
            )

            position += 1

    return pd.DataFrame(
        records
    )


# =========================================================
# 11. CREATE FINAL ML-READY TABLE
# =========================================================

def create_final_dataset(data):

    print("\n" + "=" * 70)
    print("CREATING FINAL CGMACROS ML-READY TABLE")
    print("=" * 70)

    final_columns = (
        [
            "participant_id",
            "timestamp",
            "target_timestamp_30min"
        ]
        +
        MODEL_FEATURES
        +
        [
            TARGET_COLUMN
        ]
    )

    final_data = (
        data[
            final_columns
        ]
        .copy()
    )

    print(
        "\nRows:",
        len(final_data)
    )

    print(
        "Participants:",
        final_data[
            "participant_id"
        ].nunique()
    )

    print(
        "Model features:",
        len(MODEL_FEATURES)
    )

    print(
        "Total output columns:",
        len(final_data.columns)
    )

    return final_data


# =========================================================
# 12. CREATE FINAL AUDIT SUMMARY
# =========================================================

def create_audit_summary(
    final_data,
    audit,
    invalid_mets,
    changed_current_mets,
    changed_rolling_mets
):

    feature_missing = (
        final_data[
            MODEL_FEATURES
        ]
        .isna()
    )

    rows_with_any_missing_feature = (
        feature_missing
        .any(axis=1)
        .sum()
    )

    complete_feature_rows = (
        (~feature_missing.any(axis=1))
        .sum()
    )

    current_glucose_outside = (
        final_data[
            "current_glucose"
        ].notna()
        &
        (
            (
                final_data[
                    "current_glucose"
                ] < 40
            )
            |
            (
                final_data[
                    "current_glucose"
                ] > 400
            )
        )
    ).sum()

    target_glucose_outside = (
        final_data[
            TARGET_COLUMN
        ].notna()
        &
        (
            (
                final_data[
                    TARGET_COLUMN
                ] < 40
            )
            |
            (
                final_data[
                    TARGET_COLUMN
                ] > 400
            )
        )
    ).sum()

    activity_calorie_outside = (
        final_data[
            "current_activity_calories_last_min"
        ].notna()
        &
        (
            (
                final_data[
                    "current_activity_calories_last_min"
                ] < 0
            )
            |
            (
                final_data[
                    "current_activity_calories_last_min"
                ] > 16.178
            )
        )
    ).sum()

    summary = pd.DataFrame(
        [
            {
                "metric":
                    "rows",

                "value":
                    len(final_data)
            },

            {
                "metric":
                    "participants",

                "value":
                    final_data[
                        "participant_id"
                    ].nunique()
            },

            {
                "metric":
                    "approved_model_features",

                "value":
                    len(MODEL_FEATURES)
            },

            {
                "metric":
                    "duplicate_participant_timestamps",

                "value":
                    audit[
                        "duplicate_rows"
                    ]
            },

            {
                "metric":
                    "missing_prediction_targets",

                "value":
                    audit[
                        "missing_targets"
                    ]
            },

            {
                "metric":
                    "incorrect_target_horizons",

                "value":
                    audit[
                        "incorrect_horizon"
                    ]
            },

            {
                "metric":
                    "invalid_mets_detected_in_source",

                "value":
                    len(invalid_mets)
            },

            {
                "metric":
                    "current_mets_values_changed_to_missing",

                "value":
                    changed_current_mets
            },

            {
                "metric":
                    "rolling_mets_rows_recomputed_changed",

                "value":
                    changed_rolling_mets
            },

            {
                "metric":
                    "invalid_mets_remaining",

                "value":
                    audit[
                        "invalid_mets_remaining"
                    ]
            },

            {
                "metric":
                    "rows_with_any_missing_feature",

                "value":
                    int(
                        rows_with_any_missing_feature
                    )
            },

            {
                "metric":
                    "complete_feature_rows",

                "value":
                    int(
                        complete_feature_rows
                    )
            },

            {
                "metric":
                    "current_glucose_outside_documented_range",

                "value":
                    int(
                        current_glucose_outside
                    )
            },

            {
                "metric":
                    "target_glucose_outside_documented_range",

                "value":
                    int(
                        target_glucose_outside
                    )
            },

            {
                "metric":
                    "activity_calories_marginally_outside_range",

                "value":
                    int(
                        activity_calorie_outside
                    )
            },

            {
                "metric":
                    "ready_for_modeling_handoff",

                "value":
                    "YES - split by participant and fit imputation/scaling on training data only"
            }
        ]
    )

    return summary


# =========================================================
# 13. SAVE OUTPUTS
# =========================================================

def save_outputs(
    final_data,
    feature_list,
    missingness,
    quality_flags,
    participant_summary,
    audit_summary
):

    print("\n" + "=" * 70)
    print("SAVING STEP 17 OUTPUTS")
    print("=" * 70)

    final_data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    feature_list.to_csv(
        FEATURE_LIST_FILE,
        index=False
    )

    missingness.to_csv(
        MISSINGNESS_FILE,
        index=False
    )

    quality_flags.to_csv(
        QUALITY_FLAGS_FILE,
        index=False
    )

    participant_summary.to_csv(
        PARTICIPANT_SUMMARY_FILE,
        index=False
    )

    audit_summary.to_csv(
        AUDIT_SUMMARY_FILE,
        index=False
    )

    print("\nSaved:")

    print(OUTPUT_FILE)
    print(FEATURE_LIST_FILE)
    print(MISSINGNESS_FILE)
    print(QUALITY_FLAGS_FILE)
    print(PARTICIPANT_SUMMARY_FILE)
    print(AUDIT_SUMMARY_FILE)


# =========================================================
# 14. FINAL SUMMARY
# =========================================================

def print_final_summary(
    final_data,
    missingness,
    quality_flags
):

    feature_missing = (
        final_data[
            MODEL_FEATURES
        ]
        .isna()
    )

    rows_with_any_missing = (
        feature_missing
        .any(axis=1)
        .sum()
    )

    complete_rows = (
        (~feature_missing.any(axis=1))
        .sum()
    )

    print("\n" + "=" * 70)
    print("STEP 17 FINAL SUMMARY")
    print("=" * 70)

    print(
        "\nRows:",
        len(final_data)
    )

    print(
        "Participants:",
        final_data[
            "participant_id"
        ].nunique()
    )

    print(
        "Approved model input features:",
        len(MODEL_FEATURES)
    )

    print(
        "Prediction target:",
        TARGET_COLUMN
    )

    print(
        "Missing prediction targets:",
        final_data[
            TARGET_COLUMN
        ]
        .isna()
        .sum()
    )

    print(
        "Duplicate participant/timestamps:",
        final_data.duplicated(
            subset=[
                "participant_id",
                "timestamp"
            ]
        ).sum()
    )

    horizon = (
        (
            final_data[
                "target_timestamp_30min"
            ]
            -
            final_data[
                "timestamp"
            ]
        )
        .dt.total_seconds()
        /
        60.0
    )

    print(
        "Incorrect 30-minute target horizons:",
        (
            ~np.isclose(
                horizon,
                30.0
            )
        ).sum()
    )

    print(
        "Rows with any missing input feature:",
        int(
            rows_with_any_missing
        )
    )

    print(
        "Rows with all 39 input features present:",
        int(
            complete_rows
        )
    )

    print(
        "Documented quality flags:",
        len(
            quality_flags
        )
    )

    print(
        "\nFeature groups:"
    )

    for group_name, features in (
        FEATURE_GROUPS.items()
    ):

        print(
            f"  {group_name}: "
            f"{len(features)}"
        )

    print(
        "\nIMPORTANT MODELING RULE:"
    )

    print(
        "Split data by participant, not by random rows."
    )

    print(
        "Fit imputation/scaling using training data only."
    )

    print(
        "participant_id and timestamps are metadata, not model inputs."
    )

    print(
        "target_glucose_30min is y, never X."
    )


# =========================================================
# 15. MAIN
# =========================================================

def main():

    print("=" * 70)
    print("SMART BLOOD GLUCOSE ML PROJECT")
    print("STEP 17 - FINAL CGMACROS ML-READY AUDIT")
    print("=" * 70)

    data, history = load_data()

    validate_required_columns(
        data
    )

    (
        data,
        invalid_mets,
        changed_current_mets,
        changed_rolling_mets
    ) = repair_mets(
        data,
        history
    )

    audit = (
        run_integrity_audit(
            data
        )
    )

    missingness = (
        create_missingness_report(
            data
        )
    )

    quality_flags = (
        create_quality_flags(
            data,
            invalid_mets
        )
    )

    participant_summary = (
        create_participant_summary(
            data
        )
    )

    feature_list = (
        create_feature_list()
    )

    final_data = (
        create_final_dataset(
            data
        )
    )

    audit_summary = (
        create_audit_summary(
            final_data,
            audit,
            invalid_mets,
            changed_current_mets,
            changed_rolling_mets
        )
    )

    save_outputs(
        final_data,
        feature_list,
        missingness,
        quality_flags,
        participant_summary,
        audit_summary
    )

    print_final_summary(
        final_data,
        missingness,
        quality_flags
    )

    print("\n" + "=" * 70)
    print("STEP 17 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()