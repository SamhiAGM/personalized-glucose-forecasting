from pathlib import Path
import pandas as pd


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "data_audit"

INPUT_FILE = (
    PROCESSED_DIR
    / "shanghai_temporal_features_ready.csv"
)

SUMMARY_FILE = (
    PROCESSED_DIR
    / "shanghai_summary_standardized.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "shanghai_participant_features_ready.csv"
)

SUMMARY_REPORT_FILE = (
    REPORT_DIR
    / "shanghai_participant_feature_summary.csv"
)

MISSINGNESS_FILE = (
    REPORT_DIR
    / "shanghai_participant_feature_missingness.csv"
)

DICTIONARY_FILE = (
    REPORT_DIR
    / "shanghai_participant_feature_dictionary.csv"
)

CONSISTENCY_FILE = (
    REPORT_DIR
    / "shanghai_repeated_participant_consistency.csv"
)


# =========================================================
# 2. LOAD DATA
# =========================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING STEP 7 INPUT DATA")
    print("=" * 70)

    time_series = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamp"]
    )

    summary = pd.read_csv(
        SUMMARY_FILE
    )

    # Make sure merge keys have same type
    time_series["session_id"] = (
        time_series["session_id"]
        .astype(str)
    )

    time_series["participant_id"] = (
        time_series["participant_id"]
        .astype(str)
    )

    summary["session_id"] = (
        summary["session_id"]
        .astype(str)
    )

    summary["participant_id"] = (
        summary["participant_id"]
        .astype(str)
    )

    print(
        "\nTime-series rows:",
        len(time_series)
    )

    print(
        "Time-series participants:",
        time_series["participant_id"].nunique()
    )

    print(
        "Time-series sessions:",
        time_series["session_id"].nunique()
    )

    print(
        "Summary rows:",
        len(summary)
    )

    print(
        "Summary participants:",
        summary["participant_id"].nunique()
    )

    print(
        "Summary sessions:",
        summary["session_id"].nunique()
    )

    return time_series, summary


# =========================================================
# 3. CREATE CLINICAL FEATURE TABLE
# =========================================================

def create_clinical_table(summary):

    print("\n" + "=" * 70)
    print("CREATING CLINICAL FEATURE TABLE")
    print("=" * 70)

    required_columns = [
        "session_id",
        "participant_id",
        "gender_female_1_male_2",
        "age_years",
        "bmi_kg_m2",
        "duration_of_diabetes_years",
        "hba1c_mmol_mol"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in summary.columns
    ]

    if missing_columns:

        raise ValueError(
            "Required summary columns missing: "
            + str(missing_columns)
        )

    clinical = summary[
        required_columns
    ].copy()

    clinical = clinical.rename(
        columns={
            "gender_female_1_male_2":
                "participant_sex_code",

            "age_years":
                "participant_age",

            "bmi_kg_m2":
                "participant_bmi",

            "duration_of_diabetes_years":
                "diabetes_duration_years",

            "hba1c_mmol_mol":
                "hba1c_mmol_mol"
        }
    )

    # -----------------------------------------
    # Convert numeric fields
    # -----------------------------------------

    numeric_columns = [
        "participant_sex_code",
        "participant_age",
        "participant_bmi",
        "diabetes_duration_years",
        "hba1c_mmol_mol"
    ]

    for column in numeric_columns:

        clinical[column] = pd.to_numeric(
            clinical[column],
            errors="coerce"
        )

    # -----------------------------------------
    # Human-readable sex category
    #
    # Dataset definition:
    # 1 = Female
    # 2 = Male
    # -----------------------------------------

    clinical["participant_sex"] = (
        clinical[
            "participant_sex_code"
        ]
        .map(
            {
                1: "Female",
                2: "Male"
            }
        )
    )

    # -----------------------------------------
    # Validate session ID uniqueness
    # -----------------------------------------

    duplicate_sessions = (
        clinical["session_id"]
        .duplicated()
        .sum()
    )

    print(
        "\nDuplicate summary session IDs:",
        duplicate_sessions
    )

    if duplicate_sessions != 0:

        raise ValueError(
            "Summary session_id is not unique."
        )

    print("\nClinical features selected:")

    print(" - participant_age")
    print(" - participant_sex")
    print(" - participant_bmi")
    print(" - diabetes_duration_years")
    print(" - hba1c_mmol_mol")

    return clinical


# =========================================================
# 4. CHECK REPEATED PARTICIPANT CONSISTENCY
# =========================================================

def check_repeated_participants(clinical):

    print("\n" + "=" * 70)
    print("CHECKING REPEATED PARTICIPANTS")
    print("=" * 70)

    features = [
        "participant_sex_code",
        "participant_age",
        "participant_bmi",
        "diabetes_duration_years",
        "hba1c_mmol_mol"
    ]

    repeated_ids = (
        clinical[
            "participant_id"
        ]
        .value_counts()
    )

    repeated_ids = (
        repeated_ids[
            repeated_ids > 1
        ]
        .index
        .tolist()
    )

    print(
        "\nParticipants with multiple sessions:",
        len(repeated_ids)
    )

    records = []

    for participant_id in repeated_ids:

        person = clinical[
            clinical[
                "participant_id"
            ] == participant_id
        ]

        record = {
            "participant_id":
                participant_id,

            "session_count":
                len(person)
        }

        for feature in features:

            unique_values = (
                person[feature]
                .dropna()
                .nunique()
            )

            record[
                f"{feature}_unique_values"
            ] = unique_values

            record[
                f"{feature}_consistent"
            ] = (
                unique_values <= 1
            )

        records.append(
            record
        )

    consistency = pd.DataFrame(
        records
    )

    return consistency


# =========================================================
# 5. MERGE USING SESSION ID
# =========================================================

def merge_features(
    time_series,
    clinical
):

    print("\n" + "=" * 70)
    print("MERGING CLINICAL FEATURES BY SESSION ID")
    print("=" * 70)

    starting_rows = len(
        time_series
    )

    clinical_for_merge = clinical[
        [
            "session_id",
            "participant_sex_code",
            "participant_sex",
            "participant_age",
            "participant_bmi",
            "diabetes_duration_years",
            "hba1c_mmol_mol"
        ]
    ].copy()

    data = time_series.merge(
        clinical_for_merge,
        on="session_id",
        how="left",
        validate="many_to_one"
    )

    print(
        "\nRows before merge:",
        starting_rows
    )

    print(
        "Rows after merge:",
        len(data)
    )

    if len(data) != starting_rows:

        raise ValueError(
            "Row count changed during merge."
        )

    return data


# =========================================================
# 6. CREATE PARTICIPANT-LEVEL MISSINGNESS REPORT
# =========================================================

def create_missingness_report(data):

    features = [
        "participant_age",
        "participant_sex",
        "participant_bmi",
        "diabetes_duration_years",
        "hba1c_mmol_mol"
    ]

    records = []

    # One row per participant for participant-level audit
    participant_data = (
        data.sort_values(
            [
                "participant_id",
                "timestamp"
            ]
        )
        .drop_duplicates(
            subset=[
                "participant_id"
            ]
        )
    )

    for feature in features:

        missing_rows = (
            data[
                feature
            ]
            .isna()
            .sum()
        )

        missing_participants = (
            participant_data[
                feature
            ]
            .isna()
            .sum()
        )

        records.append(
            {
                "feature":
                    feature,

                "missing_rows":
                    int(
                        missing_rows
                    ),

                "missing_row_percent":
                    round(
                        missing_rows
                        / len(data)
                        * 100,
                        2
                    ),

                "missing_participants":
                    int(
                        missing_participants
                    ),

                "total_participants":
                    int(
                        participant_data[
                            "participant_id"
                        ].nunique()
                    ),

                "missing_participant_percent":
                    round(
                        missing_participants
                        /
                        participant_data[
                            "participant_id"
                        ].nunique()
                        * 100,
                        2
                    )
            }
        )

    return pd.DataFrame(
        records
    )


# =========================================================
# 7. VALIDATE MERGE
# =========================================================

def validate_data(data):

    print("\n" + "=" * 70)
    print("VALIDATING PARTICIPANT FEATURES")
    print("=" * 70)

    feature_columns = [
        "participant_age",
        "participant_sex",
        "participant_bmi",
        "diabetes_duration_years",
        "hba1c_mmol_mol"
    ]

    # Check sessions where ALL clinical features failed
    session_level = (
        data[
            [
                "session_id"
            ]
            + feature_columns
        ]
        .drop_duplicates(
            subset=[
                "session_id"
            ]
        )
    )

    unmatched_sessions = (
        session_level[
            feature_columns
        ]
        .isna()
        .all(axis=1)
        .sum()
    )

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
        "Completely unmatched sessions:",
        unmatched_sessions
    )

    print(
        "Missing prediction targets:",
        data[
            "target_glucose_30min"
        ].isna()
        .sum()
    )

    if unmatched_sessions > 0:

        raise ValueError(
            "Some time-series sessions "
            "did not match clinical summary."
        )

    print(
        "\nParticipant feature validation passed."
    )


# =========================================================
# 8. FEATURE DICTIONARY
# =========================================================

def create_dictionary():

    dictionary = pd.DataFrame(
        [
            {
                "feature":
                    "participant_age",

                "meaning":
                    "Age recorded for participant/session",

                "unit":
                    "years"
            },

            {
                "feature":
                    "participant_sex",

                "meaning":
                    "Sex recorded in dataset; code 1 Female, 2 Male",

                "unit":
                    "categorical"
            },

            {
                "feature":
                    "participant_bmi",

                "meaning":
                    "Body mass index",

                "unit":
                    "kg/m^2"
            },

            {
                "feature":
                    "diabetes_duration_years",

                "meaning":
                    "Recorded duration of diabetes",

                "unit":
                    "years"
            },

            {
                "feature":
                    "hba1c_mmol_mol",

                "meaning":
                    "Recorded HbA1c",

                "unit":
                    "mmol/mol"
            }
        ]
    )

    dictionary[
        "uses_future_information"
    ] = "No"

    return dictionary


# =========================================================
# 9. SUMMARY REPORT
# =========================================================

def create_summary(
    data,
    missingness,
    consistency
):

    inconsistent_cells = 0

    if not consistency.empty:

        consistency_columns = [
            column
            for column
            in consistency.columns
            if column.endswith(
                "_consistent"
            )
        ]

        for column in consistency_columns:

            inconsistent_cells += (
                consistency[column]
                == False
            ).sum()

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
                    "participant_features_added",

                "value":
                    5
            },

            {
                "metric":
                    "missing_prediction_targets",

                "value":
                    data[
                        "target_glucose_30min"
                    ]
                    .isna()
                    .sum()
            },

            {
                "metric":
                    "repeated_participants_checked",

                "value":
                    len(
                        consistency
                    )
            },

            {
                "metric":
                    "inconsistent_repeated_feature_checks",

                "value":
                    int(
                        inconsistent_cells
                    )
            }
        ]
    )


# =========================================================
# 10. SAVE OUTPUTS
# =========================================================

def save_outputs(
    data,
    summary_report,
    missingness,
    dictionary,
    consistency
):

    print("\n" + "=" * 70)
    print("SAVING STEP 7 OUTPUTS")
    print("=" * 70)

    data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    summary_report.to_csv(
        SUMMARY_REPORT_FILE,
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

    consistency.to_csv(
        CONSISTENCY_FILE,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT_FILE)
    print(SUMMARY_REPORT_FILE)
    print(MISSINGNESS_FILE)
    print(DICTIONARY_FILE)
    print(CONSISTENCY_FILE)


# =========================================================
# 11. FINAL SUMMARY
# =========================================================

def print_final_summary(
    data,
    missingness,
    consistency
):

    print("\n" + "=" * 70)
    print("STEP 7 FINAL SUMMARY")
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
        "Participant-level features added: 5"
    )

    print(
        "Repeated participants checked:",
        len(consistency)
    )

    print(
        "\nParticipant feature missingness:"
    )

    print(
        missingness[
            [
                "feature",
                "missing_participants",
                "missing_participant_percent"
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nMissing prediction targets:",
        data[
            "target_glucose_30min"
        ]
        .isna()
        .sum()
    )


# =========================================================
# 12. MAIN
# =========================================================

def main():

    print("=" * 70)
    print("SMART BLOOD GLUCOSE ML PROJECT")
    print("STEP 7 - PARTICIPANT CLINICAL FEATURES")
    print("=" * 70)

    time_series, summary = (
        load_data()
    )

    clinical = (
        create_clinical_table(
            summary
        )
    )

    consistency = (
        check_repeated_participants(
            clinical
        )
    )

    data = (
        merge_features(
            time_series,
            clinical
        )
    )

    missingness = (
        create_missingness_report(
            data
        )
    )

    validate_data(
        data
    )

    dictionary = (
        create_dictionary()
    )

    summary_report = (
        create_summary(
            data,
            missingness,
            consistency
        )
    )

    save_outputs(
        data,
        summary_report,
        missingness,
        dictionary,
        consistency
    )

    print_final_summary(
        data,
        missingness,
        consistency
    )

    print("\n" + "=" * 70)
    print("STEP 7 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()