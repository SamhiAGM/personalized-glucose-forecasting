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
    / "cgmacros_temporal_features_ready.csv"
)

BIO_FILE = (
    PROCESSED_DIR
    / "cgmacros_bio_standardized.csv"
)


# =========================================================
# 3. OUTPUT FILES
# =========================================================

OUTPUT_FILE = (
    PROCESSED_DIR
    / "cgmacros_participant_features_ready.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "cgmacros_participant_feature_summary.csv"
)

MISSINGNESS_FILE = (
    REPORT_DIR
    / "cgmacros_participant_feature_missingness.csv"
)

PARTICIPANT_REPORT_FILE = (
    REPORT_DIR
    / "cgmacros_participant_feature_by_participant.csv"
)

DICTIONARY_FILE = (
    REPORT_DIR
    / "cgmacros_participant_feature_dictionary.csv"
)


# =========================================================
# 4. LOAD DATA
# =========================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING STEP 15 INPUT DATA")
    print("=" * 70)

    features = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["timestamp"],
        low_memory=False
    )

    bio = pd.read_csv(
        BIO_FILE,
        low_memory=False
    )

    features[
        "participant_id"
    ] = (
        features[
            "participant_id"
        ]
        .astype(str)
    )

    bio[
        "participant_id"
    ] = (
        bio[
            "participant_id"
        ]
        .astype(str)
    )

    print(
        "\nPrediction rows:",
        len(features)
    )

    print(
        "Prediction participants:",
        features[
            "participant_id"
        ].nunique()
    )

    print(
        "Bio rows:",
        len(bio)
    )

    print(
        "Bio participants:",
        bio[
            "participant_id"
        ].nunique()
    )

    return features, bio


# =========================================================
# 5. VALIDATE BIO PARTICIPANTS
# =========================================================

def validate_bio(
    features,
    bio
):

    print("\n" + "=" * 70)
    print("VALIDATING BIO DATA")
    print("=" * 70)

    duplicate_bio_ids = (
        bio.duplicated(
            subset=[
                "participant_id"
            ]
        ).sum()
    )

    prediction_ids = set(
        features[
            "participant_id"
        ].unique()
    )

    bio_ids = set(
        bio[
            "participant_id"
        ].unique()
    )

    missing_from_bio = sorted(
        prediction_ids
        -
        bio_ids
    )

    print(
        "\nDuplicate bio participant IDs:",
        duplicate_bio_ids
    )

    print(
        "Prediction participants missing from bio:",
        len(missing_from_bio)
    )

    if missing_from_bio:

        print(
            missing_from_bio
        )

    if duplicate_bio_ids != 0:

        raise ValueError(
            "Duplicate participant IDs detected in bio table."
        )

    if missing_from_bio:

        raise ValueError(
            "Some prediction participants are missing "
            "from bio.csv."
        )

    print(
        "\nBio participant validation passed."
    )


# =========================================================
# 6. PREPARE PARTICIPANT FEATURES
# =========================================================

def prepare_participant_features(
    bio
):

    print("\n" + "=" * 70)
    print("PREPARING PARTICIPANT / CLINICAL FEATURES")
    print("=" * 70)

    required_columns = [
        "participant_id",
        "age",
        "gender",
        "bmi"
    ]

    missing_required = [
        column
        for column in required_columns
        if column not in bio.columns
    ]

    if missing_required:

        raise ValueError(
            "Required bio columns missing: "
            f"{missing_required}"
        )

    participant = bio.copy()

    # -----------------------------------------------------
    # Numeric participant characteristics
    # -----------------------------------------------------

    participant[
        "participant_age"
    ] = pd.to_numeric(
        participant[
            "age"
        ],
        errors="coerce"
    )

    participant[
        "participant_bmi"
    ] = pd.to_numeric(
        participant[
            "bmi"
        ],
        errors="coerce"
    )

    # -----------------------------------------------------
    # Gender / sex encoding
    #
    # Male   -> 1
    # Female -> 0
    #
    # Unknown/unexpected values remain NaN.
    # Original text is preserved separately.
    # -----------------------------------------------------

    participant[
        "participant_sex_original"
    ] = (
        participant[
            "gender"
        ]
        .astype("string")
        .str.strip()
    )

    normalized_gender = (
        participant[
            "participant_sex_original"
        ]
        .str.lower()
    )

    sex_map = {
        "male": 1,
        "m": 1,
        "female": 0,
        "f": 0
    }

    participant[
        "participant_sex_code"
    ] = (
        normalized_gender
        .map(
            sex_map
        )
    )

    # -----------------------------------------------------
    # HbA1c
    # -----------------------------------------------------

    if (
        "a1c_pdl_lab"
        in participant.columns
    ):

        participant[
            "hba1c_lab"
        ] = pd.to_numeric(
            participant[
                "a1c_pdl_lab"
            ],
            errors="coerce"
        )

    else:

        participant[
            "hba1c_lab"
        ] = np.nan

    # -----------------------------------------------------
    # Fasting laboratory glucose
    # -----------------------------------------------------

    fasting_candidates = [
        "fasting_glu_pdl_lab",
        "fasting_glucose_pdl_lab"
    ]

    fasting_source = None

    for candidate in fasting_candidates:

        if candidate in participant.columns:

            fasting_source = candidate
            break

    if fasting_source is not None:

        participant[
            "fasting_glucose_lab"
        ] = pd.to_numeric(
            participant[
                fasting_source
            ],
            errors="coerce"
        )

    else:

        participant[
            "fasting_glucose_lab"
        ] = np.nan

    selected_columns = [
        "participant_id",
        "participant_age",
        "participant_sex_original",
        "participant_sex_code",
        "participant_bmi",
        "hba1c_lab",
        "fasting_glucose_lab"
    ]

    participant = participant[
        selected_columns
    ].copy()

    print(
        "\nParticipant feature rows:",
        len(participant)
    )

    print(
        "Participant IDs:",
        participant[
            "participant_id"
        ].nunique()
    )

    print(
        "\nGender values:"
    )

    print(
        participant[
            "participant_sex_original"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    return participant


# =========================================================
# 7. MERGE PARTICIPANT FEATURES
# =========================================================

def merge_participant_features(
    features,
    participant
):

    print("\n" + "=" * 70)
    print("MERGING PARTICIPANT FEATURES")
    print("=" * 70)

    original_rows = len(
        features
    )

    result = features.merge(
        participant,
        on="participant_id",
        how="left",
        validate="many_to_one"
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
            "Row count changed during participant merge."
        )

    return result


# =========================================================
# 8. VALIDATE FEATURE CONSISTENCY
# =========================================================

def validate_features(
    data
):

    print("\n" + "=" * 70)
    print("VALIDATING PARTICIPANT FEATURES")
    print("=" * 70)

    participant_columns = [
        "participant_age",
        "participant_sex_code",
        "participant_bmi",
        "hba1c_lab",
        "fasting_glucose_lab"
    ]

    # -----------------------------------------------------
    # Check participant-level consistency.
    #
    # A participant should not have multiple different ages,
    # BMIs, sex codes, etc. across prediction rows.
    # -----------------------------------------------------

    consistency_records = []

    for participant_id, group in (
        data.groupby(
            "participant_id",
            sort=True
        )
    ):

        record = {
            "participant_id":
                participant_id
        }

        for column in participant_columns:

            unique_non_missing = (
                group[
                    column
                ]
                .dropna()
                .nunique()
            )

            record[
                f"{column}_unique_values"
            ] = int(
                unique_non_missing
            )

            record[
                f"{column}_consistent"
            ] = (
                unique_non_missing
                <= 1
            )

        consistency_records.append(
            record
        )

    consistency = pd.DataFrame(
        consistency_records
    )

    consistency_columns = [
        column
        for column in consistency.columns
        if column.endswith(
            "_consistent"
        )
    ]

    inconsistent_count = (
        ~consistency[
            consistency_columns
        ]
        .all(axis=1)
    ).sum()

    print(
        "\nParticipants with inconsistent "
        "participant features:",
        inconsistent_count
    )

    print(
        "Missing prediction targets:",
        data[
            "target_glucose_30min"
        ]
        .isna()
        .sum()
    )

    if inconsistent_count != 0:

        raise ValueError(
            "Participant feature consistency validation failed."
        )

    if (
        data[
            "target_glucose_30min"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "Prediction targets became missing."
        )

    print(
        "\nParticipant feature validation passed."
    )

    return consistency


# =========================================================
# 9. CREATE MISSINGNESS REPORT
# =========================================================

def create_missingness_report(
    data
):

    feature_columns = [
        "participant_age",
        "participant_sex_code",
        "participant_bmi",
        "hba1c_lab",
        "fasting_glucose_lab"
    ]

    records = []

    total_participants = (
        data[
            "participant_id"
        ].nunique()
    )

    participant_level = (
        data[
            [
                "participant_id"
            ]
            +
            feature_columns
        ]
        .drop_duplicates(
            subset=[
                "participant_id"
            ]
        )
    )

    for column in feature_columns:

        missing_rows = (
            data[
                column
            ]
            .isna()
            .sum()
        )

        missing_participants = (
            participant_level[
                column
            ]
            .isna()
            .sum()
        )

        records.append(
            {
                "feature":
                    column,

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

                "missing_participants":
                    int(
                        missing_participants
                    ),

                "total_participants":
                    total_participants,

                "missing_participant_percent":
                    round(
                        missing_participants
                        /
                        total_participants
                        *
                        100,
                        2
                    )
            }
        )

    return pd.DataFrame(
        records
    )


# =========================================================
# 10. PARTICIPANT REPORT
# =========================================================

def create_participant_report(
    data
):

    columns = [
        "participant_id",
        "participant_age",
        "participant_sex_original",
        "participant_sex_code",
        "participant_bmi",
        "hba1c_lab",
        "fasting_glucose_lab"
    ]

    report = (
        data[
            columns
        ]
        .drop_duplicates(
            subset=[
                "participant_id"
            ]
        )
        .sort_values(
            "participant_id"
        )
        .reset_index(
            drop=True
        )
    )

    return report


# =========================================================
# 11. FEATURE DICTIONARY
# =========================================================

def create_dictionary():

    return pd.DataFrame(
        [
            {
                "feature":
                    "participant_age",

                "meaning":
                    "Participant age from CGMacros bio data",

                "unit":
                    "years",

                "uses_future_information":
                    "No"
            },

            {
                "feature":
                    "participant_sex_code",

                "meaning":
                    "Binary encoding of participant gender: female=0, male=1",

                "unit":
                    "binary",

                "uses_future_information":
                    "No"
            },

            {
                "feature":
                    "participant_bmi",

                "meaning":
                    "Participant body mass index from CGMacros bio data",

                "unit":
                    "kg/m^2",

                "uses_future_information":
                    "No"
            },

            {
                "feature":
                    "hba1c_lab",

                "meaning":
                    "Participant laboratory HbA1c measurement where available",

                "unit":
                    "dataset laboratory unit",

                "uses_future_information":
                    "No"
            },

            {
                "feature":
                    "fasting_glucose_lab",

                "meaning":
                    "Participant fasting laboratory glucose measurement where available",

                "unit":
                    "dataset laboratory unit",

                "uses_future_information":
                    "No"
            }
        ]
    )


# =========================================================
# 12. SUMMARY
# =========================================================

def create_summary(
    data,
    missingness
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
                    "participant_features_created",

                "value":
                    5
            },

            {
                "metric":
                    "missing_age_participants",

                "value":
                    int(
                        missingness.loc[
                            missingness[
                                "feature"
                            ]
                            ==
                            "participant_age",
                            "missing_participants"
                        ].iloc[0]
                    )
            },

            {
                "metric":
                    "missing_sex_participants",

                "value":
                    int(
                        missingness.loc[
                            missingness[
                                "feature"
                            ]
                            ==
                            "participant_sex_code",
                            "missing_participants"
                        ].iloc[0]
                    )
            },

            {
                "metric":
                    "missing_bmi_participants",

                "value":
                    int(
                        missingness.loc[
                            missingness[
                                "feature"
                            ]
                            ==
                            "participant_bmi",
                            "missing_participants"
                        ].iloc[0]
                    )
            },

            {
                "metric":
                    "missing_hba1c_participants",

                "value":
                    int(
                        missingness.loc[
                            missingness[
                                "feature"
                            ]
                            ==
                            "hba1c_lab",
                            "missing_participants"
                        ].iloc[0]
                    )
            },

            {
                "metric":
                    "missing_fasting_glucose_participants",

                "value":
                    int(
                        missingness.loc[
                            missingness[
                                "feature"
                            ]
                            ==
                            "fasting_glucose_lab",
                            "missing_participants"
                        ].iloc[0]
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
    missingness,
    participant_report,
    consistency,
    dictionary,
    summary
):

    print("\n" + "=" * 70)
    print("SAVING STEP 15 OUTPUTS")
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

    consistency.to_csv(
        REPORT_DIR
        /
        "cgmacros_participant_feature_consistency.csv",
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

    print(
        OUTPUT_FILE
    )

    print(
        MISSINGNESS_FILE
    )

    print(
        PARTICIPANT_REPORT_FILE
    )

    print(
        REPORT_DIR
        /
        "cgmacros_participant_feature_consistency.csv"
    )

    print(
        DICTIONARY_FILE
    )

    print(
        SUMMARY_FILE
    )


# =========================================================
# 14. FINAL SUMMARY
# =========================================================

def print_final_summary(
    data,
    missingness
):

    print("\n" + "=" * 70)
    print("STEP 15 FINAL SUMMARY")
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
        "Participant/clinical features created: 5"
    )

    print(
        "\nParticipant-level missingness:"
    )

    print(
        missingness[
            [
                "feature",
                "missing_participants",
                "total_participants",
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
# 15. MAIN
# =========================================================

def main():

    print("=" * 70)
    print("SMART BLOOD GLUCOSE ML PROJECT")
    print("STEP 15 - CGMACROS PARTICIPANT/CLINICAL FEATURES")
    print("=" * 70)

    features, bio = (
        load_data()
    )

    validate_bio(
        features,
        bio
    )

    participant = (
        prepare_participant_features(
            bio
        )
    )

    data = (
        merge_participant_features(
            features,
            participant
        )
    )

    consistency = (
        validate_features(
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
            missingness
        )
    )

    save_outputs(
        data,
        missingness,
        participant_report,
        consistency,
        dictionary,
        summary
    )

    print_final_summary(
        data,
        missingness
    )

    print("\n" + "=" * 70)
    print("STEP 15 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":

    main()