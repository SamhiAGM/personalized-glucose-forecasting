from pathlib import Path
import pandas as pd
import numpy as np
import re


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CGMACROS_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cgmacros"
    / "CGMacros"
)

BIO_FILE = (
    CGMACROS_DIR
    / "bio.csv"
)

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

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 2. OUTPUT FILES
# =========================================================

STANDARDIZED_FILE = (
    PROCESSED_DIR
    / "cgmacros_standardized_all.csv"
)

CLEAN_GLUCOSE_FILE = (
    PROCESSED_DIR
    / "cgmacros_glucose_clean.csv"
)

BIO_STANDARDIZED_FILE = (
    PROCESSED_DIR
    / "cgmacros_bio_standardized.csv"
)

PARTICIPANT_REPORT_FILE = (
    REPORT_DIR
    / "cgmacros_participant_standardization_report.csv"
)

MISSINGNESS_FILE = (
    REPORT_DIR
    / "cgmacros_standardized_missingness.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "cgmacros_standardization_summary.csv"
)


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================

def clean_column_names(df):

    df = df.copy()

    # Fix column names such as:
    # "Amount Consumed "
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df


def snake_case(name):

    name = (
        str(name)
        .strip()
        .lower()
    )

    name = re.sub(
        r"[^a-z0-9]+",
        "_",
        name
    )

    return name.strip("_")


# =========================================================
# 4. STANDARDIZE ONE PARTICIPANT FILE
# =========================================================

def standardize_participant_file(
    file_path
):

    df = pd.read_csv(
        file_path,
        low_memory=False
    )

    df = clean_column_names(
        df
    )

    # -----------------------------------------------------
    # Essential columns
    #
    # These are required for our primary glucose pipeline.
    # -----------------------------------------------------

    required_columns = [
        "Timestamp",
        "Libre GL"
    ]

    missing_required = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_required:

        raise ValueError(
            f"{file_path.name} missing essential columns: "
            f"{missing_required}"
        )

    # -----------------------------------------------------
    # Optional variables
    #
    # Some participant files do not contain every sensor,
    # activity, or nutrition column.
    # Missing optional columns are kept as NaN.
    # -----------------------------------------------------

    optional_columns = [
        "Dexcom GL",
        "HR",
        "Calories (Activity)",
        "METs",
        "Meal Type",
        "Calories",
        "Carbs",
        "Protein",
        "Fat",
        "Fiber",
        "Amount Consumed",
        "Image path"
    ]

    missing_optional = []

    for column in optional_columns:

        if column not in df.columns:

            df[column] = np.nan

            missing_optional.append(
                column
            )

    # -----------------------------------------------------
    # Select relevant columns
    # -----------------------------------------------------

    selected_columns = [
        "Timestamp",
        "Libre GL",
        "Dexcom GL",
        "HR",
        "Calories (Activity)",
        "METs",
        "Meal Type",
        "Calories",
        "Carbs",
        "Protein",
        "Fat",
        "Fiber",
        "Amount Consumed",
        "Image path"
    ]

    data = df[
        selected_columns
    ].copy()

    participant_id = (
        file_path.parent.name
    )

    # -----------------------------------------------------
    # Rename columns to consistent ML-friendly names
    # -----------------------------------------------------

    rename_map = {

        "Timestamp":
            "timestamp",

        "Libre GL":
            "glucose_mg_dl",

        "Dexcom GL":
            "dexcom_glucose_mg_dl",

        "HR":
            "heart_rate_bpm",

        "Calories (Activity)":
            "activity_calories_last_min",

        "METs":
            "mets_raw_x10",

        "Meal Type":
            "meal_type",

        "Calories":
            "meal_calories",

        "Carbs":
            "meal_carbs_g",

        "Protein":
            "meal_protein_g",

        "Fat":
            "meal_fat_g",

        "Fiber":
            "meal_fiber_g",

        "Amount Consumed":
            "meal_amount_consumed_percent",

        "Image path":
            "meal_image_path"
    }

    data = data.rename(
        columns=rename_map
    )

    # -----------------------------------------------------
    # Add participant/source identity
    # -----------------------------------------------------

    data.insert(
        0,
        "participant_id",
        participant_id
    )

    data.insert(
        1,
        "source_file",
        file_path.name
    )

    # -----------------------------------------------------
    # Parse timestamp
    # -----------------------------------------------------

    data[
        "timestamp"
    ] = pd.to_datetime(
        data[
            "timestamp"
        ],
        errors="coerce",
        format="mixed"
    )

    # -----------------------------------------------------
    # Numeric conversion
    # -----------------------------------------------------

    numeric_columns = [
        "glucose_mg_dl",
        "dexcom_glucose_mg_dl",
        "heart_rate_bpm",
        "activity_calories_last_min",
        "mets_raw_x10",
        "meal_calories",
        "meal_carbs_g",
        "meal_protein_g",
        "meal_fat_g",
        "meal_fiber_g",
        "meal_amount_consumed_percent"
    ]

    for column in numeric_columns:

        data[
            column
        ] = pd.to_numeric(
            data[
                column
            ],
            errors="coerce"
        )

    # -----------------------------------------------------
    # Convert METs
    #
    # Data dictionary says stored MET values are multiplied
    # by 10.
    # -----------------------------------------------------

    data[
        "mets"
    ] = (
        data[
            "mets_raw_x10"
        ]
        / 10.0
    )

    # -----------------------------------------------------
    # Sort chronologically
    # -----------------------------------------------------

    data = data.sort_values(
        "timestamp",
        kind="stable",
        na_position="last"
    ).reset_index(
        drop=True
    )

    return (
        data,
        missing_optional
    )


# =========================================================
# 5. STANDARDIZE ALL PARTICIPANTS
# =========================================================

def standardize_all_participants():

    print("\n" + "=" * 70)
    print("STANDARDIZING CGMACROS PARTICIPANT FILES")
    print("=" * 70)

    files = sorted(
        CGMACROS_DIR.glob(
            "CGMacros-*/*.csv"
        )
    )

    print(
        "\nParticipant CSV files found:",
        len(files)
    )

    if len(files) == 0:

        raise FileNotFoundError(
            "No CGMacros participant CSV files found inside:\n"
            f"{CGMACROS_DIR}"
        )

    standardized_parts = []

    participant_reports = []

    for number, file_path in enumerate(
        files,
        start=1
    ):

        participant_id = (
            file_path.parent.name
        )

        print(
            f"Processing "
            f"{number}/{len(files)}: "
            f"{participant_id}"
        )

        (
            data,
            missing_optional
        ) = standardize_participant_file(
            file_path
        )

        # -------------------------------------------------
        # Valid timestamps
        # -------------------------------------------------

        valid_timestamp_data = (
            data[
                data[
                    "timestamp"
                ].notna()
            ]
            .sort_values(
                "timestamp"
            )
            .copy()
        )

        # -------------------------------------------------
        # Duplicate timestamp check
        # -------------------------------------------------

        duplicate_timestamps = (
            valid_timestamp_data
            .duplicated(
                subset=[
                    "timestamp"
                ]
            )
            .sum()
        )

        # -------------------------------------------------
        # Sampling interval
        # -------------------------------------------------

        differences = (
            valid_timestamp_data[
                "timestamp"
            ]
            .diff()
            .dt.total_seconds()
            .div(60)
        )

        positive_differences = (
            differences[
                differences > 0
            ]
        )

        median_interval = (

            positive_differences.median()

            if not positive_differences.empty

            else np.nan
        )

        # -------------------------------------------------
        # Official Libre range from data dictionary:
        # 40–400 mg/dL
        #
        # Report only. Do NOT automatically delete.
        # -------------------------------------------------

        glucose = (
            data[
                "glucose_mg_dl"
            ]
        )

        out_of_range = (
            glucose.notna()
            &
            (
                (glucose < 40)
                |
                (glucose > 400)
            )
        ).sum()

        participant_reports.append(
            {
                "participant_id":
                    participant_id,

                "source_file":
                    file_path.name,

                "rows":
                    len(data),

                "valid_timestamps":
                    int(
                        data[
                            "timestamp"
                        ]
                        .notna()
                        .sum()
                    ),

                "invalid_timestamps":
                    int(
                        data[
                            "timestamp"
                        ]
                        .isna()
                        .sum()
                    ),

                "valid_libre_glucose":
                    int(
                        data[
                            "glucose_mg_dl"
                        ]
                        .notna()
                        .sum()
                    ),

                "valid_dexcom_glucose":
                    int(
                        data[
                            "dexcom_glucose_mg_dl"
                        ]
                        .notna()
                        .sum()
                    ),

                "valid_heart_rate":
                    int(
                        data[
                            "heart_rate_bpm"
                        ]
                        .notna()
                        .sum()
                    ),

                "valid_activity_calories":
                    int(
                        data[
                            "activity_calories_last_min"
                        ]
                        .notna()
                        .sum()
                    ),

                "valid_mets":
                    int(
                        data[
                            "mets"
                        ]
                        .notna()
                        .sum()
                    ),

                "meal_events":
                    int(
                        data[
                            "meal_type"
                        ]
                        .notna()
                        .sum()
                    ),

                "duplicate_timestamps":
                    int(
                        duplicate_timestamps
                    ),

                "median_sampling_interval_min":
                    (
                        float(
                            median_interval
                        )

                        if pd.notna(
                            median_interval
                        )

                        else np.nan
                    ),

                "libre_outside_documented_range":
                    int(
                        out_of_range
                    ),

                "missing_optional_columns":
                    (
                        "; ".join(
                            missing_optional
                        )

                        if missing_optional

                        else ""
                    )
            }
        )

        standardized_parts.append(
            data
        )

    combined = pd.concat(
        standardized_parts,
        ignore_index=True,
        sort=False
    )

    combined = combined.sort_values(
        [
            "participant_id",
            "timestamp"
        ],
        kind="stable",
        na_position="last"
    ).reset_index(
        drop=True
    )

    participant_report = (
        pd.DataFrame(
            participant_reports
        )
    )

    return (
        combined,
        participant_report
    )


# =========================================================
# 6. CREATE CLEAN PRIMARY GLUCOSE TABLE
# =========================================================

def create_glucose_clean_table(
    standardized
):

    print("\n" + "=" * 70)
    print("CREATING PRIMARY LIBRE GLUCOSE TABLE")
    print("=" * 70)

    starting_rows = len(
        standardized
    )

    invalid_timestamp_count = (
        standardized[
            "timestamp"
        ]
        .isna()
        .sum()
    )

    missing_glucose_count = (
        standardized[
            "glucose_mg_dl"
        ]
        .isna()
        .sum()
    )

    print(
        "\nStarting rows:",
        starting_rows
    )

    print(
        "Invalid timestamps:",
        invalid_timestamp_count
    )

    print(
        "Missing Libre glucose:",
        missing_glucose_count
    )

    # -----------------------------------------------------
    # Keep rows usable as current Libre glucose observations.
    #
    # Dexcom, HR, meals and activity are NOT required here.
    # -----------------------------------------------------

    clean = standardized[
        standardized[
            "timestamp"
        ].notna()
        &
        standardized[
            "glucose_mg_dl"
        ].notna()
    ].copy()

    clean = clean.sort_values(
        [
            "participant_id",
            "timestamp"
        ],
        kind="stable"
    ).reset_index(
        drop=True
    )

    print(
        "Rows in clean Libre table:",
        len(clean)
    )

    print(
        "Rows removed:",
        starting_rows - len(clean)
    )

    return clean


# =========================================================
# 7. STANDARDIZE BIO TABLE
# =========================================================

def standardize_bio():

    print("\n" + "=" * 70)
    print("STANDARDIZING CGMACROS BIO TABLE")
    print("=" * 70)

    if not BIO_FILE.exists():

        raise FileNotFoundError(
            f"bio.csv not found:\n"
            f"{BIO_FILE}"
        )

    bio = pd.read_csv(
        BIO_FILE,
        low_memory=False
    )

    bio = clean_column_names(
        bio
    )

    # Convert bio column names to snake_case
    bio.columns = [
        snake_case(
            column
        )
        for column in bio.columns
    ]

    if "subject" not in bio.columns:

        raise ValueError(
            "bio.csv does not contain a 'subject' column."
        )

    def make_participant_id(
        value
    ):

        if pd.isna(
            value
        ):

            return pd.NA

        try:

            number = int(
                float(
                    value
                )
            )

            return (
                f"CGMacros-{number:03d}"
            )

        except (
            ValueError,
            TypeError
        ):

            return (
                str(value)
                .strip()
            )

    bio.insert(
        0,
        "participant_id",
        bio[
            "subject"
        ].apply(
            make_participant_id
        )
    )

    print(
        "\nBio rows:",
        len(bio)
    )

    print(
        "Bio participants:",
        bio[
            "participant_id"
        ].nunique()
    )

    return bio


# =========================================================
# 8. MISSINGNESS REPORT
# =========================================================

def create_missingness_report(
    data
):

    important_columns = [
        "timestamp",
        "glucose_mg_dl",
        "dexcom_glucose_mg_dl",
        "heart_rate_bpm",
        "activity_calories_last_min",
        "mets",
        "meal_type",
        "meal_calories",
        "meal_carbs_g",
        "meal_protein_g",
        "meal_fat_g",
        "meal_fiber_g",
        "meal_amount_consumed_percent"
    ]

    records = []

    for column in important_columns:

        missing = (
            data[
                column
            ]
            .isna()
            .sum()
        )

        records.append(
            {
                "column":
                    column,

                "total_rows":
                    len(data),

                "non_null":
                    int(
                        len(data)
                        - missing
                    ),

                "missing_count":
                    int(
                        missing
                    ),

                "missing_percent":
                    round(
                        (
                            missing
                            / len(data)
                            * 100
                        )

                        if len(data) > 0

                        else np.nan,

                        2
                    )
            }
        )

    return pd.DataFrame(
        records
    )


# =========================================================
# 9. CREATE SUMMARY REPORT
# =========================================================

def create_summary(
    standardized,
    clean,
    bio,
    participant_report
):

    glucose = (
        standardized[
            "glucose_mg_dl"
        ]
    )

    out_of_range = (
        glucose.notna()
        &
        (
            (glucose < 40)
            |
            (glucose > 400)
        )
    ).sum()

    duplicate_participant_timestamps = (
        standardized[
            standardized[
                "timestamp"
            ].notna()
        ]
        .duplicated(
            subset=[
                "participant_id",
                "timestamp"
            ]
        )
        .sum()
    )

    summary = pd.DataFrame(
        [
            {
                "metric":
                    "participant_files_processed",

                "value":
                    participant_report[
                        "participant_id"
                    ].nunique()
            },

            {
                "metric":
                    "participants",

                "value":
                    standardized[
                        "participant_id"
                    ].nunique()
            },

            {
                "metric":
                    "rows_loaded",

                "value":
                    len(
                        standardized
                    )
            },

            {
                "metric":
                    "valid_libre_glucose_rows",

                "value":
                    int(
                        standardized[
                            "glucose_mg_dl"
                        ]
                        .notna()
                        .sum()
                    )
            },

            {
                "metric":
                    "valid_dexcom_glucose_rows",

                "value":
                    int(
                        standardized[
                            "dexcom_glucose_mg_dl"
                        ]
                        .notna()
                        .sum()
                    )
            },

            {
                "metric":
                    "clean_primary_glucose_rows",

                "value":
                    len(
                        clean
                    )
            },

            {
                "metric":
                    "invalid_timestamp_rows",

                "value":
                    int(
                        standardized[
                            "timestamp"
                        ]
                        .isna()
                        .sum()
                    )
            },

            {
                "metric":
                    "duplicate_participant_timestamps",

                "value":
                    int(
                        duplicate_participant_timestamps
                    )
            },

            {
                "metric":
                    "libre_outside_documented_range",

                "value":
                    int(
                        out_of_range
                    )
            },

            {
                "metric":
                    "recorded_meal_events",

                "value":
                    int(
                        standardized[
                            "meal_type"
                        ]
                        .notna()
                        .sum()
                    )
            },

            {
                "metric":
                    "bio_participants",

                "value":
                    bio[
                        "participant_id"
                    ].nunique()
            }
        ]
    )

    return summary


# =========================================================
# 10. VALIDATE OUTPUTS
# =========================================================

def validate_outputs(
    standardized,
    clean,
    bio
):

    print("\n" + "=" * 70)
    print("VALIDATING CGMACROS STANDARDIZATION")
    print("=" * 70)

    duplicate_timestamps = (
        clean.duplicated(
            subset=[
                "participant_id",
                "timestamp"
            ]
        ).sum()
    )

    print(
        "\nParticipants:",
        standardized[
            "participant_id"
        ].nunique()
    )

    print(
        "Clean glucose participants:",
        clean[
            "participant_id"
        ].nunique()
    )

    print(
        "Bio participants:",
        bio[
            "participant_id"
        ].nunique()
    )

    print(
        "Missing primary glucose in clean table:",
        clean[
            "glucose_mg_dl"
        ]
        .isna()
        .sum()
    )

    print(
        "Missing timestamps in clean table:",
        clean[
            "timestamp"
        ]
        .isna()
        .sum()
    )

    print(
        "Duplicate participant timestamps:",
        duplicate_timestamps
    )

    if (
        clean[
            "glucose_mg_dl"
        ]
        .isna()
        .sum()
        != 0
    ):

        raise ValueError(
            "Clean primary glucose table contains missing glucose."
        )

    if (
        clean[
            "timestamp"
        ]
        .isna()
        .sum()
        != 0
    ):

        raise ValueError(
            "Clean primary glucose table contains invalid timestamps."
        )

    if (
        clean[
            "participant_id"
        ]
        .nunique()
        == 0
    ):

        raise ValueError(
            "No participants remain in clean table."
        )

    print(
        "\nCGMacros validation passed."
    )


# =========================================================
# 11. SAVE OUTPUTS
# =========================================================

def save_outputs(
    standardized,
    clean,
    bio,
    participant_report,
    missingness,
    summary
):

    print("\n" + "=" * 70)
    print("SAVING STEP 10 OUTPUTS")
    print("=" * 70)

    standardized.to_csv(
        STANDARDIZED_FILE,
        index=False
    )

    clean.to_csv(
        CLEAN_GLUCOSE_FILE,
        index=False
    )

    bio.to_csv(
        BIO_STANDARDIZED_FILE,
        index=False
    )

    participant_report.to_csv(
        PARTICIPANT_REPORT_FILE,
        index=False
    )

    missingness.to_csv(
        MISSINGNESS_FILE,
        index=False
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    print("\nSaved:")

    print(
        STANDARDIZED_FILE
    )

    print(
        CLEAN_GLUCOSE_FILE
    )

    print(
        BIO_STANDARDIZED_FILE
    )

    print(
        PARTICIPANT_REPORT_FILE
    )

    print(
        MISSINGNESS_FILE
    )

    print(
        SUMMARY_FILE
    )


# =========================================================
# 12. FINAL SUMMARY
# =========================================================

def print_final_summary(
    standardized,
    clean,
    bio,
    participant_report
):

    print("\n" + "=" * 70)
    print("STEP 10 FINAL SUMMARY")
    print("=" * 70)

    print(
        "\nParticipant files:",
        participant_report[
            "participant_id"
        ].nunique()
    )

    print(
        "Participants:",
        standardized[
            "participant_id"
        ].nunique()
    )

    print(
        "Rows loaded:",
        len(
            standardized
        )
    )

    print(
        "Valid Libre glucose rows:",
        standardized[
            "glucose_mg_dl"
        ]
        .notna()
        .sum()
    )

    print(
        "Valid Dexcom glucose rows:",
        standardized[
            "dexcom_glucose_mg_dl"
        ]
        .notna()
        .sum()
    )

    print(
        "Clean primary glucose rows:",
        len(
            clean
        )
    )

    print(
        "Missing primary glucose in clean table:",
        clean[
            "glucose_mg_dl"
        ]
        .isna()
        .sum()
    )

    print(
        "Invalid timestamps in clean table:",
        clean[
            "timestamp"
        ]
        .isna()
        .sum()
    )

    print(
        "Duplicate participant timestamps:",
        clean.duplicated(
            subset=[
                "participant_id",
                "timestamp"
            ]
        ).sum()
    )

    print(
        "Recorded meal events:",
        standardized[
            "meal_type"
        ]
        .notna()
        .sum()
    )

    print(
        "Bio participants:",
        bio[
            "participant_id"
        ]
        .nunique()
    )

    median_sampling = (
        participant_report[
            "median_sampling_interval_min"
        ]
        .median()
    )

    print(
        "Median participant sampling interval (minutes):",
        median_sampling
    )

    print(
        "Libre values outside documented 40-400 range:",
        participant_report[
            "libre_outside_documented_range"
        ]
        .sum()
    )

    participants_with_missing_optional = (
        participant_report[
            "missing_optional_columns"
        ]
        .fillna("")
        .ne("")
        .sum()
    )

    print(
        "Participants missing one or more optional columns:",
        participants_with_missing_optional
    )


# =========================================================
# 13. MAIN
# =========================================================

def main():

    print("=" * 70)
    print("SMART BLOOD GLUCOSE ML PROJECT")
    print("STEP 10 - STANDARDIZE CGMACROS")
    print("=" * 70)

    (
        standardized,
        participant_report
    ) = standardize_all_participants()

    clean = (
        create_glucose_clean_table(
            standardized
        )
    )

    bio = (
        standardize_bio()
    )

    missingness = (
        create_missingness_report(
            standardized
        )
    )

    summary = (
        create_summary(
            standardized,
            clean,
            bio,
            participant_report
        )
    )

    validate_outputs(
        standardized,
        clean,
        bio
    )

    save_outputs(
        standardized,
        clean,
        bio,
        participant_report,
        missingness,
        summary
    )

    print_final_summary(
        standardized,
        clean,
        bio,
        participant_report
    )

    print("\n" + "=" * 70)
    print("STEP 10 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":

    main()