from pathlib import Path
import pandas as pd
import numpy as np
import re


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SHANGHAI_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "shanghai"
)

SESSION_DIR = (
    SHANGHAI_DIR
    / "Shanghai_T2DM"
)

SUMMARY_FILE = (
    SHANGHAI_DIR
    / "Shanghai_T2DM_Summary.xlsx"
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
# 2. COLUMN NAME CLEANING
# =========================================================

def clean_name(name):
    """
    Remove unnecessary spaces.
    """
    return re.sub(
        r"\s+",
        " ",
        str(name).strip()
    )


def canonical_name(name):
    """
    Convert different raw column names
    into consistent internal names.

    Raw files are NOT modified.
    """

    s = clean_name(name).lower()

    if s == "date" or s.startswith("date "):
        return "timestamp"

    if s.startswith("cgm"):
        return "cgm_mg_dl"

    if s.startswith("cbg"):
        return "cbg_mg_dl"

    if s.startswith("blood ketone"):
        return "blood_ketone"

    if s == "dietary intake":
        return "dietary_intake"

    if s == "饮食" or s == "进食量":
        return "dietary_intake_chinese"

    if s.startswith("insulin dose - s.c"):
        return "insulin_sc"

    if s.startswith(
        "non-insulin hypoglycemic agents"
    ):
        return "non_insulin_agents"

    if s.startswith("csii - bolus insulin"):
        return "csii_bolus"

    if (
        s.startswith("csii - basal insulin")
        or "胰岛素泵基础量" in s
    ):
        return "csii_basal"

    if s.startswith("insulin dose - i.v"):
        return "insulin_iv"

    # Generic conversion for other columns
    s = re.sub(
        r"[^a-z0-9]+",
        "_",
        s
    ).strip("_")

    return s or "unnamed"


# =========================================================
# 3. MAKE DUPLICATE COLUMN NAMES UNIQUE
# =========================================================

def make_unique_columns(columns):

    counts = {}
    result = []

    for column in columns:

        if column not in counts:
            counts[column] = 1
            result.append(column)

        else:
            counts[column] += 1

            result.append(
                f"{column}_{counts[column]}"
            )

    return result


# =========================================================
# 4. READ AND STANDARDIZE ALL SESSION FILES
# =========================================================

def standardize_sessions():

    print("\n" + "=" * 70)
    print("STANDARDIZING SHANGHAI T2DM SESSION FILES")
    print("=" * 70)

    files = sorted(
        list(
            SESSION_DIR.rglob("*.xlsx")
        )
        +
        list(
            SESSION_DIR.rglob("*.xls")
        )
    )

    print(
        f"\nSession files found: {len(files)}"
    )

    standardized_sessions = []

    processing_records = []

    for number, file_path in enumerate(
        files,
        start=1
    ):

        print(
            f"Processing "
            f"{number}/{len(files)}: "
            f"{file_path.name}"
        )

        try:

            df = pd.read_excel(
                file_path
            )

        except Exception as error:

            print(
                f"ERROR reading "
                f"{file_path.name}"
            )

            print(error)

            continue

        original_rows = len(df)

        # -------------------------------------------------
        # Standardize column names
        # -------------------------------------------------

        renamed_columns = [
            canonical_name(column)
            for column
            in df.columns
        ]

        df.columns = make_unique_columns(
            renamed_columns
        )

        # -------------------------------------------------
        # Add participant/session identity
        # -------------------------------------------------

        session_id = file_path.stem

        participant_id = (
            session_id.split("_")[0]
        )

        df.insert(
            0,
            "participant_id",
            participant_id
        )

        df.insert(
            1,
            "session_id",
            session_id
        )

        df.insert(
            2,
            "source_file",
            file_path.name
        )

        # -------------------------------------------------
        # Timestamp standardization
        # -------------------------------------------------

        if "timestamp" not in df.columns:

            print(
                "WARNING: Timestamp column "
                "not found."
            )

            continue

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        # -------------------------------------------------
        # CGM standardization
        # -------------------------------------------------

        if "cgm_mg_dl" not in df.columns:

            print(
                "WARNING: CGM column "
                "not found."
            )

            continue

        df["cgm_mg_dl"] = pd.to_numeric(
            df["cgm_mg_dl"],
            errors="coerce"
        )

        # -------------------------------------------------
        # Sort this session chronologically
        # -------------------------------------------------

        df = df.sort_values(
            "timestamp",
            kind="stable"
        ).reset_index(
            drop=True
        )

        invalid_timestamps = (
            df["timestamp"]
            .isna()
            .sum()
        )

        missing_cgm = (
            df["cgm_mg_dl"]
            .isna()
            .sum()
        )

        processing_records.append(
            {
                "session_id":
                    session_id,

                "participant_id":
                    participant_id,

                "input_rows":
                    original_rows,

                "invalid_timestamps":
                    invalid_timestamps,

                "missing_cgm":
                    missing_cgm
            }
        )

        standardized_sessions.append(
            df
        )

    # =====================================================
    # Combine sessions
    # =====================================================

    if not standardized_sessions:

        raise RuntimeError(
            "No Shanghai session files "
            "were successfully processed."
        )

    combined = pd.concat(
        standardized_sessions,
        ignore_index=True,
        sort=False
    )

    # Sort entire dataset
    combined = combined.sort_values(
        [
            "participant_id",
            "session_id",
            "timestamp"
        ],
        kind="stable"
    ).reset_index(
        drop=True
    )

    processing_report = pd.DataFrame(
        processing_records
    )

    return combined, processing_report


# =========================================================
# 5. CLEAN BASIC CGM TABLE
# =========================================================

def create_clean_cgm_table(
    standardized
):

    print("\n" + "=" * 70)
    print("CREATING CLEAN CGM BASE TABLE")
    print("=" * 70)

    clean = standardized.copy()

    starting_rows = len(clean)

    invalid_timestamp_mask = (
        clean["timestamp"].isna()
    )

    missing_cgm_mask = (
        clean["cgm_mg_dl"].isna()
    )

    invalid_timestamp_count = (
        invalid_timestamp_mask.sum()
    )

    missing_cgm_count = (
        missing_cgm_mask.sum()
    )

    print(
        f"\nStarting rows: "
        f"{starting_rows}"
    )

    print(
        "Invalid timestamps:",
        invalid_timestamp_count
    )

    print(
        "Missing CGM values:",
        missing_cgm_count
    )

    # -----------------------------------------
    # Remove rows impossible to use for
    # glucose forecasting
    # -----------------------------------------

    clean = clean[
        clean["timestamp"].notna()
        &
        clean["cgm_mg_dl"].notna()
    ].copy()

    clean = clean.reset_index(
        drop=True
    )

    # -----------------------------------------
    # Check duplicate timestamps
    # WITHIN each session
    # -----------------------------------------

    duplicate_mask = (
        clean.duplicated(
            subset=[
                "session_id",
                "timestamp"
            ],
            keep=False
        )
    )

    duplicate_count = (
        duplicate_mask.sum()
    )

    print(
        "Duplicate session timestamps:",
        duplicate_count
    )

    if duplicate_count > 0:

        print(
            "\nWARNING:"
            " Duplicate timestamps found."
        )

        print(
            "No duplicates were "
            "automatically deleted."
        )

    # -----------------------------------------
    # Final chronological ordering
    # -----------------------------------------

    clean = clean.sort_values(
        [
            "participant_id",
            "session_id",
            "timestamp"
        ],
        kind="stable"
    ).reset_index(
        drop=True
    )

    print(
        "\nRows remaining:",
        len(clean)
    )

    print(
        "Rows removed:",
        starting_rows - len(clean)
    )

    return clean


# =========================================================
# 6. STANDARDIZE PARTICIPANT SUMMARY
# =========================================================

def standardize_summary():

    print("\n" + "=" * 70)
    print("STANDARDIZING PARTICIPANT SUMMARY")
    print("=" * 70)

    summary = pd.read_excel(
        SUMMARY_FILE,
        sheet_name="T2DM"
    )

    # Convert summary column names
    # into safe snake_case names

    new_columns = []

    for column in summary.columns:

        cleaned = clean_name(
            column
        ).lower()

        cleaned = re.sub(
            r"[^a-z0-9]+",
            "_",
            cleaned
        ).strip("_")

        new_columns.append(
            cleaned
        )

    summary.columns = (
        make_unique_columns(
            new_columns
        )
    )

    # -----------------------------------------
    # Find patient/session identifier
    # -----------------------------------------

    if "patient_number" in summary.columns:

        summary[
            "session_id"
        ] = (
            summary[
                "patient_number"
            ]
            .astype(str)
        )

        summary[
            "participant_id"
        ] = (
            summary[
                "session_id"
            ]
            .str.split("_")
            .str[0]
        )

    print(
        "\nSummary rows:",
        len(summary)
    )

    print(
        "Summary columns:",
        len(summary.columns)
    )

    return summary


# =========================================================
# 7. QUALITY REPORT
# =========================================================

def create_quality_report(
    standardized,
    clean,
    processing_report
):

    report = {
        "session_files_processed":
            processing_report[
                "session_id"
            ].nunique(),

        "unique_participants":
            standardized[
                "participant_id"
            ].nunique(),

        "raw_loaded_rows":
            len(standardized),

        "invalid_timestamp_rows":
            standardized[
                "timestamp"
            ].isna().sum(),

        "missing_cgm_rows":
            standardized[
                "cgm_mg_dl"
            ].isna().sum(),

        "clean_cgm_rows":
            len(clean),

        "rows_removed":
            (
                len(standardized)
                - len(clean)
            ),

        "duplicate_session_timestamps":
            clean.duplicated(
                subset=[
                    "session_id",
                    "timestamp"
                ]
            ).sum()
    }

    quality = pd.DataFrame(
        [
            {
                "metric": key,
                "value": value
            }
            for key, value
            in report.items()
        ]
    )

    return quality


# =========================================================
# 8. SAVE OUTPUT FILES
# =========================================================

def save_outputs(
    standardized,
    clean,
    summary,
    processing_report,
    quality_report
):

    print("\n" + "=" * 70)
    print("SAVING PROCESSED OUTPUTS")
    print("=" * 70)

    standardized_file = (
        PROCESSED_DIR
        / "shanghai_standardized_all.csv"
    )

    clean_file = (
        PROCESSED_DIR
        / "shanghai_cgm_clean.csv"
    )

    summary_file = (
        PROCESSED_DIR
        / "shanghai_summary_standardized.csv"
    )

    processing_file = (
        REPORT_DIR
        / "shanghai_standardization_report.csv"
    )

    quality_file = (
        REPORT_DIR
        / "shanghai_cleaning_summary.csv"
    )

    standardized.to_csv(
        standardized_file,
        index=False
    )

    clean.to_csv(
        clean_file,
        index=False
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    processing_report.to_csv(
        processing_file,
        index=False
    )

    quality_report.to_csv(
        quality_file,
        index=False
    )

    print("\nSaved:")
    print(standardized_file)
    print(clean_file)
    print(summary_file)
    print(processing_file)
    print(quality_file)


# =========================================================
# 9. FINAL SUMMARY
# =========================================================

def print_final_summary(
    standardized,
    clean
):

    print("\n" + "=" * 70)
    print("STEP 3 FINAL SUMMARY")
    print("=" * 70)

    print(
        "\nSessions:",
        standardized[
            "session_id"
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
        len(standardized)
    )

    print(
        "Valid cleaned CGM rows:",
        len(clean)
    )

    print(
        "Removed rows:",
        (
            len(standardized)
            - len(clean)
        )
    )

    print(
        "Missing CGM remaining:",
        clean[
            "cgm_mg_dl"
        ].isna().sum()
    )

    print(
        "Missing timestamps remaining:",
        clean[
            "timestamp"
        ].isna().sum()
    )

    print(
        "Duplicate session timestamps:",
        clean.duplicated(
            subset=[
                "session_id",
                "timestamp"
            ]
        ).sum()
    )


# =========================================================
# 10. MAIN
# =========================================================

def main():

    print("=" * 70)
    print("SMART BLOOD GLUCOSE ML PROJECT")
    print("STEP 3 - STANDARDIZE SHANGHAI T2DM")
    print("=" * 70)

    standardized, processing_report = (
        standardize_sessions()
    )

    clean = create_clean_cgm_table(
        standardized
    )

    summary = standardize_summary()

    quality_report = (
        create_quality_report(
            standardized,
            clean,
            processing_report
        )
    )

    save_outputs(
        standardized,
        clean,
        summary,
        processing_report,
        quality_report
    )

    print_final_summary(
        standardized,
        clean
    )

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()