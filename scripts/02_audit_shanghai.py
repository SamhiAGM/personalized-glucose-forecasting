from pathlib import Path
import pandas as pd
import numpy as np
import re


# ---------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SHANGHAI_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "shanghai"
)

SUMMARY_FILE = (
    SHANGHAI_DIR
    / "Shanghai_T2DM_Summary.xlsx"
)

SESSION_DIR = (
    SHANGHAI_DIR
    / "Shanghai_T2DM"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "data_audit"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. COLUMN-NAME HELPERS
# ---------------------------------------------------------

def clean_name(name):
    """
    Remove unnecessary spaces from column names.
    """

    return re.sub(
        r"\s+",
        " ",
        str(name).strip()
    )


def canonical_name(name):
    """
    Convert different versions of the same column
    into one standard internal name.

    IMPORTANT:
    This does NOT modify the raw Excel files.
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

    # Generic safe name for other columns
    cleaned = re.sub(
        r"[^a-z0-9]+",
        "_",
        s
    ).strip("_")

    return cleaned or "unnamed"


# ---------------------------------------------------------
# 3. AUDIT SUMMARY / CLINICAL FILE
# ---------------------------------------------------------

def audit_summary_file():

    print("\n" + "=" * 70)
    print("AUDITING SHANGHAI T2DM SUMMARY FILE")
    print("=" * 70)

    if not SUMMARY_FILE.exists():

        print(
            f"ERROR: Summary file not found:\n"
            f"{SUMMARY_FILE}"
        )

        return None

    summary = pd.read_excel(
        SUMMARY_FILE,
        sheet_name="T2DM"
    )

    print(
        f"\nRows in summary file: "
        f"{len(summary)}"
    )

    print(
        f"Columns in summary file: "
        f"{len(summary.columns)}"
    )

    print("\nSummary columns:")

    for column in summary.columns:
        print(f" - {column}")

    # ---------------------------------------
    # Missing-value report
    # ---------------------------------------

    missing_report = pd.DataFrame(
        {
            "variable": summary.columns,
            "missing_count": (
                summary.isna().sum().values
            ),
            "missing_percent": (
                summary.isna().mean().values
                * 100
            ),
            "dtype": [
                str(dtype)
                for dtype
                in summary.dtypes
            ]
        }
    )

    missing_report[
        "missing_percent"
    ] = missing_report[
        "missing_percent"
    ].round(2)

    output_file = (
        REPORT_DIR
        / "shanghai_summary_missingness.csv"
    )

    missing_report.to_csv(
        output_file,
        index=False
    )

    print(
        "\nSaved participant-level "
        "missingness report:"
    )
    print(output_file)

    # ---------------------------------------
    # Participant/session IDs
    # ---------------------------------------

    if "Patient Number" in summary.columns:

        session_ids = (
            summary["Patient Number"]
            .astype(str)
        )

        # Example:
        # 2001_0_20201102
        # participant ID = 2001

        participant_ids = (
            session_ids
            .str.split("_")
            .str[0]
        )

        print(
            "\nNumber of summary records / sessions:",
            session_ids.nunique()
        )

        print(
            "Number of unique base participants:",
            participant_ids.nunique()
        )

    return summary


# ---------------------------------------------------------
# 4. AUDIT EACH CGM SESSION FILE
# ---------------------------------------------------------

def audit_session_files():

    print("\n" + "=" * 70)
    print("AUDITING SHANGHAI T2DM CGM FILES")
    print("=" * 70)

    files = sorted(
        list(SESSION_DIR.rglob("*.xlsx"))
        + list(SESSION_DIR.rglob("*.xls"))
    )

    print(
        f"\nExcel session files found: "
        f"{len(files)}"
    )

    session_records = []
    column_records = []
    missing_records = []

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

            df = pd.read_excel(file_path)

        except Exception as error:

            print(
                f"Could not read "
                f"{file_path.name}"
            )

            print(error)

            continue

        # -------------------------------------
        # Store original columns
        # -------------------------------------

        original_columns = list(
            df.columns
        )

        column_mapping = {
            column: canonical_name(column)
            for column in original_columns
        }

        for original, canonical in (
            column_mapping.items()
        ):

            column_records.append(
                {
                    "file_name":
                        file_path.name,

                    "original_column":
                        original,

                    "canonical_column":
                        canonical
                }
            )

        # Rename only IN MEMORY.
        # Raw files remain unchanged.

        df = df.rename(
            columns=column_mapping
        )

        session_id = file_path.stem

        participant_id = (
            session_id.split("_")[0]
        )

        # -------------------------------------
        # Check timestamp column
        # -------------------------------------

        if "timestamp" not in df.columns:

            print(
                "WARNING: No timestamp "
                f"column in {file_path.name}"
            )

            continue

        timestamps = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        valid_timestamp_count = (
            timestamps.notna().sum()
        )

        invalid_timestamp_count = (
            timestamps.isna().sum()
        )

        duplicate_timestamps = (
            timestamps[
                timestamps.notna()
            ]
            .duplicated()
            .sum()
        )

        # -------------------------------------
        # CGM values
        # -------------------------------------

        if "cgm_mg_dl" in df.columns:

            cgm = pd.to_numeric(
                df["cgm_mg_dl"],
                errors="coerce"
            )

        else:

            cgm = pd.Series(
                np.nan,
                index=df.index
            )

        valid_cgm_count = (
            cgm.notna().sum()
        )

        missing_cgm_count = (
            cgm.isna().sum()
        )

        if len(df) > 0:

            missing_cgm_percent = (
                missing_cgm_count
                / len(df)
                * 100
            )

        else:

            missing_cgm_percent = np.nan

        # -------------------------------------
        # Start / end timestamps
        # -------------------------------------

        valid_times = timestamps.dropna()

        if len(valid_times) > 0:

            start_time = (
                valid_times.min()
            )

            end_time = (
                valid_times.max()
            )

            duration_days = (
                (
                    end_time
                    - start_time
                ).total_seconds()
                / 86400
            )

        else:

            start_time = pd.NaT
            end_time = pd.NaT
            duration_days = np.nan

        # -------------------------------------
        # Sampling interval analysis
        # -------------------------------------

        cgm_times = (
            timestamps[
                timestamps.notna()
                & cgm.notna()
            ]
            .sort_values()
            .drop_duplicates()
        )

        intervals = (
            cgm_times
            .diff()
            .dt.total_seconds()
            .div(60)
            .dropna()
        )

        if len(intervals) > 0:

            median_interval = (
                intervals.median()
            )

            min_interval = (
                intervals.min()
            )

            max_interval = (
                intervals.max()
            )

            # approximately 15 minutes
            approx_15_percent = (
                intervals
                .between(14, 16)
                .mean()
                * 100
            )

            gap_over_30_count = (
                intervals > 30
            ).sum()

        else:

            median_interval = np.nan
            min_interval = np.nan
            max_interval = np.nan
            approx_15_percent = np.nan
            gap_over_30_count = 0

        # -------------------------------------
        # Meal entries
        # -------------------------------------

        if (
            "dietary_intake"
            in df.columns
        ):

            meal_entries = (
                df[
                    "dietary_intake"
                ]
                .notna()
                .sum()
            )

        else:

            meal_entries = 0

        # -------------------------------------
        # Missingness for every column
        # -------------------------------------

        for column in df.columns:

            missing_count = (
                df[column]
                .isna()
                .sum()
            )

            if len(df) > 0:

                missing_percent = (
                    missing_count
                    / len(df)
                    * 100
                )

            else:

                missing_percent = np.nan

            missing_records.append(
                {
                    "session_id":
                        session_id,

                    "participant_id":
                        participant_id,

                    "variable":
                        column,

                    "total_rows":
                        len(df),

                    "missing_count":
                        missing_count,

                    "missing_percent":
                        round(
                            missing_percent,
                            2
                        )
                }
            )

        # -------------------------------------
        # Save one row for this session
        # -------------------------------------

        session_records.append(
            {
                "session_id":
                    session_id,

                "participant_id":
                    participant_id,

                "rows":
                    len(df),

                "valid_timestamps":
                    valid_timestamp_count,

                "invalid_timestamps":
                    invalid_timestamp_count,

                "duplicate_timestamps":
                    duplicate_timestamps,

                "valid_cgm":
                    valid_cgm_count,

                "missing_cgm":
                    missing_cgm_count,

                "missing_cgm_percent":
                    round(
                        missing_cgm_percent,
                        2
                    ),

                "start_time":
                    start_time,

                "end_time":
                    end_time,

                "duration_days":
                    round(
                        duration_days,
                        2
                    )
                    if pd.notna(
                        duration_days
                    )
                    else np.nan,

                "median_interval_min":
                    round(
                        median_interval,
                        2
                    )
                    if pd.notna(
                        median_interval
                    )
                    else np.nan,

                "min_interval_min":
                    round(
                        min_interval,
                        2
                    )
                    if pd.notna(
                        min_interval
                    )
                    else np.nan,

                "max_interval_min":
                    round(
                        max_interval,
                        2
                    )
                    if pd.notna(
                        max_interval
                    )
                    else np.nan,

                "approx_15_min_percent":
                    round(
                        approx_15_percent,
                        2
                    )
                    if pd.notna(
                        approx_15_percent
                    )
                    else np.nan,

                "gaps_over_30_min":
                    int(
                        gap_over_30_count
                    ),

                "meal_entries":
                    int(
                        meal_entries
                    )
            }
        )

    # -----------------------------------------
    # Convert reports into DataFrames
    # -----------------------------------------

    session_report = pd.DataFrame(
        session_records
    )

    column_report = pd.DataFrame(
        column_records
    )

    missing_report = pd.DataFrame(
        missing_records
    )

    # -----------------------------------------
    # Save reports
    # -----------------------------------------

    session_report.to_csv(
        REPORT_DIR
        / "shanghai_session_audit.csv",
        index=False
    )

    column_report.to_csv(
        REPORT_DIR
        / "shanghai_column_inventory.csv",
        index=False
    )

    missing_report.to_csv(
        REPORT_DIR
        / "shanghai_timeseries_missingness.csv",
        index=False
    )

    return (
        session_report,
        column_report,
        missing_report
    )


# ---------------------------------------------------------
# 5. PRINT OVERALL SUMMARY
# ---------------------------------------------------------

def print_overall_summary(
    session_report
):

    print("\n" + "=" * 70)
    print("SHANGHAI T2DM OVERALL AUDIT SUMMARY")
    print("=" * 70)

    if session_report.empty:

        print(
            "No session records were "
            "successfully audited."
        )

        return

    print(
        "\nSessions:",
        session_report[
            "session_id"
        ].nunique()
    )

    print(
        "Unique participants:",
        session_report[
            "participant_id"
        ].nunique()
    )

    print(
        "Total rows:",
        session_report[
            "rows"
        ].sum()
    )

    print(
        "Total valid CGM readings:",
        session_report[
            "valid_cgm"
        ].sum()
    )

    print(
        "Total duplicate timestamps:",
        session_report[
            "duplicate_timestamps"
        ].sum()
    )

    print(
        "Median session duration "
        "(days):",
        round(
            session_report[
                "duration_days"
            ].median(),
            2
        )
    )

    print(
        "Median CGM sampling "
        "interval (minutes):",
        round(
            session_report[
                "median_interval_min"
            ].median(),
            2
        )
    )

    print(
        "Total meal entries:",
        session_report[
            "meal_entries"
        ].sum()
    )

    print(
        "\nReports saved inside:"
    )

    print(REPORT_DIR)


# ---------------------------------------------------------
# 6. MAIN PROGRAM
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("SMART BLOOD GLUCOSE ML PROJECT")
    print("STEP 2 - SHANGHAI T2DM DATA AUDIT")
    print("=" * 70)

    audit_summary_file()

    (
        session_report,
        column_report,
        missing_report
    ) = audit_session_files()

    print_overall_summary(
        session_report
    )

    print("\n" + "=" * 70)
    print("STEP 2 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()