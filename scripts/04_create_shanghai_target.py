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

INPUT_FILE = (
    PROCESSED_DIR
    / "shanghai_cgm_clean.csv"
)

OUTPUT_ALL = (
    PROCESSED_DIR
    / "shanghai_with_target_all.csv"
)

OUTPUT_READY = (
    PROCESSED_DIR
    / "shanghai_target_ready.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "shanghai_target_summary.csv"
)

SESSION_REPORT_FILE = (
    REPORT_DIR
    / "shanghai_target_by_session.csv"
)


# =========================================================
# 2. LOAD CLEAN DATA
# =========================================================

def load_clean_data():

    print("\n" + "=" * 70)
    print("LOADING CLEAN SHANGHAI CGM DATA")
    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamp"]
    )

    print(
        f"\nRows loaded: {len(df)}"
    )

    print(
        "Participants:",
        df["participant_id"].nunique()
    )

    print(
        "Sessions:",
        df["session_id"].nunique()
    )

    print(
        "Missing timestamps:",
        df["timestamp"].isna().sum()
    )

    print(
        "Missing CGM:",
        df["cgm_mg_dl"].isna().sum()
    )

    return df


# =========================================================
# 3. CREATE 30-MINUTE TARGET
# =========================================================

def create_target(df):

    print("\n" + "=" * 70)
    print("CREATING 30-MINUTE TARGET")
    print("=" * 70)

    data = df.copy()

    # -----------------------------------------------------
    # Expected future timestamp
    # -----------------------------------------------------

    data[
        "target_timestamp_30min"
    ] = (
        data["timestamp"]
        + pd.Timedelta(minutes=30)
    )

    # -----------------------------------------------------
    # Build a lookup table:
    #
    # session_id
    # target_timestamp_30min
    # target_glucose_30min
    #
    # -----------------------------------------------------

    target_lookup = (
        data[
            [
                "session_id",
                "timestamp",
                "cgm_mg_dl"
            ]
        ]
        .rename(
            columns={
                "timestamp":
                    "target_timestamp_30min",

                "cgm_mg_dl":
                    "target_glucose_30min"
            }
        )
    )

    # -----------------------------------------------------
    # Match future glucose using BOTH:
    #
    # session_id
    # target timestamp
    #
    # This prevents data from another session/person
    # from being used accidentally.
    # -----------------------------------------------------

    data = data.merge(
        target_lookup,
        on=[
            "session_id",
            "target_timestamp_30min"
        ],
        how="left",
        validate="many_to_one"
    )

    # -----------------------------------------------------
    # Target availability flag
    # -----------------------------------------------------

    data[
        "target_available_30min"
    ] = (
        data[
            "target_glucose_30min"
        ]
        .notna()
        .astype(int)
    )

    available = (
        data[
            "target_available_30min"
        ].sum()
    )

    missing = (
        len(data)
        - available
    )

    availability_percent = (
        available
        / len(data)
        * 100
    )

    print(
        f"\nRows checked: {len(data)}"
    )

    print(
        "Valid 30-minute targets:",
        available
    )

    print(
        "Rows without valid target:",
        missing
    )

    print(
        "Target availability:",
        f"{availability_percent:.2f}%"
    )

    return data


# =========================================================
# 4. CREATE ML-TARGET-READY TABLE
# =========================================================

def create_target_ready_table(data):

    print("\n" + "=" * 70)
    print("CREATING TARGET-READY TABLE")
    print("=" * 70)

    ready = data[
        data[
            "target_glucose_30min"
        ].notna()
    ].copy()

    ready = ready.reset_index(
        drop=True
    )

    print(
        f"\nRows with usable targets: "
        f"{len(ready)}"
    )

    print(
        "Missing targets remaining:",
        ready[
            "target_glucose_30min"
        ].isna().sum()
    )

    return ready


# =========================================================
# 5. CREATE SESSION-LEVEL TARGET REPORT
# =========================================================

def create_session_report(data):

    session_report = (
        data.groupby(
            [
                "participant_id",
                "session_id"
            ],
            as_index=False
        )
        .agg(
            total_rows=(
                "timestamp",
                "size"
            ),

            valid_targets=(
                "target_available_30min",
                "sum"
            )
        )
    )

    session_report[
        "missing_targets"
    ] = (
        session_report[
            "total_rows"
        ]
        -
        session_report[
            "valid_targets"
        ]
    )

    session_report[
        "target_availability_percent"
    ] = (
        session_report[
            "valid_targets"
        ]
        /
        session_report[
            "total_rows"
        ]
        * 100
    ).round(2)

    return session_report


# =========================================================
# 6. CREATE OVERALL SUMMARY
# =========================================================

def create_summary(
    data,
    ready
):

    total_rows = len(data)

    valid_targets = len(ready)

    missing_targets = (
        total_rows
        - valid_targets
    )

    availability_percent = (
        valid_targets
        / total_rows
        * 100
    )

    summary = pd.DataFrame(
        [
            {
                "metric":
                    "input_rows",

                "value":
                    total_rows
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
                    "valid_30min_targets",

                "value":
                    valid_targets
            },

            {
                "metric":
                    "missing_30min_targets",

                "value":
                    missing_targets
            },

            {
                "metric":
                    "target_availability_percent",

                "value":
                    round(
                        availability_percent,
                        2
                    )
            }
        ]
    )

    return summary


# =========================================================
# 7. VALIDATE TARGETS
# =========================================================

def validate_targets(ready):

    print("\n" + "=" * 70)
    print("VALIDATING TARGET CREATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Verify target timestamp is exactly 30 minutes ahead
    # -----------------------------------------------------

    difference = (
        ready[
            "target_timestamp_30min"
        ]
        -
        ready[
            "timestamp"
        ]
    )

    difference_minutes = (
        difference
        .dt.total_seconds()
        / 60
    )

    incorrect_target_times = (
        difference_minutes
        .ne(30)
        .sum()
    )

    print(
        "\nTargets not exactly "
        "30 minutes ahead:",
        incorrect_target_times
    )

    # -----------------------------------------------------
    # Check target missingness
    # -----------------------------------------------------

    print(
        "Missing target glucose:",
        ready[
            "target_glucose_30min"
        ].isna().sum()
    )

    # -----------------------------------------------------
    # Participant/session identity
    # -----------------------------------------------------

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

    if incorrect_target_times != 0:

        raise ValueError(
            "Target timing validation failed."
        )

    print(
        "\nTarget validation passed."
    )


# =========================================================
# 8. SAVE OUTPUTS
# =========================================================

def save_outputs(
    data,
    ready,
    summary,
    session_report
):

    print("\n" + "=" * 70)
    print("SAVING STEP 4 OUTPUTS")
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

    session_report.to_csv(
        SESSION_REPORT_FILE,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT_ALL)
    print(OUTPUT_READY)
    print(SUMMARY_FILE)
    print(SESSION_REPORT_FILE)


# =========================================================
# 9. PRINT FINAL SUMMARY
# =========================================================

def print_final_summary(
    data,
    ready
):

    print("\n" + "=" * 70)
    print("STEP 4 FINAL SUMMARY")
    print("=" * 70)

    print(
        "\nInput CGM rows:",
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
        "Valid 30-minute targets:",
        len(ready)
    )

    print(
        "Rows without targets:",
        len(data) - len(ready)
    )

    print(
        "Target availability:",
        f"{len(ready) / len(data) * 100:.2f}%"
    )

    print(
        "Missing targets in ready file:",
        ready[
            "target_glucose_30min"
        ].isna().sum()
    )


# =========================================================
# 10. MAIN PROGRAM
# =========================================================

def main():

    print("=" * 70)
    print("SMART BLOOD GLUCOSE ML PROJECT")
    print("STEP 4 - CREATE 30-MINUTE TARGET")
    print("=" * 70)

    df = load_clean_data()

    data = create_target(
        df
    )

    ready = (
        create_target_ready_table(
            data
        )
    )

    validate_targets(
        ready
    )

    session_report = (
        create_session_report(
            data
        )
    )

    summary = create_summary(
        data,
        ready
    )

    save_outputs(
        data,
        ready,
        summary,
        session_report
    )

    print_final_summary(
        data,
        ready
    )

    print("\n" + "=" * 70)
    print("STEP 4 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()