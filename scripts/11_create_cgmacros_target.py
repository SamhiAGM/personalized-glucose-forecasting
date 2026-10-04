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

INPUT_FILE = (
    PROCESSED_DIR
    / "cgmacros_glucose_clean.csv"
)

OUTPUT_ALL_FILE = (
    PROCESSED_DIR
    / "cgmacros_with_target.csv"
)

OUTPUT_READY_FILE = (
    PROCESSED_DIR
    / "cgmacros_target_ready.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "cgmacros_target_summary.csv"
)

PARTICIPANT_REPORT_FILE = (
    REPORT_DIR
    / "cgmacros_target_by_participant.csv"
)


# =========================================================
# 2. SETTINGS
# =========================================================

TARGET_HORIZON_MINUTES = 30


# =========================================================
# 3. LOAD INPUT
# =========================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING CGMACROS CLEAN GLUCOSE DATA")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamp"],
        low_memory=False
    )

    df["participant_id"] = (
        df["participant_id"]
        .astype(str)
    )

    df = df.sort_values(
        [
            "participant_id",
            "timestamp"
        ],
        kind="stable"
    ).reset_index(drop=True)

    print(
        "\nInput rows:",
        len(df)
    )

    print(
        "Participants:",
        df["participant_id"].nunique()
    )

    print(
        "Missing glucose:",
        df["glucose_mg_dl"].isna().sum()
    )

    print(
        "Missing timestamps:",
        df["timestamp"].isna().sum()
    )

    return df


# =========================================================
# 4. VALIDATE INPUT UNIQUENESS
# =========================================================

def validate_input(df):

    print("\n" + "=" * 70)
    print("VALIDATING TARGET INPUT")
    print("=" * 70)

    duplicate_count = (
        df.duplicated(
            subset=[
                "participant_id",
                "timestamp"
            ]
        ).sum()
    )

    print(
        "\nDuplicate participant/timestamps:",
        duplicate_count
    )

    if duplicate_count != 0:

        raise ValueError(
            "Duplicate participant/timestamp rows detected. "
            "Do not create targets until they are resolved."
        )

    if df["timestamp"].isna().any():

        raise ValueError(
            "Invalid timestamps detected."
        )

    if df["glucose_mg_dl"].isna().any():

        raise ValueError(
            "Missing primary glucose detected."
        )

    print(
        "Input validation passed."
    )


# =========================================================
# 5. CREATE EXACT 30-MINUTE TARGET
# =========================================================

def create_target(df):

    print("\n" + "=" * 70)
    print("CREATING EXACT 30-MINUTE TARGET")
    print("=" * 70)

    data = df.copy()

    # This is the exact future time we want.
    data[
        "target_timestamp_30min"
    ] = (
        data["timestamp"]
        +
        pd.Timedelta(
            minutes=TARGET_HORIZON_MINUTES
        )
    )

    # -----------------------------------------------------
    # Create lookup table:
    #
    # participant + timestamp -> glucose
    # -----------------------------------------------------

    lookup = df[
        [
            "participant_id",
            "timestamp",
            "glucose_mg_dl"
        ]
    ].copy()

    lookup = lookup.rename(
        columns={
            "timestamp":
                "target_timestamp_30min",

            "glucose_mg_dl":
                "target_glucose_30min"
        }
    )

    # -----------------------------------------------------
    # Exact timestamp merge
    #
    # Example:
    # t = 10:00
    # requested target timestamp = 10:30
    #
    # Only glucose recorded exactly at 10:30 is accepted.
    # -----------------------------------------------------

    data = data.merge(
        lookup,
        on=[
            "participant_id",
            "target_timestamp_30min"
        ],
        how="left",
        validate="many_to_one"
    )

    data[
        "target_available"
    ] = (
        data[
            "target_glucose_30min"
        ]
        .notna()
        .astype(int)
    )

    return data


# =========================================================
# 6. VALIDATE TARGET
# =========================================================

def validate_target(data):

    print("\n" + "=" * 70)
    print("VALIDATING 30-MINUTE TARGET")
    print("=" * 70)

    available = (
        data[
            "target_glucose_30min"
        ].notna()
    )

    valid_targets = (
        available.sum()
    )

    missing_targets = (
        (~available).sum()
    )

    target_availability = (
        valid_targets
        / len(data)
        * 100
    )

    print(
        "\nValid targets:",
        valid_targets
    )

    print(
        "Missing targets:",
        missing_targets
    )

    print(
        "Target availability:",
        round(
            target_availability,
            2
        ),
        "%"
    )

    # -----------------------------------------------------
    # Confirm target time is exactly +30 minutes
    # -----------------------------------------------------

    time_difference = (
        data.loc[
            available,
            "target_timestamp_30min"
        ]
        -
        data.loc[
            available,
            "timestamp"
        ]
    )

    difference_minutes = (
        time_difference
        .dt.total_seconds()
        / 60.0
    )

    incorrect_horizon = (
        ~np.isclose(
            difference_minutes,
            TARGET_HORIZON_MINUTES
        )
    ).sum()

    print(
        "Incorrect target horizons:",
        incorrect_horizon
    )

    if incorrect_horizon != 0:

        raise ValueError(
            "Some targets are not exactly 30 minutes ahead."
        )

    print(
        "\nTarget validation passed."
    )


# =========================================================
# 7. CREATE TARGET-READY TABLE
# =========================================================

def create_target_ready(data):

    print("\n" + "=" * 70)
    print("CREATING TARGET-READY DATASET")
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
        "\nRows before target filtering:",
        len(data)
    )

    print(
        "Rows ready for ML:",
        len(ready)
    )

    print(
        "Rows removed due to unavailable "
        "30-minute target:",
        len(data) - len(ready)
    )

    print(
        "Missing targets in ready table:",
        ready[
            "target_glucose_30min"
        ]
        .isna()
        .sum()
    )

    return ready


# =========================================================
# 8. PARTICIPANT REPORT
# =========================================================

def create_participant_report(data):

    records = []

    for participant_id, group in (
        data.groupby(
            "participant_id",
            sort=True
        )
    ):

        total_rows = len(
            group
        )

        valid_targets = (
            group[
                "target_glucose_30min"
            ]
            .notna()
            .sum()
        )

        missing_targets = (
            total_rows
            - valid_targets
        )

        records.append(
            {
                "participant_id":
                    participant_id,

                "total_rows":
                    total_rows,

                "valid_targets":
                    int(
                        valid_targets
                    ),

                "missing_targets":
                    int(
                        missing_targets
                    ),

                "target_availability_percent":
                    round(
                        (
                            valid_targets
                            / total_rows
                            * 100
                        )
                        if total_rows > 0
                        else np.nan,
                        2
                    )
            }
        )

    return pd.DataFrame(
        records
    )


# =========================================================
# 9. SUMMARY REPORT
# =========================================================

def create_summary(
    data,
    ready
):

    valid_targets = (
        data[
            "target_glucose_30min"
        ]
        .notna()
        .sum()
    )

    missing_targets = (
        data[
            "target_glucose_30min"
        ]
        .isna()
        .sum()
    )

    availability = (
        valid_targets
        / len(data)
        * 100
    )

    summary = pd.DataFrame(
        [
            {
                "metric":
                    "input_glucose_rows",

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
                    "prediction_horizon_minutes",

                "value":
                    TARGET_HORIZON_MINUTES
            },

            {
                "metric":
                    "valid_30min_targets",

                "value":
                    int(
                        valid_targets
                    )
            },

            {
                "metric":
                    "missing_30min_targets",

                "value":
                    int(
                        missing_targets
                    )
            },

            {
                "metric":
                    "target_availability_percent",

                "value":
                    round(
                        availability,
                        2
                    )
            },

            {
                "metric":
                    "target_ready_rows",

                "value":
                    len(
                        ready
                    )
            },

            {
                "metric":
                    "missing_targets_in_ready_file",

                "value":
                    int(
                        ready[
                            "target_glucose_30min"
                        ]
                        .isna()
                        .sum()
                    )
            }
        ]
    )

    return summary


# =========================================================
# 10. SAVE OUTPUTS
# =========================================================

def save_outputs(
    data,
    ready,
    participant_report,
    summary
):

    print("\n" + "=" * 70)
    print("SAVING STEP 11 OUTPUTS")
    print("=" * 70)

    data.to_csv(
        OUTPUT_ALL_FILE,
        index=False
    )

    ready.to_csv(
        OUTPUT_READY_FILE,
        index=False
    )

    participant_report.to_csv(
        PARTICIPANT_REPORT_FILE,
        index=False
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    print("\nSaved:")

    print(
        OUTPUT_ALL_FILE
    )

    print(
        OUTPUT_READY_FILE
    )

    print(
        PARTICIPANT_REPORT_FILE
    )

    print(
        SUMMARY_FILE
    )


# =========================================================
# 11. FINAL SUMMARY
# =========================================================

def print_final_summary(
    data,
    ready
):

    valid_targets = (
        data[
            "target_glucose_30min"
        ]
        .notna()
        .sum()
    )

    missing_targets = (
        data[
            "target_glucose_30min"
        ]
        .isna()
        .sum()
    )

    availability = (
        valid_targets
        / len(data)
        * 100
    )

    print("\n" + "=" * 70)
    print("STEP 11 FINAL SUMMARY")
    print("=" * 70)

    print(
        "\nInput glucose rows:",
        len(data)
    )

    print(
        "Participants:",
        data[
            "participant_id"
        ].nunique()
    )

    print(
        "Prediction horizon:",
        TARGET_HORIZON_MINUTES,
        "minutes"
    )

    print(
        "Valid 30-minute targets:",
        valid_targets
    )

    print(
        "Rows without exact 30-minute target:",
        missing_targets
    )

    print(
        "Target availability:",
        round(
            availability,
            2
        ),
        "%"
    )

    print(
        "Target-ready rows:",
        len(
            ready
        )
    )

    print(
        "Missing targets in ready file:",
        ready[
            "target_glucose_30min"
        ]
        .isna()
        .sum()
    )

    print(
        "Incorrect target horizons: 0"
    )


# =========================================================
# 12. MAIN
# =========================================================

def main():

    print("=" * 70)
    print("SMART BLOOD GLUCOSE ML PROJECT")
    print("STEP 11 - CGMACROS 30-MINUTE TARGET")
    print("=" * 70)

    data = load_data()

    validate_input(
        data
    )

    data = create_target(
        data
    )

    validate_target(
        data
    )

    ready = create_target_ready(
        data
    )

    participant_report = (
        create_participant_report(
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
        participant_report,
        summary
    )

    print_final_summary(
        data,
        ready
    )

    print("\n" + "=" * 70)
    print("STEP 11 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":

    main()