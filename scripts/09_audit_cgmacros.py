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
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "data_audit"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 2. OUTPUT FILES
# =========================================================

FILE_INVENTORY_FILE = (
    REPORT_DIR
    / "cgmacros_file_inventory.csv"
)

TABLE_INVENTORY_FILE = (
    REPORT_DIR
    / "cgmacros_table_inventory.csv"
)

COLUMN_INVENTORY_FILE = (
    REPORT_DIR
    / "cgmacros_column_inventory.csv"
)

MISSINGNESS_FILE = (
    REPORT_DIR
    / "cgmacros_missingness.csv"
)

CANDIDATE_COLUMNS_FILE = (
    REPORT_DIR
    / "cgmacros_candidate_columns.csv"
)

TIMESTAMP_AUDIT_FILE = (
    REPORT_DIR
    / "cgmacros_timestamp_audit.csv"
)


# =========================================================
# 3. HELPERS
# =========================================================

def clean_text(value):

    return re.sub(
        r"\s+",
        " ",
        str(value).strip()
    )


def safe_sample_values(series):

    values = (
        series.dropna()
        .astype(str)
        .drop_duplicates()
        .head(3)
        .tolist()
    )

    result = " | ".join(
        clean_text(value)[:80]
        for value in values
    )

    return result


def classify_column(column):

    name = clean_text(
        column
    ).lower()

    categories = []

    # Participant / subject identifiers
    if any(
        keyword in name
        for keyword in [
            "participant",
            "subject",
            "patient",
            "user",
            "person",
            "id"
        ]
    ):
        categories.append(
            "participant_id"
        )

    # Time
    if any(
        keyword in name
        for keyword in [
            "timestamp",
            "datetime",
            "date",
            "time"
        ]
    ):
        categories.append(
            "timestamp"
        )

    # Glucose
    if any(
        keyword in name
        for keyword in [
            "glucose",
            "cgm",
            "blood_glucose",
            "bg"
        ]
    ):
        categories.append(
            "glucose"
        )

    # Meal / food
    if any(
        keyword in name
        for keyword in [
            "meal",
            "food",
            "diet"
        ]
    ):
        categories.append(
            "meal"
        )

    # Carbohydrates
    if any(
        keyword in name
        for keyword in [
            "carb",
            "carbohydrate"
        ]
    ):
        categories.append(
            "carbohydrate"
        )

    # Protein
    if "protein" in name:
        categories.append(
            "protein"
        )

    # Fat
    if any(
        keyword in name
        for keyword in [
            "fat",
            "lipid"
        ]
    ):
        categories.append(
            "fat"
        )

    # Fibre
    if any(
        keyword in name
        for keyword in [
            "fiber",
            "fibre"
        ]
    ):
        categories.append(
            "fibre"
        )

    # Energy / calories
    if any(
        keyword in name
        for keyword in [
            "calorie",
            "kcal",
            "energy"
        ]
    ):
        categories.append(
            "energy"
        )

    # Heart rate
    if any(
        keyword in name
        for keyword in [
            "heart",
            "heart_rate",
            "heartrate",
            "hr"
        ]
    ):
        categories.append(
            "heart_rate"
        )

    # Steps / activity
    if any(
        keyword in name
        for keyword in [
            "step",
            "activity",
            "exercise",
            "active",
            "movement",
            "met"
        ]
    ):
        categories.append(
            "activity"
        )

    # Clinical / demographic
    if "age" in name:
        categories.append(
            "age"
        )

    if any(
        keyword in name
        for keyword in [
            "sex",
            "gender"
        ]
    ):
        categories.append(
            "sex"
        )

    if "bmi" in name:
        categories.append(
            "bmi"
        )

    if "hba1c" in name:
        categories.append(
            "hba1c"
        )

    return categories


# =========================================================
# 4. RAW FILE INVENTORY
# =========================================================

def build_file_inventory():

    print("\n" + "=" * 70)
    print("CGMACROS RAW FILE INVENTORY")
    print("=" * 70)

    records = []

    for path in CGMACROS_DIR.rglob("*"):

        if not path.is_file():
            continue

        records.append(
            {
                "file_name":
                    path.name,

                "extension":
                    path.suffix.lower(),

                "size_mb":
                    round(
                        path.stat().st_size
                        / (1024 * 1024),
                        4
                    ),

                "relative_path":
                    str(
                        path.relative_to(
                            PROJECT_ROOT
                        )
                    )
            }
        )

    inventory = pd.DataFrame(
        records
    )

    inventory.to_csv(
        FILE_INVENTORY_FILE,
        index=False
    )

    print(
        "\nTotal files:",
        len(inventory)
    )

    if not inventory.empty:

        print(
            "\nFiles by extension:"
        )

        print(
            inventory[
                "extension"
            ]
            .value_counts()
        )

    return inventory


# =========================================================
# 5. READ TABULAR FILE
# =========================================================

def read_tables(path):

    suffix = path.suffix.lower()

    tables = []

    try:

        if suffix == ".csv":

            df = pd.read_csv(
                path,
                low_memory=False
            )

            tables.append(
                (
                    "default",
                    df
                )
            )

        elif suffix == ".tsv":

            df = pd.read_csv(
                path,
                sep="\t",
                low_memory=False
            )

            tables.append(
                (
                    "default",
                    df
                )
            )

        elif suffix in {
            ".xlsx",
            ".xls"
        }:

            workbook = pd.ExcelFile(
                path
            )

            for sheet in workbook.sheet_names:

                df = pd.read_excel(
                    path,
                    sheet_name=sheet
                )

                tables.append(
                    (
                        sheet,
                        df
                    )
                )

        elif suffix == ".parquet":

            df = pd.read_parquet(
                path
            )

            tables.append(
                (
                    "default",
                    df
                )
            )

    except Exception as error:

        return [], str(error)

    return tables, None


# =========================================================
# 6. AUDIT TABULAR DATA
# =========================================================

def audit_tables():

    print("\n" + "=" * 70)
    print("AUDITING CGMACROS TABLES")
    print("=" * 70)

    supported = {
        ".csv",
        ".tsv",
        ".xlsx",
        ".xls",
        ".parquet"
    }

    table_records = []
    column_records = []
    missing_records = []
    candidate_records = []
    timestamp_records = []

    files = sorted(
        [
            path
            for path in CGMACROS_DIR.rglob("*")
            if (
                path.is_file()
                and
                path.suffix.lower()
                in supported
            )
        ]
    )

    print(
        "\nPotential tabular files:",
        len(files)
    )

    for number, path in enumerate(
        files,
        start=1
    ):

        print(
            f"Processing "
            f"{number}/{len(files)}: "
            f"{path.name}"
        )

        tables, error = (
            read_tables(
                path
            )
        )

        if error is not None:

            table_records.append(
                {
                    "file_name":
                        path.name,

                    "sheet":
                        None,

                    "rows":
                        None,

                    "columns":
                        None,

                    "status":
                        "ERROR",

                    "error":
                        error,

                    "relative_path":
                        str(
                            path.relative_to(
                                PROJECT_ROOT
                            )
                        )
                }
            )

            continue

        for sheet_name, df in tables:

            table_records.append(
                {
                    "file_name":
                        path.name,

                    "sheet":
                        sheet_name,

                    "rows":
                        len(df),

                    "columns":
                        len(df.columns),

                    "status":
                        "OK",

                    "error":
                        None,

                    "relative_path":
                        str(
                            path.relative_to(
                                PROJECT_ROOT
                            )
                        )
                }
            )

            # ---------------------------------
            # Every column
            # ---------------------------------

            for column in df.columns:

                categories = (
                    classify_column(
                        column
                    )
                )

                non_null = (
                    df[column]
                    .notna()
                    .sum()
                )

                missing = (
                    df[column]
                    .isna()
                    .sum()
                )

                unique_values = (
                    df[column]
                    .nunique(
                        dropna=True
                    )
                )

                column_records.append(
                    {
                        "file_name":
                            path.name,

                        "sheet":
                            sheet_name,

                        "column":
                            column,

                        "dtype":
                            str(
                                df[column].dtype
                            ),

                        "non_null":
                            int(
                                non_null
                            ),

                        "unique_values":
                            int(
                                unique_values
                            ),

                        "sample_values":
                            safe_sample_values(
                                df[column]
                            )
                    }
                )

                missing_records.append(
                    {
                        "file_name":
                            path.name,

                        "sheet":
                            sheet_name,

                        "column":
                            column,

                        "rows":
                            len(df),

                        "missing_count":
                            int(
                                missing
                            ),

                        "missing_percent":
                            round(
                                (
                                    missing
                                    / len(df)
                                    * 100
                                )
                                if len(df) > 0
                                else np.nan,
                                2
                            )
                    }
                )

                # ---------------------------------
                # Candidate feature columns
                # ---------------------------------

                for category in categories:

                    candidate_records.append(
                        {
                            "category":
                                category,

                            "file_name":
                                path.name,

                            "sheet":
                                sheet_name,

                            "column":
                                column,

                            "dtype":
                                str(
                                    df[column].dtype
                                ),

                            "non_null":
                                int(
                                    non_null
                                ),

                            "unique_values":
                                int(
                                    unique_values
                                )
                        }
                    )

                # ---------------------------------
                # Timestamp quality
                # ---------------------------------

                if "timestamp" in categories:

                    parsed = pd.to_datetime(
                        df[column],
                        errors="coerce"
                    )

                    valid_count = (
                        parsed.notna().sum()
                    )

                    timestamp_records.append(
                        {
                            "file_name":
                                path.name,

                            "sheet":
                                sheet_name,

                            "column":
                                column,

                            "rows":
                                len(df),

                            "valid_datetime":
                                int(
                                    valid_count
                                ),

                            "invalid_datetime":
                                int(
                                    parsed.isna().sum()
                                ),

                            "valid_percent":
                                round(
                                    (
                                        valid_count
                                        / len(df)
                                        * 100
                                    )
                                    if len(df) > 0
                                    else np.nan,
                                    2
                                ),

                            "minimum":
                                (
                                    parsed.min()
                                    if valid_count > 0
                                    else pd.NaT
                                ),

                            "maximum":
                                (
                                    parsed.max()
                                    if valid_count > 0
                                    else pd.NaT
                                )
                        }
                    )

    table_inventory = pd.DataFrame(
        table_records
    )

    column_inventory = pd.DataFrame(
        column_records
    )

    missingness = pd.DataFrame(
        missing_records
    )

    candidates = pd.DataFrame(
        candidate_records
    )

    timestamp_audit = pd.DataFrame(
        timestamp_records
    )

    table_inventory.to_csv(
        TABLE_INVENTORY_FILE,
        index=False
    )

    column_inventory.to_csv(
        COLUMN_INVENTORY_FILE,
        index=False
    )

    missingness.to_csv(
        MISSINGNESS_FILE,
        index=False
    )

    candidates.to_csv(
        CANDIDATE_COLUMNS_FILE,
        index=False
    )

    timestamp_audit.to_csv(
        TIMESTAMP_AUDIT_FILE,
        index=False
    )

    return (
        table_inventory,
        column_inventory,
        missingness,
        candidates,
        timestamp_audit
    )


# =========================================================
# 7. PRINT CANDIDATE COLUMN SUMMARY
# =========================================================

def print_candidate_summary(
    candidates
):

    print("\n" + "=" * 70)
    print("CGMACROS CANDIDATE FEATURE COLUMNS")
    print("=" * 70)

    if candidates.empty:

        print(
            "\nNo candidate columns "
            "were automatically detected."
        )

        return

    counts = (
        candidates[
            "category"
        ]
        .value_counts()
    )

    print(
        "\nCandidate columns by category:"
    )

    print(
        counts
    )

    important_categories = [
        "participant_id",
        "timestamp",
        "glucose",
        "meal",
        "carbohydrate",
        "protein",
        "fat",
        "fibre",
        "energy",
        "heart_rate",
        "activity",
        "age",
        "sex",
        "bmi",
        "hba1c"
    ]

    for category in important_categories:

        subset = candidates[
            candidates[
                "category"
            ] == category
        ]

        if subset.empty:
            continue

        print(
            "\n"
            + "-" * 60
        )

        print(
            category.upper()
        )

        print(
            subset[
                [
                    "file_name",
                    "sheet",
                    "column"
                ]
            ]
            .drop_duplicates()
            .head(20)
            .to_string(
                index=False
            )
        )


# =========================================================
# 8. FINAL SUMMARY
# =========================================================

def print_final_summary(
    file_inventory,
    table_inventory,
    candidates,
    timestamp_audit
):

    print("\n" + "=" * 70)
    print("STEP 9 CGMACROS AUDIT FINAL SUMMARY")
    print("=" * 70)

    successful_tables = (
        table_inventory[
            table_inventory[
                "status"
            ] == "OK"
        ]
        if not table_inventory.empty
        else pd.DataFrame()
    )

    failed_tables = (
        table_inventory[
            table_inventory[
                "status"
            ] == "ERROR"
        ]
        if not table_inventory.empty
        else pd.DataFrame()
    )

    print(
        "\nRaw files:",
        len(
            file_inventory
        )
    )

    print(
        "Successfully audited tables/sheets:",
        len(
            successful_tables
        )
    )

    print(
        "Failed table reads:",
        len(
            failed_tables
        )
    )

    if not successful_tables.empty:

        print(
            "Rows across audited tables:",
            int(
                successful_tables[
                    "rows"
                ]
                .fillna(0)
                .sum()
            )
        )

    print(
        "Candidate feature references:",
        len(
            candidates
        )
    )

    print(
        "Timestamp columns audited:",
        len(
            timestamp_audit
        )
    )

    print(
        "\nReports saved inside:"
    )

    print(
        REPORT_DIR
    )


# =========================================================
# 9. MAIN
# =========================================================

def main():

    print("=" * 70)
    print("SMART BLOOD GLUCOSE ML PROJECT")
    print("STEP 9 - CGMACROS DATA AUDIT")
    print("=" * 70)

    if not CGMACROS_DIR.exists():

        raise FileNotFoundError(
            f"CGMacros folder not found:\n"
            f"{CGMACROS_DIR}"
        )

    file_inventory = (
        build_file_inventory()
    )

    (
        table_inventory,
        column_inventory,
        missingness,
        candidates,
        timestamp_audit
    ) = audit_tables()

    print_candidate_summary(
        candidates
    )

    print_final_summary(
        file_inventory,
        table_inventory,
        candidates,
        timestamp_audit
    )

    print("\n" + "=" * 70)
    print("STEP 9 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()