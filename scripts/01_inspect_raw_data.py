from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports" / "data_audit"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def build_file_inventory():
    print("\n" + "=" * 70)
    print("RAW DATA FILE INVENTORY")
    print("=" * 70)

    records = []

    for file_path in RAW_DIR.rglob("*"):
        if file_path.is_file():
            records.append(
                {
                    "dataset": (
                        file_path.relative_to(RAW_DIR).parts[0]
                        if file_path.relative_to(RAW_DIR).parts
                        else "unknown"
                    ),
                    "file_name": file_path.name,
                    "extension": file_path.suffix.lower(),
                    "size_mb": round(
                        file_path.stat().st_size / (1024 * 1024), 3
                    ),
                    "relative_path": str(file_path.relative_to(PROJECT_ROOT)),
                }
            )

    inventory = pd.DataFrame(records)

    if inventory.empty:
        print("No files found inside data/raw/")
        return inventory

    print(f"\nTotal raw files: {len(inventory)}")

    print("\nFiles by dataset:")
    print(inventory["dataset"].value_counts())

    print("\nFiles by extension:")
    print(inventory["extension"].value_counts())

    output_path = REPORT_DIR / "file_inventory.csv"
    inventory.to_csv(output_path, index=False)

    print(f"\nSaved inventory to:")
    print(output_path)

    return inventory


def inspect_excel_file(file_path):
    print("\n" + "-" * 70)
    print(f"FILE: {file_path.name}")
    print(f"PATH: {file_path}")
    print("-" * 70)

    try:
        excel_file = pd.ExcelFile(file_path)

        print("Sheets:")
        print(excel_file.sheet_names)

        for sheet in excel_file.sheet_names[:3]:
            print(f"\nSheet: {sheet}")

            df = pd.read_excel(
                file_path,
                sheet_name=sheet,
                nrows=5
            )

            print("Columns:")
            for column in df.columns:
                print(f"  - {column}")

            print("\nFirst 5 rows:")
            print(df.head())

            print("\nDetected data types:")
            print(df.dtypes)

    except Exception as error:
        print(f"Could not inspect {file_path.name}")
        print(error)


def inspect_csv_file(file_path):
    print("\n" + "-" * 70)
    print(f"FILE: {file_path.name}")
    print(f"PATH: {file_path}")
    print("-" * 70)

    try:
        df = pd.read_csv(file_path, nrows=5)

        print("Columns:")
        for column in df.columns:
            print(f"  - {column}")

        print("\nFirst 5 rows:")
        print(df.head())

        print("\nDetected data types:")
        print(df.dtypes)

    except Exception as error:
        print(f"Could not inspect {file_path.name}")
        print(error)


def inspect_sample_files():
    print("\n" + "=" * 70)
    print("SAMPLE TABLE INSPECTION")
    print("=" * 70)

    excel_files = list(RAW_DIR.rglob("*.xlsx"))
    csv_files = list(RAW_DIR.rglob("*.csv"))

    print(f"\nExcel files found: {len(excel_files)}")
    print(f"CSV files found: {len(csv_files)}")

    # Inspect only a few files in Step 1.
    # We do NOT want to process the whole dataset yet.

    for file_path in excel_files[:5]:
        inspect_excel_file(file_path)

    for file_path in csv_files[:5]:
        inspect_csv_file(file_path)


def main():
    print("=" * 70)
    print("SMART BLOOD GLUCOSE ML PROJECT")
    print("STEP 1 - RAW DATA INSPECTION")
    print("=" * 70)

    build_file_inventory()
    inspect_sample_files()

    print("\n" + "=" * 70)
    print("STEP 1 INSPECTION FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()