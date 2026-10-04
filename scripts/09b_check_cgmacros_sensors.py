from pathlib import Path
import pandas as pd

ROOT = Path(
    r"data\raw\cgmacros\CGMacros"
)

files = sorted(
    ROOT.glob("CGMacros-*/*.csv")
)

records = []

for file_path in files:

    df = pd.read_csv(
        file_path,
        low_memory=False
    )

    # Remove accidental spaces such as
    # "Amount Consumed "
    df.columns = df.columns.str.strip()

    participant_id = file_path.parent.name

    libre_count = (
        df["Libre GL"].notna().sum()
        if "Libre GL" in df.columns
        else 0
    )

    dexcom_count = (
        df["Dexcom GL"].notna().sum()
        if "Dexcom GL" in df.columns
        else 0
    )

    records.append(
        {
            "participant":
                participant_id,

            "rows":
                len(df),

            "libre_nonnull":
                libre_count,

            "dexcom_nonnull":
                dexcom_count
        }
    )

result = pd.DataFrame(records)

print(result.to_string(index=False))

print("\n" + "=" * 50)

print(
    "Participant files:",
    len(result)
)

print(
    "Participants with Libre:",
    (result["libre_nonnull"] > 0).sum()
)

print(
    "Participants with Dexcom:",
    (result["dexcom_nonnull"] > 0).sum()
)

print(
    "Participants with both:",
    (
        (result["libre_nonnull"] > 0)
        &
        (result["dexcom_nonnull"] > 0)
    ).sum()
)

print(
    "Participants with neither:",
    (
        (result["libre_nonnull"] == 0)
        &
        (result["dexcom_nonnull"] == 0)
    ).sum()
)

print(
    "\nTotal Libre readings:",
    result["libre_nonnull"].sum()
)

print(
    "Total Dexcom readings:",
    result["dexcom_nonnull"].sum()
)