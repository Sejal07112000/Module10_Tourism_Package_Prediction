import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def register_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop the stray index column that sometimes gets saved with the csv
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    print("Dataset registered successfully.")
    print(f"Path            : {path}")
    print(f"Rows, Columns   : {df.shape}")
    print(f"Columns         : {list(df.columns)}")
    print(f"Missing values  : {int(df.isnull().sum().sum())}")
    print("Target balance  :")
    print(df["ProdTaken"].value_counts(normalize=True).round(3).to_string())

    return df


if __name__ == "__main__":
    register_dataset()
