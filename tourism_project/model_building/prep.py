
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

# Identifier column - carries no predictive signal
DROP_COLUMNS = ["CustomerID"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")].copy()
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    # Normalize inconsistent category labels found during EDA
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    df = df.drop_duplicates()
    return df


def prepare_data(data_path: str = DATA_PATH, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(data_path)
    df = clean_data(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("Data preparation complete.")
    print(f"Train shape: {Xtrain.shape}, Test shape: {Xtest.shape}")
    print("Train target balance:")
    print(ytrain.value_counts(normalize=True).round(3).to_string())

    return Xtrain, Xtest, ytrain, ytest


if __name__ == "__main__":
    prepare_data()
