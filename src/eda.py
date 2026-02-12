import pandas as pd
import os
import sys

PROCESSED_PATH = "data/processed/clean.csv"


def validate_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data file not found at: {path}")


def load_processed_data(path: str) -> pd.DataFrame:
    try:
        validate_file_exists(path)
        df = pd.read_csv(path)
        print(f"[INFO] Processed data loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load processed data: {e}")


def summarize_structure(df: pd.DataFrame) -> None:
    print("\n===== DATASET STRUCTURE =====")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isna().sum())


def analyze_target(df: pd.DataFrame) -> None:
    print("\n===== TARGET: fare_amount =====")
    print(df["fare_amount"].describe())

    high_fare = df[df["fare_amount"] > 200].shape[0]
    print(f"Rows with fare > 200: {high_fare}")


def analyze_trip_distance(df: pd.DataFrame) -> None:
    print("\n===== FEATURE: trip_distance =====")
    print(df["trip_distance"].describe())

    zero_distance = df[df["trip_distance"] == 0].shape[0]
    print(f"Trips with zero distance: {zero_distance}")

    extreme_distance = df[df["trip_distance"] > 100].shape[0]
    print(f"Trips with extreme distance (>100 miles): {extreme_distance}")


def analyze_passenger(df: pd.DataFrame) -> None:
    print("\n===== FEATURE: passenger_count =====")
    print(df["passenger_count"].value_counts().sort_index())

    invalid_passengers = df[
        (df["passenger_count"] < 1) | (df["passenger_count"] > 6)
    ].shape[0]
    print(f"Invalid passenger rows (should be 0 after cleaning): {invalid_passengers}")


def analyze_time(df: pd.DataFrame) -> None:
    print("\n===== FEATURE: tpep_pickup_datetime =====")

    try:
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    except Exception:
        print("[WARNING] Could not parse datetime column cleanly.")
        return

    print("Min timestamp:", df["tpep_pickup_datetime"].min())
    print("Max timestamp:", df["tpep_pickup_datetime"].max())


def analyze_locations(df: pd.DataFrame) -> None:
    print("\n===== LOCATION IDS =====")

    print("Unique PU locations:", df["PULocationID"].nunique())
    print("Unique DO locations:", df["DOLocationID"].nunique())


def main():
    try:
        df = load_processed_data(PROCESSED_PATH)

        summarize_structure(df)
        analyze_target(df)
        analyze_trip_distance(df)
        analyze_passenger(df)
        analyze_time(df)
        analyze_locations(df)

        print("\n[SUCCESS] EDA completed.")

    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
