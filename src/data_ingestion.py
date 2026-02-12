import pandas as pd
import os
import sys
from typing import List

RAW_PATH = 'data/raw/raw.csv'
OUTPUT_PATH = 'data/processed/clean.csv'

REQUIRED_FEATURES : List[str] = [
	"fare_amount",
	"trip_distance",
    "PULocationID" ,
	"DOLocationID" ,
    "passenger_count",
	"tpep_pickup_datetime",
	"VendorID",
	"RatecodeID"
]

def validate_path(path:str)->None:
	if not os.path.exists(path):
		raise FileNotFoundError(f"Raw data file not found at {path}")

def load_data(path:str) -> pd.DataFrame:
	try:
		validate_path(path)
		df = pd.read_csv(path)
		print(f"File loaded succesfully with {df.shape} columns")
		return df
	except pd.errors.EmptyDataError:
		raise RuntimeError('Raw File is Empty')
	except Exception as e:
		raise RuntimeError(f"Failed to load file at {path} , {e}")
	
def validate_schema(df:pd.DataFrame) -> None:
	missing = [col for col in REQUIRED_FEATURES if col not in df.columns]
	if missing:
		raise RuntimeError(f"Required Columns {missing} are missing")

def clean_data(df):
	try:
		validate_schema(df)

		df = df[REQUIRED_FEATURES].copy()
		df['fare_amount'] = pd.to_numeric(df['fare_amount'],errors='coerce')
		df = df.dropna(subset='fare_amount')
		df = df[df['fare_amount']>0]
		df = df[df['fare_amount']<500]
		df = df[df['passenger_count']>=1]
		df = df[df['passenger_count']<=6]
		print(f"Data loaded with {df.shape} columns")
		print(df['fare_amount'].describe())
		print(df['fare_amount'].isna().sum())
		return df
	except Exception as e:
		raise RuntimeError(f"Data cleaning failed {e}")

def save_processed(df:pd.DataFrame,path:str)->None:
	try:
		os.makedirs(os.path.dirname(path),exist_ok=True)
		df.to_csv(path,index=False)
		print(f"Data saved at path {path}")
	except Exception as e:
		raise RuntimeError(f"Failed to save processed data {e}")

def main():
	try:
		df = load_data(RAW_PATH)
		df = clean_data(df)
		save_processed(df,OUTPUT_PATH)

	except Exception as e:
		raise RuntimeError(f"Process failed , {e}")

if __name__=="__main__":
	main()