import pandas as pd

RAW_PATH = 'data/raw/raw.csv'
OUTPUT_PATH = 'data/processed/clean.csv'

REQUIRED_FEATURES = [
	"fare_amount",
	"trip_distance",
    "PULocationID" ,
	"DOLocationID" ,
    "passenger_count",
	"tpep_pickup_datetime",
	"VendorID",
	"RatecodeID"
]

def load_data(path):
	return pd.read_csv(path)

def clean_data(df):
	df = df[REQUIRED_FEATURES]
	df = df.dropna(subset=['fare_amount'])
	df = df[df['fare_amount']>0]
	df = df[df['passenger_count']>=1]
	df = df[df['passenger_count']<=6]
	return df

def save_processed(df):
	df.to_csv(OUTPUT_PATH)

def main():
	df = load_data(RAW_PATH)
	print('Data Loaded with shape : ', df.shape)
	df = clean_data(df)
	print('Clean Data Shape : ',df.shape)
	save_processed(df)
	print('Data saved to : ', OUTPUT_PATH)

if __name__=="__main__":
	main()