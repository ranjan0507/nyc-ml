import pandas as pd
from src.feature_engineering import apply_feature_engineering
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

TARGET_COLUMN = 'fare_amount'

NUMERIC_COLUMNS = [
    "trip_distance",
    "log_distance",
    "passenger_count",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month"
]

CATEGORICAL_COLUMNS = [
    "PULocationID",
    "DOLocationID",
    "VendorID",
    "RatecodeID",
    "is_weekend",
    "is_zero_distance",
    "is_single_passenger"
]

def load_data(path:str)->pd.DataFrame:
	df = pd.read_csv(path)
	return df

def prepare_feature(df:pd.DataFrame)->pd.DataFrame:
	df = apply_feature_engineering(df)
	X = df.drop(columns=[TARGET_COLUMN])
	y = df[TARGET_COLUMN]
	return X , y

def build_preprocessing_pipeline():
	numeric_pipeline = Pipeline(steps=[
		("imputer",SimpleImputer(strategy='median')) ,
		("scaler",StandardScaler())
	])

	categorical_pipeline = Pipeline(steps=[
		("imputer",SimpleImputer(strategy='most_frequent')) ,
		('encoder',OneHotEncoder(handle_unknown='ignore'))
	])

	preprocessor = ColumnTransformer(
		transformers=[
			('num',numeric_pipeline,NUMERIC_COLUMNS),
			('cat',categorical_pipeline,CATEGORICAL_COLUMNS)
		]
	)

	return preprocessor