import pandas as pd
import numpy as np

def extract_time_features(df:pd.DataFrame)->pd.DataFrame:
	df = df.copy()
	df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
	df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
	df['pickup_dayofweek'] = df['tpep_pickup_datetime'].dt.dayofweek
	df['pickup_month'] = df['tpep_pickup_datetime'].dt.month
	df['is_weekend'] = df['pickup_dayofweek'].isin([5,6]).astype(int)
	df.drop(columns=['tpep_pickup_datetime'],inplace=True)
	return df

def add_distance_features(df:pd.DataFrame)->pd.DataFrame:
	df = df.copy()
	df['is_zero_distance'] = (df['trip_distance']==0).astype(int)
	df['log_distance'] = np.log1p(df['trip_distance'])
	return df

def add_passenger_feature(df:pd.DataFrame)->pd.DataFrame:
	df = df.copy()
	df['is_single_passenger'] = (df['passenger_count']==1).astype(int)
	return df

def apply_feature_engineering(df:pd.DataFrame)->pd.DataFrame:
	df = extract_time_features(df)
	df = add_distance_features(df)
	df = add_passenger_feature(df)
	return df

