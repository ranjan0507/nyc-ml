import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.preprocessing import build_preprocessing_pipeline

train_path = 'data/processed/train.csv'
val_path = 'data/processed/val.csv'

model_dump_path = 'models/model_v1.joblib'

SEED = 42

def load_dataset(path):
	df = pd.read_csv(path)
	return df

def rmse(y_true,y_pred):
	return np.sqrt(mean_squared_error(y_true,y_pred))

def train_baseline(X_train,y_train,X_val,y_val):
	baseline = DummyRegressor(strategy='median')
	baseline.fit(X_train,y_train)
	preds = baseline.predict(X_val)
	score = rmse(y_val,preds)
	print(f"Baseline Model score {score}")
	return score

def train_linear_model(preprocessor,X_train,y_train,X_val,y_val):
	pipeline = Pipeline(steps=[
		('preprocessing',preprocessor),
		('model',LinearRegression())
	])

	pipeline.fit(X_train,y_train)
	preds = pipeline.predict(X_val)
	score = rmse(y_val,preds)
	return pipeline , score

def train_random_forest(preprocessor,X_train,y_train,X_val,y_val):
	pipeline = Pipeline(steps=[
		('preprocessing',preprocessor),
		('model',RandomForestRegressor(
			n_estimators=40,
			random_state=SEED,
			n_jobs=1,
			max_depth=12
		))
	])
	pipeline.fit(X_train,y_train)
	pred = pipeline.predict(X_val)
	score = rmse(y_val,pred)
	return pipeline,score

def main():
	print(f"Loading training and validating data..")
	train = pd.read_csv(train_path)
	val = pd.read_csv(val_path)

	print(f"Preparing data features ...")
	X_train = train.drop(columns=['fare_amount'])
	y_train = train['fare_amount']
	X_val = val.drop(columns=['fare_amount'])
	y_val = val['fare_amount']

	print(f"Building preprocessor pipelines ...")
	preprocessor = build_preprocessing_pipeline()

	baseline_score = train_baseline(X_train,y_train,X_val,y_val)

	lr_model , lr_score = train_linear_model(preprocessor,X_train,y_train,X_val,y_val)

	rf_model , rf_score = train_random_forest(preprocessor,X_train,y_train,X_val,y_val)

	print(f"Model Comparison")
	print(f"Baseline Model RMSE: ",baseline_score)
	print(f"Linear Regression Model RMSE: ",lr_score)
	print(f"Random Forest Model RMSE: ",rf_score)

	print(f"Saving trained Pipeline..")
	best_model = rf_model if rf_score<lr_score else lr_model
	joblib.dump(best_model,model_dump_path)
	print(f"Pipeline saved successfully")

if __name__=='__main__':
	main()


