import pandas as pd
import os

CLEAN_DATA_PATH = 'data/processed/clean.csv'
OUTPUT_PATH = 'data/processed'

SEED = 42

def validate_path(path:str)->None:
	if not os.path.exists(path):
		raise FileNotFoundError(f"File Noe found at {path}")
	
def load_clean_data(path:str)->pd.DataFrame:
	try:
		validate_path(path)
		df = pd.read_csv(path)
		print(f"Clean data read with {df.shape} shape")
		return df
	except Exception as e:
		raise RuntimeError(f"Failed to load clean data from , {e}")
	
def create_buckets(df:pd.DataFrame)->pd.DataFrame:
	try:
		bins=[0,5,10,20,50,100,float("inf")]
		labels=["0-5","5-10","10-20","20-50","50-100","100+"]
		df['fare_bucket'] = pd.cut(df['fare_amount'],bins=bins,labels=labels)
		if df['fare_bucket'].isna().sum() > 0:
			raise ValueError(f"Some rows could'nt be bucketed , check fare values")
		return df
	except Exception as e:
		raise RuntimeError(f"Some problem occured while creating buckets , {e}")
	
def stratify_split(df:pd.DataFrame):
	try:
		from sklearn.model_selection import train_test_split
		train_val,test = train_test_split(
			df,
			test_size=0.15,
			stratify=df['fare_bucket'],
			random_state=SEED
		)
		train,val = train_test_split(
			train_val,
			test_size=0.1765,
			stratify=train_val['fare_bucket'] ,
			random_state=SEED
		)
		print()
		return train,val,test
	except Exception as e:
		raise RuntimeError(f"A Problem occured doing Stratified , {e}")
	
def save_split(train:pd.DataFrame,val:pd.DataFrame,test:pd.DataFrame):
	try:
		os.makedirs(os.path.dirname(OUTPUT_PATH),exist_ok=True)
		train = train.drop(columns=['fare_bucket'])
		val = val.drop(columns=['fare_bucket'])
		test = test.drop(columns=['fare_bucket'])
		print(f"Training set shape : {train.shape}")
		print(f"Validate set shape : {val.shape}")
		print(f"Testing set shape : {test.shape}")
		train.to_csv(os.path.join(OUTPUT_PATH,'train.csv'),index=False)
		val.to_csv(os.path.join(OUTPUT_PATH,'val.csv'),index=False)
		test.to_csv(os.path.join(OUTPUT_PATH,'test.csv'),index=False)
	except Exception as e:
		raise RuntimeError(f"A Problem occured while saving the processed split files , {e}")

def display_info(df:pd.DataFrame,name:str)->None:
	print(f"{name} , Fare Distribution")
	print(df['fare_amount'].info())

def main():
	try:
		df = load_clean_data(CLEAN_DATA_PATH)
		df = create_buckets(df)
		train , val , test = stratify_split(df)
		display_info(train,'Train')
		display_info(val,'Val')
		display_info(test,'Test')
		save_split(train,val,test)

	except Exception as e:
	    raise RuntimeError(f"Some problem occured , {e}")


if __name__ == '__main__':
	main()