import requests
from dotenv import load_dotenv
import os
from utilities.logging_config import logger
from io import StringIO
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utilities'))
load_dotenv()
base_url = os.getenv("BASE_URL")

minio_access_key = os.getenv("MINIO_ACCESS_KEY")
minio_secret_key = os.getenv("MINIO_SECRET_KEY")
minio_url = os.getenv("MINIO_URL")
minio_bucket = os.getenv("MINIO_BUCKET")
from utilities.utils import get_minio_client, upload_to_minio
minio_client = get_minio_client(minio_url, minio_access_key, minio_secret_key)

def main():
    folder_path = r"D:\Data Engineering Bootcamp\mlflow-demo\data"
    train_data_path = os.path.join(folder_path, "train.csv")
    test_data_path = os.path.join(folder_path, "test.csv")
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    upload_to_minio(minio_client, minio_bucket, train_data, "train.parquet")
    upload_to_minio(minio_client, minio_bucket, test_data, "test.parquet")
    logger.info("Data ingestion completed successfully.")
        
    
if __name__=="__main__":
    main()
    