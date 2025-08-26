import torch
import torch.nn as nn
from minio import Minio
import pandas as pd
import io
from utilities.logging_config import logger


class MLPModel(nn.Module):
    def __init__(self, in_features, hidden_units=[128, 32]):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def get_minio_client(minio_url, minio_access_key, minio_secret_key):
    try:
        minio_client = Minio(
            minio_url,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False,
        )
        logger.info("Connected to MinIO successfully.")
        return minio_client
    except Exception as e:
        logger.error(f"Failed to connect to MinIO: {e}")
        raise

def upload_to_minio(minio_client, bucket_name, data, object_name):
    if not minio_client.bucket_exists(bucket_name):
        logger.info(f"Bucket {bucket_name} does not exist. Creating it.")
        minio_client.make_bucket(bucket_name)
    parquet_buffer = io.BytesIO()
    data.to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)
    minio_client.put_object(
        bucket_name,
        object_name,
        parquet_buffer,
        length=parquet_buffer.getbuffer().nbytes,
        content_type="application/octet-stream",
    )
    logger.info(f"Data uploaded to MinIO bucket {bucket_name} as {object_name}")



def setup_db(con, minio_url, minio_access_key, minio_secret_key):
    con.execute(f"""
        SET s3_endpoint='{minio_url}';
        SET s3_access_key_id='{minio_access_key}';
        SET s3_secret_access_key='{minio_secret_key}';
        SET s3_use_ssl=false;
        SET s3_url_style='path';
    """)
    con.execute("ATTACH 'ducklake:/home/prahald/Documents/Data Engineering Bootcamp/mlflow-demo/catalog.ducklake' AS my_lake (DATA_PATH '/home/prahald/Documents/Data Engineering Bootcamp/mlflow-demo/catalog_data')")
    con.execute("USE my_lake")
    con.execute("CREATE SCHEMA IF NOT EXISTS bank_schema")
    logger.info(f"Database setup completed")
    return None