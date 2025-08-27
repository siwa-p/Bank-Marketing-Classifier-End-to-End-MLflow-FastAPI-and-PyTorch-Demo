import torch
import torch.nn as nn
from minio import Minio
import pandas as pd
import io
import os
from utilities.logging_config import logger


class MLPModel(nn.Module):
    def __init__(self, in_features, hidden_units=[256, 64, 16], dropout_prob=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_units[0])
        self.bn1 = nn.BatchNorm1d(hidden_units[0])
        self.dropout1 = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.bn2 = nn.BatchNorm1d(hidden_units[1])
        self.dropout2 = nn.Dropout(dropout_prob)

        self.fc3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.bn3 = nn.BatchNorm1d(hidden_units[2])
        self.dropout3 = nn.Dropout(dropout_prob)

        self.fc4 = nn.Linear(hidden_units[2], 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
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
    folder_path = r"D:\Data Engineering Bootcamp\mlflow-demo"
    ducklake_db_path = os.path.join(folder_path, "catalog.ducklake")
    ducklake_data_path = os.path.join(folder_path, "catalog_data")

    con.execute(f"""
        SET s3_endpoint='{minio_url}';
        SET s3_access_key_id='{minio_access_key}';
        SET s3_secret_access_key='{minio_secret_key}';
        SET s3_use_ssl=false;
        SET s3_url_style='path';
    """)
    con.execute(f"ATTACH 'ducklake:{ducklake_db_path}' AS my_lake (DATA_PATH '{ducklake_data_path}')")
    con.execute("USE my_lake")
    con.execute("CREATE SCHEMA IF NOT EXISTS bank_schema")
    logger.info(f"Database setup completed")
    return None

def process_features(data):
    if "y" in data.columns:
        features = data.drop('y', axis=1)
    else:
        features = data.copy()
    numerical_features = features.select_dtypes(include=['number'])
    categorical_features = features.select_dtypes(exclude=['number'])
    if not categorical_features.empty:
        encoded_features = pd.get_dummies(categorical_features)
        features_tot = pd.concat([numerical_features, encoded_features], axis=1)
    else:
        features_tot = numerical_features
    features_tot = features_tot.apply(pd.to_numeric, errors='coerce').fillna(0)
    return features_tot

