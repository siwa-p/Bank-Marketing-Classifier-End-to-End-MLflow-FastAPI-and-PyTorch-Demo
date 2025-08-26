import duckdb
import sys
import os
from minio import Minio
from dotenv import load_dotenv
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, ".."))
sys.path.append(parent_path)
from utilities.logging_config import logger
from utilities.utils import get_minio_client, setup_db
import pandas as pd
load_dotenv(override=True)

minio_access_key = os.getenv("MINIO_ACCESS_KEY")
minio_secret_key = os.getenv("MINIO_SECRET_KEY")
minio_url = os.getenv("MINIO_URL")
minio_bucket_name = os.getenv("MINIO_BUCKET")

minio_client = get_minio_client(minio_url, minio_access_key, minio_secret_key)

duckdb.install_extension("ducklake")
duckdb.load_extension("ducklake")

con = duckdb.connect(":memory:")

con.execute("INSTALL 'ducklake'") \
   .execute("LOAD 'ducklake'") \
   .execute("INSTALL 'httpfs'") \
   .execute("LOAD 'httpfs'")

def load_parquet_from_minio_to_duckdb(con, table_name, schema_name):
    object_name = f"{table_name}.parquet"
    s3_path = f"s3://{minio_bucket_name}/{object_name}"
    try:
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} AS
            SELECT * FROM read_parquet('{s3_path}')
        """)
        logger.info(f"Loaded {table_name} into DuckDB table {schema_name}.{table_name}")
    except Exception as e:
        logger.error(f"Failed to load {table_name} into DuckDB: {e}")
        raise

def main():
    setup_db(con, minio_url, minio_access_key, minio_secret_key)
    load_parquet_from_minio_to_duckdb(con, "train", "bank_schema")
    load_parquet_from_minio_to_duckdb(con, "test", "bank_schema")
    logger.info("Data loading completed successfully.")
    
if __name__ == "__main__":
    main()