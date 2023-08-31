# -*- coding: utf-8 -*-
from azure.storage.blob import BlobServiceClient
from datetime import datetime
import os

def connect_to_azure(file_name):
    account_url = 'https://ogtaidata.blob.core.windows.net'
    account_key = 'Vd3ySqOsoDnbyfvlmgt4YM9Tp6HKoGFQr37PDZ9jxMJaJZzLf1XI8ZAI5XTmgV2wtECWUqUXxcQq+ASt7Fa1Pg=='
    
    blob_service = BlobServiceClient(
        account_url=account_url, credential=account_key)
    
    container_name = 'aijourneygidpredictions'
    # Create a local directory to hold blob data
    local_path = "./"
    os.makedirs(local_path, exist_ok=True)
    
    # Create a file in the local data directory to upload and download
    local_file_name = file_name
    upload_file_path = os.path.join(local_path, local_file_name)
    
    folder_name = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Create a blob client using the local file name as the name for the blob and the folder name in minute format
    blob_client = blob_service.get_blob_client(
        container=container_name, blob=f'{folder_name}/{local_file_name}')
    
    print("\nUploading to Azure Storage as blob:\n\t" + local_file_name)
    
    # Upload the created file
    with open(file=upload_file_path, mode="rb") as data:
        blob_client.upload_blob(data)
    print("\nCSV file uploaded succesfuly uploaded to azure")
