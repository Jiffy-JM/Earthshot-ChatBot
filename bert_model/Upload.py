import boto3
import json
from botocore.exceptions import NoCredentialsError

def upload_to_s3(file_path, bucket_name, s3_key, endpoint_url):

    # Open the JSON file
    with open('/Users/wrk/Desktop/Earthshot-ChatBot/bert_model/secrets.json') as file:
    # Read the contents of the file
        data = json.load(file)

    # AWS credentials
    aws_access_key_id = data['aws_access_key_id'][0]
    aws_secret_access_key = data['aws_secret_access_key'][0]

    # Create an S3 client with the specified endpoint URL
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url=endpoint_url
    )

    try:
        # Upload the file to S3
        s3.upload_file(file_path, bucket_name, s3_key)
        print(f"File uploaded successfully to S3 bucket: {bucket_name}")
    except NoCredentialsError:
        print("AWS credentials not available.")

# Usage example
file_path = '/Users/wrk/Desktop/Refined_txt.txt'
bucket_name = 'test'
s3_key = 'ml/Refined_txt.txt'
endpoint_url = 'https://gateway.storjshare.io'

upload_to_s3(file_path, bucket_name, s3_key, endpoint_url)
