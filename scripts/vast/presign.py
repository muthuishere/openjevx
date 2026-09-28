#!/usr/bin/env python3
"""Print a time-limited GET link for an object in our private S3 bucket.

Run under sec so credentials exist only in this process:
  sec run S3_ENDPOINT S3_ACCESS_KEY S3_SECRET_KEY S3_REGION BACKUP_S3_BUCKET -- python3 presign.py KEY [SECONDS]
The link is written to stdout for the caller to redirect into a gitignored file.
"""
import os
import sys

import boto3

key = sys.argv[1]
seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 86400
endpoint = os.environ["S3_ENDPOINT"]
endpoint = endpoint if "://" in endpoint else "https://" + endpoint
client = boto3.client("s3", endpoint_url=endpoint, region_name=os.environ["S3_REGION"],
                      aws_access_key_id=os.environ["S3_ACCESS_KEY"], aws_secret_access_key=os.environ["S3_SECRET_KEY"])
print(client.generate_presigned_url("get_object", Params={"Bucket": os.environ["BACKUP_S3_BUCKET"], "Key": key},
                                    ExpiresIn=seconds))
