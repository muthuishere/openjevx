#!/usr/bin/env python3
"""Cloudflare R2 over its S3 API: upload, download, and time-limited links for the GPU box.

Credentials come from $CLOUDFLARE_ALLPURPOSE_TOKEN (loaded from the login shell if this process
does not have it). R2 derives S3 keys from an API token: access key = token id,
secret = sha256(token). Values never leave this process; the box only ever gets signed links.

  r2.py put FILE KEY [--bucket B]       r2.py get KEY FILE [--bucket B]
  r2.py ls PREFIX [--bucket B]
"""
import hashlib, json, os, subprocess, sys, urllib.request
from pathlib import Path

ACCOUNT = os.environ.get("CF_ACCOUNT_ID", "4cf2da436d01cd836a29ae0ae1eee8fc")
PRIVATE = os.environ.get("OPENJEVX_R2_BUCKET", "openjevx-train")
_client = None


def _token():
    tok = os.environ.get("CLOUDFLARE_ALLPURPOSE_TOKEN")
    if not tok:
        tok = subprocess.run(["zsh", "-ic", "printf %s \"$CLOUDFLARE_ALLPURPOSE_TOKEN\""], text=True,
                             capture_output=True).stdout.strip()
    if not tok:
        raise SystemExit("CLOUDFLARE_ALLPURPOSE_TOKEN is not set")
    return tok


def client():
    global _client
    if _client is None:
        import boto3
        from botocore.config import Config
        tok = _token()
        req = urllib.request.Request("https://api.cloudflare.com/client/v4/user/tokens/verify",
                                     headers={"Authorization": "Bearer " + tok})
        token_id = json.load(urllib.request.urlopen(req))["result"]["id"]
        _client = boto3.client("s3", endpoint_url=f"https://{ACCOUNT}.r2.cloudflarestorage.com", region_name="auto",
                               aws_access_key_id=token_id, aws_secret_access_key=hashlib.sha256(tok.encode()).hexdigest(),
                               config=Config(signature_version="s3v4"))
    return _client


def put(path, key, bucket=PRIVATE):
    from boto3.s3.transfer import TransferConfig
    client().upload_file(str(path), bucket, key, Config=TransferConfig(
        multipart_threshold=64 << 20, multipart_chunksize=64 << 20, max_concurrency=8))


def get(key, path, bucket=PRIVATE):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    client().download_file(bucket, key, str(path))


def ls(prefix, bucket=PRIVATE):
    pages = client().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
    return [o["Key"] for p in pages for o in p.get("Contents", [])]


def link_get(key, seconds=86400, bucket=PRIVATE):
    return client().generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=seconds)


def link_put(key, seconds=86400, bucket=PRIVATE):
    return client().generate_presigned_url("put_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=seconds)


if __name__ == "__main__":
    a = sys.argv[1:]
    bucket = PRIVATE
    if "--bucket" in a:
        i = a.index("--bucket"); bucket = a[i + 1]; del a[i:i + 2]
    if a[:1] == ["put"]: put(a[1], a[2], bucket)
    elif a[:1] == ["get"]: get(a[1], a[2], bucket)
    elif a[:1] == ["ls"]: print("\n".join(ls(a[1] if len(a) > 1 else "", bucket)))
    else: sys.exit(__doc__)
