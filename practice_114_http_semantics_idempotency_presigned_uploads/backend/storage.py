"""S3 / MinIO Storage Integration for Presigned Uploads.

Handles:
1. Bucket initialization and CORS policy enforcement for direct browser uploads.
2. Presigned S3 PUT URL generation.
3. Server-side object verification (ETag & ContentLength check via HeadObject).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "datasets")
S3_REGION = os.getenv("S3_REGION", "us-east-1")

# Configure Boto3 client for MinIO S3 API
s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name=S3_REGION,
    config=Config(signature_version="s3v4"),
)


def init_storage() -> None:
    """Ensure the target bucket exists and has correct CORS configuration.

    CORS allows browser fetch/XHR from http://localhost:5173 to perform direct PUTs
    and read the ETag response header.
    """
    try:
        # Check if bucket exists
        s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
    except ClientError:
        try:
            s3_client.create_bucket(Bucket=S3_BUCKET_NAME)
            print(f"[Storage] Created S3 bucket: {S3_BUCKET_NAME}")
        except Exception as e:
            print(f"[Storage] Warning creating bucket: {e}")

    # Enforce CORS rules for direct-to-S3 uploads
    cors_configuration = {
        "CORSRules": [
            {
                "AllowedHeaders": ["*"],
                "AllowedMethods": ["GET", "PUT", "POST", "DELETE", "HEAD"],
                "AllowedOrigins": [
                    "http://localhost:5173",
                    "http://127.0.0.1:5173",
                    "http://localhost:8114",
                ],
                "ExposeHeaders": ["ETag", "Content-Length", "Content-Type"],
                "MaxAgeSeconds": 3600,
            }
        ]
    }
    try:
        s3_client.put_bucket_cors(
            Bucket=S3_BUCKET_NAME,
            CORSConfiguration=cors_configuration,
        )
        print(f"[Storage] Successfully applied S3 CORS rules on '{S3_BUCKET_NAME}'")
    except Exception as e:
        print(f"[Storage] Note: Could not apply S3 CORS via boto3 ({e}). Container init may have applied it.")


def generate_presigned_put_url(
    s3_key: str,
    content_type: str,
    expires_in: int = 900,
) -> str:
    """Generate a presigned S3 PUT URL for direct browser-to-storage upload.

    ========================================================================
    TODO(human): Exercise 3 (Backend) - Generate Presigned PUT URL
    ========================================================================
    Context:
    Proxying multi-hundred megabyte file uploads through FastAPI exhausts
    server memory and keeps async event loop workers blocked on network I/O.
    Instead, FastAPI acts as an authority that grants a temporary signed URL.
    The browser streams bytes directly to MinIO using this URL.

    Instructions:
    Use `s3_client.generate_presigned_url(...)`:
      - `ClientMethod`: 'put_object'
      - `Params`:
          {
              'Bucket': S3_BUCKET_NAME,
              'Key': s3_key,
              'ContentType': content_type,
          }
      - `ExpiresIn`: expires_in (default 900 seconds / 15 minutes)

    Return the generated string URL.
    ========================================================================
    """
    # --- IMPLEMENTATION GOES HERE ---
    raise NotImplementedError(
        "Exercise 3: Implement generate_presigned_put_url in backend/storage.py"
    )


def verify_uploaded_object(s3_key: str, expected_etag: str) -> dict[str, Any]:
    """Verify that the object was successfully uploaded to MinIO and matches the client ETag.

    Returns the S3 HeadObject metadata dict.
    Raises ValueError if object is missing or ETag mismatches.
    """
    try:
        response = s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
    except ClientError as e:
        raise ValueError(f"Object '{s3_key}' not found in storage bucket '{S3_BUCKET_NAME}': {e}")

    s3_etag = response.get("ETag", "").strip('"')
    clean_expected = expected_etag.strip('"')

    if s3_etag and clean_expected and s3_etag != clean_expected:
        raise ValueError(
            f"ETag mismatch for '{s3_key}'. Client reported {clean_expected}, "
            f"but MinIO calculated {s3_etag}. Binary upload may be corrupted."
        )

    return {
        "s3_key": s3_key,
        "etag": s3_etag,
        "content_length": response.get("ContentLength", 0),
        "content_type": response.get("ContentType", "application/octet-stream"),
        "last_modified": str(response.get("LastModified", "")),
    }


def delete_object(s3_key: str) -> None:
    """Remove an object from MinIO."""
    try:
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
    except Exception as e:
        print(f"[Storage] Error deleting {s3_key}: {e}")
