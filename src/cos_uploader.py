"""
IBM Cloud Object Storage (COS) Uploader
Stores uploaded resumes/documents in IBM COS and retrieves them for RAG ingestion.

IBM Cloud Lite provides 25 GB free COS storage.

Environment variables required (add to .env):
    COS_API_KEY          — IBM Cloud API key (same key works)
    COS_INSTANCE_CRN     — COS service instance CRN
    COS_ENDPOINT         — Regional endpoint  e.g. https://s3.us-south.cloud-object-storage.appdomain.cloud
    COS_BUCKET           — Target bucket name  e.g. interview-trainer-docs

Usage:
    cos = COSUploader()
    url = cos.upload_file("resume.pdf")
    content = cos.download_as_bytes("resume.pdf")
    cos.list_files()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

try:
    import ibm_boto3
    from ibm_botocore.client import Config
    _HAS_COS = True
except ImportError:
    _HAS_COS = False


class COSUploader:
    """
    Thin wrapper around IBM COS (ibm-cos-sdk).
    Falls back gracefully when the SDK is not installed or COS is not configured.
    """

    def __init__(self):
        self.api_key       = os.getenv("COS_API_KEY")
        self.instance_crn  = os.getenv("COS_INSTANCE_CRN")
        self.endpoint      = os.getenv("COS_ENDPOINT", "https://s3.us-south.cloud-object-storage.appdomain.cloud")
        self.bucket        = os.getenv("COS_BUCKET", "interview-trainer-docs")
        self._client       = None

        self.enabled = bool(_HAS_COS and self.api_key and self.instance_crn)

        if not _HAS_COS:
            print("[COS] ibm-cos-sdk not installed — COS features disabled. "
                  "Run: pip install ibm-cos-sdk")
        elif not self.enabled:
            print("[COS] COS_API_KEY or COS_INSTANCE_CRN not set — COS features disabled.")

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def _get_client(self):
        if self._client is None:
            if not _HAS_COS:
                raise ImportError("ibm-cos-sdk not installed. Run: pip install ibm-cos-sdk")
            self._client = ibm_boto3.client(
                "s3",
                ibm_api_key_id=self.api_key,
                ibm_service_instance_id=self.instance_crn,
                config=Config(signature_version="oauth"),
                endpoint_url=self.endpoint,
            )
        return self._client

    def _ensure_bucket(self) -> None:
        """Create the bucket if it doesn't exist."""
        client = self._get_client()
        existing = [b["Name"] for b in client.list_buckets().get("Buckets", [])]
        if self.bucket not in existing:
            client.create_bucket(Bucket=self.bucket)
            print(f"[COS] Created bucket '{self.bucket}'.")

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------
    def upload_file(self, local_path: str, object_key: Optional[str] = None) -> str:
        """
        Upload a local file to COS.
        Returns the COS object URL (not publicly accessible — use download_as_bytes).
        """
        if not self.enabled:
            raise RuntimeError("COS is not configured. Set COS_API_KEY and COS_INSTANCE_CRN.")

        self._ensure_bucket()
        key    = object_key or Path(local_path).name
        client = self._get_client()
        client.upload_file(local_path, self.bucket, key)
        url = f"{self.endpoint}/{self.bucket}/{key}"
        print(f"[COS] Uploaded '{local_path}' → {url}")
        return url

    def upload_bytes(self, data: bytes, object_key: str) -> str:
        """Upload raw bytes (e.g. Streamlit UploadedFile content) to COS."""
        if not self.enabled:
            raise RuntimeError("COS is not configured.")
        self._ensure_bucket()
        self._get_client().put_object(Bucket=self.bucket, Key=object_key, Body=data)
        url = f"{self.endpoint}/{self.bucket}/{object_key}"
        print(f"[COS] Uploaded bytes → {url}")
        return url

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------
    def download_as_bytes(self, object_key: str) -> bytes:
        """Download a COS object and return raw bytes."""
        if not self.enabled:
            raise RuntimeError("COS is not configured.")
        response = self._get_client().get_object(Bucket=self.bucket, Key=object_key)
        return response["Body"].read()

    def download_to_file(self, object_key: str, local_path: str) -> None:
        """Download a COS object to a local file."""
        if not self.enabled:
            raise RuntimeError("COS is not configured.")
        self._get_client().download_file(self.bucket, object_key, local_path)
        print(f"[COS] Downloaded '{object_key}' → '{local_path}'.")

    # ------------------------------------------------------------------
    # List / Delete
    # ------------------------------------------------------------------
    def list_files(self) -> List[str]:
        """Return a list of object keys in the configured bucket."""
        if not self.enabled:
            return []
        response = self._get_client().list_objects_v2(Bucket=self.bucket)
        return [obj["Key"] for obj in response.get("Contents", [])]

    def delete_file(self, object_key: str) -> None:
        """Delete an object from COS."""
        if not self.enabled:
            raise RuntimeError("COS is not configured.")
        self._get_client().delete_object(Bucket=self.bucket, Key=object_key)
        print(f"[COS] Deleted '{object_key}' from bucket '{self.bucket}'.")
