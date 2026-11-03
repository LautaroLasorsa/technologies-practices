"""FastAPI Application for Practice 114: HTTP Semantics, Idempotency, and Presigned Uploads.

Endpoints demonstrate:
- POST /api/uploads/presign: Phase 1 S3 Presigned PUT generation
- POST /api/datasets/commit: Phase 3 Idempotent Commit with Idempotency-Key
- GET  /api/datasets/{id}: Conditional requests with ETag & If-None-Match (304 Not Modified)
- PUT  /api/datasets/{id}: Optimistic concurrency locking with If-Match (412 Precondition Failed)
- Chaos & Telemetry APIs for inspecting network drops, retries, and headers
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import idempotency
import storage


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class PresignRequest(BaseModel):
    filename: str
    content_type: str = "text/csv"
    size_bytes: int = Field(gt=0)


class PresignResponse(BaseModel):
    upload_url: str
    s3_key: str
    http_method: str = "PUT"
    expires_in_seconds: int = 900


class CommitDatasetRequest(BaseModel):
    name: str
    description: str
    s3_key: str
    etag: str
    size_bytes: int


class UpdateDatasetRequest(BaseModel):
    name: str
    description: str


class Dataset(BaseModel):
    id: str
    name: str
    description: str
    s3_key: str
    size_bytes: int
    etag: str
    version: int
    created_at: str
    updated_at: str


class ChaosConfig(BaseModel):
    drop_post_commit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    delay_ms: int = Field(default=0, ge=0, le=10000)


# ---------------------------------------------------------------------------
# In-Memory Database & State
# ---------------------------------------------------------------------------
datasets_db: dict[str, Dataset] = {}
chaos_settings = ChaosConfig(drop_post_commit_rate=0.0, delay_ms=0)


def compute_dataset_etag(dataset: Dataset) -> str:
    """Generate a strong ETag representing the resource version."""
    fingerprint = f"{dataset.id}:{dataset.version}:{dataset.name}:{dataset.description}:{dataset.updated_at}"
    digest = hashlib.md5(fingerprint.encode("utf-8")).hexdigest()[:10]
    return f'"{dataset.id}-v{dataset.version}-{digest}"'


def seed_sample_datasets() -> None:
    """Seed sample datasets to test conditional caching and optimistic locking."""
    datasets_db.clear()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    sample1 = Dataset(
        id="ds_sales_2025",
        name="Q1 Global Sales Dataset",
        description="Consolidated regional retail transactions across North America and EMEA.",
        s3_key="raw/sample_sales.csv",
        size_bytes=1048576,
        etag="",
        version=1,
        created_at=now,
        updated_at=now,
    )
    sample1.etag = compute_dataset_etag(sample1)
    datasets_db[sample1.id] = sample1

    sample2 = Dataset(
        id="ds_sensor_telemetry",
        name="IoT Fleet Sensor Telemetry",
        description="High-frequency temperature and pressure readings sampled at 100Hz.",
        s3_key="raw/sample_sensors.csv",
        size_bytes=4194304,
        etag="",
        version=1,
        created_at=now,
        updated_at=now,
    )
    sample2.etag = compute_dataset_etag(sample2)
    datasets_db[sample2.id] = sample2


# ---------------------------------------------------------------------------
# Application Lifespan & App Setup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize MinIO S3 bucket and CORS
    storage.init_storage()
    seed_sample_datasets()
    yield


app = FastAPI(
    title="Practice 114: Resilient Ingestion API",
    description="HTTP Semantics, Idempotency Keys, Conditional Caching & S3 Presigned Uploads",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for React frontend (Vite port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["ETag", "Idempotency-Key", "Idempotent-Replay"],
)


# ---------------------------------------------------------------------------
# Phase 1: Presigned Upload URL Generation
# ---------------------------------------------------------------------------
@app.post("/api/uploads/presign", response_model=PresignResponse)
async def create_presigned_upload(request: PresignRequest):
    """Phase 1: Negotiate a temporary presigned PUT URL directly to S3/MinIO.

    The React client sends file metadata, and FastAPI delegates binary streaming
    directly to MinIO by creating an authorized S3 PUT URL.
    """
    if chaos_settings.delay_ms > 0:
        await asyncio.sleep(chaos_settings.delay_ms / 1000.0)

    # Generate isolated S3 object key
    unique_id = uuid.uuid4().hex[:12]
    clean_filename = "".join(c for c in request.filename if c.isalnum() or c in "._-")
    s3_key = f"datasets/{unique_id}_{clean_filename}"

    # Delegate to storage module
    upload_url = storage.generate_presigned_put_url(
        s3_key=s3_key,
        content_type=request.content_type,
        expires_in=900,
    )

    return PresignResponse(
        upload_url=upload_url,
        s3_key=s3_key,
        http_method="PUT",
        expires_in_seconds=900,
    )


# ---------------------------------------------------------------------------
# Phase 3: Idempotent Commit Endpoint
# ---------------------------------------------------------------------------
@app.post("/api/datasets/commit", status_code=status.HTTP_201_CREATED)
async def commit_dataset(
    request: CommitDatasetRequest,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    """Phase 3: Register dataset metadata and link S3 object with Idempotency-Key protection.

    Demonstrates:
    - In-flight locking (409 Conflict)
    - Payload tampering detection (422 Unprocessable Entity)
    - Response replay from cache (201 Created with Idempotent-Replay: true)
    - Simulated post-commit network drops (Chaos switch)
    """
    if chaos_settings.delay_ms > 0:
        await asyncio.sleep(chaos_settings.delay_ms / 1000.0)

    payload_dict = request.model_dump()
    payload_hash = idempotency.compute_payload_hash(payload_dict)

    if idempotency_key:
        decision, record = idempotency.evaluate_idempotency(idempotency_key, payload_hash)

        if decision == "IN_FLIGHT":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A mutation with this Idempotency-Key is currently in-flight. Please wait.",
            )
        elif decision == "PAYLOAD_MISMATCH":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Idempotency-Key reuse error: Request payload has changed.",
            )
        elif decision == "REPLAY" and record is not None:
            # Replay cached response without executing commit business logic
            return JSONResponse(
                content=record.body,
                status_code=record.status_code,
                headers={"Idempotent-Replay": "true", **record.headers},
            )

    # Verify S3 object exists in MinIO and matches the client-reported ETag
    try:
        storage.verify_uploaded_object(request.s3_key, request.etag)
    except Exception as e:
        if idempotency_key:
            idempotency.mark_idempotency_failed(idempotency_key)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Storage verification failed: {e}",
        )

    # Perform Database Commit
    dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    dataset = Dataset(
        id=dataset_id,
        name=request.name,
        description=request.description,
        s3_key=request.s3_key,
        size_bytes=request.size_bytes,
        etag="",
        version=1,
        created_at=now,
        updated_at=now,
    )
    dataset.etag = compute_dataset_etag(dataset)
    datasets_db[dataset_id] = dataset

    response_body = dataset.model_dump()
    response_headers = {"ETag": dataset.etag, "Idempotency-Key": idempotency_key or ""}

    # Save to idempotency store if key was provided
    if idempotency_key:
        idempotency.store_idempotency_response(
            key=idempotency_key,
            status_code=status.HTTP_201_CREATED,
            body=response_body,
            headers=response_headers,
        )

    # Chaos Injection: Simulated network drop AFTER commit has succeeded on the server
    if chaos_settings.drop_post_commit_rate > 0.0:
        if random.random() < chaos_settings.drop_post_commit_rate:
            # The database write succeeded, but the connection abruptly dies before reaching the client!
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="[Chaos Simulation] Connection dropped immediately after database commit! Frontend must retry with same Idempotency-Key.",
            )

    return JSONResponse(
        content=response_body,
        status_code=status.HTTP_201_CREATED,
        headers=response_headers,
    )


# ---------------------------------------------------------------------------
# Exercise 2: HTTP Conditional Requests (ETag, If-None-Match, If-Match)
# ---------------------------------------------------------------------------
@app.get("/api/datasets/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    if_none_match: Optional[str] = Header(None, alias="If-None-Match"),
):
    """Fetch dataset metadata with ETag conditional validation (304 Not Modified).

    ========================================================================
    TODO(human): Exercise 2 (Backend part A) - Implement If-None-Match (304)
    ========================================================================
    Context:
    If the client already holds a cached representation with an ETag matching
    the current resource, returning the full JSON payload wastes bandwidth.
    Instead, return HTTP 304 Not Modified with an empty body and the ETag header.

    Instructions:
    1. If `dataset_id not in datasets_db`, raise HTTPException 404 Not Found.
    2. Retrieve `dataset = datasets_db[dataset_id]`.
    3. Compute `current_etag = compute_dataset_etag(dataset)`.
    4. If `if_none_match` is present:
       Compare `if_none_match.strip('"')` with `current_etag.strip('"')`
       (or check if `if_none_match == "*"`).
       If they match:
         Return `Response(status_code=status.HTTP_304_NOT_MODIFIED, headers={"ETag": current_etag})`.
    5. If no match or header absent:
       Return `JSONResponse(content=dataset.model_dump(), headers={"ETag": current_etag})`.
    ========================================================================
    """
    # --- IMPLEMENTATION GOES HERE ---
    raise NotImplementedError(
        "Exercise 2: Implement get_dataset with If-None-Match in backend/main.py"
    )


@app.put("/api/datasets/{dataset_id}")
async def update_dataset(
    dataset_id: str,
    request: UpdateDatasetRequest,
    if_match: Optional[str] = Header(None, alias="If-Match"),
):
    """Update dataset metadata with optimistic concurrency locking (If-Match / 412).

    ========================================================================
    TODO(human): Exercise 2 (Backend part B) - Implement If-Match Optimistic Locking
    ========================================================================
    Context:
    Prevents the Lost Update Problem. If Alice and Bob both read revision A,
    and Alice saves revision B, Bob's save must be rejected if his `If-Match`
    header still presents revision A.

    Instructions:
    1. If `dataset_id not in datasets_db`, raise HTTPException 404 Not Found.
    2. Retrieve `dataset = datasets_db[dataset_id]`.
    3. Compute `current_etag = compute_dataset_etag(dataset)`.
    4. Check If-Match condition:
       If `if_match is None` OR `if_match.strip('"') != current_etag.strip('"')`:
         Raise `HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail="Precondition Failed: Resource has been modified by another client.")`.
    5. Update dataset fields:
       `dataset.name = request.name`
       `dataset.description = request.description`
       `dataset.version += 1`
       `dataset.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())`
       `dataset.etag = compute_dataset_etag(dataset)`
    6. Return `JSONResponse(content=dataset.model_dump(), headers={"ETag": dataset.etag})`.
    ========================================================================
    """
    # --- IMPLEMENTATION GOES HERE ---
    raise NotImplementedError(
        "Exercise 2: Implement update_dataset with If-Match in backend/main.py"
    )


# ---------------------------------------------------------------------------
# General Listing, Chaos & Telemetry Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/datasets")
async def list_datasets():
    """List all registered datasets."""
    return list(datasets_db.values())


@app.get("/api/chaos/config")
async def get_chaos_config():
    """Get active network chaos configuration."""
    return chaos_settings


@app.post("/api/chaos/config")
async def update_chaos_config(config: ChaosConfig):
    """Update network chaos configuration."""
    global chaos_settings
    chaos_settings = config
    return {"message": "Chaos configuration updated", "config": chaos_settings}


@app.get("/api/chaos/telemetry")
async def get_telemetry():
    """Return backend idempotency records and active datasets for UI telemetry."""
    return {
        "idempotency_records": idempotency.get_all_records(),
        "total_datasets": len(datasets_db),
        "chaos_config": chaos_settings.model_dump(),
    }


@app.post("/api/chaos/reset")
async def reset_chaos_and_state():
    """Reset datasets to initial seed and flush idempotency store."""
    idempotency.clear_idempotency_store()
    seed_sample_datasets()
    global chaos_settings
    chaos_settings = ChaosConfig(drop_post_commit_rate=0.0, delay_ms=0)
    return {"message": "State reset successfully"}
