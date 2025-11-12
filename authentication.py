"""JWT authentication module for Axsy Inference API."""
import os
import json
import logging
from functools import lru_cache
from typing import Optional, Dict, Any

import jwt
from google.cloud import secretmanager
from google.auth import default
from google.oauth2 import service_account

logger = logging.getLogger("uvicorn.error")

# Check if JWT verification is enabled via environment variable
JWT_VERIFICATION_ENABLED = os.getenv("JWT_VERIFICATION_ENABLED", "false").lower() in ("true", "1", "yes", "on")


def get_project_id() -> str:
    """Get the GCP project ID from environment, service account key, or default credentials."""
    project_id = os.getenv("PROJECT_ID") or os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if project_id:
        logger.debug(f"Project ID from env: {project_id}")
        return project_id
    
    # Try to get project ID from service account key in Secret Manager
    # Use env vars to avoid circular dependency with get_project_id()
    try:
        logger.debug("Fetching project ID from service account key in Secret Manager...")
        # Get project from env vars first to avoid circular dependency
        temp_project = get_project_name() or os.getenv("PROJECT_ID") or os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or "smart-vision-training"
        sa_key_dict = get_service_account_key_sync(project=temp_project)
        project_id = sa_key_dict.get("project_id")
        if project_id:
            logger.debug(f"Project ID from service account key: {project_id}")
            return project_id
    except Exception as e:
        logger.debug(f"Could not get project ID from Secret Manager: {e}")
    
    # Fallback to default credentials
    try:
        logger.debug("Fetching project ID using GoogleAuth...")
        _, project_id = default()
        if project_id:
            logger.debug(f"Project ID fetched: {project_id}")
            return project_id
    except Exception as e:
        logger.error(f"Error fetching project ID: {e}")
    
    # Fallback default
    default_project = "smart-vision-training"
    logger.warning(f"Using default project ID: {default_project}")
    return default_project


def get_project_name() -> Optional[str]:
    """Get the project name from environment variable."""
    project_name = os.getenv("PROJECT_NAME")
    if project_name:
        logger.debug(f"Project name fetched: {project_name}")
    return project_name


@lru_cache(maxsize=1)
def _get_secret_manager_client():
    """Get or create a Secret Manager client (cached)."""
    return secretmanager.SecretManagerServiceClient()


async def get_jwt_public_key() -> str:
    """Get the JWT public key from Secret Manager.
    
    Uses PROJECT_NAME environment variable if set, otherwise falls back to PROJECT_ID.
    This function runs the synchronous Secret Manager call in a thread pool to avoid blocking.
    """
    from fastapi.concurrency import run_in_threadpool
    return await run_in_threadpool(get_jwt_public_key_sync)


async def verify_jwt_token(token: str) -> Dict[str, Any]:
    """Verify a JWT token using the public key from Secret Manager.
    
    Args:
        token: The JWT token string to verify
        
    Returns:
        Decoded token payload as a dictionary
        
    Raises:
        jwt.ExpiredSignatureError: If token is expired
        jwt.InvalidTokenError: If token is invalid
        Exception: For other verification errors
    """
    try:
        logger.debug(f"Verifying JWT token: {token[:20]}...")
        token = token.strip()
        public_key = await get_jwt_public_key()
        decoded = jwt.decode(token, public_key, algorithms=["RS256"])
        logger.debug(f"JWT token successfully verified")
        return decoded
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        raise ValueError("Token expired. Please log in again.")
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise ValueError("Invalid token. Please log in again.")
    except Exception as e:
        logger.error(f"JWT verification error: {e}")
        raise ValueError("An error occurred during token verification.")


def get_jwt_public_key_sync() -> str:
    """Synchronous version of get_jwt_public_key for use in non-async contexts or thread pools."""
    project_name = get_project_name()
    project_id = get_project_id()
    project = project_name or project_id
    
    logger.debug(f"Getting JWT public key from Secret Manager for project: {project}")
    
    try:
        client = _get_secret_manager_client()
        secret_name = f"projects/{project}/secrets/axsy_jwt_pub/versions/latest"
        
        response = client.access_secret_version(request={"name": secret_name})
        public_key = response.payload.data.decode("utf-8")
        logger.debug("Successfully retrieved JWT public key")
        return public_key
    except Exception as e:
        logger.error(f"Failed to get JWT public key from Secret Manager: {e}")
        raise


def get_service_account_key_sync(project: Optional[str] = None) -> Dict[str, Any]:
    """Get the service account key from Secret Manager.
    
    Args:
        project: Optional project ID/name. If not provided, will try to get from env vars
                or use default. Avoids calling get_project_id() to prevent circular dependency.
    
    Returns:
        Service account key as a dictionary (JSON parsed)
        
    Raises:
        Exception: If the secret cannot be retrieved or parsed
    """
    if not project:
        # Try to get project from env vars first to avoid circular dependency
        project = get_project_name() or os.getenv("PROJECT_ID") or os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or "smart-vision-training"
    
    logger.debug(f"Getting service account key from Secret Manager for project: {project}")
    
    try:
        client = _get_secret_manager_client()
        secret_name = f"projects/{project}/secrets/service_account_key/versions/latest"
        
        response = client.access_secret_version(request={"name": secret_name})
        sa_key_json = response.payload.data.decode("utf-8")
        sa_key_dict = json.loads(sa_key_json)
        logger.debug("Successfully retrieved service account key from Secret Manager")
        return sa_key_dict
    except Exception as e:
        logger.error(f"Failed to get service account key from Secret Manager: {e}")
        raise


@lru_cache(maxsize=1)
def get_google_credentials():
    """Get Google Auth credentials, preferring service account key from Secret Manager.
    
    This function caches the credentials to avoid repeated Secret Manager calls.
    Falls back to default credentials if Secret Manager access fails.
    """
    # First, try to get service account key from Secret Manager
    try:
        sa_key_dict = get_service_account_key_sync()
        credentials = service_account.Credentials.from_service_account_info(sa_key_dict)
        logger.debug("Using service account credentials from Secret Manager")
        return credentials
    except Exception as e:
        logger.warning(f"Could not get service account key from Secret Manager: {e}. Falling back to default credentials.")
        try:
            credentials, _ = default()
            logger.debug("Using default application credentials")
            return credentials
        except Exception as e2:
            logger.error(f"Failed to get default credentials: {e2}")
            raise

