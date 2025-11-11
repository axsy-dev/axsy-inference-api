"""JWT authentication module for Axsy Inference API."""
import os
import logging
from functools import lru_cache
from typing import Optional, Dict, Any

import jwt
from google.cloud import secretmanager
from google.auth import default

logger = logging.getLogger("uvicorn.error")

# Check if JWT verification is enabled via environment variable
JWT_VERIFICATION_ENABLED = os.getenv("JWT_VERIFICATION_ENABLED", "false").lower() in ("true", "1", "yes", "on")


def get_project_id() -> str:
    """Get the GCP project ID from environment or default credentials."""
    project_id = os.getenv("PROJECT_ID") or os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if project_id:
        logger.debug(f"Project ID from env: {project_id}")
        return project_id
    
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

