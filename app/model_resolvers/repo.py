from fastapi import HTTPException
from .parser import RepoURI


async def resolve_repo(
    uri: RepoURI, source_uri: str, host_url: str, host_api_key: str
) -> str:
    """
    Stub for repo:// resolver.

    Expected behavior in Phase 1 (D-016):
    1. Call Data Repository GET /api/resolve?uri={source_uri} to obtain a harbor_ref.
    2. Pull the OCI artifact from Harbor using ORAS (harbor-oci-client) into the
       host's managed models directory.
    3. Return the resolved local:// path.
    """
    raise HTTPException(
        status_code=501,
        detail=(
            f"repo:// resolver not yet available. Data Repository integration "
            f"will be completed in Phase 1. Use local:// or huggingface:// for now. "
            f"URI: {source_uri}"
        ),
    )
