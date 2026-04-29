from fastapi import HTTPException
from .parser import RepoURI


async def resolve_repo(
    uri: RepoURI, source_uri: str, host_url: str, host_api_key: str
) -> str:
    """
    Stub for repo:// resolver. Will be implemented in S-013.
    """
    raise HTTPException(
        status_code=502,
        detail=f"repo:// resolver not yet available (S-013 pending). URI: {source_uri}",
    )
