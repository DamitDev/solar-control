from .parser import parse, RepoURI, HuggingFaceURI, LocalURI
from .huggingface import resolve_huggingface
from .repo import resolve_repo


async def resolve(uri_str: str, host_url: str, host_api_key: str) -> str:
    """
    Parses a URI and dispatches to the correct resolver.
    Returns a resolved local:// URI.
    """
    parsed = parse(uri_str)

    if isinstance(parsed, LocalURI):
        # local:// is passed through to the host to be validated there
        return uri_str

    elif isinstance(parsed, HuggingFaceURI):
        return await resolve_huggingface(parsed, uri_str, host_url, host_api_key)

    elif isinstance(parsed, RepoURI):
        return await resolve_repo(parsed, uri_str, host_url, host_api_key)

    else:
        # Should not happen if parser is correct
        return uri_str
