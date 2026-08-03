from .parser import parse, RepoURI, HuggingFaceURI, LocalURI
from .huggingface import resolve_huggingface
from .repo import resolve_repo


async def resolve(
    uri_str: str,
    host_url: str,
    host_api_key: str,
    backend_type: str | None = None,
) -> str:
    """
    Parses a URI and dispatches to the correct resolver.
    Returns a resolved local:// URI.

    ``backend_type`` is forwarded to the host pull for ``repo://`` URIs so
    llama.cpp artifacts resolve to their largest ``*.gguf`` (the host needs
    a file, not a directory).  ``local://`` and ``huggingface://`` are never
    affected.
    """
    parsed = parse(uri_str)

    if isinstance(parsed, LocalURI):
        # local:// is passed through to the host to be validated there
        return uri_str

    elif isinstance(parsed, HuggingFaceURI):
        return await resolve_huggingface(parsed, uri_str, host_url, host_api_key)

    elif isinstance(parsed, RepoURI):
        return await resolve_repo(uri_str, host_url, host_api_key, backend_type)

    else:
        # Should not happen if parser is correct
        return uri_str
