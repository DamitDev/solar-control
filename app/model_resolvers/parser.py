from dataclasses import dataclass
from typing import Union
from fastapi import HTTPException


@dataclass(frozen=True)
class RepoURI:
    name: str
    version: str
    subpath: str = ""
    scheme: str = "repo"


@dataclass(frozen=True)
class HuggingFaceURI:
    model_id: str
    scheme: str = "huggingface"


@dataclass(frozen=True)
class LocalURI:
    path: str
    scheme: str = "local"


ParsedURI = Union[RepoURI, HuggingFaceURI, LocalURI]


def parse(uri: str) -> ParsedURI:
    """
    Parses a model source URI according to the spec Section 2.4.

    repo://{name}:{version}
    huggingface://{model_id}
    local://{path}
    """
    if not uri:
        raise HTTPException(status_code=400, detail="Empty model source URI")

    if uri.startswith("repo://"):
        content = uri[len("repo://") :]
        if ":" not in content:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid repo URI: '{uri}'. Missing version. Expected 'repo://name:version'",
            )
        name, version = content.split(":", 1)
        if not name or not version:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid repo URI: '{uri}'. Name and version must be non-empty.",
            )
        # Optional subpath: repo://name:version/subpath selects a file inside
        # the artifact directory (e.g. a GGUF file). It is resolved by the
        # host after the pull; Data Repository lookup uses name:version only.
        subpath = ""
        if "/" in version:
            version, subpath = version.split("/", 1)
            if not version or not subpath:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid repo URI: '{uri}'. "
                        "Version and subpath must be non-empty."
                    ),
                )
        return RepoURI(name=name, version=version, subpath=subpath)

    elif uri.startswith("huggingface://"):
        model_id = uri[len("huggingface://") :]
        if not model_id:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid huggingface URI: '{uri}'. Missing model_id.",
            )
        return HuggingFaceURI(model_id=model_id)

    elif uri.startswith("local://"):
        # triple slash for absolute, double slash for relative per spec
        # we just capture the path part
        path = uri[len("local://") :]
        if not path:
            raise HTTPException(
                status_code=400, detail=f"Invalid local URI: '{uri}'. Missing path."
            )
        return LocalURI(path=path)

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model source scheme in URI: '{uri}'. Supported: repo://, huggingface://, local://",
        )
