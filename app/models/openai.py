from pydantic import BaseModel, Field, ConfigDict
from typing import Any
from datetime import datetime, timezone


class ModelInfo(BaseModel):
    """Model information for OpenAI compatibility"""

    id: str
    object: str = "model"
    created: int = Field(
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp())
    )
    owned_by: str = "solar"


class ModelsResponse(BaseModel):
    """OpenAI /v1/models response"""

    object: str = "list"
    data: list[ModelInfo]


class ChatMessage(BaseModel):
    """Chat message"""

    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = None
    name: str | None = None


class StreamOptions(BaseModel):
    """OpenAI stream_options for including usage in streaming responses"""

    model_config = ConfigDict(extra="allow")

    include_usage: bool | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request"""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool | None = False
    stream_options: StreamOptions | None = None
    n: int | None = None
    stop: list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None


class CompletionRequest(BaseModel):
    """OpenAI completion request"""

    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool | None = False
    n: int | None = None
    stop: list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    user: str | None = None


class ProxyRequest(BaseModel):
    """Generic proxy request"""

    endpoint: str
    method: str = "POST"
    data: dict[str, Any] | None = None


class ClassifyRequest(BaseModel):
    """Classification request for HuggingFace classification models"""

    model_config = ConfigDict(extra="allow")

    model: str
    input: Any = Field(..., description="Text or list of texts to classify")
    return_all_scores: bool = Field(
        default=False,
        description="Return scores for all classes, not just top prediction",
    )


class ClassifyScoreItem(BaseModel):
    """Individual class score."""

    label: str
    score: float


class ClassifyChoice(BaseModel):
    """Classification result for a single input"""

    index: int
    label: str
    score: float
    all_scores: list[ClassifyScoreItem] | None = Field(
        default=None, description="Scores for all classes (when return_all_scores=True)"
    )


class ClassifyResponse(BaseModel):
    """Classification response"""

    id: str
    object: str = "classification"
    model: str
    choices: list[ClassifyChoice]
    usage: dict[str, int]


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request for HuggingFace embedding models"""

    model_config = ConfigDict(extra="allow")

    model: str
    input: Any = Field(..., description="Text or list of texts to embed")
    encoding_format: str | None = Field(
        default="float", description="Encoding format: 'float' or 'base64'"
    )
    dimensions: int | None = Field(
        default=None, description="Optional dimension truncation"
    )


class EmbeddingData(BaseModel):
    """Individual embedding result"""

    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response"""

    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: dict[str, int]


class RerankRequest(BaseModel):
    """OpenAI-compatible rerank request for reranker models"""

    model_config = ConfigDict(extra="allow")

    model: str
    query: str = Field(..., description="Query text to rank documents against")
    documents: list[str] = Field(..., description="List of documents to rerank")
    top_n: int | None = Field(
        default=None, description="Number of top results to return (default: all)"
    )
    return_documents: bool | None = Field(
        default=True, description="Whether to return document text in results"
    )


class RerankResult(BaseModel):
    """Individual rerank result"""

    index: int
    document: str | None = None
    relevance_score: float


class RerankResponse(BaseModel):
    """OpenAI-compatible rerank response"""

    id: str
    results: list[RerankResult]
    model: str
    usage: dict[str, int]
