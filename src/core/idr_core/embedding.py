"""嵌入端口。"""

from collections.abc import Sequence
from typing import Protocol

EMBEDDING_DIM = 1024


class EmbeddingError(RuntimeError):
    """嵌入服务返回的结果与请求不符。"""


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
