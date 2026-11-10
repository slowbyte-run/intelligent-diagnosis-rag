"""嵌入端口。向量写入在后续票接入，本票只保留注入点。"""

from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
