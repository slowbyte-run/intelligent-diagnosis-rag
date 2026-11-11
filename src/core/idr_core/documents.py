"""文献标识与文献名。

文献的稳定标识由内容哈希推导：同一份 PDF 不管放在哪个目录、被处理几次，标识
都不变。入库据此断点续跑，检索据此把引用对回文献名——两边必须用同一条规则，
否则图谱里的 ``doc_id`` 与目录里的文件对不上。

文献名只用于展示与引用，不进入图谱：图谱里存的是标识，名字由调用方按标识从
语料目录解析，检索因此不依赖文件名。
"""

from __future__ import annotations

from hashlib import sha256
from typing import Protocol

DOCUMENT_ID_LENGTH = 16


def document_id(content: bytes) -> str:
    """由文献内容推导稳定标识；长度固定，便于在图与日志里阅读。"""
    return sha256(content).hexdigest()[:DOCUMENT_ID_LENGTH]


class DocumentNames(Protocol):
    """按标识查文献名；查不到时返回 ``None``，不编造名字。"""

    def name_for(self, doc_id: str) -> str | None: ...
