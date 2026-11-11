"""向量检索的读模型与端口。

向量表以证据段标识为键，只存向量，不存文献名、页码与原文：召回只负责
「哪些证据段与问题相近」，出处由调用方回图存储按标识补上。这样两边的元数据
只有一个事实源，引用不会因为向量表落伍而指错页码。

召回是顺序扫描（ADR-0005 的修订：6806 条向量的近似索引收益现在无法验证），
``distance`` 是余弦距离，越小越相近，结果按它升序。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class VectorMatch:
    """一条向量召回结果：证据段标识与到查询向量的余弦距离。"""

    evidence_id: str
    distance: float


class VectorReader(Protocol):
    """向量检索端口：给一个查询向量，取最相近的若干证据段。"""

    def search_similar(
        self, vector: Sequence[float], *, limit: int
    ) -> tuple[VectorMatch, ...]:
        """按余弦距离升序返回至多 ``limit`` 条；库里没有向量时返回空。"""
        ...
