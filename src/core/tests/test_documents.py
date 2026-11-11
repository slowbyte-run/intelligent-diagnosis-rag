"""文献标识：入库与检索必须用同一条内容哈希规则。"""

from __future__ import annotations

from idr_core.documents import DOCUMENT_ID_LENGTH, document_id


def test_document_id_is_the_short_content_hash() -> None:
    # 期望值由 sha256("paper") 的前 16 位十六进制独立算出；图谱里已有的
    # doc_id 都按这条规则生成，改动它等于让全部引用失效。
    assert document_id(b"paper") == "382635c9325bf327"
    assert len(document_id(b"paper")) == DOCUMENT_ID_LENGTH


def test_document_id_depends_only_on_content() -> None:
    assert document_id(b"same bytes") == document_id(b"same bytes")
    assert document_id(b"same bytes") != document_id(b"other bytes")
