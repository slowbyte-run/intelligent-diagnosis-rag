"""故障模式的 location / damage_form：抽取、本体校验与持久化。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from idr_core.models import BBox
from idr_core.parsing import ParsedPage
from idr_rotating import IngestDeps, ingest
from idr_rotating.pending import PendingKind, RejectionReason

from fakes import FakeEmbedder, FakeGraphStore, FakeLLM, FakeParser, FakeVectorStore

TEXT = "本体里的故障模式带有发生部位与损伤形式两个属性。"
CONTENT = b"%PDF-1.4 attribute fixture"


def make_deps(
    entities: Sequence[dict[str, object]], *, graph_store: FakeGraphStore
) -> IngestDeps:
    response = json.dumps(
        {"entities": list(entities), "relations": []}, ensure_ascii=False
    )
    return IngestDeps(
        llm=FakeLLM(response),
        parser=FakeParser(
            [ParsedPage(number=1, text=TEXT, bbox=BBox(0.0, 0.0, 595.0, 842.0))]
        ),
        graph_store=graph_store,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )


def entity(
    name: str,
    *,
    location: str | None = None,
    damage_form: str | None = None,
) -> dict[str, object]:
    document_id = hashlib.sha256(CONTENT).hexdigest()[:16]
    payload: dict[str, object] = {
        "name": name,
        "type": "fault_mode",
        "evidence_ids": [f"ev:{document_id}:1:0-{len(TEXT)}"],
        "confidence": 0.9,
    }
    if location is not None:
        payload["location"] = location
    if damage_form is not None:
        payload["damage_form"] = damage_form
    return payload


def test_extracted_attributes_are_normalised_and_persisted(
    tmp_path: Path,
) -> None:
    (tmp_path / "paper.pdf").write_bytes(CONTENT)
    graph_store = FakeGraphStore()

    report = ingest(
        tmp_path,
        deps=make_deps(
            [entity("新鲜剥落", location="外圈", damage_form="磨损")],
            graph_store=graph_store,
        ),
    )

    document = report.documents[0]
    assert document.error is None
    assert document.entities == 1

    (concept,) = graph_store.batches[0].concepts
    # 提示词同时给出标识与标签，两者都要接受；持久化时统一用本体标识。
    assert concept.attributes == {
        "location": "outer_race",
        "damage_form": "wear",
    }


def test_canonical_instance_supplies_attributes_the_model_omitted(
    tmp_path: Path,
) -> None:
    (tmp_path / "paper.pdf").write_bytes(CONTENT)
    graph_store = FakeGraphStore()

    ingest(
        tmp_path,
        deps=make_deps([entity("外圈剥落")], graph_store=graph_store),
    )

    (concept,) = graph_store.batches[0].concepts
    assert concept.name == "轴承外圈疲劳剥落"
    assert concept.attributes == {
        "location": "outer_race",
        "damage_form": "rolling_contact_fatigue",
    }


@pytest.mark.parametrize(
    ("attribute", "invalid_value"),
    [("location", "外太空"), ("damage_form", "未知损伤形式")],
)
def test_attribute_value_outside_the_ontology_goes_to_pending_review(
    tmp_path: Path,
    attribute: str,
    invalid_value: str,
) -> None:
    (tmp_path / "paper.pdf").write_bytes(CONTENT)
    graph_store = FakeGraphStore()
    payload: dict[str, object] = entity("外圈剥落")
    payload[attribute] = invalid_value

    report = ingest(
        tmp_path, deps=make_deps([payload], graph_store=graph_store)
    )

    document = report.documents[0]
    assert document.error is None, "本体越界是待审数据，不是整篇失败"
    assert document.entities == 0
    assert document.rejected_entities == 1
    assert graph_store.batches[0].concepts == ()

    (review,) = document.pending_reviews
    assert review.kind is PendingKind.ENTITY
    assert review.reason is RejectionReason.ATTRIBUTE_NOT_IN_ONTOLOGY
    assert review.value == f"fault_mode.{attribute}"
    assert invalid_value in review.label
    assert review.examples[0].text == TEXT
